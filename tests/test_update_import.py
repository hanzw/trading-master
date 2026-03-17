"""Tests for portfolio update import: parser and diff logic."""

import pytest

from trading_master.portfolio.update_import import (
    parse_portfolio_text,
    diff_portfolio,
    apply_portfolio_update,
)


# ── Parser tests ────────────────────────────────────────────────────


SAMPLE_FIDELITY = """\
AAPL
Apple Inc
150.5000
$25,234.50
$167.67
+1.23%
+$2.05
MSFT
Microsoft Corporation
50
$21,000.00
$420.00
-0.50%
-$2.10
GOOGL
Alphabet Inc Class A
25.0000
$4,375.00
$175.00
+0.75%
+$1.31
"""

SAMPLE_WITH_CASH = """\
AAPL
Apple Inc
100
$17,500.00
$175.00
+1.00%
+$1.75
USD
$5,432.10
"""

SAMPLE_BRK = """\
BRK.B
Berkshire Hathaway Inc Cl B
10
$4,500.00
$450.00
+0.20%
+$0.90
"""

SAMPLE_FCASH = """\
TSLA
Tesla Inc
20
$5,000.00
$250.00
+2.00%
+$5.00
FCASH
Fidelity Cash
$10,000.00
"""

SAMPLE_NA_PRICE = """\
VTI
Vanguard Total Stock Market ETF
100
$22,500.00
N/A
+0.50%
+$1.12
"""


def test_parse_basic_format():
    positions = parse_portfolio_text(SAMPLE_FIDELITY)
    assert len(positions) == 3

    aapl = next(p for p in positions if p["ticker"] == "AAPL")
    assert aapl["shares"] == 150.5
    assert aapl["price"] == 167.67
    assert aapl["value"] == 25234.50

    msft = next(p for p in positions if p["ticker"] == "MSFT")
    assert msft["shares"] == 50
    assert msft["price"] == 420.00

    googl = next(p for p in positions if p["ticker"] == "GOOGL")
    assert googl["shares"] == 25.0


def test_parse_cash_usd():
    positions = parse_portfolio_text(SAMPLE_WITH_CASH)
    stocks = [p for p in positions if p["ticker"] != "_CASH"]
    cash = [p for p in positions if p["ticker"] == "_CASH"]

    assert len(stocks) == 1
    assert stocks[0]["ticker"] == "AAPL"
    assert len(cash) == 1
    assert cash[0]["value"] == 5432.10


def test_parse_brk_b_conversion():
    positions = parse_portfolio_text(SAMPLE_BRK)
    assert len(positions) == 1
    assert positions[0]["ticker"] == "BRK-B"
    assert positions[0]["shares"] == 10


def test_parse_fcash():
    positions = parse_portfolio_text(SAMPLE_FCASH)
    stocks = [p for p in positions if p["ticker"] != "_CASH"]
    cash = [p for p in positions if p["ticker"] == "_CASH"]

    assert len(stocks) == 1
    assert stocks[0]["ticker"] == "TSLA"
    assert len(cash) == 1
    assert cash[0]["value"] == 10000.00


def test_parse_na_price():
    """Should handle N/A price by calculating from value/shares."""
    positions = parse_portfolio_text(SAMPLE_NA_PRICE)
    assert len(positions) == 1
    vti = positions[0]
    assert vti["ticker"] == "VTI"
    assert vti["shares"] == 100
    assert vti["value"] == 22500.00
    # Price should be calculated from value/shares
    assert abs(vti["price"] - 225.0) < 0.01


def test_parse_empty():
    assert parse_portfolio_text("") == []
    assert parse_portfolio_text("  \n  \n  ") == []


# ── Diff tests ──────────────────────────────────────────────────────


def test_diff_added():
    current = {}
    new = [{"ticker": "AAPL", "shares": 100, "value": 17500, "price": 175.0}]
    result = diff_portfolio(current, new)
    assert len(result["added"]) == 1
    assert result["added"][0]["ticker"] == "AAPL"
    assert result["removed"] == []
    assert result["changed"] == []


def test_diff_removed():
    current = {"AAPL": {"quantity": 100, "avg_cost": 150.0}}
    new = []
    result = diff_portfolio(current, new)
    assert len(result["removed"]) == 1
    assert result["removed"][0]["ticker"] == "AAPL"
    assert result["added"] == []


def test_diff_changed():
    current = {"AAPL": {"quantity": 100, "avg_cost": 150.0}}
    new = [{"ticker": "AAPL", "shares": 120, "value": 21000, "price": 175.0}]
    result = diff_portfolio(current, new)
    assert len(result["changed"]) == 1
    assert result["changed"][0]["diff"] == 20.0
    assert result["changed"][0]["old_shares"] == 100
    assert result["changed"][0]["new_shares"] == 120


def test_diff_unchanged():
    current = {"AAPL": {"quantity": 100, "avg_cost": 150.0}}
    new = [{"ticker": "AAPL", "shares": 100, "value": 17500, "price": 175.0}]
    result = diff_portfolio(current, new)
    assert result["added"] == []
    assert result["removed"] == []
    assert result["changed"] == []
    assert len(result["unchanged"]) == 1


def test_diff_mixed():
    current = {
        "AAPL": {"quantity": 100, "avg_cost": 150.0},
        "GOOGL": {"quantity": 50, "avg_cost": 160.0},
        "TSLA": {"quantity": 30, "avg_cost": 200.0},
    }
    new = [
        {"ticker": "AAPL", "shares": 120, "value": 21000, "price": 175.0},  # changed +20
        {"ticker": "GOOGL", "shares": 50, "value": 8750, "price": 175.0},   # unchanged
        {"ticker": "MSFT", "shares": 25, "value": 10500, "price": 420.0},   # added
        # TSLA removed
    ]
    result = diff_portfolio(current, new)
    assert len(result["added"]) == 1
    assert result["added"][0]["ticker"] == "MSFT"
    assert len(result["removed"]) == 1
    assert result["removed"][0]["ticker"] == "TSLA"
    assert len(result["changed"]) == 1
    assert result["changed"][0]["ticker"] == "AAPL"
    assert len(result["unchanged"]) == 1
    assert result["unchanged"][0]["ticker"] == "GOOGL"


def test_diff_ignores_cash():
    """Cash entries should not appear in the diff."""
    current = {"AAPL": {"quantity": 100, "avg_cost": 150.0}}
    new = [
        {"ticker": "AAPL", "shares": 100, "value": 17500, "price": 175.0},
        {"ticker": "_CASH", "shares": 0, "value": 5000, "price": 0},
    ]
    result = diff_portfolio(current, new)
    assert result["added"] == []
    assert result["removed"] == []
    assert result["changed"] == []
    assert len(result["unchanged"]) == 1


def test_diff_decreased_shares():
    current = {"AAPL": {"quantity": 100, "avg_cost": 150.0}}
    new = [{"ticker": "AAPL", "shares": 80, "value": 14000, "price": 175.0}]
    result = diff_portfolio(current, new)
    assert len(result["changed"]) == 1
    assert result["changed"][0]["diff"] == -20.0
