"""Tests for budget enforcement."""

import pytest

from trading_master.budget import BudgetExceededError, CostBudget


def test_record_accumulates():
    budget = CostBudget(max_cost_usd=10.0, warn_cost_usd=5.0, max_tokens=100_000)
    budget.record(tokens=1000, cost=0.50)
    budget.record(tokens=2000, cost=0.75)
    assert budget.accumulated_cost == pytest.approx(1.25)
    assert budget.accumulated_tokens == 3000
    assert budget.call_count == 2


def test_budget_exceeded_cost():
    budget = CostBudget(max_cost_usd=1.0, warn_cost_usd=0.5, max_tokens=1_000_000)
    budget.record(tokens=100, cost=0.80)
    with pytest.raises(BudgetExceededError, match="Cost budget exceeded"):
        budget.record(tokens=100, cost=0.30)


def test_budget_exceeded_tokens():
    budget = CostBudget(max_cost_usd=100.0, warn_cost_usd=50.0, max_tokens=1000)
    budget.record(tokens=800, cost=0.01)
    with pytest.raises(BudgetExceededError, match="Token budget exceeded"):
        budget.record(tokens=300, cost=0.01)


def test_remaining_budget():
    budget = CostBudget(max_cost_usd=5.0, warn_cost_usd=2.0, max_tokens=100_000)
    assert budget.remaining_budget() == 5.0
    budget.record(tokens=100, cost=1.50)
    assert budget.remaining_budget() == pytest.approx(3.50)
    budget.record(tokens=100, cost=3.00)
    assert budget.remaining_budget() == pytest.approx(0.50)


def test_remaining_budget_does_not_go_negative():
    budget = CostBudget(max_cost_usd=1.0, warn_cost_usd=0.5, max_tokens=1_000_000)
    budget.accumulated_cost = 2.0  # Force over budget
    assert budget.remaining_budget() == 0.0


def test_summary():
    budget = CostBudget(max_cost_usd=10.0, warn_cost_usd=5.0, max_tokens=100_000)
    budget.record(tokens=500, cost=0.25)
    s = budget.summary()
    assert s["cost"] == pytest.approx(0.25)
    assert s["tokens"] == 500
    assert s["calls"] == 1
    assert s["remaining"] == pytest.approx(9.75)


def test_estimate_run_cost():
    budget = CostBudget()
    # gpt-4o-mini: input=$0.15/M, output=$0.60/M
    est = budget.estimate_run_cost(n_tickers=1, model="gpt-4o-mini")
    assert est > 0
    # 1 ticker, 7 calls, 1500 tok/call = 10500 tokens
    # 60% input = 6300, 40% output = 4200
    # cost = (6300/1M)*0.15 + (4200/1M)*0.60 = 0.000945 + 0.00252 = ~0.003465
    assert est == pytest.approx(0.003465, rel=0.01)


def test_estimate_run_cost_scales_with_tickers():
    budget = CostBudget()
    est1 = budget.estimate_run_cost(n_tickers=1, model="gpt-4o-mini")
    est5 = budget.estimate_run_cost(n_tickers=5, model="gpt-4o-mini")
    assert est5 == pytest.approx(est1 * 5, rel=0.01)


def test_estimate_run_cost_unknown_model():
    budget = CostBudget()
    # Unknown model should use fallback rates and not crash
    est = budget.estimate_run_cost(n_tickers=1, model="unknown-model-xyz")
    assert est > 0
