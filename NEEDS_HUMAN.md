# Needs Human Input

Items that require manual action (API keys, paid services, permissions).
When you're back, review and resolve these — then mark them [x].

## Pending

- [x] **Real Fama-French factor data** from Ken French's data library
  - **Resolved**: v0.4.3 — `fetch_french_factors()` auto-downloads from Ken French's website
  - **Fallback**: If download fails, `generate_synthetic_factors()` is used as backup
  - **Added**: 2026-03-18 | **Resolved**: 2026-03-18

