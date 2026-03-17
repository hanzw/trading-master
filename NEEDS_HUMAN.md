# Needs Human Input

Items that require manual action (API keys, paid services, permissions).
When you're back, review and resolve these — then mark them [x].

## Pending

- [ ] **Real Fama-French factor data** from Ken French's data library
  - **Why**: `tm quant ff5` currently uses synthetic/random factor data — results are illustrative only, not real factor exposures
  - **Free alternative used**: `generate_synthetic_factors()` produces realistic-looking but random factor returns
  - **How to fix**: Download CSV from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html, parse into (n, 5) array, cache locally
  - **Added**: 2026-03-18

