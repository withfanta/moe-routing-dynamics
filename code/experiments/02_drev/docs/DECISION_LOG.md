# DREV-P0 Decision Log

One line per decision. The result decision is appended once, after the formal run,
and is not reinterpreted afterwards.

## Scaffold

- **Project created.** `delayed_rejected_evidence` initialized as a fresh
  repository with a minimal structure. No research tree, no future-method roadmap,
  no dormant architecture nodes.
- **REDV treated as read-only.** Validated implementation pieces were copied from
  `/home/h-li/work/rejected_expert_delayed_value` (commits recorded in
  `README.md`); that repository was not edited and its conclusions were not
  imported. DREV-P0 is not REDV-V3.
- **Charter accepted as locked.** The user-locked question, fixed source layer and
  horizons, primary quantity `D_m`, interpretation limit, and out-of-scope list in
  `docs/RESEARCH_CHARTER.md` are frozen.

## Result (final)

- **DREV-P0 does not show that Layer-4 rejected evidence becomes more informative
  about routing regret at later depths than at the immediate next layer.** Verdict
  NOT_PROMISING under the frozen four-condition rule, with all four conditions
  failing: 0 of 3 delayed horizons have R2_real > 0; 0 of 3 have D > 0;
  mean_D_delayed = -0.05931 < +0.02; and mean_D_delayed (-0.05931) is not greater
  than D_immediate (-0.04340). At every horizon the equal-dimension derangement
  control predicted regret at least as well as the correctly matched rejected
  evidence, and D was most negative at the largest distance (Δ = 8). The
  delayed-rejected-evidence hypothesis is stopped; no DREV-P1, no added layers or
  horizons, no PCA/alpha/n changes, no memory or attention built.
- **Measurement fix confirmed effective, independent of the outcome.** The frozen
  StandardScaler+PCA(64) compression put the probes at p = 128 against n = 512,
  out of the p >> n regime; held-out R² values are near zero rather than large and
  negative, so the matched comparison is interpretable. Routing regret was present
  at every horizon (mean G 0.047-0.054, positive for 74-78% of samples), so the
  null result is not caused by a degenerate target.
