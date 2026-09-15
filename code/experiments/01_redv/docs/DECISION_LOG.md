# REDV-V1 Decision Log

One line per decision. Result decisions are appended exactly once, after the
formal run, and are not reinterpreted afterwards.

## Scaffold

- **Project created.** `rejected_expert_delayed_value` initialized as a fresh
  repository with a minimal structure. No research tree, no dormant future-method
  nodes, no roadmap documents. ResearchOps stays smaller than the experiment.
- **Charter accepted as locked.** The user-locked question, primary comparison,
  primary outcome, and out-of-scope list in `docs/RESEARCH_CHARTER.md` are frozen
  and are not subject to agent reinterpretation.

## Result

- **REDV-V1 supports delayed predictive value of rejected near-miss expert
  evidence under the frozen OLMoE/WikiText setting.** Verdict SUPPORTED by the
  pre-registered rule: mean Delta_R2 = +0.47406 (4->5 +0.82645, 8->9 +0.44654,
  12->13 +0.14919), all three > 0 and mean >= +0.02. The prerequisite phenomenon
  is supported. Direction stops here; no method is implemented and a later
  experiment requires explicit user authorization.
- **Caveat recorded, verdict unchanged.** All six held-out R² values are negative
  (both probes worse than predicting the test mean), and a post-hoc control found
  that 2048 columns of scale-matched random noise yield a larger Delta_R2 than the
  real rejected-evidence vector at every transition. Under this protocol Delta_R2
  does not isolate incremental information. The frozen rule and thresholds were
  not modified and no rescue experiment was run.

## REDV-V2 result (final)

- **REDV-V2 does not support sample-specific delayed predictive value of rejected
  near-miss expert evidence under the frozen OLMoE/WikiText setting with a
  dimension-matched shuffled control.** Verdict NOT_SUPPORTED by the pre-registered
  rule, with all three conditions failing: every R2_real is negative
  (-0.94185, -0.49803, -0.27184); D is negative at 4->5 (-0.19634) and 8->9
  (-0.13224) with only +0.02939 at 12->13; and mean D = -0.09973 < +0.02. On the
  two earlier transitions rejected evidence from a *different* sample predicted
  regret better than a sample's own. The entire rejected-expert delayed-value
  direction is stopped. No REDV-V3, no rescue, no method implemented.
- **REDV-V1 left untouched.** Its SUPPORTED verdict and artifacts remain exactly as
  recorded; REDV-V2 read them only. The confound REDV-V2 was authorized to repair
  is confirmed as the explanation for REDV-V1's apparent positive result: the
  `real - baseline` gain reappeared (+2.44, +1.02, +0.79) and disappeared entirely
  against the equal-dimension control.
