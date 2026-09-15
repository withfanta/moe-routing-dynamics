# XEC-P0 Decision Log

One line per decision. The result decision is appended once, after the formal run, and is
not reinterpreted afterwards.

## Scaffold

- **Project created.** `cross_layer_expert_cache` initialized as a fresh repository with a
  minimal structure. No research tree, no speculative architecture nodes.
- **Fresh data confirmed available.** EIPC-P0 used 2048 train blocks from a 20628-block
  window; excluding all of them leaves 18580 untouched blocks against the 3584 XEC-P0
  needs. REDV-V1, REDV-V2 and DREV-P0 were each re-checked and all record
  `train_split_used = False`.
- **Target layer inherited, not chosen here.** Layer 12 comes from the EIPC-P0 result
  (+0.17506 R² vs +0.10923 at Layer 8), which was recorded and committed before XEC-P0
  existed. No other layer is tested.
- **Prior projects treated as closed and read-only.** No file in any prior repository is
  modified, and no prior conclusion beyond the established phenomenon is imported.
- **Charter accepted as locked.** The five actions, the four compared methods, true
  next-token NLL as the primary outcome, the interpretation limit, and the out-of-scope
  list in `docs/RESEARCH_CHARTER.md` are frozen.

## Result (final)

- **XEC-P0 is NOT_PROMISING under the frozen seven-condition rule.** All seven conditions
  failed: mean d_native = -0.001506 (threshold -0.005) with CI [-0.005743, +0.002760];
  mean d_current = +0.000347 with CI [-0.000761, +0.001670]; mean d_fused = +0.001056 with
  CI [-0.000023, +0.002156]; and EXPERT_CACHE lost to FUSED_CACHE in two of three seeds.
  The expert-cache method direction is stopped: no more actions, no target-layer change, no
  cache-dimension tuning, no larger sample count, no attention heads, no optimizer change,
  no Layer 8, no other model or dataset, no rescue.
- **The null is attributable to a near-noise target, not a handicapped mechanism.** All
  nine checkpoints selected epoch 0 because validation CE never improved on ~1.62 (chance
  1.609) while train CE collapsed to 0.03-0.08. The oracle label's median best-vs-second
  margin is 0.0087 nats, a third of samples have under 0.001 nats of oracle gain, and total
  oracle headroom is ~0.049 nats. The three variants were parameter-matched at 153600 and
  shared a per-seed initialization, so the cache mechanism was not disadvantaged; but since
  the checkpoints are effectively untrained, comparisons among the learned variants are
  uninformative about the mechanism itself.
- **EIPC-P0 not reinterpreted.** Its PROMISING information-content verdict stands unchanged
  and its artifacts were read-only throughout. XEC-P0 shows only that this information did
  not convert into an improved routing decision under this frozen setting and restricted
  action space.
- **Numerical note.** The selected-expert reconstruction invariant showed max ABSOLUTE
  error 2.12e-02 (train) / 2.24e-02 (validation) / 1.90e-02 (test), which a diagnostic on
  real contexts attributed to fp16 accumulation: relative error is ~1e-3 at every layer, at
  fp16 eps, with the largest absolute deviations exactly where |y| reaches 5-12.7. The
  decisive gate was exact: action 0 reproduced the plain native NLL at 0.0e+00 error on
  every split.
