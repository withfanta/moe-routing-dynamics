# EPD-P0 Decision Log

One line per decision. The exploratory result is appended once, after the analysis.

## Scaffold

- **Project created.** `expert_provenance_decomposition` initialized as a fresh repository
  with a minimal structure. Exploratory scale; no architecture authorized.
- **EIPC projection reused, not regenerated.** The EIPC-P0 artifact was read and its hash
  verified as `fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282`
  (seed 20260917, R^2048 -> R^32), keeping EPD-P0 directly comparable to EIPC-P0.
- **Fresh data confirmed.** EIPC-P0 used 2048 train blocks and XEC-P0 used 3584, disjoint
  from each other; excluding both leaves 14996 fresh blocks against the 768 EPD-P0 needs.
  All prior projects share the same 40000-record window, so block ids align. REDV-V1,
  REDV-V2 and DREV-P0 were re-verified to record `train_split_used = False`.
- **Prior projects treated as read-only.** REDV-V1/V2, DREV-P0, EIPC-P0 and XEC-P0 are
  closed; no file in any of them is modified and no result is reinterpreted. EPD-P0 asks a
  narrower diagnostic question about the EIPC signal and executes no rejected experts, no
  counterfactual routing, and no oracle actions.
- **No formal verdict by design.** The protocol reports a decomposition pattern
  (PATH-DOMINANT / CONTENT-DOMINANT / INTERACTION / NO CLEAN DECOMPOSITION), not a
  SUPPORTED/NOT_SUPPORTED outcome.

## Exploratory result

- **PATTERN: PATH_DOMINANT.** On TEST, R2_id +0.66969, R2_full +0.34007, R2_fused +0.20114,
  R2_content +0.06732. Descriptive gaps: FULL-FUSED +0.13893, FULL-CONTENT +0.27275,
  FULL-ID -0.32962. Layerwise, ID_PATH rises monotonically 0.14 -> 0.60 across Layers 1-11
  while CONTENT_RANK stays near zero at every depth (mean +0.00173). The EIPC-P0 provenance
  advantage appears carried by *which experts were selected* across depth, not by what they
  computed.
- **The ID > FULL ordering is a compression artifact, verified not an information paradox.**
  Binarizing FULL_PROVENANCE's slot occupancy reproduces ID_PATH exactly, so FULL strictly
  contains ID_PATH. At the shared 32-dim budget PCA retains 52% of ID_PATH's variance but only
  15% of FULL's, since FULL is 22528 dense dimensions dominated by activation magnitudes. A
  post-hoc control confirmed FULL remains below ID_PATH at 64 and 128 components. Two
  observations keep the path reading robust: CONTENT_RANK is near zero at every layer, and
  concatenating content onto the path moves nothing (+0.66969 -> +0.66728, delta -0.0024).
- **Classifier gap corrected before reporting, no measurement changed.** The original
  `classify_pattern` required ID_PATH to be close to FULL *from below*, so a case where the
  pure path exceeds full provenance fell through to NO_CLEAN_DECOMPOSITION. That was a gap in
  the rule, not ambiguity in the data. The condition now reads "at least as good as FULL",
  with a test pinning the exceed-case; re-running produced identical R2 values and the pattern
  PATH_DOMINANT.
- **Nothing built from the answer.** No cache, no attention, no method. This is one small
  pilot at one layer in one model on one dataset with linear probes; a plausible untested
  reading is that the path is predictable partly because routing is autocorrelated across
  depth, which would make it informative without being useful. XEC-P0's NOT_PROMISING
  actionability result stands unchanged.
