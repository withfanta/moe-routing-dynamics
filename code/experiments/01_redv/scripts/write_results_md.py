"""Render artifacts/REDV_V1/RESULTS.md from results.json. No new statistic."""
import json, os, sys

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "REDV_V1")
r = json.load(open(os.path.join(ART, "results.json")))

L = []
L.append("# REDV-V1 Results\n")
L.append("Rejected-Expert Delayed Value V1. Frozen setting, single run, no rescue")
L.append("modification. Verdict follows the rule committed in")
L.append("`protocols/REDV_V1_PREREGISTRATION.md` before these numbers existed.\n")
L.append("## Provenance\n")
L.append(f"- model: `{r['model_id']}` revision `{r['model_revision']}`")
L.append("- data: WikiText-103-raw, validation (probe fitting) / test (held-out); train unused")
L.append(f"- data manifest sha256: `{r['data_manifest_sha256']}`")
L.append(f"- probe: `{r['probe']}`\n")
L.append("## Primary result\n")
L.append("| transition | R² baseline | R² augmented | Delta_R2 |")
L.append("|---|---|---|---|")
for t in r["transitions"]:
    L.append(f"| {t['transition_human']} | {t['r2_baseline']:+.5f} | "
             f"{t['r2_augmented']:+.5f} | {t['delta_r2']:+.5f} |")
L.append(f"\n**mean Delta_R2 = {r['mean_delta_r2']:+.5f}**\n")
L.append("Baseline feature dim 2112 = [h 2048 ; g 64]. Augmented dim 4160 adds r (2048).")
L.append("Both probes fit on 512 validation contexts, evaluated once on 512 test contexts.\n")
L.append("## Routing regret (descriptive only)\n")
L.append("These values do not change the verdict.\n")
L.append("| transition | n val | n test | mean G | median G | proportion G > 0 |")
L.append("|---|---|---|---|---|---|")
for t in r["transitions"]:
    L.append(f"| {t['transition_human']} | {t['n_validation']} | {t['n_test']} | "
             f"{t['mean_G']:.5f} | {t['median_G']:.5f} | {t['proportion_G_positive']:.4f} |")
L.append("\n## Verdict\n")
L.append(f"**{r['verdict']}**\n")
L.append(f"{r['decision_sentence']}\n")
L.append("Rule applied:\n")
L.append(f"- SUPPORTED: {r['judgement_rule']['supported']}")
L.append(f"- NOT_SUPPORTED: {r['judgement_rule']['not_supported']}")
L.append(f"- INCONCLUSIVE: {r['judgement_rule']['inconclusive']}\n")
L.append(f"{r['stopping_note']}\n")
L.append(f"Rescue modifications: {r['rescue_modifications']}.\n")

open(os.path.join(ART, "RESULTS.md"), "w").write("\n".join(L))
print("RESULTS.md written")
