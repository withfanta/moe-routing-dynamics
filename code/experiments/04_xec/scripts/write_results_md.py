"""Render artifacts/XEC_P0/RESULTS.md from results.json. No new statistic."""
import json, os
ART=os.path.join(os.path.dirname(__file__),"..","artifacts","XEC_P0")
r=json.load(open(os.path.join(ART,"results.json")))
S=r["per_seed_mean_nll"]; C=r["comparisons"]; D=r["descriptive"]
seeds=[str(s) for s in r["policy_seeds"]]
L=[]
L.append("# XEC-P0 Results\n")
L.append("Cross-Layer Expert Cache Actionability Pilot. The verdict follows the")
L.append("seven-condition rule committed in `protocols/XEC_P0_PREREGISTRATION.md` before")
L.append("these numbers existed.\n")
L.append("## Provenance\n")
L.append(f"- model: `{r['model_id']}` revision `{r['model_revision']}`, frozen, FP16, not quantized")
L.append(f"- **no OLMoE parameter received gradients**: `{r['olmoe_receives_gradients']}`")
L.append(f"- target: Layer {r['target_layer']} (code {r['target_layer_code']}), history Layers 1-11")
L.append(f"- data: WikiText-103-raw train, EIPC-P0 blocks excluded ({r['eipc_exclusion']['eipc_blocks_excluded']})")
L.append(f"- manifest sha256: `{r['data_manifest_sha256']}`")
for k in ("train","validation","test"):
    L.append(f"  - {k} fingerprint `{r['fingerprints'][k][:32]}…`")
L.append(f"- n_train {r['n_train']}, n_validation {r['n_validation']}, n_test {r['n_test']}")
L.append(f"- projection sha256: `{r['projection_sha256']}`")
L.append(f"- policy seeds: {r['policy_seeds']}\n")
L.append("## Policy NLL table (true next-token NLL on TEST, lower is better)\n")
hdr="| method | " + " | ".join(f"seed {s}" for s in seeds) + " | mean |"
L.append(hdr); L.append("|---|" + "---|"*(len(seeds)+1))
nat=D["native_mean_nll"]
L.append(f"| Native OLMoE | {nat:.6f} | {nat:.6f} | {nat:.6f} | **{nat:.6f}** |")
L[-1]=L[-1].replace(f"| {nat:.6f} | {nat:.6f} | {nat:.6f} |", f"| {nat:.6f} (det.) | — | — |")
for v,label in (("CURRENT_ONLY","Current Only"),("FUSED_CACHE","Fused Cache"),("EXPERT_CACHE","Expert Cache")):
    row=" | ".join(f"{S[v][s]:.6f}" for s in seeds)
    L.append(f"| {label} | {row} | **{S[v]['mean']:.6f}** |")
orc=D["five_action_oracle_mean_nll"]
L.append(f"| Five-action Oracle | {orc:.6f} (det.) | — | — | **{orc:.6f}** |")
L.append("\nNative and the oracle are deterministic, so their seed columns are marked (det.).\n")
L.append(f"Oracle headroom vs native: {D['oracle_headroom_vs_native']:+.6f} nats/token.\n")
L.append("## Primary paired comparisons\n")
L.append("Per TEST sample, EXPERT_CACHE averaged over the three seeds. Negative is better.\n")
L.append("| comparison | mean paired ΔNLL | bootstrap 95% CI |")
L.append("|---|---|---|")
for key,label in (("d_native","Expert − Native"),("d_current","Expert − Current"),("d_fused","Expert − Fused")):
    st=C[key]
    L.append(f"| {label} | **{st['mean']:+.6f}** | [{st['ci_low']:+.6f}, {st['ci_high']:+.6f}] |")
L.append(f"\nPaired bootstrap: {C['d_native']['n_bootstrap']} resamples over TEST sample indices, "
         f"seed {C['d_native']['bootstrap_seed']}; seeds not resampled independently.\n")
L.append("## Verdict\n")
L.append(f"**{r['final_verdict']}**\n")
L.append("| condition | requirement | observed | outcome |")
L.append("|---|---|---|---|")
for c in r["conditions"]:
    L.append(f"| {c['condition']} | {c['requirement']} | {c['observed']} | **{'PASS' if c['passed'] else 'FAIL'}** |")
L.append("")
L.append(f"{r['decision_sentence']}\n")
L.append("## Descriptive metrics (not used for the verdict)\n")
L.append(f"- native mean NLL: {D['native_mean_nll']:.6f}")
L.append(f"- five-action oracle mean NLL: {orc:.6f}")
L.append(f"- oracle headroom vs native: {D['oracle_headroom_vs_native']:+.6f}")
od=D["test_oracle_action_distribution"]
L.append(f"- TEST oracle action distribution: {od['counts']} "
         f"(native {od['fraction_native_action']:.4f}, swap {od['fraction_swap_actions']:.4f})")
L.append("- five-action accuracy vs TEST oracle (seed-averaged):")
for v,a in D["action_accuracy_vs_test_oracle"].items():
    L.append(f"  - {v}: {a:.4f}")
L.append("- predicted action distribution (pooled over seeds):")
for v,pd in D["predicted_action_distribution"].items():
    L.append(f"  - {v}: {pd['counts']} (native {pd['fraction_native_action']:.4f}, swap {pd['fraction_swap_actions']:.4f})")
L.append(f"- EXPERT_CACHE vs native: improved {D['expert_cache_fraction_improved_vs_native']:.4f}, "
         f"worsened {D['expert_cache_fraction_worsened_vs_native']:.4f}, "
         f"equal {D['expert_cache_fraction_equal_vs_native']:.4f}\n")
L.append("## Numerical note recorded with the result\n")
L.append("The selected-expert reconstruction invariant (weighted sum of the eight separately")
L.append("executed experts vs the stock fused MoE output) showed a maximum ABSOLUTE error of")
L.append("2.24e-02 on validation and 1.90e-02 on test across ~11.5M values per split. A")
L.append("diagnostic on real contexts showed the RELATIVE error is ~1e-3 at every layer, at")
L.append("fp16 eps (9.77e-04), with the largest absolute deviations occurring exactly where")
L.append("|y| is large (5 to 12.7). This is fp16 accumulation noise, not a defect.\n")
L.append("The decisive correctness gate is independent of that tolerance: **action 0 reproduced")
L.append("the plain native NLL at 0.0e+00 error** on every split, so the intervention machinery")
L.append("is exact where it matters for the metric.\n")
L.append("## Interpretation limit\n")
L.append(f"{r['stopping_note']}\n")
L.append("## Regenerating ignored caches\n")
L.append("`.npz` caches are git-ignored. sha256:\n")
for k,v in r["artifact_sha256"].items():
    L.append(f"- `{k}`: `{v}`")
L.append("\nRegenerate with the command block in `README.md`.\n")
L.append("## Relationship to prior projects\n")
L.append(f"Prior projects are {r['prior_projects_status']}. The Layer-12 target was inherited")
L.append("from the committed EIPC-P0 result, and no rejected expert was executed here.\n")
L.append(f"Rescue modifications: {r['rescue_modifications']}.\n")
open(os.path.join(ART,"RESULTS.md"),"w").write("\n".join(L))
print("RESULTS.md written")
