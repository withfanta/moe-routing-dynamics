"""Generate code/source_audit.json by re-auditing every source repository read-only.

Run from the packaged repository root. Only reads the sources: `git rev-parse`,
`git status --porcelain`, `git ls-files`. Never writes to them.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

WORK = "/home/h-li/work"
PKG = os.path.dirname(os.path.abspath(__file__))

REPOS = [
    ("01_redv", "rejected_expert_delayed_value", ["REDV-V1", "REDV-V2"], "910482e"),
    ("02_drev", "delayed_rejected_evidence", ["DREV-P0"], "4cce387"),
    ("03_eipc", "expert_identity_preservation", ["EIPC-P0"], "517f74e"),
    ("04_xec", "cross_layer_expert_cache", ["XEC-P0"], "a664cb5"),
    ("05_epd", "expert_provenance_decomposition", ["EPD-P0"], "995d23e"),
    ("06_rmo", "routing_markov_order", ["RMO-P0"], "a8e9d83"),
    ("07_rmc", "routing_memory_crossmodel", ["RMC-P0"], "cc60edd"),
    ("08_nhd", "nonlinear_history_decoding", ["NHD-P0"], "f1a7d4a"),
]

RESULT_HINTS = ("results.json", "RESULTS.md", "protocol.md", "PREREGISTRATION",
                "PROTOCOL", "data_manifest.json", "model_provenance.json",
                "environment.json", "source_verification.json",
                "router_identification.json")


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True).stdout.strip()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_one(slug: str, name: str, experiments: list, expected: str) -> dict:
    repo = os.path.join(WORK, name)
    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain")
    entries = [l for l in status.splitlines() if l.strip()]
    tracked = [f for f in git(repo, "ls-files").splitlines() if f]

    sizes = {}
    for f in tracked:
        p = os.path.join(repo, f)
        if os.path.isfile(p):
            sizes[f] = os.path.getsize(p)

    # Large files present in the source tree but not packaged.
    omitted = []
    for root, dirs, files in os.walk(repo):
        if ".git" in dirs:
            dirs.remove(".git")
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, repo)
            if rel in sizes:
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if sz >= 1 << 20:
                omitted.append({"path": rel, "bytes": sz,
                                "sha256": sha256_file(p)})
    omitted.sort(key=lambda d: -d["bytes"])

    pkg_dir = os.path.join(PKG, "experiments", slug)
    packaged = []
    for root, dirs, files in os.walk(pkg_dir):
        for f in files:
            packaged.append(os.path.relpath(os.path.join(root, f), pkg_dir))

    return {
        "slug": slug,
        "source_repository": repo,
        "experiments": experiments,
        "head_sha": head,
        "head_short": head[:7],
        "expected_short": expected,
        "head_matches_expected": head.startswith(expected),
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "n_commits": int(git(repo, "rev-list", "--count", "HEAD") or 0),
        "commit_log": git(repo, "log", "--oneline").splitlines(),
        "git_status_porcelain": entries,
        "working_tree_clean": len(entries) == 0,
        "uncommitted_modifications": len(entries) > 0,
        "canonical_source": f"git archive {expected} (frozen commit, not working tree)",
        "n_tracked_files": len(tracked),
        "tracked_bytes_total": sum(sizes.values()),
        "largest_tracked_file": (max(sizes.items(), key=lambda kv: kv[1])
                                 if sizes else None),
        "result_and_protocol_files": sorted(
            f for f in tracked if any(h in f for h in RESULT_HINTS)),
        "n_packaged_files": len(packaged),
        "packaged_bytes_total": sum(
            os.path.getsize(os.path.join(pkg_dir, f)) for f in packaged),
        "large_artifacts_omitted": omitted,
        "omitted_bytes_total": sum(d["bytes"] for d in omitted),
    }


def main() -> int:
    repos = [audit_one(*r) for r in REPOS]
    out = {
        "audit_generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "packaging_task": "repository packaging / reproducibility, not a new experiment",
        "sources_treated_as": "READ-ONLY",
        "sources_modified": False,
        "packaging_method": "git archive <frozen sha> | tar -x",
        "all_sources_clean": all(r["working_tree_clean"] for r in repos),
        "all_heads_match_expected": all(r["head_matches_expected"] for r in repos),
        "n_repositories": len(repos),
        "n_packaged_files_total": sum(r["n_packaged_files"] for r in repos),
        "packaged_bytes_total": sum(r["packaged_bytes_total"] for r in repos),
        "omitted_bytes_total": sum(r["omitted_bytes_total"] for r in repos),
        "max_packaged_file_bytes": 25 * 1024 * 1024,
        "git_lfs_used": False,
        "repositories": repos,
        "excluded_projects": [
            "SensorLLM", "video-language H2", "MSR-VTT", "wristmotion-3act",
            "intervention_relation_learning",
        ],
    }
    path = os.path.join(PKG, "source_audit.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")
    print(f"  repos {out['n_repositories']}  clean {out['all_sources_clean']}  "
          f"heads match {out['all_heads_match_expected']}")
    print(f"  packaged {out['n_packaged_files_total']} files, "
          f"{out['packaged_bytes_total'] / 1e6:.2f} MB")
    print(f"  omitted {out['omitted_bytes_total'] / 1e6:.1f} MB of large artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
