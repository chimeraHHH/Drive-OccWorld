"""Descriptive lower bound from completed baseline records; no model or new metric.

For each original anchor-instance-horizon, preserve its original point denominator.
Uncovered points have identically zero predictions in the fixed-owner candidate.
Their error contribution is therefore unavoidable by changing covered-object poses.
The lower bound is relaxed: it assumes zero error for every covered point, which
need not be achievable with a single rigid transform per predicted box.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

COMPLETE = "edfdf0b38145ad0b1d5fb69370ba8e8d91fda973f33540c1f5149917f58077f6"
SUMMARY = "ba67afc75b02ef481b0846914207fc01c6d04a4d0c675d3a7c00d1129b81141e"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return tuple(row[x] for x in (
        "scene_token", "sample_token", "instance_token", "horizon_seconds"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive(root):
    require(digest(root / "complete.json") == COMPLETE, "completion SHA mismatch")
    require(digest(root / "summary.json") == SUMMARY, "summary SHA mismatch")
    complete = json.loads((root / "complete.json").read_text())
    require(complete["files_sha256"]["summary.json"] == SUMMARY, "ledger mismatch")
    data = json.loads((root / "summary.json").read_text())
    coverage = data["coverage_object_records"]
    physical = [r for r in data["physical_object_records"] if r["arm"] == "CRN_CV"]
    by_key = {key(r): r for r in physical}
    require(len(coverage) == len(physical) == len(by_key) == 16074, "row count")
    require(len({key(r) for r in coverage}) == 16074, "duplicate coverage rows")
    require({key(r) for r in coverage} == set(by_key), "support keys differ")
    require(len({r["scene_token"] for r in coverage}) == 100, "scene count")
    require(len({r["sample_token"] for r in coverage}) == 200, "anchor count")
    per_object = []
    max_reconstruction_error = 0.0
    for row in coverage:
        ref = by_key[key(row)]
        n = row["source_points"]
        nc = row["covered"]["source_points"]
        nu = row["uncovered"]["source_points"]
        require(n > 0 and 0 <= nc <= n and nc + nu == n, "point counts")
        require(nc == row["covered_points"], "covered count disagreement")
        require(n == ref["source_points"] and row["group"] == ref["group"], "support")
        derived = {"key": key(row), "group": row["group"], "h": row["horizon_seconds"],
                   "points": n, "covered": nc, "uncovered": nu}
        for axis, metric in [("xy", "xy_m"), ("3d", "xyz_m")]:
            floor = 0.0
            if nu:
                zero = row["uncovered"]["zero"][metric]
                baseline = row["uncovered"]["CRN_CV"][metric]
                require(math.isfinite(zero) and zero >= 0 and zero == baseline,
                        "uncovered prediction must equal zero baseline")
                floor = nu / n * zero
            covered = nc / n * row["covered"]["CRN_CV"][metric] if nc else 0.0
            observed = ref[f"epe_{axis}_m"]
            require(math.isfinite(covered) and covered >= 0, "invalid covered error")
            diff = abs(floor + covered - observed)
            max_reconstruction_error = max(max_reconstruction_error, diff)
            require(math.isclose(floor + covered, observed, rel_tol=1e-12, abs_tol=1e-10),
                    "coverage decomposition does not reconstruct original object EPE")
            derived[axis] = {"floor": floor, "covered": covered, "CV": observed}
        per_object.append(derived)

    bins = []
    for h in (0.5, 1.0, 1.5, 2.0):
        for group in ("all", "stationary", "ambiguous", "moving"):
            rows = [r for r in per_object if r["h"] == h and
                    (group == "all" or r["group"] == group)]
            require(bool(rows), "unexpected empty original group")
            values = {}
            for axis in ("xy", "3d"):
                means = {field: math.fsum(r[axis][field] for r in rows) / len(rows)
                         for field in ("floor", "covered", "CV")}
                values[axis] = {
                    "CRN_CV_mean_epe_m": means["CV"],
                    "unavoidable_uncovered_contribution_m": means["floor"],
                    "covered_contribution_m": means["covered"],
                    "relaxed_lower_bound_fraction_of_CV": means["floor"] / means["CV"]
                    if means["CV"] else None,
                    "maximum_removable_CV_error_m_relaxed": means["covered"],
                }
            bins.append({"horizon_seconds": h, "group": group,
                         "object_anchor_pairs": len(rows),
                         "source_points": sum(r["points"] for r in rows),
                         "covered_points": sum(r["covered"] for r in rows),
                         "uncovered_points": sum(r["uncovered"] for r in rows),
                         "metrics": values})
    return {
        "schema": "fixed-owner-error-floor-v1",
        "status": "COMPLETE_DESCRIPTIVE_BASELINE_DECOMPOSITION",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": {"complete_sha256": COMPLETE, "summary_sha256": SUMMARY,
                    "script_sha256": digest(__file__)},
        "support": {"anchors": 200, "scenes": 100, "object_horizon_rows": 16074},
        "checks": {"all_original_CRN_CV_object_errors_reconstructed": True,
                   "max_abs_reconstruction_error_m": max_reconstruction_error,
                   "absolute_tolerance_m": 1e-10, "relative_tolerance": 1e-12},
        "formula": "mean_i[(uncovered_count_i / original_point_count_i) * zero_EPE_uncovered_i]",
        "aggregation": "Original per-object mean-point EPE, then equal anchor-instance per group/horizon",
        "interpretation": [
            "Fixed-owner physical branch only; occupancy Gaussian fields and O are not bounded by this number.",
            "Relaxed lower bound, not an achievable oracle score: covered points may not admit perfect shared rigid transforms.",
            "No candidate predictions, no new support or metric, no fitted threshold, no checkpoint selection.",
            "Fixed exposed dev200 descriptive result; no training-seed or population uncertainty claim.",
            "Never divide the primary candidate EPE by coverage or remove uncovered points to improve scores.",
        ],
        "bins": bins,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = derive(args.reference)
    with args.out.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out": str(args.out), "checks": result["checks"],
                      "moving": [r for r in result["bins"] if r["group"] == "moving"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
