#!/usr/bin/env python3
"""Render a descriptive diagnostic from the frozen completed analysis only.

No observations are synthesized, and no statistics are re-estimated. The plot
copies completed summary statistics, validates their shared population, and
exports the exact plotted values together with source provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "history_camera_evidence_train_analysis_v1.json"
EXPECTED_SHA256 = "20b49f30bab4ae8c539e385c288bb8e355feb587cabfdd3327423dc68b2905bd"
STEM = ROOT / "figures" / "history_camera_evidence_train_v1"
FEATURES = (
    ("score_true", "True ZNCC score", "#2166AC", "o"),
    ("score_broken", "Broken correspondence", "#8B6AA8", "s"),
    ("lsq_speed_xy_mps", "D speed", "#D17B27", "^"),
)
REASONS = (
    ("valid", "Valid camera score", "#648FAE"),
    ("low_current_zero_texture", "Low current texture", "#9CABB5"),
    ("hypothesis_projection_invalid", "Invalid hypothesis projection", "#ABB7BF"),
    ("no_current_zero_view", "No current zero-motion view", "#BBC4CB"),
    ("missing_previous_keyframe", "Missing previous keyframe", "#CBD1D6"),
)


def extract_data() -> dict:
    raw = SOURCE.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError(f"Frozen source changed: {sha}")
    source = json.loads(raw)
    assert (source["samples"], source["scenes"]) == (512, 256)
    rows = []
    for horizon in source["horizons"]:
        group = horizon["strata"]["no_radar"]["all"]
        summary = next(row for row in group["bins"] if row["all_speeds"])
        features = summary["features_same_valid_population"]
        support_fields = ("positive_weight", "negative_weight", "positive_points", "negative_points")
        first_support = {key: features[FEATURES[0][0]][key] for key in support_fields}
        for feature, *_ in FEATURES:
            assert {key: features[feature][key] for key in support_fields} == first_support
        rows.append({
            "horizon_seconds": horizon["horizon_seconds"],
            "original_object_count": group["original_object_count"],
            "valid_camera_points": summary["by_camera_reason"]["valid"]["points"],
            "shared_auc_support": first_support,
            "weighted_auc": {feature: features[feature]["auc"] for feature, *_ in FEATURES},
            "true_minus_broken_auc": summary["paired_true_minus_broken_auc"],
        })
    assert [row["horizon_seconds"] for row in rows] == [0.5, 1.0, 1.5, 2.0]
    horizon_2s = next(h for h in source["horizons"] if h["horizon_seconds"] == 2.0)
    summary_2s = next(row for row in horizon_2s["strata"]["no_radar"]["all"]["bins"] if row["all_speeds"])
    total_cost = summary_2s["support"]["FULL_denominator_negative_cost_xy_m"]
    reasons = []
    for reason, label, _ in REASONS:
        values = summary_2s["by_camera_reason"][reason]
        cost = values["FULL_denominator_negative_cost_xy_m"]
        fraction = values["negative_cost_fraction_of_this_radar_group"]
        assert math.isclose(cost / total_cost, fraction, rel_tol=1e-12, abs_tol=1e-12)
        assert values["original_object_count"] == summary_2s["support"]["original_object_count"]
        reasons.append({
            "reason": reason,
            "label": label,
            "full_denominator_negative_cost_xy_m": cost,
            "fraction_of_no_radar_uncovered_negative_cost": fraction,
            "percentage": 100.0 * fraction,
            "points": values["points"],
        })
    assert math.isclose(sum(row["fraction_of_no_radar_uncovered_negative_cost"] for row in reasons), 1.0, abs_tol=1e-12)
    return {
        "schema": "history-camera-evidence-train-figure-data-v1",
        "source_path": str(SOURCE),
        "source_sha256": sha,
        "source_status": source["status"],
        "samples": source["samples"],
        "scenes": source["scenes"],
        "population": "No-radar AND CRN-uncovered (owner<0), all GT motion groups, all D-speed bins; original future-valid support",
        "panel_a": {
            "metric": "Object-point-weighted binary AUC; positive benefit>0, negative benefit<0; fixed score directions",
            "population_rule": "All three features share exactly the same reason_code==0 population within each horizon; horizons have different future-valid populations",
            "feature_definitions": {feature: source["rules"]["definitions"][feature] for feature, *_ in FEATURES},
            "chance_auc": 0.5,
            "rows": rows,
        },
        "panel_b": {
            "horizon_seconds": 2.0,
            "metric": "Share of negative cost, not missing-point or missing-object share",
            "cost_definition": "sum(weight * max(-benefit_xy_m, 0)) / ALL original valid object count",
            "original_object_count": summary_2s["support"]["original_object_count"],
            "full_denominator_negative_cost_xy_m": total_cost,
            "rows": reasons,
        },
        "original_point_weight": source["rules"]["point_weight"],
        "denominator": source["rules"]["denominator"],
        "limitations": source["rules"]["limitations"] + [
            "Descriptive diagnostic only; no significance test or uncertainty interval is shown",
            "AUC is not future occupancy IoU",
        ],
    }


def render(data: dict) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#6A737D",
        "axes.linewidth": 0.7,
        "xtick.color": "#364152",
        "ytick.color": "#364152",
        "text.color": "#202B3A",
        "axes.labelcolor": "#202B3A",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(12.4, 5.4), facecolor="white")
    grid = fig.add_gridspec(1, 2, left=0.075, right=0.975, top=0.74, bottom=0.25, wspace=0.64, width_ratios=[1, 1.04])
    ax_a, ax_b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    fig.text(0.075, 0.945, "Historical-camera evidence", fontsize=17, fontweight="bold")
    fig.text(0.075, 0.89, "train512 / 256 scenes  |  No radar + CRN-uncovered  |  Diagnostic only", fontsize=11, color="#506072")
    ax_a.set_title("a   Benefit-sign discrimination", loc="left", pad=17, fontweight="bold")
    rows = data["panel_a"]["rows"]
    x = [row["horizon_seconds"] for row in rows]
    ax_a.axhline(0.5, color="#9CA5AD", linewidth=1, linestyle=(0, (4, 3)), zorder=0)
    for key, label, color, marker in FEATURES:
        y = [row["weighted_auc"][key] for row in rows]
        ax_a.plot(x, y, label=label, color=color, marker=marker, linewidth=1.8, markersize=5.5)
        for xx, yy in zip(x, y):
            offset = -15 if key == "score_broken" else 10
            ax_a.annotate(f"{yy:.3f}", (xx, yy), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=8.5, color=color)
    ax_a.set(xlim=(0.39, 2.11), ylim=(0.45, 0.68), xlabel="Future horizon (s)", ylabel="Weighted AUC")
    ax_a.set_xticks(x, ["0.5", "1.0", "1.5", "2.0"])
    ax_a.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_a.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper right", ncol=1, frameon=False, fontsize=8.5, handlelength=2, borderpad=0.2, labelspacing=0.35)
    ax_a.text(0.99, 0.012, "Dashed line: chance = 0.5", transform=ax_a.transAxes, ha="right", va="bottom", fontsize=8, color="#66717D")

    ax_b.set_title("b   Negative-cost partition at 2 s", loc="left", pad=17, fontweight="bold")
    reason_rows = data["panel_b"]["rows"]
    y_positions = list(range(len(reason_rows)))
    values = [row["percentage"] for row in reason_rows]
    ax_b.barh(y_positions, values, color=[color for _, _, color in REASONS], height=0.54)
    ax_b.set_yticks(y_positions, [label for _, label, _ in REASONS], fontsize=9)
    ax_b.invert_yaxis()
    ax_b.set(xlim=(0, 104), xlabel="Share of no-radar uncovered negative cost")
    ax_b.xaxis.set_major_locator(MultipleLocator(25))
    ax_b.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax_b.grid(axis="x", color="#E5E9ED", linewidth=0.7)
    ax_b.set_axisbelow(True)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    for yy, value in zip(y_positions, values):
        ax_b.text(value + 1.7, yy, f"{value:.3f}%", va="center", fontsize=9, color="#334155")

    fig.text(0.075, 0.158, "AUC: identical camera-valid population across features within each horizon; original object-point weights.", fontsize=9, color="#4D5A68")
    fig.text(0.075, 0.112, "Cost: all 10,332 original objects remain in the denominator at 2 s, including covered / camera-missing support.", fontsize=9, color="#4D5A68")
    fig.text(0.075, 0.057, "Training-internal descriptive evidence. AUC is not future IoU; these summaries do not establish routing or motion improvement.", fontsize=9, color="#334155")
    STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".png"), dpi=220)
    fig.savefig(STEM.with_suffix(".pdf"), metadata={"Title": "Historical-camera evidence: descriptive diagnostic", "Subject": "Frozen completed analysis; SHA256 " + data["source_sha256"]})
    plt.close(fig)


if __name__ == "__main__":
    data = extract_data()
    render(data)
    output = STEM.with_suffix(".data.json")
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"source_sha256": data["source_sha256"], "outputs": [str(STEM.with_suffix(suffix)) for suffix in (".png", ".pdf", ".data.json")]}, indent=2))
