"""Plot actual completed-reference decomposition; no missing or mock values."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data_path = ROOT / "fixed_owner_error_floor_dev200_v1.json"
data = json.loads(data_path.read_text())
rows = [r for r in data["bins"] if r["group"] == "moving"]
x = [r["horizon_seconds"] for r in rows]
floor = [r["metrics"]["xy"]["unavoidable_uncovered_contribution_m"] for r in rows]
covered = [r["metrics"]["xy"]["covered_contribution_m"] for r in rows]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.4, 5.2))
fig.subplots_adjust(left=.13, right=.97, bottom=.32, top=.78)
ax.bar(x, floor, width=.30, color="#BA5A54", label="Uncovered: fixed zero prediction")
ax.bar(x, covered, bottom=floor, width=.30, color="#517D9A", label="Covered: potentially reducible error")
for row, xx, low, high in zip(rows, x, floor, covered):
    frac = row["metrics"]["xy"]["relaxed_lower_bound_fraction_of_CV"]
    ax.text(xx, low / 2, f"{frac:.1%}", ha="center", va="center", color="white", weight="bold")
    ax.text(xx, low + high + .065, f"{low + high:.2f}", ha="center", va="bottom")
ax.set_xticks(x, [f"{h:.1f}" for h in x])
ax.set_xlabel("Forecast horizon (s)")
ax.set_ylabel("Contribution to full-support XY EPE (m)")
ax.set_ylim(0, 3.9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=.18)
ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.text(.13, .94, "Where the strong baseline's motion error remains", fontsize=14, weight="bold")
fig.text(.13, .885, "Existing CRN-CV baseline · moving objects · fixed development set", fontsize=10)
fig.text(.13, .16, "Red share is a relaxed error floor for the current fixed-owner physical branch.", fontsize=9)
fig.text(.13, .12, "Original point denominators; equal weight per object–anchor. All uncovered points retained.", fontsize=8.5)
fig.text(.13, .08, "Not Cpl/Fix results. Not an attainable oracle score or a sensor limitation.", fontsize=9, color="#8A3734")

out = ROOT / "figures"
out.mkdir(exist_ok=True)
stem = out / "fixed_owner_error_floor_dev200_v1"
fig.savefig(stem.with_suffix(".png"), dpi=220)
fig.savefig(stem.with_suffix(".pdf"))
plt.close(fig)
provenance = {"data": str(data_path), "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "mock_values": False, "candidate_results": False,
              "files_sha256": {ext: hashlib.sha256(stem.with_suffix(ext).read_bytes()).hexdigest()
                               for ext in (".png", ".pdf")}}
stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
print(stem.with_suffix(".png"))
