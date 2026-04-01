import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
SPLIT_ID = 1

BASE_DIR = Path(
    "/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment/results"
)

OUT_DIR = BASE_DIR / f"split_{SPLIT_ID}"
OUT_DIR.mkdir(exist_ok=True)

FUNCTIONS = ["DIR", "DM", "INTJ", "AS"]

CONDITIONS = [
    "REAL_ONLY",
    "REAL_PLUS_NEAR",
    "REAL_PLUS_MIDDLE",
    "REAL_PLUS_FAR",
    "REAL_PLUS_RANDOM",
    "REAL_PLUS_DISTANCE_BALANCED"
]

# =========================
# COLLECT METRICS
# =========================
rows = []

for cond in CONDITIONS:
    metrics_path = OUT_DIR / cond / "metrics.json"

    if not metrics_path.exists():
        print(f"⚠️ Missing results for {cond}")
        continue

    with open(metrics_path) as f:
        metrics = json.load(f)

    report = metrics["classification_report"]

    for func in FUNCTIONS:
        rows.append({
            "condition": cond,
            "function": func,
            "precision": report[func]["precision"],
            "recall": report[func]["recall"],
            "f1": report[func]["f1-score"]
        })

df = pd.DataFrame(rows)

# =========================
# SAVE FULL TABLE
# =========================
csv_path = OUT_DIR / "full_results_table.csv"
df.to_csv(csv_path, index=False)

print("\nSaved full results table to:")
print(csv_path)

# =========================
# PLOT — F1 BY CONDITION
# =========================
plt.figure(figsize=(10, 5))

for func in FUNCTIONS:
    subset = df[df["function"] == func]
    plt.plot(
        subset["condition"],
        subset["f1"],
        marker="o",
        label=func
    )

plt.xticks(rotation=45, ha="right")
plt.ylabel("F1 score")
plt.title(f"Split {SPLIT_ID}: Per-function performance across conditions")
plt.legend(title="Function")
plt.tight_layout()

plot_path = OUT_DIR / "full_results_plot.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print("Saved full results plot to:")
print(plot_path)
