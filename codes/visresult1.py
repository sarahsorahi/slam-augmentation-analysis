import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Load data
# -----------------------------
BASE_DIR = Path("/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment")
df = pd.read_csv(BASE_DIR / "combined_results.csv")

# -----------------------------
# Create overall F (NEW 🔥)
# -----------------------------
df["F"] = (
    df["F1_DIR"] +
    df["F1_DM"] +
    df["F1_INTJ"] +
    df["F1_AS"]
) / 4

# -----------------------------
# Metrics to plot
# -----------------------------
metrics = [
    "F",             # NEW ⭐
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "accuracy",
    "macro_auc"
]

metric_labels = [
    "F (avg functions)",
    "Macro F1",
    "Micro F1",
    "Weighted F1",
    "Accuracy",
    "Macro AUC"
]

# -----------------------------
# Compute mean and std
# -----------------------------
grouped = df.groupby("condition")[metrics]

means = grouped.mean()
stds = grouped.std()

# sort by F (NEW ⭐)
means = means.sort_values(by="F", ascending=False)
stds = stds.loc[means.index]

conditions = means.index.tolist()

# -----------------------------
# Plot
# -----------------------------
x = np.arange(len(conditions))
width = 0.12

fig, ax = plt.subplots(figsize=(13, 6))

for i, metric in enumerate(metrics):
    ax.bar(
        x + i * width,
        means[metric],
        width,
        yerr=stds[metric],
        capsize=4,
        label=metric_labels[i]
    )

# -----------------------------
# Formatting
# -----------------------------
ax.set_xlabel("Condition")
ax.set_ylabel("Score")
ax.set_title("Performance Across Conditions (Mean ± Std)")

ax.set_xticks(x + width * (len(metrics)/2))
ax.set_xticklabels(conditions, rotation=30)

ax.legend()

plt.tight_layout()

# -----------------------------
# Save + show
# -----------------------------
output_path = BASE_DIR / "main_metrics_with_F.png"
plt.savefig(output_path, dpi=300)

plt.show()

print("✅ Plot saved to:", output_path)