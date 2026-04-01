import pandas as pd
from pathlib import Path

# -----------------------------
# File path
# -----------------------------
BASE_DIR = Path("/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment")

# -----------------------------
# Load combined data
# -----------------------------
df = pd.read_csv(BASE_DIR / "combined_results.csv")

# -----------------------------
# Metrics to summarize
# -----------------------------
metrics = [
    "accuracy",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "macro_auc",
    "F1_DIR",
    "F1_DM",
    "F1_INTJ",
    "F1_AS",
    "AUC_DIR",
    "AUC_DM",
    "AUC_INTJ",
    "AUC_AS",
]

# -----------------------------
# Check missing columns
# -----------------------------
missing_cols = [col for col in metrics if col not in df.columns]
if missing_cols:
    raise ValueError(f"These columns are missing from combined_results.csv: {missing_cols}")

# -----------------------------
# Compute mean and std
# -----------------------------
summary = df.groupby("condition")[metrics].agg(["mean", "std"])

# Flatten multi-index columns
summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
summary = summary.reset_index()

# -----------------------------
# Optional overall class-F1 mean
# (average of per-function F1 means)
# -----------------------------
summary["F1_functions_mean"] = (
    summary["F1_DIR_mean"]
    + summary["F1_DM_mean"]
    + summary["F1_INTJ_mean"]
    + summary["F1_AS_mean"]
) / 4

# -----------------------------
# Sort results
# -----------------------------
summary = summary.sort_values(by="macro_f1_mean", ascending=False)

# -----------------------------
# Pretty printing
# -----------------------------
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n=== SUMMARY TABLE ===")
print(summary)

# -----------------------------
# Save full summary
# -----------------------------
output_path = BASE_DIR / "summary_stats.csv"
summary.to_csv(output_path, index=False)

print(f"\n✅ Saved as: {output_path}")