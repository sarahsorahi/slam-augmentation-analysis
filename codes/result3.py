import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel

# -----------------------------
# Load data
# -----------------------------
BASE_DIR = Path("/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment")
df = pd.read_csv(BASE_DIR / "combined_results.csv")

# -----------------------------
# Create overall metrics
# -----------------------------
df["F"] = (
    df["F1_DIR"] +
    df["F1_DM"] +
    df["F1_INTJ"] +
    df["F1_AS"]
) / 4

df["AUC"] = (
    df["AUC_DIR"] +
    df["AUC_DM"] +
    df["AUC_INTJ"] +
    df["AUC_AS"]
) / 4

# -----------------------------
# Metrics
# -----------------------------
overall_metrics = ["F", "accuracy", "AUC"]
function_metrics = ["F1_DIR", "F1_DM", "F1_INTJ", "F1_AS"]

# -----------------------------
# Conditions (augmented only)
# -----------------------------
conditions = [
    "REAL_PLUS_NEAR",
    "REAL_PLUS_RANDOM",
    "REAL_PLUS_MIDDLE",
    "REAL_PLUS_FAR",
    "REAL_PLUS_DISTANCE_BALANCED"
]

# -----------------------------
# Helper function
# -----------------------------
def run_test(real_vals, aug_vals):
    diff = aug_vals - real_vals

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Wilcoxon (main)
    try:
        _, p_wilcoxon = wilcoxon(aug_vals, real_vals)
    except:
        p_wilcoxon = np.nan

    # t-test (support)
    _, p_t = ttest_rel(aug_vals, real_vals)

    # effect size
    d = mean_diff / std_diff if std_diff != 0 else np.nan

    # wins
    wins = np.sum(diff > 0)

    return mean_diff, p_wilcoxon, p_t, d, wins

# -----------------------------
# Run analysis
# -----------------------------
results = []

for cond in conditions:

    for metric in overall_metrics + function_metrics:

        real = df[df["condition"] == "REAL_ONLY"].sort_values("split")[metric].values
        aug = df[df["condition"] == cond].sort_values("split")[metric].values

        mean_diff, p_w, p_t, d, wins = run_test(real, aug)

        results.append({
            "condition": cond,
            "metric": metric,
            "mean_diff": round(mean_diff, 4),
            "wilcoxon_p": round(p_w, 4),
            "t_test_p": round(p_t, 4),
            "cohens_d": round(d, 4),
            "wins(out of 5)": int(wins)
        })

# -----------------------------
# Create dataframe
# -----------------------------
results_df = pd.DataFrame(results)

# -----------------------------
# Pretty print
# -----------------------------
pd.set_option("display.float_format", "{:.4f}".format)

print("\n=== SIGNIFICANCE OF CHANGE (Augmented vs REAL_ONLY) ===")
print(results_df)

# -----------------------------
# Save
# -----------------------------
output_path = BASE_DIR / "significance_change_results.csv"
results_df.to_csv(output_path, index=False)

print("\n✅ Results saved to:", output_path)