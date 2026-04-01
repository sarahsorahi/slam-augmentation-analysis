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
# Setup
# -----------------------------
conditions = [
    "REAL_PLUS_NEAR",
    "REAL_PLUS_RANDOM",
    "REAL_PLUS_MIDDLE",
    "REAL_PLUS_FAR",
    "REAL_PLUS_DISTANCE_BALANCED"
]

overall_metrics = ["F", "accuracy", "AUC"]
function_metrics = ["F1_DIR", "F1_DM", "F1_INTJ", "F1_AS"]

# -----------------------------
# Helper
# -----------------------------
def compute_stats(real, aug):
    diff = aug - real

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    try:
        _, p_w = wilcoxon(aug, real)
    except:
        p_w = np.nan

    _, p_t = ttest_rel(aug, real)

    d = mean_diff / std_diff if std_diff != 0 else np.nan

    return p_w, p_t, d

def mark(p):
    if np.isnan(p):
        return "nan"
    return f"{p:.4f} ✓" if p < 0.05 else f"{p:.4f} ✗"

# -----------------------------
# Build tables
# -----------------------------
def build_tables(metrics):

    wilcoxon_rows = []
    ttest_rows = []
    effect_rows = []

    for cond in conditions:

        row_w = {"condition": cond}
        row_t = {"condition": cond}
        row_d = {"condition": cond}

        for metric in metrics:

            real = df[df["condition"] == "REAL_ONLY"].sort_values("split")[metric].values
            aug = df[df["condition"] == cond].sort_values("split")[metric].values

            p_w, p_t, d = compute_stats(real, aug)

            col_name = metric.replace("F1_", "")

            row_w[col_name] = mark(p_w)
            row_t[col_name] = mark(p_t)
            row_d[col_name] = round(d, 4)

        wilcoxon_rows.append(row_w)
        ttest_rows.append(row_t)
        effect_rows.append(row_d)

    return (
        pd.DataFrame(wilcoxon_rows),
        pd.DataFrame(ttest_rows),
        pd.DataFrame(effect_rows)
    )

# -----------------------------
# Generate tables
# -----------------------------
overall_w, overall_t, overall_d = build_tables(overall_metrics)
func_w, func_t, func_d = build_tables(function_metrics)

# -----------------------------
# Save
# -----------------------------
overall_w.to_csv(BASE_DIR / "overall_wilcoxon_marked.csv", index=False)
overall_t.to_csv(BASE_DIR / "overall_ttest_marked.csv", index=False)
overall_d.to_csv(BASE_DIR / "overall_effect_size.csv", index=False)

func_w.to_csv(BASE_DIR / "function_wilcoxon_marked.csv", index=False)
func_t.to_csv(BASE_DIR / "function_ttest_marked.csv", index=False)
func_d.to_csv(BASE_DIR / "function_effect_size.csv", index=False)

# -----------------------------
# Done
# -----------------------------
print("✅ Saved with significance marks!")