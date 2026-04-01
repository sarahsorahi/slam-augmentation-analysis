import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================
RAW_DATA_PATH = Path(
    "/Users/aysasorahi/Documents/master/SLAM LAB/REZA/data/look_data.ods"
)

OUT_DIR = Path(
    "/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment/synthetic"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "synthetic_pool.csv"

# =========================
# Load raw ODS file
# =========================
df = pd.read_excel(RAW_DATA_PATH, engine="odf")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Fix truncated column if needed
if "functio" in df.columns:
    df = df.rename(columns={"functio": "function"})

# =========================
# Select synthetic rows
# =========================
if "label" not in df.columns:
    raise ValueError("Column 'label' not found in raw data.")

syn_df = df[df["label"] == "L"].copy()
print(f"Raw synthetic rows: {len(syn_df)}")

# =========================
# Keep only relevant columns
# =========================
required_cols = {"sample", "function"}
missing = required_cols - set(syn_df.columns)

if missing:
    raise ValueError(f"Missing required columns in synthetic data: {missing}")

syn_df = syn_df[["sample", "function"]]

# Drop unusable rows
syn_df = syn_df.dropna(subset=["sample", "function"])
syn_df = syn_df.reset_index(drop=True)

# Add explicit source column
syn_df["source"] = "synthetic"

print(f"Usable synthetic rows: {len(syn_df)}")

# =========================
# Save synthetic pool
# =========================
syn_df.to_csv(OUT_PATH, index=False)

print("\n✅ Synthetic pool created:")
print(OUT_PATH)

print("\nFunction distribution:")
print(syn_df["function"].value_counts())
