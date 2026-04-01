import pandas as pd
from pathlib import Path

# -----------------------------
# 1. Define base directory
# -----------------------------
BASE_DIR = Path("/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment")

# -----------------------------
# 2. Define file names
# -----------------------------
files = [
    "results_summary_split_1.csv",
    "results_summary_split_2.csv",
    "results_summary_split_3.csv",
    "results_summary_split_4.csv",
    "results_summary_split_5.csv"
]

# -----------------------------
# 3. Load and combine data
# -----------------------------
dfs = []

for i, file in enumerate(files, start=1):
    file_path = BASE_DIR / file

    print(f"Loading: {file_path}")

    df = pd.read_csv(file_path)

    # Add split column
    df["split"] = i

    dfs.append(df)

# Combine all splits
master_df = pd.concat(dfs, ignore_index=True)

# -----------------------------
# 4. Clean / standardize condition names
# -----------------------------
master_df["condition"] = (
    master_df["condition"]
    .astype(str)
    .str.strip()
    .str.replace("+", "_PLUS_", regex=False)
    .str.replace(" ", "_")
    .str.upper()
)

# -----------------------------
# 5. Sort and reset index
# -----------------------------
master_df = master_df.sort_values(by=["condition", "split"])
master_df.reset_index(drop=True, inplace=True)

# -----------------------------
# 6. Save combined dataset
# -----------------------------
output_path = BASE_DIR / "combined_results.csv"
master_df.to_csv(output_path, index=False)

print("\n✅ Combined dataset saved to:")
print(output_path)

# -----------------------------
# 7. Sanity checks
# -----------------------------
print("\n--- HEAD ---")
print(master_df.head())

print("\n--- CONDITIONS ---")
print(master_df["condition"].unique())

print("\n--- SHAPE ---")
print(master_df.shape)

print("\n--- SPLIT COUNTS ---")
print(master_df.groupby("condition")["split"].nunique())