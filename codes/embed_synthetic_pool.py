import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# =========================
# Paths
# =========================
SYN_PATH = Path(
    "/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment/synthetic/synthetic_pool.csv"
)

OUT_DIR = SYN_PATH.parent

EMB_OUT = OUT_DIR / "synthetic_embeddings.npy"
META_OUT = OUT_DIR / "synthetic_metadata.csv"

# =========================
# Load synthetic pool
# =========================
df = pd.read_csv(SYN_PATH)

required_cols = {"sample", "function", "source"}
missing = required_cols - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["sample"]).reset_index(drop=True)

texts = df["sample"].tolist()

print(f"Synthetic samples to embed: {len(texts)}")

# =========================
# Load embedding model
# =========================
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# =========================
# Compute embeddings
# =========================
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# =========================
# Save outputs
# =========================
np.save(EMB_OUT, embeddings)
df.to_csv(META_OUT, index=False)

print("\n✅ Synthetic embeddings saved:")
print(EMB_OUT)
print(META_OUT)
