import torch
import pandas as pd
import json
from pathlib import Path

from transformers import RobertaTokenizer, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
SPLIT_ID = 1

BASE_DIR = Path("/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment")

MODEL_NAME = "roberta-base"
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_FUNCTIONS = ["DIR", "DM", "INTJ", "AS"]

# =========================
# PATHS
# =========================
split_dir = BASE_DIR / "splits" / f"split_{SPLIT_ID}"

train_path = split_dir / "real_train.csv"
test_path  = split_dir / "real_test.csv"

out_dir = BASE_DIR / "results" / f"split_{SPLIT_ID}" / "REAL_ONLY_ROBERTA"
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD + CLEAN
# =========================
train_df = pd.read_csv(train_path)[["sample", "function"]].dropna()
test_df  = pd.read_csv(test_path)[["sample", "function"]].dropna()

train_df["function"] = train_df["function"].astype(str).str.strip().str.upper()
test_df["function"]  = test_df["function"].astype(str).str.strip().str.upper()

train_df = train_df[train_df["function"].isin(ALL_FUNCTIONS)].reset_index(drop=True)
test_df  = test_df[test_df["function"].isin(ALL_FUNCTIONS)].reset_index(drop=True)

print("\nTrain label distribution:")
print(train_df["function"].value_counts())
print("\nTest label distribution:")
print(test_df["function"].value_counts())

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
le.fit(ALL_FUNCTIONS)

y_train = le.transform(train_df["function"])
y_test  = le.transform(test_df["function"])

# =========================
# LOAD ROBERTA (FROZEN)
# =========================
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()  # 🔒 frozen

@torch.no_grad()
def encode_sentences(sentences, batch_size=16, max_length=128):
    embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state  # (B, T, H)
        attention_mask = encoded["attention_mask"].unsqueeze(-1)

        # Mean pooling (masked)
        pooled = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
        embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0).numpy()

# =========================
# EMBEDDING
# =========================
X_train = encode_sentences(train_df["sample"].tolist())
X_test  = encode_sentences(test_df["sample"].tolist())

# =========================
# TRAIN CLASSIFIER
# =========================
clf = LogisticRegression(
    max_iter=2000,
    multi_class="multinomial",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

report = classification_report(
    y_test, y_pred,
    labels=list(range(len(ALL_FUNCTIONS))),
    target_names=ALL_FUNCTIONS,
    output_dict=True,
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred, labels=list(range(len(ALL_FUNCTIONS))))

print("\n=== RESULTS (RoBERTa) ===")
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print("Per-class F1:", {lbl: round(report[lbl]["f1-score"], 4) for lbl in ALL_FUNCTIONS})

# =========================
# SAVE RESULTS
# =========================
metrics = {
    "split": SPLIT_ID,
    "encoder": "roberta-base (frozen)",
    "condition": "REAL_ONLY_BASELINE",
    "accuracy": acc,
    "macro_f1": macro_f1,
    "train_label_distribution": train_df["function"].value_counts().to_dict(),
    "test_label_distribution": test_df["function"].value_counts().to_dict(),
    "classification_report": report,
    "confusion_matrix_labels": ALL_FUNCTIONS,
    "confusion_matrix": cm.tolist()
}

with open(out_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

pred_df = test_df.copy()
pred_df["predicted_function"] = le.inverse_transform(y_pred)
pred_df.to_csv(out_dir / "predictions.csv", index=False)

print("\nSaved to:", out_dir)
