import pandas as pd
import json
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# =========================
# REPRODUCIBILITY
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# =========================
# CONFIG
# =========================
BASE_DIR = Path("/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment")

SPLITS = [1, 2, 3, 4, 5]

ALL_FUNCTIONS = ["DIR", "DM", "INTJ", "AS"]

CONDITIONS = [
    "REAL_ONLY",
    "REAL_PLUS_NEAR",
    "REAL_PLUS_MIDDLE",
    "REAL_PLUS_FAR",
    "REAL_PLUS_RANDOM",
    "REAL_PLUS_DISTANCE_BALANCED"
]

MODEL_NAME = "bert-base-uncased"

# =========================
# DATASET CLASS
# =========================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# =========================
# MAIN LOOP
# =========================
summary_results = []

for SPLIT_ID in SPLITS:

    print(f"\n########################")
    print(f"Running Split {SPLIT_ID}")
    print(f"########################")

    split_dir = BASE_DIR / "splits" / f"split_{SPLIT_ID}"
    test_path = split_dir / "real_test.csv"

    test_df = pd.read_csv(test_path)[["sample", "function"]].dropna()
    test_df["function"] = test_df["function"].astype(str).str.strip().str.upper()
    test_df = test_df[test_df["function"].isin(ALL_FUNCTIONS)]

    label_encoder = LabelEncoder()
    label_encoder.fit(ALL_FUNCTIONS)

    y_test = label_encoder.transform(test_df["function"])

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    for CONDITION in CONDITIONS:

        print(f"\nCondition: {CONDITION}")

        train_path = split_dir / "conditions" / CONDITION / "train.csv"

        train_df = pd.read_csv(train_path)[["sample", "function"]].dropna()
        train_df["function"] = train_df["function"].astype(str).str.strip().str.upper()
        train_df = train_df[train_df["function"].isin(ALL_FUNCTIONS)]

        y_train = label_encoder.transform(train_df["function"])

        # -------------------------
        # Compute class weights
        # -------------------------
        class_weights_np = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.tensor(class_weights_np, dtype=torch.float)

        train_dataset = TextDataset(
            train_df["sample"].tolist(), y_train, tokenizer
        )
        test_dataset = TextDataset(
            test_df["sample"].tolist(), y_test, tokenizer
        )

        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(ALL_FUNCTIONS)
        ).to(DEVICE)

        # -------------------------
        # Custom Trainer
        # -------------------------
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")

                loss_fct = nn.CrossEntropyLoss(
                    weight=class_weights.to(logits.device)
                )
                loss = loss_fct(
                    logits.view(-1, model.num_labels),
                    labels.view(-1)
                )
                return (loss, outputs) if return_outputs else loss

        # -------------------------
        # Metrics
        # -------------------------
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
            preds = np.argmax(probs, axis=1)

            acc = accuracy_score(labels, preds)
            macro_f1 = f1_score(labels, preds, average="macro")

            auc_per_class = {}
            for i, label_name in enumerate(ALL_FUNCTIONS):
                try:
                    auc_per_class[label_name] = roc_auc_score(
                        (labels == i).astype(int),
                        probs[:, i]
                    )
                except:
                    auc_per_class[label_name] = np.nan

            macro_auc = np.nanmean(list(auc_per_class.values()))

            return {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "macro_auc": macro_auc,
                **{f"AUC_{k}": v for k, v in auc_per_class.items()}
            }

        training_args = TrainingArguments(
            output_dir=str(BASE_DIR / "tmp_checkpoints"),
            do_train=True,
            do_eval=True,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_steps=50,
            seed=SEED,
            save_strategy="no"
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics = trainer.evaluate()

        summary_results.append({
            "split": SPLIT_ID,
            "condition": CONDITION,
            "accuracy": metrics["eval_accuracy"],
            "macro_f1": metrics["eval_macro_f1"],
            "macro_auc": metrics["eval_macro_auc"],
            "AUC_DIR": metrics.get("eval_AUC_DIR"),
            "AUC_DM": metrics.get("eval_AUC_DM"),
            "AUC_INTJ": metrics.get("eval_AUC_INTJ"),
            "AUC_AS": metrics.get("eval_AUC_AS"),
        })

# =========================
# SAVE RESULTS
# =========================
summary_df = pd.DataFrame(summary_results)
summary_path = BASE_DIR / "results_summary_all_splits.csv"
summary_df.to_csv(summary_path, index=False)

print("\n=== FINAL SUMMARY (ALL SPLITS) ===")
print(summary_df)
print("\nSaved to:", summary_path)