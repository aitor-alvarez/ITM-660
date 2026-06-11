"""
Yelp Reviews Text Classification using BERT
Fine-tunes bert-base-uncased on 5-class star rating prediction.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
_BASE_DIR    = Path(__file__).parent if "__file__" in dir() else Path.cwd()
DATA_PATH    = _BASE_DIR / "Session 6" / "yelp_reviews.parquet"
MODEL_NAME   = "bert-base-uncased"
OUTPUT_DIR   = _BASE_DIR / "bert_yelp_output"
NUM_LABELS   = 5
MAX_LEN      = 128
SAMPLE_SIZE  = 20_000   # rows to sample from the full 650 k for faster training
                         # set to None to use the entire dataset
TEST_SIZE    = 0.2
BATCH_SIZE   = 16
NUM_EPOCHS   = 3
LR           = 2e-5
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Dataset ───────────────────────────────────────────────────────────────────
class YelpDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Yelp Reviews — BERT Fine-tuning")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # 1. Load data
    print(f"Loading data from {DATA_PATH} …")
    df = pd.read_parquet(DATA_PATH)
    print(f"Full dataset: {df.shape}  |  columns: {df.columns.tolist()}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}\n")

    # 2. Optional subsample (stratified so class balance is preserved)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        df, _ = train_test_split(
            df, train_size=SAMPLE_SIZE, stratify=df["label"], random_state=SEED
        )
        print(f"Sampled {len(df):,} rows (stratified)\n")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=SEED,
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # 4. Tokenise
    print(f"Loading tokenizer: {MODEL_NAME} …")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    print("Tokenising train …")
    train_enc = tokenizer(
        X_train, truncation=True, padding=True, max_length=MAX_LEN
    )
    print("Tokenising test …")
    test_enc  = tokenizer(
        X_test,  truncation=True, padding=True, max_length=MAX_LEN
    )

    train_dataset = YelpDataset(train_enc, y_train)
    test_dataset  = YelpDataset(test_enc,  y_test)

    # 5. Load pre-trained BERT for classification
    print(f"\nLoading model: {MODEL_NAME} with {NUM_LABELS} output classes …")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # 6. Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. Train
    print("\n── Training ──────────────────────────────────────────────")
    trainer.train()

    # 8. Evaluate
    print("\n── Evaluation ────────────────────────────────────────────")
    results = trainer.evaluate()
    print(f"Test accuracy: {results['eval_accuracy']:.4f}")

    preds_output = trainer.predict(test_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[f"★{i+1}" for i in range(NUM_LABELS)]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 9. Save best model
    best_path = OUTPUT_DIR / "best_model"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    print(f"\nBest model saved to {best_path}")


if __name__ == "__main__":
    main()