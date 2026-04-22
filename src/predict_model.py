import json
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from config import *
from data_loader import Encodeded_Dataset


OUTPUT_DIR = RESULTS_DIR
MODEL_NAME =MODEL
LABELS = OUTPUT_LABELS

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

SEEDS = SEEDS_SETS  # determine how many seeds should be create in the ensemble cluster
MAX_LENGTH = MAX_LENGTH_c
NUM_EPOCHS = NUM_EPOCHS_c
LEARNING_RATE = LEARNING_RATE_c
WEIGHT_DECAY = WEIGHT_DECAY_c

if torch.cuda.is_available():
    TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE_c
    EVAL_BATCH_SIZE = EVAL_BATCH_SIZE_c
else:
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 8

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# the difference between load_jsonl in data_process_ready is this version provide dimention control
def load_jsonl_for_model(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = str(obj.get("input", "")).strip().replace("_comma_", ",")
            label = str(obj.get("output", "")).strip().lower()
            if not text or label not in label2id:
                continue
            rows.append({"text": text, "label": label2id[label]})
    return rows

def build_dataset(rows, tokenizer):
    texts = [r["text"] for r in rows]
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return Encodeded_Dataset(encodings, labels)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_count = 0
# AI generated: provided the torch cuda setting
# ++++++++++++++++++++++++++++++
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)

            bs = batch["labels"].size(0)
            total_loss += loss.item() * bs
            total_count += bs

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())

# ++++++++++++++++++++++++++++++
    avg_loss = total_loss / max(total_count, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
    }




# AI polished code: focus on torch setup
# ++++++++++++++++++++++++++++++
def train_one_seed(seed, train_rows, val_rows, test_rows,SPECIFIC_DATASET):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== Seed {seed} | device = {device} =====")

    # get the pretrained model to run better
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    train_ds = build_dataset(train_rows, tokenizer)
    val_ds = build_dataset(val_rows, tokenizer)
    test_ds = build_dataset(test_rows, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            bs = batch["labels"].size(0)
            running_loss += loss.item() * bs
            seen += bs

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device)
# ++++++++++++++++++++++++++++++

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # restore best weights
    model.load_state_dict(best_state)

    # final eval on test
    test_metrics = evaluate(model, test_loader, device)

    # save only one final model per seed
    seed_dir = Path(OUTPUT_DIR)/ Path(SPECIFIC_DATASET) / f"seed_{seed}" / "best_model"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(seed_dir))
    tokenizer.save_pretrained(str(seed_dir))

    print(f"\nBest epoch for seed {seed}: {best_epoch}")
    print(f"Saved to: {seed_dir}")
    print(f"Test acc: {test_metrics['accuracy']:.4f}")
    print(f"Test f1 : {test_metrics['f1_macro']:.4f}")
    print("Confusion matrix:")
    print(test_metrics["confusion_matrix"])

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_f1": best_val_f1,
        "test_acc": test_metrics["accuracy"],
        "test_f1": test_metrics["f1_macro"],
        "model_dir": str(seed_dir),
    }


def execute_model_training(SPECIFIC_DATASET):
    TRAIN_PATH = f"{SPLIT_DATA_DIR}/{SPECIFIC_DATASET}/split/train.jsonl"
    VAL_PATH = f"{SPLIT_DATA_DIR}/{SPECIFIC_DATASET}/split/validation.jsonl"
    TEST_PATH = f"{SPLIT_DATA_DIR}/{SPECIFIC_DATASET}/split/test.jsonl"

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Dataset_link is not exist")

    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    else:
        raise ValueError("cuda is not available. Please use ")
    print("model:", MODEL_NAME)

    train_rows = load_jsonl_for_model(TRAIN_PATH)
    val_rows = load_jsonl_for_model(VAL_PATH)
    test_rows = load_jsonl_for_model(TEST_PATH)

    if not train_rows or not val_rows or not test_rows:
        raise ValueError("one of the file is missing")
    print("train:", len(train_rows))
    print("val:", len(val_rows))
    print("test:", len(test_rows))

    results = []
    for seed in SEEDS:
        results.append(train_one_seed(seed, train_rows, val_rows, test_rows,SPECIFIC_DATASET))

    print("\n===== Summary =====")
    for r in results:
        print(
            f"seed={r['seed']} | best_epoch={r['best_epoch']} | "
            f"val_f1={r['val_f1']:.4f} | test_f1={r['test_f1']:.4f} | test_acc={r['test_acc']:.4f}"
        )
