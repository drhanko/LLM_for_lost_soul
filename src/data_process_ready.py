import json
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from config import SPLIT_DATA_DIR,OUTPUT_LABELS
from pathlib import Path
# read the data from the processed data

def load_jsonl(path):
    rows = []
    LABELS = OUTPUT_LABELS
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # different from reprocess, this data fetch is for roberta-base to improve performance
            text = str(obj.get("input", "")).strip().replace("_comma_", ",")
            label = str(obj.get("output", "")).strip().lower()
            if not text or label not in LABELS:
                continue
            rows.append({"input": text, "output": label})
    return rows


# save the data into json file
def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def process_data_split()-> pd.DataFrame:
    # split_Data
    dir = Path(SPLIT_DATA_DIR)
    dataset_names = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
    print(dataset_names)
    for dataset_name in dataset_names:
        dataset_name = Path(dataset_name)
        INPUT_PATH = f"{dir}/{dataset_name}/ready_to_split.jsonl"
        OUTPUT_DIR = f"{dir}/{dataset_name}/split"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        data = load_jsonl(INPUT_PATH)

        texts = [x["input"] for x in data]
        labels = [x["output"] for x in data]

        # Split the train data into validation & test data
        x_train, x_temp, y_train, y_temp = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        # Split the rest data into validation & test data
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )

        train_rows = [{"input": x, "output": y} for x, y in zip(x_train, y_train)]
        val_rows = [{"input": x, "output": y} for x, y in zip(x_val, y_val)]
        test_rows = [{"input": x, "output": y} for x, y in zip(x_test, y_test)]

        save_jsonl(train_rows, f"{OUTPUT_DIR}/train.jsonl")
        save_jsonl(val_rows, f"{OUTPUT_DIR}/validation.jsonl")
        save_jsonl(test_rows, f"{OUTPUT_DIR}/test.jsonl")

        print("The data description")
        print(f"Dataset : {dataset_name}")
        print("train:", len(train_rows))
        print("val:", len(val_rows))
        print("test:", len(test_rows))