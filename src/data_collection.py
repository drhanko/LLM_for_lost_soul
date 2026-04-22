# src/data_collection.py
import pandas as pd
from datasets import load_dataset
import kagglehub
import shutil
from pathlib import Path


def run_data_collection(DATA_DIR,EMPATHETIC_URL,DAILYDIALOG_URL,GOEMOTIONS_URL):
    # empathetic_dialogues
    print("empathetic_dialogues downloading......")
    dir = Path(DATA_DIR)
    empathetic_dialogues = load_dataset(EMPATHETIC_URL, trust_remote_code=True)

    for split in ["train", "validation", "test"]:
        rows = []
        for item in empathetic_dialogues[split]:
            rows.append({
                "dataset": "empathetic_dialogues",
                "conversation_id": item["conv_id"],
                "role": "user" if item["speaker_idx"] == 0 else "assistant",
                "text": item["utterance"],
                "emotion": item["context"]
            })

        df = pd.DataFrame(rows)
        df.to_csv(dir / f"empathetic_dialogues_{split}.csv", index=False)

    # daily_dialog
    print("dailydialog downloading......")
    kagglehub.dataset_download(
        DAILYDIALOG_URL,
        output_dir=str(dir / "dailydialog"),
    )
    # go_emotions
    print("go_emotions downloading......")
    go_emotions = load_dataset( GOEMOTIONS_URL,trust_remote_code=True)

    for split in ["train", "validation", "test"]:
        rows = []
        for idx, item in enumerate(go_emotions[split]):
            rows.append({
                "dataset": "go_emotions",
                "conversation_id": idx,
                "role": "user",
                "text": item["text"],
                "emotion": item["labels"]
            })

        df = pd.DataFrame(rows)
        df.to_csv(dir / f"go_emotions_{split}.csv", index=False)


def move_and_rename_dailydialog(src_dir,dst_dir):

    mapping = {
        "train.csv": "dailydialog_train.csv",
        "validation.csv": "dailydialog_validation.csv",
        "test.csv": "dailydialog_test.csv",
    }

    for src_name, dst_name in mapping.items():
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)
        src_file = src_dir / src_name
        dst_file = dst_dir / dst_name

        if src_file.exists():
            shutil.move(str(src_file), str(dst_file))
            print(f"Moved: {src_name} -> {dst_name}")
        else:
            print(f"Not found: {src_file}")

def test_data_collection(dir):
    print("test_data_collection")
    dir = Path(dir)
    # check whether the csv is output successfully or not

    expected_files = [
        "empathetic_dialogues_train.csv",
        "go_emotions_train.csv",
        "dailydialog_train.csv"
    ]

    for f in expected_files:
        assert (dir / f).exists(), f"{f} not found"