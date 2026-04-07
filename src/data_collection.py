# src/data_collection.py
import pandas as pd
from datasets import load_dataset
import kagglehub
import shutil
from pathlib import Path


def run_data_collection():
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)

    # empathetic_dialogues
    print("empathetic_dialogues")
    empathetic_dialogues = load_dataset("empathetic_dialogues", trust_remote_code=True)

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
        df.to_csv(data_dir / f"empathetic_dialogues_{split}.csv", index=False)

    # daily_dialog
    kagglehub.dataset_download(
        "thedevastator/dailydialog-unlock-the-conversation-potential-in",
        output_dir=str(data_dir / "dailydialog"),
    )
    # go_emotions
    print("go_emotions")
    go_emotions = load_dataset("go_emotions", trust_remote_code=True)

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
        df.to_csv(data_dir / f"go_emotions_{split}.csv", index=False)


def move_and_rename_dailydialog():
    src_dir = Path("../data/dailydialog")
    dst_dir = Path("../data")

    mapping = {
        "train.csv": "dailydialog_train.csv",
        "validation.csv": "dailydialog_validation.csv",
        "test.csv": "dailydialog_test.csv",
    }

    for src_name, dst_name in mapping.items():
        src_file = src_dir / src_name
        dst_file = dst_dir / dst_name

        if src_file.exists():
            shutil.move(str(src_file), str(dst_file))
            print(f"Moved: {src_name} -> {dst_name}")
        else:
            print(f"Not found: {src_file}")