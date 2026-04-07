# src/data_collection.py

import pandas as pd
from datasets import load_dataset
import kagglehub
from pathlib import Path


def run_data_collection():
    data_dir = Path("./data")
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
    print("daily_dialog")
    kagglehub.dataset_download(
        "thedevastator/dailydialog-unlock-the-conversation-potential-in",
        path=str(data_dir / "dailydialog")
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

