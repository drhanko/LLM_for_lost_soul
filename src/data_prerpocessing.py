import pandas as pd



label_map = {
    # ----------------
    # anger
    # ----------------
    "annoyed": "anger",
    "annoyance": "anger",
    "angry": "anger",
    "anger": "anger",
    "furious": "anger",
    "disapproval": "anger",

    # ----------------
    # sadness
    # ----------------
    "sad": "sadness",
    "sadness": "sadness",
    "sentimental": "sadness",
    "devastated": "sadness",
    "disappointed": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "lonely": "sadness",
    "guilty": "sadness",

    # ----------------
    # joy
    # ----------------
    "joy": "joy",
    "joyful": "joy",
    "happiness": "joy",
    "amusement": "joy",
    "admiration": "joy",
    "love": "joy",
    "proud": "joy",
    "grateful": "joy",
    "gratitude": "joy",
    "excited": "joy",
    "excitement": "joy",
    "optimism": "joy",
    "relief": "joy",
    "content": "joy",
    "confident": "joy",
    "faithful": "joy",
    "trusting": "joy",
    "caring": "joy",
    "hopeful": "joy",
    "impressed": "joy",

    # ----------------
    # fear
    # ----------------
    "fear": "fear",
    "afraid": "fear",
    "terrified": "fear",
    "nervous": "fear",
    "nervousness": "fear",
    "anxious": "fear",
    "apprehensive": "fear",

    # ----------------
    # disgust
    # ----------------
    "disgust": "disgust",
    "disgusted": "disgust",

    # ----------------
    # surprise
    # ----------------
    "surprised": "surprise",
    "surprise": "surprise",
    "realization": "surprise",

    # ----------------
    # neutral
    # ----------------
    "neutral": "neutral",
    "approval": "neutral",
    "prepared": "neutral",
    "curiosity": "neutral",
    "confusion": "neutral",
    "desire": "neutral",
    "anticipating": "neutral",
    "nostalgic": "neutral",
    "embarrassed": "neutral",
    "embarrassment": "neutral",
}
# for empathetic
emotion_map_emp = [
    "afraid",
    "angry",
    "annoyed",
    "anticipating",
    "anxious",
    "apprehensive",
    "ashamed",
    "caring",
    "confident",
    "content",
    "devastated",
    "disappointed",
    "disgusted",
    "embarrassed",
    "excited",
    "faithful",
    "furious",
    "grateful",
    "guilty",
    "hopeful",
    "impressed",
    "jealous",
    "joyful",
    "lonely",
    "nostalgic",
    "prepared",
    "proud",
    "sad",
    "sentimental",
    "surprised",
    "terrified",
    "trusting"
]

# for daily dialog
act_map = {
    1: "inform",
    2: "question",
    3: "directive",
    4: "commissive"
}

emotion_map_daily = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}

#
emotion_map_go = {
  0: "admiration",
  1: "amusement",
  2: "anger",
  3: "annoyance",
  4: "approval",
  5: "caring",
  6: "confusion",
  7: "curiosity",
  8: "desire",
  9: "disappointment",
  10: "disapproval",
  11: "disgust",
  12: "embarrassment",
  13: "excitement",
  14: "fear",
  15: "gratitude",
  16: "grief",
  17: "joy",
  18: "love",
  19: "nervousness",
  20: "optimism",
  21: "pride",
  22: "realization",
  23: "relief",
  24: "remorse",
  25: "sadness",
  26: "surprise",
  27: "neutral"
}
def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_csv(path):
    return pd.read_csv(path)

def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)

# =========================
#  Standardize column for these datasets
#     "source": "empathetic_dialogues"/"dailydialog"/"go_emotions"
#     "split": "train"/"validation"/"test",
#     "instruction": "Respond empathetically."/"Predict the emotion."/"Respond naturally." (Alpaca / Instruction tuning)
#     "input": user_input, (input → output mapping)
#     "output": emotions,
#     "meta": appendix
# =========================

def format_empathetic(df, split_name):
    rows = []

    for _, r in df.iterrows():
        text = clean_text(r.get("text", ""))
        emotion = str(r.get("emotion", "")).strip()

        if not text or not emotion:
            continue

        try:
            gen_emo = label_map.get(emotion.lower(), "neutral")
        except:
            gen_emo = "neutral"

        rows.append({
            "source": "empathetic_dialogues",
            "split": split_name,
            "instruction": "Respond empathetically.",
            "input": text,
            "output": gen_emo,
            "meta": str(r.get("conversation_id", ""))
        })

    return rows


def split_dialog_text(dialog_text):
    # 先用標點切句
    utterances = re.split(r'(?<=[.!?])\s+', dialog_text.strip())
    utterances = [x.strip() for x in utterances if x.strip()]
    return utterances

def format_dailydialog(df, split_name):
    rows = []

    for _, r in df.iterrows():
        dialog = r.get("dialog", "")
        acts = r.get("act", "")
        emotions = r.get("emotion", "")

        # dialog 是 list，而且只有一個完整字串
        if isinstance(dialog, str):
            try:
                dialog = literal_eval(dialog)
            except:
                dialog = [dialog]

        if isinstance(dialog, list) and len(dialog) == 1:
            dialog_text = dialog[0]
            dialog = split_dialog_text(dialog_text)

        if not isinstance(dialog, list):
            continue

        # act / emotion 轉 list
        if isinstance(acts, str):
            try:
                acts = literal_eval(acts)
            except:
                acts = list(map(int, acts.strip("[]").split()))

        if isinstance(emotions, str):
            try:
                emotions = literal_eval(emotions)
            except:
                emotions = list(map(int, emotions.strip("[]").split()))

        n = min(len(dialog), len(acts), len(emotions))

        for i in range(n):
            text = clean_text(dialog[i])


            if not text:
                continue

            try:
                raw_emotion = emotion_map_daily[int(emotions[i])]
                gen_emo = label_map.get(raw_emotion.lower(), "neutral")
            except:
                gen_emo = "neutral"

            try:
                act_text = act_map.get(int(acts[i]), "unknown")
            except:
                act_text = "unknown"

            rows.append({
                "source": "dailydialog",
                "split": split_name,
                "instruction": "Predict the emotion.",
                "input": text,
                "output": gen_emo,
                "meta": f"act={act_text}"
            })

    return rows

def format_goemotions(df, split_name):
    rows = []

    for _, r in df.iterrows():
        text = clean_text(r.get("text", ""))
        raw_labels = r.get("emotion", "")

        if not text:
            continue

        output_label = goemotion_to_single_label(
            raw_labels,
            label_map,
            emotion_map_go
        )
        gen_emo = label_map.get(output_label.lower(), "neutral")


        # print(output_label)
        # print(gen_emo)

        try:
            gen_emo = label_map.get(output_label.lower(), "neutral")
        except:
            gen_emo = "neutral"

        rows.append({
            "source": "go_emotions",
            "split": split_name,
            "instruction": "Predict the emotion label from the text.",
            "input": text,
            "output": gen_emo,
            "meta": ""
        })

    return rows
# There are multiple labels for 1 sentence. Therefore, either pick the first emotion as label or use multi-label method

# goemotion_to_single_label: pick the first emotion as label
def goemotion_to_single_label(raw_labels, label_map, emotion_map_go):
    # 1. 轉成 list
    raw_labels = ast.literal_eval(raw_labels)
    first_label = raw_labels[0]

    try:
        # 3. 數字 → 原始 label
        raw_name = emotion_map_go[int(first_label)]

        # 4. 原始 label → 統一 label
        final_label = label_map.get(raw_name.lower(), "neutral")

        return final_label

    except Exception:
        return "neutral"


import os
import re
import ast
import pandas as pd
from ast import literal_eval


# ---------- 你原本的 mapping 保留 ----------
# label_map, emotion_map_daily, emotion_map_go, act_map, clean_text, load_csv, save_df
# format_empathetic, format_dailydialog, format_goemotions, goemotion_to_single_label
# 都可以沿用


def global_balanced_sample(df, label_col="output", total_n=5000, random_state=404):
    if df.empty:
        return df

    labels = sorted(df[label_col].dropna().unique().tolist())
    num_labels = len(labels)

    if num_labels == 0:
        return df.iloc[0:0].copy()

    base = total_n // num_labels
    remainder = total_n % num_labels

    target_counts = {}
    for i, label in enumerate(labels):
        target_counts[label] = base + (1 if i < remainder else 0)

    sampled_parts = []
    for label, group in df.groupby(label_col):
        n = target_counts.get(label, 0)
        if n == 0:
            continue

        replace = len(group) < n
        sampled_parts.append(group.sample(n=n, replace=replace, random_state=random_state))

    result = pd.concat(sampled_parts, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result


def build_split_df(base, split_name):
    if split_name == "train":
        ed = load_csv(f"{base}/empathetic_dialogues_train.csv")
        dd = load_csv(f"{base}/dailydialog_train.csv")
        ge = load_csv(f"{base}/go_emotions_train.csv")
    elif split_name == "validation":
        ed = load_csv(f"{base}/empathetic_dialogues_validation.csv")
        dd = load_csv(f"{base}/dailydialog_validation.csv")
        ge = load_csv(f"{base}/go_emotions_validation.csv")
    elif split_name == "test":
        ed = load_csv(f"{base}/empathetic_dialogues_test.csv")
        dd = load_csv(f"{base}/dailydialog_test.csv")
        ge = load_csv(f"{base}/go_emotions_test.csv")
    else:
        raise ValueError(f"Unknown split: {split_name}")

    rows = []
    rows += format_empathetic(ed, split_name)
    rows += format_dailydialog(dd, split_name)
    rows += format_goemotions(ge, split_name)

    return pd.DataFrame(rows)


def data_prerpocessing():
    base = "../data"

    # 先做三個 split
    train_df = build_split_df(base, "train")
    val_df = build_split_df(base, "validation")
    test_df = build_split_df(base, "test")

    # 再各自做全局平衡
    train_df_balanced = global_balanced_sample(train_df, total_n=15000, random_state=404)
    val_df_balanced   = global_balanced_sample(val_df,   total_n=3000,  random_state=404)
    test_df_balanced  = global_balanced_sample(test_df,  total_n=3000,  random_state=404)

    # 存檔
    save_df(train_df_balanced, "../data/clean_data/train.jsonl")
    save_df(val_df_balanced,   "../data/clean_data/validation.jsonl")
    save_df(test_df_balanced,  "../data/clean_data/test.jsonl")

    print("Saved processed datasets.")
    print("train:", len(train_df_balanced))
    print("validation:", len(val_df_balanced))
    print("test:", len(test_df_balanced))
    print("train label counts:\n", train_df_balanced["output"].value_counts())
    print("validation label counts:\n", val_df_balanced["output"].value_counts())
    print("test label counts:\n", test_df_balanced["output"].value_counts())

