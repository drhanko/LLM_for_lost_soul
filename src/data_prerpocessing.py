import os
import re
import ast
import pandas as pd
from ast import literal_eval

# 3-class sentiment mapping
POSITIVE = {
    "joy", "joyful", "happiness", "amusement", "admiration", "love", "proud",
    "grateful", "gratitude", "excited", "excitement", "optimism", "relief",
    "content", "confident", "faithful", "trusting", "caring", "hopeful",
    "impressed", "approval", "desire"
}

NEGATIVE = {
    "sad", "sadness", "sentimental", "devastated", "disappointed", "grief",
    "remorse", "lonely", "guilty", "anger", "annoyed", "annoyance", "angry",
    "furious", "disapproval", "fear", "afraid", "terrified", "nervous",
    "nervousness", "anxious", "apprehensive", "disgust", "disgusted",
    "embarrassed", "embarrassment", "ashamed", "jealous"
}

NEUTRAL = {
    "neutral", "prepared", "curiosity", "confusion", "nostalgic"
}

# exclude the surprise label. This label can be categorized neither POSITIVE, NEGATIVE and NEUTRAL.
SURPRISE = {"surprise", "surprised", "realization"}



# condense the label from multiple to 3
def emotion_to_sentiment(emotion):
    if emotion is None:
        return None
    e = str(emotion).strip().lower()
    if not e:
        return None
    if e in SURPRISE:
        return None
    if e in POSITIVE:
        return "positive"
    if e in NEGATIVE:
        return "negative"
    if e in NEUTRAL:
        return "neutral"
    return None


# clean the section of the data
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


def split_dialog_text(dialog_text):
    utterances = re.split(r'(?<=[.!?])\s+', dialog_text.strip())
    utterances = [x.strip() for x in utterances if x.strip()]
    return utterances


# format functions
def format_empathetic(df, split_name):
    rows = []

    for _, r in df.iterrows():
        text = clean_text(r.get("text", ""))
        emotion = str(r.get("emotion", "")).strip()

        if not text or not emotion:
            continue

        gen_emo = emotion_to_sentiment(emotion)
        if gen_emo is None:
            continue  # skip surprise and unknown labels

        rows.append({
            "source": "empathetic_dialogues",
            "split": split_name,
            "instruction": "Classify the sentiment.",
            "input": text,
            "output": gen_emo,
            "meta": str(r.get("conversation_id", ""))
        })

    return rows


def format_dailydialog(df, split_name):
    rows = []

    # transform the numbers into text labels
    emotion_map_daily = {
        0: "neutral",
        1: "anger",
        2: "disgust",
        3: "fear",
        4: "happiness",
        5: "sadness",
        6: "surprise"
    }

    act_map = {
        1: "inform",
        2: "question",
        3: "directive",
        4: "commissive"
    }

    for _, r in df.iterrows():
        dialog = r.get("dialog", "")
        acts = r.get("act", "")
        emotions = r.get("emotion", "")

        if isinstance(dialog, str):
            try:
                dialog = literal_eval(dialog)
            except:
                dialog = [dialog]

        if isinstance(dialog, list) and len(dialog) == 1:
            dialog = split_dialog_text(dialog[0])

        if not isinstance(dialog, list):
            continue

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
                gen_emo = emotion_to_sentiment(raw_emotion)
            except:
                gen_emo = None

            if gen_emo is None:
                continue  # skip surprise / None labels

            try:
                act_text = act_map.get(int(acts[i]), "unknown")
            except:
                act_text = "unknown"

            # create prompt for the model training
            rows.append({
                "source": "dailydialog",
                "split": split_name,
                "instruction": "Classify the sentiment.",
                "input": text,
                "output": gen_emo,
                "meta": f"act={act_text}"
            })

    return rows


def goemotion_to_single_label(raw_labels, emotion_map_go):
    try:
        raw_labels = ast.literal_eval(raw_labels)
        if not isinstance(raw_labels, list):
            return None
    except:
        return None

    for first_label in raw_labels:
        try:
            raw_name = emotion_map_go[int(first_label)]
            sent = emotion_to_sentiment(raw_name)
            if sent is not None:
                return sent
        except:
            continue

    return None


def format_goemotions(df, split_name):
    rows = []

    # transform the numbers into text labels
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

    for _, r in df.iterrows():
        text = clean_text(r.get("text", ""))
        raw_labels = r.get("emotion", "")

        if not text:
            continue

        gen_emo = goemotion_to_single_label(raw_labels, emotion_map_go)
        if gen_emo is None:
            continue  # skip surprise / None labels

        rows.append({
            "source": "go_emotions",
            "split": split_name,
            "instruction": "Classify the sentiment.",
            "input": text,
            "output": gen_emo,
            "meta": ""
        })

    return rows

# avoid  chan_len > 1000 row data. This kind of data will ruin the results
def add_char_len(df):
    tmp = df.copy()
    tmp["char_len"] = (
        "<s>[INST] " + tmp["instruction"].astype(str) + "\n\n" +
        tmp["input"].astype(str) + " [/INST] " +
        tmp["output"].astype(str)
    ).str.len()
    return tmp

def filter_by_char_len(df, max_len=1000):
    df = add_char_len(df)
    before = len(df)
    df = df[df["char_len"] <= max_len].copy()
    after = len(df)
    print(f"Filter char_len > {max_len}: {before} -> {after}")
    return df.drop(columns=["char_len"])
#####


def global_balanced_sample(df, label_col="output", total_n=10000, random_state=404):
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
    rows = []

    if split_name == "mixed":
        ed = load_csv(f"{base}/empathetic_dialogues_train.csv")
        dd = load_csv(f"{base}/dailydialog_train.csv")
        ge = load_csv(f"{base}/go_emotions_train.csv")
        rows += format_empathetic(ed, "train")
        rows += format_dailydialog(dd, "train")
        rows += format_goemotions(ge, "train")
    elif split_name == "empathetic":
        ed = load_csv(f"{base}/empathetic_dialogues_train.csv")
        rows += format_empathetic(ed, "train")
    elif split_name == "dailydialog":
        dd = load_csv(f"{base}/dailydialog_train.csv")
        rows += format_dailydialog(dd, "train")
    elif split_name == "goemotions":
        ge = load_csv(f"{base}/go_emotions_train.csv")
        rows += format_goemotions(ge, "train")
    else:
        raise ValueError(f"Unknown split: {split_name}")

    return pd.DataFrame(rows)


def data_prerpocessing(name):
    base = "../data"

    #build
    ready_to_split_df = build_split_df(base,name)

    #remove the test length > 1000
    ready_to_split_df = filter_by_char_len(ready_to_split_df, max_len=1000)

    # balanced sample
    ready_to_split_df_balanced = global_balanced_sample(ready_to_split_df, total_n=10000, random_state=404)
    save_df(ready_to_split_df_balanced, f"../data/clean_data/{name}/ready_to_split.jsonl")

    print("Saved processed datasets.")
    print("train:", len(ready_to_split_df_balanced))
    print("train label counts:\n", ready_to_split_df_balanced["output"].value_counts())
