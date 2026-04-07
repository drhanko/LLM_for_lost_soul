import pandas as pd
from transformers import AutoTokenizer

JSONL_PATH = "../data/clean_data/train.jsonl"
TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_LEN = 256

def build_prompt(instruction, input_text):
    return (
        f"<s>[INST] {instruction}\n\n"
        f"{input_text} [/INST] "
    )

def check_dataform():
    try:
        df = pd.read_json(JSONL_PATH, lines=True)
    except Exception as e:
        print("Fail to read the file", e)
        return

    print("Read the file successfully")
    print("rows =", len(df))
    print("columns =", df.columns.tolist())
    print()

    # 2) 欄位檢查
    required = {"instruction", "input", "output"}
    missing = required - set(df.columns)
    if missing:
        print("Missing the column", missing)
        return
    print("Column are all set")
    print()

    # data form check
    bad_rows = []
    for idx, row in df.iterrows():
        for col in ["instruction", "input", "output"]:
            val = row.get(col, None)
            if pd.isna(val) or str(val).strip() == "":
                bad_rows.append((idx, col))
    if bad_rows:
        print("Nan values")
        print(bad_rows[:20])
    else:
        print("Nan values are not exist")
    print()

    #check the label
    labels = df["output"].astype(str).str.strip().str.lower()
    print("label distribution:")
    print(labels.value_counts())
    print()


    print("sample rows:")
    for i in range(min(3, len(df))):
        inst = str(df.iloc[i]["instruction"]).strip()
        inp = str(df.iloc[i]["input"]).strip()
        out = str(df.iloc[i]["output"]).strip()
        prompt = build_prompt(inst, inp)
        print(f"\n--- sample {i} ---")
        print("PROMPT:")
        print(prompt)
        print("OUTPUT:")
        print(out)
    print()

    # tokenizer check
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("tokenizer worked")
    except Exception as e:
        print("tokenizer failed", e)
        return

    # token length check
    too_long = 0
    lengths = []
    for i in range(len(df)):
        inst = str(df.iloc[i]["instruction"]).strip()
        inp = str(df.iloc[i]["input"]).strip()
        out = str(df.iloc[i]["output"]).strip()
        text = build_prompt(inst, inp) + out + tokenizer.eos_token

        try:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            lengths.append(len(ids))
            if len(ids) > MAX_LEN:
                too_long += 1
        except Exception as e:
            print(f"tokenize failed at row {i}: {e}")
            return

    if lengths:
        print("token length stats")
        print("min =", min(lengths))
        print("max =", max(lengths))
        print("avg =", sum(lengths) / len(lengths))
        print(f"over MAX_LEN({MAX_LEN}) =", too_long)


    df = pd.read_json(JSONL_PATH, lines=True)
    df["text_for_check"] = (
            "<s>[INST] " + df["instruction"].astype(str) + "\n\n" +
            df["input"].astype(str) + " [/INST] " +
            df["output"].astype(str)
    )

    df["char_len"] = df["text_for_check"].str.len()
    print(df.sort_values("char_len", ascending=False)[["source", "split", "input", "output", "char_len"]].head(10))
