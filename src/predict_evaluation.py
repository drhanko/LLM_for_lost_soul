import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from config import *
from dotenv import load_dotenv
import gdown



MAX_LENGTH = MAX_LENGTH_c
BATCH_SIZE = TRAIN_BATCH_SIZE_c
LABELS = OUTPUT_LABELS
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

def load_jsonl_for_evaluation(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = str(obj.get("input", "")).strip().replace("_comma_", ",")
            label = str(obj.get("output", "")).strip().lower()
            row = {"text": text}
            if label in label2id:
                row["labels"] = label2id[label]
            rows.append(row)
    return rows

def predict_probs(model, tokenizer, texts):
    device = next(model.parameters()).device
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)

def evaluate_probs(probs, labels):
    preds = np.argmax(probs, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2]),
    }

def plot_multiclass_roc(y_true, y_prob, class_names, SAVE_PATH=None):
    """
    y_true: shape (N,)
    y_prob: shape (N, C)
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(8, 6))

    roc_auc_dict = {}
    valid_class_count = 0

    for i, name in enumerate(class_names):
        # skip if a class does not appear in y_true
        if len(np.unique(y_true_bin[:, i])) < 2:
            print(f"Skipping ROC for class '{name}' because only one class is present in y_true.")
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_dict[name] = roc_auc
        valid_class_count += 1

        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.4f})")

    # micro-average ROC
    if valid_class_count > 0:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(
            fpr_micro,
            tpr_micro,
            linewidth=2,
            label=f"micro-average (AUC = {auc_micro:.4f})",
        )

        try:
            auc_macro = roc_auc_score(
                y_true_bin,
                y_prob,
                average="macro",
                multi_class="ovr",
            )
            print(f"macro-average AUC = {auc_macro:.4f}")
        except Exception as e:
            print("macro-average AUC could not be computed:", e)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


    plt.title("Multiclass ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if SAVE_PATH is not None:
        plt.savefig(SAVE_PATH, dpi=200, bbox_inches="tight")
        print(f"ROC figure saved to: {SAVE_PATH}")

    plt.show()
    return auc_macro

def plot_precision_recall_bars(y_true, y_pred, class_names, title, SAVE_PATH=None):
    precision, recall, _, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0
    )

    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, precision, width, label="Precision")
    plt.bar(x + width / 2, recall, width, label="Recall")

    plt.xticks(x, class_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if SAVE_PATH is not None:
        plt.savefig(SAVE_PATH, dpi=200, bbox_inches="tight")
        print(f"Precision/Recall figure saved to: {SAVE_PATH}")

    plt.show()

    print("Per-class precision:", dict(zip(class_names, np.round(precision, 4))))
    print("Per-class recall   :", dict(zip(class_names, np.round(recall, 4))))
    print("Support            :", dict(zip(class_names, support)))


def plot_f1_acc_auc_metrics(metrics_dict, title, save_path=None):

    datasets = list(metrics_dict.keys())

    acc = metrics_dict["accuracy"]
    f1  = metrics_dict["f1_macro"]
    auc = metrics_dict["auc"]

    y = np.arange(len(datasets))
    height = 0.25

    plt.figure(figsize=(8,6))

    plt.barh(y - height, acc, height, label="Accuracy")
    plt.barh(y,          f1,  height, label="Macro-F1")
    plt.barh(y + height, auc, height, label="AUC")

    plt.yticks(y, datasets)
    plt.xlim(0, 1)
    plt.xlabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()

def evaluation(mode ,SPECIFIC_DATASET):

    if mode == "local":

        MODEL_DIRS = [
            f"{RESULTS_DIR}/seed_42/best_model",
            f"{RESULTS_DIR}/seed_43/best_model",
            f"{RESULTS_DIR}/seed_44/best_model",
        ]

        VAL_PATH = f"{SPLIT_DATA_DIR}/{SPECIFIC_DATASET}/splits/validation.jsonl"
        TEST_PATH = f"{SPLIT_DATA_DIR}/{SPECIFIC_DATASET}/splits/test.jsonl"

        print(MODEL_DIRS, VAL_PATH, TEST_PATH)

    elif mode == "google":

        load_dotenv()  # 讀 .env

        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

        url = f"https://drive.google.com/drive/folders/{folder_id}"
        print("[INFO] Downloading folder...")
        gdown.download_folder(url, output=RESULTS_DIR, quiet=False, use_cookies=False)

        MODEL_DIRS = [
            f"{RESULTS_DIR}/{SPECIFIC_DATASET}/final_models/seed_42/best_model",
            f"{RESULTS_DIR}/{SPECIFIC_DATASET}/final_models/seed_43/best_model",
            f"{RESULTS_DIR}/{SPECIFIC_DATASET}/final_models/seed_44/best_model",
        ]

        VAL_PATH = f"{RESULTS_DIR}/{SPECIFIC_DATASET}/validation.jsonl"
        TEST_PATH = f"{RESULTS_DIR}/{SPECIFIC_DATASET}/test.jsonl"

        print(MODEL_DIRS,VAL_PATH,TEST_PATH)

    raise ValueError("Invalid mode")
    val_data = load_jsonl_for_evaluation(VAL_PATH)
    test_data = load_jsonl_for_evaluation(TEST_PATH)

    val_texts = [x["text"] for x in val_data if "labels" in x]
    test_texts = [x["text"] for x in test_data if "labels" in x]
    val_labels = np.array([x["labels"] for x in val_data if "labels" in x], dtype=np.int64)
    test_labels = np.array([x["labels"] for x in test_data if "labels" in x], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRS[0], local_files_only=True)

    models = []
    for d in MODEL_DIRS:
        print("loading:", d)
        model = AutoModelForSequenceClassification.from_pretrained(
            d,
            local_files_only=True
        )
        model.to(device)
        models.append(model)

    print("\nRunning validation inference...")
    val_probs_sum = None
    for i, model in enumerate(models, 1):
        print(f"  model {i}/{len(models)}")
        probs = predict_probs(model, tokenizer, val_texts)
        val_probs_sum = probs if val_probs_sum is None else (val_probs_sum + probs)

    val_mix = val_probs_sum / len(models)
    val_metrics = evaluate_probs(val_mix, val_labels)

    print("\n===== Equal-weight val result =====")
    print(f"val_acc = {val_metrics['accuracy']:.4f}")
    print(f"val_f1  = {val_metrics['f1_macro']:.4f}")
    print("\nVal confusion matrix:")
    print(val_metrics["confusion_matrix"])

    print("\nPlotting validation ROC AUC...")
    auc_val = plot_multiclass_roc(
        val_labels,
        val_mix,
        LABELS,
        save_path="/content/roc_auc_val.png"
    )

    val_preds = np.argmax(val_mix, axis=-1)
    plot_precision_recall_bars(
        val_labels,
        val_preds,
        LABELS,
        title="Validation Precision / Recall",
        save_path="/content/precision_recall_val.png"
    )

    val_metrics["auc"] = auc_val
    plot_f1_acc_auc_metrics(
        val_metrics,
        title="Validation Stage",
        save_path="/content/f1_acc_auc_metrics_val.png"
    )

    print("\nRunning test inference...")
    test_probs_sum = None
    for i, model in enumerate(models, 1):
        print(f"  model {i}/{len(models)}")
        probs = predict_probs(model, tokenizer, test_texts)
        test_probs_sum = probs if test_probs_sum is None else (test_probs_sum + probs)

    test_mix = test_probs_sum / len(models)
    test_metrics = evaluate_probs(test_mix, test_labels)

    print("\n===== Equal-weight test result =====")
    print(f"test_acc = {test_metrics['accuracy']:.4f}")
    print(f"test_f1  = {test_metrics['f1_macro']:.4f}")
    print("\nConfusion matrix:")
    print(test_metrics["confusion_matrix"])

    print("\nPlotting test ROC AUC...")
    auc_test = plot_multiclass_roc(
        test_labels,
        test_mix,
        LABELS,
        save_path="/content/roc_auc_test.png"
    )

    test_preds = np.argmax(test_mix, axis=-1)
    plot_precision_recall_bars(
        test_labels,
        test_preds,
        LABELS,
        title="Test Precision / Recall",
        save_path="/content/precision_recall_test.png"
    )

    test_metrics["auc"] = auc_test

    print(test_metrics)

    plot_f1_acc_auc_metrics(
        test_metrics,
        title="Test Stage",
        save_path="/content/f1_acc_auc_metrics_test.png"
    )

    preds = np.argmax(test_mix, axis=-1)
    print("\nSample predictions:")
    for i in range(min(10, len(test_texts))):
        print(test_texts[i], "=>", id2label[int(preds[i])])
