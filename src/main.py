# include your tests here
# for example for your Progress report you should be able to load data from at least one API source.
from data_collection import run_data_collection,move_and_rename_dailydialog,test_data_collection
from data_prerpocessing import data_prerpocessing
from data_process_ready import process_data_split
from predict_model import execute_model_training
from predict_evaluation import evaluation
from config import *
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate the model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pre_train
    pre_train_parser = subparsers.add_parser("pre_train", help="Run all the process before training")

    # train
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--dataset_link", type=str, default=None, help="Link to a dataset")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Run evaluation pipeline")
    eval_parser.add_argument("--switch", choices=["local","google"], default=None, help="Link to a model")
    eval_parser.add_argument("--dataset_link", type=str, default=None, help="Link to a dataset")
    return parser.parse_args()

def data_initialization():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_data_collection(DATA_DIR, EMPATHETIC_URL, DAILYDIALOG_URL, GOEMOTIONS_URL)
    move_and_rename_dailydialog(DAILYDIALOG_DIR, DATA_DIR)
    test_data_collection(DATA_DIR)
    data_prerpocessing("mixed")
    data_prerpocessing("empathetic")
    data_prerpocessing("dailydialog")
    data_prerpocessing("goemotions")
    process_data_split()

if __name__ == "__main__":
    args = parse_args()

    if  args.command == "pre_train":
        data_initialization()
        print("\n===== Pre-Train =====")
    elif args.command == "train":
        execute_model_training(args.dataset_link)
        print("\n===== Train =====")
    elif args.command == "eval":
        evaluation(args.switch ,args.dataset_link)
        print("\n===== Eval =====")


