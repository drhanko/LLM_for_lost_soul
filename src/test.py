# include your tests here
# for example for your Progress report you should be able to load data from at least one API source.
from data_collection import run_data_collection,move_and_rename_dailydialog
from data_prerpocessing import data_prerpocessing
from check_jsonl import check_dataform
from pathlib import Path


def test_data_collection():
    print("test_data_collection")
    run_data_collection()

    # check whether the csv is output successfully or not
    data_dir = Path("../data")

    expected_files = [
        "empathetic_dialogues_train.csv",
        "empathetic_dialogues_validation.csv",
        "empathetic_dialogues_test.csv",
        "go_emotions_train.csv",
        "go_emotions_validation.csv",
        "go_emotions_test.csv",
    ]

    for f in expected_files:
        print("dwd")
        assert (data_dir / f).exists(), f"{f} not found"

if __name__ == "__main__":
    test_data_collection()
    move_and_rename_dailydialog()
    data_prerpocessing()
    check_dataform()