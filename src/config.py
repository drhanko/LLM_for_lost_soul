from pathlib import Path
from dotenv import load_dotenv
import os

# project configuration from .env (secret part)
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)  # loads into os.environ

# project configuration
DATA_DIR = "../data"
SPLIT_DATA_DIR = "../data/clean_data"
DAILYDIALOG_DIR="../data/dailydialog"
RESULTS_DIR = "../results"

# I put GOOGLE_DRIVE_FOLDER_ID in config.py instead of .env in order to pass the final. After the final I will use .env instead of config.py
GOOGLE_DRIVE_FOLDER_ID= "1-lOi108h0d-qtnNML3PddIV6Pa_p-EN7"


GOOGLE_DRIVE_FOLDER_ID_mixed_data = "12KzpJvHXgyJbRS6rGDvLgOZCGs9Iqn19"
GOOGLE_DRIVE_FOLDER_ID_empathetic = "17FHjOw0nTmZS4OfttqGqBgUgoH6r6wct"
GOOGLE_DRIVE_FOLDER_ID_goemotions = "1mi9JczNMSsKejEA8-ppNCEEsSqmKpD5j"
GOOGLE_DRIVE_FOLDER_ID_dailydialog = "1UiiZyRopWVYF7HDyDph4yx3KBWZs8KGW"

# data sources configuration
EMPATHETIC_URL = "empathetic_dialogues"
DAILYDIALOG_URL = "thedevastator/dailydialog-unlock-the-conversation-potential-in"
GOEMOTIONS_URL = "go_emotions"



#LABEL config
OUTPUT_LABELS = ["negative", "neutral", "positive"]

#Model config
MODEL = "roberta-base"
SEEDS_SETS = [42, 43, 44]
MAX_LENGTH_c = 160
NUM_EPOCHS_c = 5
LEARNING_RATE_c = 2e-5
WEIGHT_DECAY_c = 0.01

#Model config-torch.cuda config. If cuda is not available, both size will become 8
TRAIN_BATCH_SIZE_c = 16
EVAL_BATCH_SIZE_c = 32



# === base path ===
BASE_DIR = Path(__file__).resolve().parent

# === load .env ===
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)


# === env variables ===
DEBUG = os.getenv("DEBUG", "False") == "True"