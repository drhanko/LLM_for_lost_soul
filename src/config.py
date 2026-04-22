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
# data sources configuration
EMPATHETIC_URL = "empathetic_dialogues"
DAILYDIALOG_URL = "thedevastator/dailydialog-unlock-the-conversation-potential-in"
GOEMOTIONS_URL = "go_emotions"



#LABEL config
OUTPUT_LABELS = ["negative", "neutral", "positive"]

#Model config
SPECIFIC_DATASET = "dailydialog"
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