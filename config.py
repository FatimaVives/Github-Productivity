# config.py
from pathlib import Path
import os

# project paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

# data files 
RAW_CSV = DATA_DIR / "repositories.csv"

# training config
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# model hyperparams (defaults)
RF_PARAMS = {"n_estimators": 100, "random_state": RANDOM_SEED, "n_jobs": -1}
