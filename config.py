"""Part 1 configuration. Paths are relative to this project directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "part1-data"
OUTPUT_DIR = ROOT / "outputs"
TRAIN_PATH = DATA_DIR / "part1_train.csv"
VAL_PATH = DATA_DIR / "part1_val.csv"
TEST_PATH = DATA_DIR / "part1_test.csv"
RANDOM_STATE = 42
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
FEATURES = [f"emb_{i}" for i in range(150)]
K = 8
SMOOTHING = 5.0
LOW_CONFIDENCE = 0.2
HIGH_CONFIDENCE = 0.8
MIN_LABELED_CLUSTER = 5
PSEUDO_WEIGHT = 0.3
SUBMISSION_THRESHOLDS = [0.20, 0.20, 0.20, 0.40, 0.40]
