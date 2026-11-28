from pathlib import Path


# Main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Data directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Original processed dataset
DATASET_PATH = (
    PROCESSED_DATA_DIR
    / "GSE38642_final.csv"
)


# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"

FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_DIR = RESULTS_DIR / "metrics"

MODELS_DIR = RESULTS_DIR / "models"


# Dataset configuration
TARGET_COLUMN = "Diabetes_Status"

ID_COLUMN = "Sample_ID"


# Machine learning configuration
RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5


# Create output directories if they do not exist
for directory in [
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )