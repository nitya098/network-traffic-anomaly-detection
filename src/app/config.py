from pathlib import Path

# src/app/config.py -> repository root is two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = REPO_ROOT / "models"
JOBLIB_PATH = MODELS_DIR / "nsl_kdd_random_forest_500.joblib"
METADATA_PATH = MODELS_DIR / "nsl_kdd_random_forest_500_metadata.json"
CONTRACT_PATH = MODELS_DIR / "nsl_kdd_random_forest_500_input_contract.json"

ALLOWED_OUTPUT_CLASSES = ("Normal", "DoS", "Probe", "R2L", "U2R")
