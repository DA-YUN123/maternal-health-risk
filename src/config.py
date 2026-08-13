from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "maternal_health_risk.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "maternal_health_risk_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
REPORT_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORT_DIR / "metrics"
FIGURE_DIR = REPORT_DIR / "figures"

UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/863/maternal%2Bhealth%2Brisk"
UCI_DOI = "https://doi.org/10.24432/C5DP5D"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip"
EXPECTED_ZIP_SHA256 = "84f0de0d647bb217ff7224ff265afadf27eb65c5483baaf1cfcc59abadeb75b1"
EXPECTED_CSV_SHA256 = "a1f7025719f84715096e0d1f95ae2e56b57809b9b15449e1836c96a7d976ae9b"

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"
EXPECTED_COLUMNS = FEATURES + [TARGET]
LABELS = ["low risk", "mid risk", "high risk"]
KOREAN_LABELS = {
    "low risk": "저위험",
    "mid risk": "중위험",
    "high risk": "고위험",
}
FEATURE_LABELS = {
    "Age": "나이(세)",
    "SystolicBP": "수축기 혈압(mmHg)",
    "DiastolicBP": "이완기 혈압(mmHg)",
    "BS": "혈당(BS, mmol/L)",
    "BodyTemp": "체온(°F)",
    "HeartRate": "심박수(bpm)",
}
RANDOM_STATE = 42


def ensure_output_directories() -> None:
    for path in (PROCESSED_DATA_PATH.parent, MODEL_DIR, METRICS_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)

