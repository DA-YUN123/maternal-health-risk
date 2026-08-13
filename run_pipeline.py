from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib")
)

import joblib
import pandas as pd
import sklearn

from src.config import (
    EXPECTED_CSV_SHA256,
    FEATURES,
    FIGURE_DIR,
    LABELS,
    METADATA_PATH,
    METRICS_DIR,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    REPORT_DIR,
    TARGET,
    UCI_DATASET_PAGE,
    UCI_DOI,
    ensure_output_directories,
)
from src.data import clean_data, load_raw_data, sha256_file
from src.modeling import train_and_evaluate
from src.reporting import json_ready, save_figures, write_results_markdown


def _write_json(path, payload) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ensure_output_directories()
    raw_frame = load_raw_data(RAW_DATA_PATH)
    raw_hash = sha256_file(RAW_DATA_PATH)
    if raw_hash != EXPECTED_CSV_SHA256:
        raise ValueError(
            "Raw CSV checksum differs from the verified UCI file. "
            "Run scripts/download_data.py --force and review the source."
        )

    clean_frame, audit = clean_data(raw_frame, remove_exact_duplicates=True)
    clean_frame.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")
    processed_hash = sha256_file(PROCESSED_DATA_PATH)

    result = train_and_evaluate(clean_frame)
    # Lossless compression keeps the trained estimator unchanged while making
    # the distributable model artifact substantially smaller.
    joblib.dump(result.best_model, MODEL_PATH, compress=3)

    result.comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    result.cv_results.to_csv(METRICS_DIR / "cross_validation_results.csv", index=False)
    result.feature_importance.to_csv(
        METRICS_DIR / "permutation_feature_importance.csv", index=False
    )
    result.test_predictions.to_csv(METRICS_DIR / "holdout_predictions.csv", index=False)
    _write_json(METRICS_DIR / "preprocessing_audit.json", audit)
    _write_json(METRICS_DIR / "classification_reports.json", result.detailed_metrics)
    _write_json(METRICS_DIR / "confusion_matrices.json", result.confusion_matrices)

    feature_ranges = {
        feature: {
            "min": float(clean_frame[feature].min()),
            "max": float(clean_frame[feature].max()),
            "median": float(clean_frame[feature].median()),
        }
        for feature in FEATURES
    }
    selected_row = result.comparison.loc[
        result.comparison["model"] == result.best_model_name
    ].iloc[0]
    metadata = {
        "project": "Maternal Health Risk Classification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": {
            "name": "Maternal Health Risk",
            "provider": "UCI Machine Learning Repository",
            "dataset_page": UCI_DATASET_PAGE,
            "doi": UCI_DOI,
            "raw_csv_sha256": raw_hash,
            "processed_csv_sha256": processed_hash,
            "note": (
                "The UCI metadata page lists 1,013 instances, while the downloaded "
                "official CSV contains 1,014 data rows. This project reports the file audit."
            ),
        },
        "preprocessing": audit,
        "features": FEATURES,
        "target": TARGET,
        "labels": LABELS,
        "feature_ranges_observed_in_clean_data": feature_ranges,
        "selected_model": result.best_model_name,
        "best_parameters": result.best_params,
        "selection_rule": result.split_summary["model_selection_metric"],
        "split": result.split_summary,
        "selected_model_metrics": {
            "cv_macro_f1_mean": float(selected_row["cv_macro_f1_mean"]),
            "cv_macro_f1_std": float(selected_row["cv_macro_f1_std"]),
            "holdout_accuracy": float(selected_row["holdout_accuracy"]),
            "holdout_balanced_accuracy": float(
                selected_row["holdout_balanced_accuracy"]
            ),
            "holdout_macro_f1": float(selected_row["holdout_macro_f1"]),
            "holdout_weighted_f1": float(selected_row["holdout_weighted_f1"]),
        },
        "permutation_feature_importance": result.feature_importance.to_dict(
            orient="records"
        ),
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "intended_use": "Education and model-development demonstration only",
        "warning": "Not a medical device and not for diagnosis or treatment decisions.",
    }
    _write_json(METADATA_PATH, metadata)

    save_figures(
        clean_frame,
        result.comparison,
        result.best_model_name,
        result.confusion_matrices[result.best_model_name],
        result.feature_importance,
        FIGURE_DIR,
    )
    write_results_markdown(
        REPORT_DIR / "RESULTS.md",
        audit,
        result.comparison,
        result.best_model_name,
        result.detailed_metrics,
    )

    print("Pipeline completed successfully.")
    print(f"Raw rows: {audit['raw_rows']:,}")
    print(f"Clean rows: {audit['clean_rows']:,}")
    print(f"Selected model: {result.best_model_name}")
    print(f"CV macro F1: {selected_row['cv_macro_f1_mean']:.4f}")
    print(f"Holdout macro F1: {selected_row['holdout_macro_f1']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
