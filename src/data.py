from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from src.config import (
    EXPECTED_COLUMNS,
    EXPECTED_CSV_SHA256,
    EXPECTED_ZIP_SHA256,
    FEATURES,
    LABELS,
    TARGET,
    UCI_ZIP_URL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_official_data(destination: Path, force: bool = False) -> Path:
    """Download and verify the official UCI ZIP, then extract its CSV."""
    destination = Path(destination)
    if destination.exists() and not force:
        current_hash = sha256_file(destination)
        if current_hash != EXPECTED_CSV_SHA256:
            raise ValueError(
                f"Existing CSV checksum differs from the verified source: {current_hash}"
            )
        return destination

    request = Request(UCI_ZIP_URL, headers={"User-Agent": "maternal-risk-project/1.0"})
    with urlopen(request, timeout=60) as response:
        zip_bytes = response.read()

    zip_hash = hashlib.sha256(zip_bytes).hexdigest()
    if zip_hash != EXPECTED_ZIP_SHA256:
        raise ValueError(
            "The UCI ZIP has changed. Review the new file before using it. "
            f"Expected {EXPECTED_ZIP_SHA256}, received {zip_hash}."
        )

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in the UCI ZIP, found {csv_names}.")
        csv_bytes = archive.read(csv_names[0])

    csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    if csv_hash != EXPECTED_CSV_SHA256:
        raise ValueError(
            f"Extracted CSV checksum mismatch: expected {EXPECTED_CSV_SHA256}, received {csv_hash}."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_bytes)
    return destination


def load_raw_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data was not found at {path}. Run scripts/download_data.py first."
        )
    frame = pd.read_csv(path)
    _validate_schema(frame)
    return frame


def _validate_schema(frame: pd.DataFrame) -> None:
    actual = list(frame.columns)
    if actual != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns. Expected {EXPECTED_COLUMNS}, received {actual}.")


def clean_data(
    raw_frame: pd.DataFrame,
    *,
    remove_exact_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Validate and clean the UCI data while returning an auditable summary."""
    _validate_schema(raw_frame)
    frame = raw_frame.copy()

    raw_missing = {column: int(value) for column, value in frame.isna().sum().items()}
    raw_duplicate_rows = int(frame.duplicated().sum())
    raw_heart_rate_7_rows = int((frame["HeartRate"] == 7).sum())

    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame[TARGET] = frame[TARGET].astype("string").str.strip().str.lower()

    invalid_labels = sorted(set(frame[TARGET].dropna()) - set(LABELS))
    if invalid_labels:
        raise ValueError(f"Unexpected risk labels: {invalid_labels}")
    if frame[EXPECTED_COLUMNS].isna().any().any():
        missing = frame[EXPECTED_COLUMNS].isna().sum()
        raise ValueError(f"Missing or non-numeric values found:\n{missing[missing > 0]}")

    if remove_exact_duplicates:
        frame = frame.drop_duplicates().copy()

    # Togunwa et al. (2023) identified 7 bpm as biologically implausible in this
    # dataset and replaced the affected values with the mode, 70 bpm.
    corrected_heart_rate_rows = int((frame["HeartRate"] == 7).sum())
    frame.loc[frame["HeartRate"] == 7, "HeartRate"] = 70

    frame = frame.drop_duplicates().reset_index(drop=True)
    integer_features = ["Age", "SystolicBP", "DiastolicBP", "HeartRate"]
    frame[integer_features] = frame[integer_features].astype("int64")
    frame[["BS", "BodyTemp"]] = frame[["BS", "BodyTemp"]].astype("float64")
    frame[TARGET] = pd.Categorical(frame[TARGET], categories=LABELS, ordered=True)

    audit = {
        "raw_rows": int(len(raw_frame)),
        "raw_columns": int(raw_frame.shape[1]),
        "raw_missing_values": raw_missing,
        "raw_exact_duplicate_rows": raw_duplicate_rows,
        "raw_heart_rate_7_rows": raw_heart_rate_7_rows,
        "remove_exact_duplicates": remove_exact_duplicates,
        "heart_rate_7_rows_corrected_after_deduplication": corrected_heart_rate_rows,
        "clean_rows": int(len(frame)),
        "clean_exact_duplicate_rows": int(frame.duplicated().sum()),
        "clean_missing_values": int(frame.isna().sum().sum()),
        "class_counts": {
            label: int((frame[TARGET] == label).sum()) for label in LABELS
        },
    }
    return frame, audit

