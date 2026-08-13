from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_PATH  # noqa: E402
from src.data import download_official_data, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the verified UCI source CSV.")
    parser.add_argument("--force", action="store_true", help="Replace the existing CSV.")
    args = parser.parse_args()
    path = download_official_data(RAW_DATA_PATH, force=args.force)
    print(f"Saved: {path}")
    print(f"SHA-256: {sha256_file(path)}")


if __name__ == "__main__":
    main()

