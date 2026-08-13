from __future__ import annotations

import unittest

from src.config import FEATURES, LABELS, RAW_DATA_PATH, TARGET
from src.data import clean_data, load_raw_data


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw_data(RAW_DATA_PATH)
        cls.clean, cls.audit = clean_data(cls.raw)

    def test_official_file_audit(self):
        self.assertEqual(len(self.raw), 1014)
        self.assertEqual(self.audit["raw_exact_duplicate_rows"], 562)
        self.assertEqual(self.audit["raw_heart_rate_7_rows"], 2)

    def test_clean_data_integrity(self):
        self.assertEqual(len(self.clean), 452)
        self.assertEqual(self.clean.duplicated().sum(), 0)
        self.assertEqual(self.clean.isna().sum().sum(), 0)
        self.assertEqual((self.clean["HeartRate"] == 7).sum(), 0)

    def test_expected_features_and_labels(self):
        self.assertEqual(list(self.clean.columns), FEATURES + [TARGET])
        self.assertEqual(set(self.clean[TARGET].astype("string")), set(LABELS))


if __name__ == "__main__":
    unittest.main()

