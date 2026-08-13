from __future__ import annotations

import json
import unittest

import joblib
import pandas as pd

from src.config import FEATURES, METADATA_PATH, MODEL_PATH


class ModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise unittest.SkipTest("Run python run_pipeline.py before testing the model.")
        cls.model = joblib.load(MODEL_PATH)
        cls.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    def test_prediction_and_probabilities(self):
        ranges = self.metadata["feature_ranges_observed_in_clean_data"]
        sample = pd.DataFrame(
            [{feature: ranges[feature]["median"] for feature in FEATURES}],
            columns=FEATURES,
        )
        prediction = self.model.predict(sample)
        probability = self.model.predict_proba(sample)
        self.assertIn(prediction[0], self.metadata["labels"])
        self.assertAlmostEqual(float(probability[0].sum()), 1.0, places=7)

    def test_artifact_schema(self):
        self.assertEqual(self.metadata["features"], FEATURES)
        self.assertEqual(list(self.model.classes_), sorted(self.metadata["labels"]))


if __name__ == "__main__":
    unittest.main()

