
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from intelligent_infrastructure_monitor.features import FEATURE_COLUMNS, build_features
from intelligent_infrastructure_monitor.model import load_model, score_logs, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class IntelligentInfrastructureMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.logs = pd.read_csv(PROJECT_ROOT / "data" / "sample_infrastructure_logs.csv")

    def test_feature_matrix_is_numeric_and_stable(self) -> None:
        features = build_features(self.logs.head(25))
        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertEqual(features.shape, (25, len(FEATURE_COLUMNS)))
        self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes))
        self.assertFalse(features.isna().any().any())

    def test_training_and_scoring_produce_both_model_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            report = train_model(self.logs, model_path=model_path)
            bundle = load_model(model_path)
            scored = score_logs(self.logs.head(100), bundle)

            self.assertTrue(model_path.exists())
            self.assertIn("selected_config", report)
            self.assertIn("is_anomaly_predicted", scored.columns)
            self.assertIn("failure_risk_probability", scored.columns)
            self.assertIn("maintenance_risk", scored.columns)
            self.assertTrue(scored["is_anomaly_predicted"].isin([0, 1]).all())


if __name__ == "__main__":
    unittest.main()
