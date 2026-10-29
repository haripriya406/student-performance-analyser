import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app import create_app
from data_utils import PREDICTION_FEATURES
from train_model import train


def sample_frame(size: int = 40) -> pd.DataFrame:
    rows = []
    for index in range(size):
        rows.append(
            {
                "student_id": f"TEST-{index:04d}",
                "age": 15 + index % 5,
                "gender": "Female" if index % 2 else "Male",
                "study_hours_per_day": 1 + (index % 6),
                "attendance_percentage": 60 + (index * 7 % 41),
                "previous_score": 40 + (index * 11 % 61),
                "assignments_completed_out_of_10": index % 11,
                "sleep_hours_per_day": 5 + (index % 5),
                "internet_access": "Yes" if index % 2 else "No",
                "parental_support": "High" if index % 3 else "Low",
                "extracurricular_hours_per_week": index % 12,
                "family_income_level": "Low" if index % 2 else "High",
                "final_score": 35 + (index * 13 % 66),
                "academic_risk": "At Risk" if index % 2 else "Not At Risk",
            }
        )
    return pd.DataFrame(rows)


class StudentPerformanceAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_path = self.root / "student_performance_dataset.csv"
        sample_frame().to_csv(self.dataset_path, index=False)
        self.model_path = self.root / "model.joblib"
        self.metrics_path = self.root / "metrics.json"
        self.upload_path = self.root / "instance" / "replacement_dataset.csv"
        self.app = create_app(
            {
                "TESTING": True,
                "DATASET_PATH": self.dataset_path,
                "UPLOAD_DATASET_PATH": self.upload_path,
                "MODEL_PATH": self.model_path,
                "METRICS_PATH": self.metrics_path,
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_dashboard_returns_dataset_derived_values(self):
        response = self.client.get("/api/dashboard")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["total_students"], 40)
        self.assertEqual(body["risk_counts"]["At Risk"], 20)
        self.assertEqual(body["average_final_score"], round(sample_frame()["final_score"].mean(), 2))
        self.assertEqual(len(body["sample_students"]), 8)

    def test_student_search_filter_and_pagination(self):
        response = self.client.get("/api/students?search=TEST-000&risk=At%20Risk&page=1&per_page=5")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["total"], 5)
        self.assertLessEqual(len(body["records"]), 5)
        self.assertTrue(all("TEST-000" in row["student_id"] for row in body["records"]))
        self.assertTrue(all(row["academic_risk"] == "At Risk" for row in body["records"]))

    def test_student_filters_reject_invalid_values(self):
        response = self.client.get("/api/students?min_attendance=90&max_attendance=20")
        self.assertEqual(response.status_code, 400)

    def test_missing_dataset_is_reported(self):
        self.app.config["DATASET_PATH"] = self.root / "missing.csv"
        response = self.client.get("/api/dashboard")
        self.assertEqual(response.status_code, 503)
        self.assertIn("Dataset not found", response.get_json()["error"])

    def test_upload_requires_confirmation_and_preserves_original(self):
        original_bytes = self.dataset_path.read_bytes()
        payload = sample_frame(6).to_csv(index=False).encode("utf-8")
        response = self.client.post(
            "/api/upload-dataset",
            data={"file": (io.BytesIO(payload), "records.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

        response = self.client.post(
            "/api/upload-dataset",
            data={
                "file": (io.BytesIO(payload), "records.csv"),
                "confirm_replace": "true",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.dataset_path.read_bytes(), original_bytes)
        self.assertEqual(self.client.get("/api/dashboard").get_json()["total_students"], 6)

    def test_upload_rejects_unexpected_schema(self):
        wrong = pd.DataFrame({"student_id": ["id"], "academic_risk": ["At Risk"]})
        response = self.client.post(
            "/api/upload-dataset",
            data={
                "file": (io.BytesIO(wrong.to_csv(index=False).encode("utf-8")), "bad.csv"),
                "confirm_replace": "true",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 422)

    def test_training_saves_pipeline_metrics_and_prediction_works(self):
        train(self.dataset_path, self.model_path, self.metrics_path)
        self.assertTrue(self.model_path.is_file())
        self.assertTrue(self.metrics_path.is_file())
        metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["training_records"] + metrics["testing_records"], 40)
        self.assertNotIn("final_score", metrics["prediction_features"])

        record = sample_frame(1).iloc[0].to_dict()
        prediction = {field: record[field] for field in PREDICTION_FEATURES}
        response = self.client.post("/api/predict", json=prediction)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn(body["prediction"], ("At Risk", "Not At Risk"))
        self.assertIn("class_probabilities", body)
        self.assertIn("not calibrated", body["probability_notice"])

    def test_prediction_rejects_extra_and_out_of_range_fields(self):
        response = self.client.post("/api/predict", json={"age": 14, "unexpected": True})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
