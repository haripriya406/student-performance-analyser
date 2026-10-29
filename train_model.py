import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PREDICTION_FEATURES,
    RISK_LABELS,
    read_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json",
            prefix="metrics-", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(data, temporary_file, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        raise


def train(
    dataset_path: Path,
    model_path: Path,
    metrics_path: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    frame = read_dataset(dataset_path)
    total_before_deduplication = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    if len(frame) < 4:
        raise ValueError("At least four unique records are required for a train/test split.")
    class_counts = frame["academic_risk"].value_counts()
    if len(class_counts) != 2:
        raise ValueError("Training data must contain both supported academic_risk classes.")

    can_stratify = class_counts.min() >= 2 and int(len(frame) * 0.2) >= 2
    stratify = frame["academic_risk"] if can_stratify else None
    x_train, x_test, y_train, y_test = train_test_split(
        frame[PREDICTION_FEATURES],
        frame["academic_risk"],
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics: dict[str, Any] = {
        "model": "Random Forest Classifier",
        "random_state": random_state,
        "training_records": int(len(x_train)),
        "testing_records": int(len(x_test)),
        "records_before_duplicate_removal": int(total_before_deduplication),
        "duplicate_records_removed_before_split": int(total_before_deduplication - len(frame)),
        "class_distribution": {
            label: int(frame["academic_risk"].eq(label).sum()) for label in RISK_LABELS
        },
        "train_test_stratified": bool(can_stratify),
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(
                precision_score(y_test, predictions, labels=RISK_LABELS, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, predictions, labels=RISK_LABELS, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_test, predictions, labels=RISK_LABELS, average="macro", zero_division=0)
            ),
        },
        "confusion_matrix": {
            "labels": RISK_LABELS,
            "values": confusion_matrix(y_test, predictions, labels=RISK_LABELS).tolist(),
        },
        "classification_report": classification_report(
            y_test, predictions, labels=RISK_LABELS, output_dict=True, zero_division=0
        ),
        "prediction_features": PREDICTION_FEATURES,
        "excluded_columns": ["student_id", "final_score", "academic_risk"],
        "evaluation_note": (
            "Held-out test metrics on the supplied data do not establish real-world reliability. "
            "Synthetic labels may reflect artificial rules; external validation is required."
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".joblib", prefix="model-", dir=model_path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            joblib.dump(pipeline, temporary_file)
        os.replace(temporary_path, model_path)
    except OSError:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        raise
    atomic_write_json(metrics_path, metrics)
    print(f"Training records: {metrics['training_records']}")
    print(f"Testing records: {metrics['testing_records']}")
    print(f"Records removed as exact duplicates before split: {metrics['duplicate_records_removed_before_split']}")
    print(f"Class distribution: {metrics['class_distribution']}")
    print("Test metrics:")
    for name, value in metrics["metrics"].items():
        print(f"  {name}: {value:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(metrics["evaluation_note"])
    return metrics


def main() -> None:
    configured_dataset = os.environ.get("STUDENT_DATASET_PATH")
    uploaded_dataset = PROJECT_ROOT / "instance" / "replacement_dataset.csv"
    default_dataset = (
        Path(configured_dataset)
        if configured_dataset
        else uploaded_dataset if uploaded_dataset.is_file()
        else PROJECT_ROOT / "student_performance_dataset.csv"
    )
    parser = argparse.ArgumentParser(description="Train and evaluate the student risk model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Validated CSV path (defaults to the active replacement dataset when present).",
    )
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "student_performance_model.joblib")
    parser.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "reports" / "model_metrics.json")
    args = parser.parse_args()
    train(args.dataset, args.model, args.metrics)


if __name__ == "__main__":
    main()
