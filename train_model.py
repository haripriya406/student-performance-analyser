import argparse
import json
import os
import tempfile
from math import ceil
from pathlib import Path
from typing import Any

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PREDICTION_FEATURES,
    RISK_LABELS,
    read_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_pipeline(random_state: int = 42) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="metrics-",
            dir=path.parent,
            delete=False,
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
    labelled = frame.loc[frame["academic_risk"].notna()].drop_duplicates().reset_index(drop=True)
    class_counts = labelled["academic_risk"].value_counts()
    if len(class_counts) < 2:
        raise ValueError("At least two labelled academic-risk categories are required.")

    features = labelled[PREDICTION_FEATURES]
    target = labelled["academic_risk"]
    test_count = ceil(len(labelled) * 0.2)
    training_count = len(labelled) - test_count
    stratify = (
        target
        if class_counts.min() >= 2
        and test_count >= len(class_counts)
        and training_count >= len(class_counts)
        else None
    )

    evaluation_available = False
    split_training_count = len(labelled)
    testing_count = 0
    metrics: dict[str, float | None] = {
        "accuracy": None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
    }
    matrix_values: list[list[int]] = []
    report: dict[str, Any] = {}

    if len(labelled) >= 4 and test_count >= 1 and training_count >= 2:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=random_state,
            stratify=stratify,
        )
        if y_train.nunique() >= 2:
            evaluation_model = build_pipeline(random_state)
            evaluation_model.fit(x_train, y_train)
            predictions = evaluation_model.predict(x_test)
            evaluation_available = True
            split_training_count = len(x_train)
            testing_count = len(x_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision_macro": float(
                    precision_score(
                        y_test,
                        predictions,
                        labels=RISK_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "recall_macro": float(
                    recall_score(
                        y_test,
                        predictions,
                        labels=RISK_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "f1_macro": float(
                    f1_score(
                        y_test,
                        predictions,
                        labels=RISK_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
            matrix_values = confusion_matrix(
                y_test, predictions, labels=RISK_LABELS
            ).tolist()
            report = classification_report(
                y_test,
                predictions,
                labels=RISK_LABELS,
                output_dict=True,
                zero_division=0,
            )

    final_model = build_pipeline(random_state)
    final_model.fit(features, target)
    metrics_data: dict[str, Any] = {
        "model": "Logistic Regression",
        "random_state": random_state,
        "training_records": int(split_training_count),
        "testing_records": int(testing_count),
        "labelled_records": int(len(labelled)),
        "class_distribution": {
            label: int(labelled["academic_risk"].eq(label).sum()) for label in RISK_LABELS
        },
        "train_test_stratified": bool(stratify is not None and evaluation_available),
        "evaluation_available": evaluation_available,
        "metrics": metrics,
        "confusion_matrix": {"labels": RISK_LABELS, "values": matrix_values},
        "classification_report": report,
        "prediction_features": PREDICTION_FEATURES,
        "excluded_columns": ["student_id", "final_score", "academic_risk"],
        "evaluation_note": (
            "Metrics use a fixed-seed held-out test split and are model estimates, not a "
            "guarantee. Synthetic labels and this dataset do not establish real-world validity."
            if evaluation_available
            else "Evaluation metrics are unavailable because this dataset is too small or "
            "imbalanced to create a held-out test split with both classes in training. "
            "The prediction model was fitted to all labelled records."
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".joblib",
            prefix="model-",
            dir=model_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            joblib.dump(final_model, temporary_file)
        os.replace(temporary_path, model_path)
    except OSError:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        raise
    atomic_write_json(metrics_path, metrics_data)
    return metrics_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the student risk model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "student_performance_dataset.csv",
        help="Path to the validated student CSV dataset.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "student_performance_model.joblib",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "reports" / "model_metrics.json",
    )
    args = parser.parse_args()
    metrics = train(args.dataset, args.model, args.metrics)
    print(f"Model: {metrics['model']}")
    print(f"Training records: {metrics['training_records']}")
    print(f"Testing records: {metrics['testing_records']}")
    if metrics["evaluation_available"]:
        for name, value in metrics["metrics"].items():
            print(f"{name}: {value:.4f}")
    else:
        print(metrics["evaluation_note"])
    print(f"Saved model: {args.model}")
    print(f"Saved metrics: {args.metrics}")


if __name__ == "__main__":
    main()
