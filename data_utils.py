from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_COLUMNS = [
    "student_id",
    "age",
    "gender",
    "study_hours_per_day",
    "attendance_percentage",
    "previous_score",
    "assignments_completed_out_of_10",
    "sleep_hours_per_day",
    "internet_access",
    "parental_support",
    "extracurricular_hours_per_week",
    "family_income_level",
    "final_score",
    "academic_risk",
]

TARGET_COLUMN = "academic_risk"
PREDICTION_FEATURES = [
    "age",
    "gender",
    "study_hours_per_day",
    "attendance_percentage",
    "previous_score",
    "assignments_completed_out_of_10",
    "sleep_hours_per_day",
    "internet_access",
    "parental_support",
    "extracurricular_hours_per_week",
    "family_income_level",
]
CATEGORICAL_FEATURES = [
    "gender",
    "internet_access",
    "parental_support",
    "family_income_level",
]
NUMERIC_FEATURES = [
    feature for feature in PREDICTION_FEATURES if feature not in CATEGORICAL_FEATURES
]
RISK_LABELS = ["At Risk", "Not At Risk"]
NUMERIC_BOUNDS = {
    "age": (10, 100),
    "study_hours_per_day": (0, 24),
    "attendance_percentage": (0, 100),
    "previous_score": (0, 100),
    "assignments_completed_out_of_10": (0, 10),
    "sleep_hours_per_day": (0, 24),
    "extracurricular_hours_per_week": (0, 168),
    "final_score": (0, 100),
}
INTEGER_COLUMNS = {"age", "assignments_completed_out_of_10"}


class DatasetValidationError(ValueError):
    """Raised when a dataset does not match the supported schema or values."""


def validate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise DatasetValidationError("The CSV contains no student records.")

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    extra_columns = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    if missing_columns or extra_columns:
        problems = []
        if missing_columns:
            problems.append(f"Missing columns: {', '.join(missing_columns)}.")
        if extra_columns:
            problems.append(f"Unexpected columns: {', '.join(map(str, extra_columns))}.")
        raise DatasetValidationError(" ".join(problems))

    cleaned = frame.loc[:, EXPECTED_COLUMNS].copy()
    cleaned["student_id"] = cleaned["student_id"].astype("string").str.strip()
    if cleaned["student_id"].isna().any() or cleaned["student_id"].eq("").any():
        raise DatasetValidationError("student_id cannot be empty.")

    for column in CATEGORICAL_FEATURES + [TARGET_COLUMN]:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        if cleaned[column].isna().any() or cleaned[column].eq("").any():
            raise DatasetValidationError(f"{column} contains missing or empty values.")
    invalid_labels = sorted(set(cleaned[TARGET_COLUMN].dropna()) - set(RISK_LABELS))
    if invalid_labels:
        raise DatasetValidationError(
            f"{TARGET_COLUMN} must contain only: {', '.join(RISK_LABELS)}."
        )

    for column, (minimum, maximum) in NUMERIC_BOUNDS.items():
        original = cleaned[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise DatasetValidationError(f"{column} must contain numeric values.")
        if numeric.isna().any():
            raise DatasetValidationError(f"{column} contains missing values.")
        if ((numeric < minimum) | (numeric > maximum)).any():
            raise DatasetValidationError(
                f"{column} values must be between {minimum} and {maximum}."
            )
        if column in INTEGER_COLUMNS and (numeric % 1 != 0).any():
            raise DatasetValidationError(f"{column} must contain whole numbers.")
        cleaned[column] = numeric.astype(int if column in INTEGER_COLUMNS else float)

    return cleaned


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path.name}")
    try:
        frame = pd.read_csv(path, dtype={"student_id": "string"})
    except pd.errors.EmptyDataError as error:
        raise DatasetValidationError("The CSV file is empty.") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DatasetValidationError("The file is not a readable CSV.") from error
    return validate_dataframe(frame)


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
