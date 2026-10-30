import io
import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from data_utils import (
    CATEGORICAL_FEATURES,
    INTEGER_COLUMNS,
    NUMERIC_BOUNDS,
    PREDICTION_FEATURES,
    RISK_LABELS,
    DatasetValidationError,
    json_safe,
    read_dataset,
    validate_dataframe,
)
PROJECT_ROOT = Path(__file__).resolve().parent
RECORDS_PER_PAGE = 20
MAX_RECORDS_PER_PAGE = 100
COLUMN_DESCRIPTIONS = {
    "student_id": "Unique student record identifier",
    "age": "Student age in years",
    "gender": "Student-reported gender category",
    "study_hours_per_day": "Average study hours per day",
    "attendance_percentage": "Attendance as a percentage",
    "previous_score": "Previous assessment score",
    "assignments_completed_out_of_10": "Completed assignments out of 10",
    "sleep_hours_per_day": "Average sleep hours per day",
    "internet_access": "Whether internet access is available",
    "parental_support": "Recorded level of parental support",
    "extracurricular_hours_per_week": "Extracurricular hours per week",
    "family_income_level": "Recorded family income category",
    "final_score": "Final score (excluded from model inputs)",
    "academic_risk": "Observed academic-risk category",
}


class ModelArtifactError(RuntimeError):
    """Raised when a packaged model or evaluation report cannot be read."""


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: json_safe(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _category_values(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        column: sorted(frame[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURES
    }


def _dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    risk_counts = frame["academic_risk"].value_counts().reindex(RISK_LABELS, fill_value=0)
    labels = list(RISK_LABELS)

    def category_averages(column: str) -> list[float | None]:
        averages = frame.groupby("academic_risk", dropna=True)[column].mean()
        return [
            round(float(averages[label]), 2)
            if label in averages.index and pd.notna(averages[label])
            else None
            for label in labels
        ]

    return {
        "total_students": int(len(frame)),
        "risk_counts": {label: int(risk_counts[label]) for label in labels},
        "average_attendance": (
            round(float(frame["attendance_percentage"].mean()), 2)
            if frame["attendance_percentage"].notna().any()
            else None
        ),
        "average_study_hours": (
            round(float(frame["study_hours_per_day"].mean()), 2)
            if frame["study_hours_per_day"].notna().any()
            else None
        ),
        "average_final_score": (
            round(float(frame["final_score"].mean()), 2)
            if frame["final_score"].notna().any()
            else None
        ),
        "risk_distribution": {
            "labels": labels,
            "values": [int(risk_counts[label]) for label in labels],
        },
        "study_hours_by_risk": {
            "labels": labels,
            "values": category_averages("study_hours_per_day"),
        },
        "attendance_by_risk": {
            "labels": labels,
            "values": category_averages("attendance_percentage"),
        },
        "sample_students": _records(frame.tail(8).iloc[::-1]),
    }


def _dataset_info(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "column_descriptions": COLUMN_DESCRIPTIONS,
        "missing_values": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "duplicate_records": int(frame.duplicated().sum()),
        "class_distribution": {
            label: int(frame["academic_risk"].eq(label).sum()) for label in RISK_LABELS
        },
        "categories": _category_values(frame),
        "upload_behavior": (
            "On Vercel, an uploaded dataset is stored temporarily in /tmp and may not "
            "persist between function instances. It does not replace the packaged CSV "
            "or automatically retrain the packaged model."
            if os.environ.get("VERCEL")
            else "A confirmed valid upload becomes the active dataset for analytics. "
            "It does not replace the original CSV or automatically retrain the packaged model."
        ),
    }


def _parse_integer(value: str | None, name: str, minimum: int, maximum: int) -> int:
    if value is None:
        raise ValueError(f"{name} must be a whole number.")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a whole number.") from error
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _validate_prediction(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    unexpected = sorted(set(data) - set(PREDICTION_FEATURES))
    if unexpected:
        errors.append(f"Unexpected field(s): {', '.join(unexpected)}.")

    missing = [field for field in PREDICTION_FEATURES if field not in data]
    if missing:
        errors.append(f"Please provide: {', '.join(missing)}.")

    cleaned: dict[str, Any] = {}
    for field in PREDICTION_FEATURES:
        if field in CATEGORICAL_FEATURES:
            value = data.get(field)
            if value is None or not str(value).strip():
                errors.append(f"{field.replace('_', ' ').capitalize()} is required.")
            else:
                cleaned[field] = str(value).strip()
            continue

        raw_value = data.get(field)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{field.replace('_', ' ').capitalize()} must be a number.")
            continue
        if not math.isfinite(value):
            errors.append(f"{field.replace('_', ' ').capitalize()} must be a finite number.")
            continue
        minimum, maximum = NUMERIC_BOUNDS[field]
        if value < minimum or value > maximum:
            errors.append(
                f"{field.replace('_', ' ').capitalize()} must be between "
                f"{minimum} and {maximum}."
            )
            continue
        if field in INTEGER_COLUMNS and not value.is_integer():
            errors.append(f"{field.replace('_', ' ').capitalize()} must be a whole number.")
            continue
        cleaned[field] = int(value) if field in INTEGER_COLUMNS else value
    return cleaned, errors


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    configured_dataset = Path(
        os.environ.get(
            "STUDENT_DATASET_PATH",
            PROJECT_ROOT / "data" / "student_performance_dataset.csv",
        )
    )
    if not configured_dataset.is_absolute():
        configured_dataset = PROJECT_ROOT / configured_dataset
    upload_dataset_path = (
        Path(os.environ.get("TMPDIR", "/tmp"))
        / "student-performance-analyser"
        / "replacement_dataset.csv"
        if os.environ.get("VERCEL")
        else PROJECT_ROOT / "instance" / "replacement_dataset.csv"
    )
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "local-student-analyser"),
        DATASET_PATH=configured_dataset,
        UPLOAD_DATASET_PATH=upload_dataset_path,
        MODEL_PATH=PROJECT_ROOT / "models" / "student_performance_model.joblib",
        METRICS_PATH=PROJECT_ROOT / "reports" / "model_metrics.json",
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    )
    if test_config:
        app.config.update(test_config)
    app.extensions["student_model_cache"] = {}

    @app.template_filter("number_or_dash")
    def number_or_dash(value: Any, decimals: int = 1) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}"

    def active_dataset_path() -> Path:
        upload_path = Path(app.config["UPLOAD_DATASET_PATH"])
        if upload_path.is_file():
            return upload_path
        return Path(app.config["DATASET_PATH"])

    def load_active_dataset() -> pd.DataFrame:
        return read_dataset(active_dataset_path())

    def model_bundle() -> Any:
        path = Path(app.config["MODEL_PATH"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            file_stat = path.stat()
        except FileNotFoundError as error:
            raise ModelArtifactError(
                f"Trained model artifact not found at {path}. Run "
                "'python train_model.py' locally and include the generated model "
                "in the deployment."
            ) from error
        except OSError as error:
            raise ModelArtifactError(f"Cannot access trained model artifact {path}: {error}") from error
        cache_key = (str(path.resolve()), file_stat.st_mtime_ns, file_stat.st_size)
        cache: dict[Any, Any] = app.extensions["student_model_cache"]
        if cache_key not in cache:
            try:
                model = joblib.load(path)
            except (
                OSError,
                EOFError,
                ValueError,
                ImportError,
                pickle.UnpicklingError,
                AttributeError,
                KeyError,
                TypeError,
            ) as error:
                raise ModelArtifactError(
                    f"Cannot load trained model artifact {path}: {error}"
                ) from error
            cache[cache_key] = model
        return cache[cache_key]

    def load_model_metrics() -> dict[str, Any]:
        path = Path(app.config["METRICS_PATH"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            with path.open(encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)
        except FileNotFoundError as error:
            raise ModelArtifactError(
                f"Evaluation report not found at {path}. Run "
                "'python train_model.py' locally and include the generated report "
                "in the deployment."
            ) from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ModelArtifactError(
                f"Cannot load evaluation report {path}: {error}"
            ) from error
        if not isinstance(metrics, dict):
            raise ModelArtifactError(f"Evaluation report {path} must contain a JSON object.")
        return metrics

    def html_dataset_error(error: Exception, status: int = 503):
        return (
            render_template(
                "error.html",
                status="Dataset unavailable",
                message=str(error),
            ),
            status,
        )

    def html_model_error(error: ModelArtifactError, status: int = 503):
        return (
            render_template(
                "error.html",
                status="Model unavailable",
                message=str(error),
            ),
            status,
        )

    def api_dataset_error(error: Exception, status: int = 503):
        return jsonify({"error": str(error)}), status

    @app.route("/")
    def index():
        return redirect(url_for("dashboard"))

    @app.route("/dashboard")
    def dashboard():
        try:
            frame = load_active_dataset()
        except (DatasetValidationError, FileNotFoundError, OSError) as error:
            return html_dataset_error(error)
        summary = _dataset_summary(frame)
        return render_template("dashboard.html", summary=summary)

    @app.route("/students")
    def students():
        try:
            frame = load_active_dataset()
        except (DatasetValidationError, FileNotFoundError, OSError) as error:
            return html_dataset_error(error)

        search = request.args.get("search", "").strip()
        risk = request.args.get("risk", "").strip()
        if risk and risk not in RISK_LABELS:
            return html_dataset_error(ValueError("Choose a valid academic-risk category."), 400)
        try:
            page = _parse_integer(
                request.args.get("page", "1"), "Page", 1, 1_000_000
            )
        except ValueError as error:
            return html_dataset_error(error, 400)

        filtered = frame
        if search:
            filtered = filtered.loc[
                filtered["student_id"].str.contains(search, case=False, regex=False, na=False)
            ]
        if risk:
            filtered = filtered.loc[filtered["academic_risk"].eq(risk)]

        total = len(filtered)
        total_pages = math.ceil(total / RECORDS_PER_PAGE)
        current_page = min(page, total_pages) if total_pages else 1
        start = (current_page - 1) * RECORDS_PER_PAGE
        records = _records(filtered.iloc[start : start + RECORDS_PER_PAGE])
        return render_template(
            "students.html",
            records=records,
            search=search,
            selected_risk=risk,
            risk_labels=RISK_LABELS,
            total=total,
            page=current_page,
            total_pages=total_pages,
            per_page=RECORDS_PER_PAGE,
        )

    @app.route("/dataset")
    def dataset():
        try:
            frame = load_active_dataset()
            path = active_dataset_path()
        except (DatasetValidationError, FileNotFoundError, OSError) as error:
            return html_dataset_error(error)
        info = _dataset_info(frame, path)
        return render_template(
            "dataset.html",
            info=info,
            preview=_records(frame.head(10)),
            columns=list(frame.columns),
        )

    @app.get("/predict")
    def predict():
        try:
            frame = load_active_dataset()
        except (DatasetValidationError, FileNotFoundError, OSError) as error:
            return html_dataset_error(error)
        categories = _category_values(frame)
        return render_template(
            "predict.html",
            categories=categories,
            values={},
            errors=[],
        )

    @app.post("/result")
    def result():
        try:
            frame = load_active_dataset()
        except (DatasetValidationError, FileNotFoundError, OSError) as error:
            return html_dataset_error(error)
        student_data, errors = _validate_prediction(request.form.to_dict())
        if errors:
            return (
                render_template(
                    "predict.html",
                    categories=_category_values(frame),
                    values=request.form.to_dict(),
                    errors=errors,
                ),
                400,
            )
        try:
            model = model_bundle()
        except ModelArtifactError as error:
            return html_model_error(error)
        input_frame = pd.DataFrame([student_data])
        prediction = str(model.predict(input_frame)[0])
        probabilities = {
            str(label): float(probability)
            for label, probability in zip(
                model.classes_,
                model.predict_proba(input_frame)[0],
            )
        }
        return render_template(
            "result.html",
            prediction=prediction,
            probabilities=probabilities,
            student_data=student_data,
        )

    @app.route("/evaluation")
    def evaluation():
        try:
            metrics = load_model_metrics()
        except ModelArtifactError as error:
            return render_template(
                "evaluation.html",
                metrics=None,
                error=str(error),
            )
        return render_template("evaluation.html", metrics=metrics, error=None)

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.get("/api/dashboard")
    def api_dashboard():
        try:
            frame = load_active_dataset()
        except (DatasetValidationError, FileNotFoundError, OSError) as error:
            return api_dataset_error(error)
        return jsonify(_dataset_summary(frame))

    @app.get("/api/students")
    def api_students():
        try:
            frame = load_active_dataset()
        except (DatasetValidationError, FileNotFoundError, OSError) as error:
            return api_dataset_error(error)

        search = request.args.get("search", "").strip()
        risk = request.args.get("risk", "").strip()
        if risk and risk not in RISK_LABELS:
            return jsonify({"error": "Choose a valid academic-risk category."}), 400
        try:
            page = _parse_integer(request.args.get("page", "1"), "Page", 1, 1_000_000)
            per_page = _parse_integer(
                request.args.get("per_page", str(RECORDS_PER_PAGE)),
                "Records per page",
                1,
                MAX_RECORDS_PER_PAGE,
            )
            minimum_attendance = request.args.get("min_attendance")
            maximum_attendance = request.args.get("max_attendance")
            minimum = float(minimum_attendance) if minimum_attendance else None
            maximum = float(maximum_attendance) if maximum_attendance else None
            for name, value in (("Minimum attendance", minimum), ("Maximum attendance", maximum)):
                if value is not None and (not math.isfinite(value) or value < 0 or value > 100):
                    raise ValueError(f"{name} must be between 0 and 100.")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError("Minimum attendance cannot exceed maximum attendance.")
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        filtered = frame
        if search:
            filtered = filtered.loc[
                filtered["student_id"].str.contains(search, case=False, regex=False, na=False)
            ]
        if risk:
            filtered = filtered.loc[filtered["academic_risk"].eq(risk)]
        if minimum is not None:
            filtered = filtered.loc[filtered["attendance_percentage"] >= minimum]
        if maximum is not None:
            filtered = filtered.loc[filtered["attendance_percentage"] <= maximum]

        sort_options = {
            "student_id": "student_id",
            "score": "final_score",
            "attendance": "attendance_percentage",
        }
        sort_key = request.args.get("sort", "student_id")
        order = request.args.get("order", "asc")
        if sort_key not in sort_options or order not in ("asc", "desc"):
            return jsonify({"error": "Choose a valid sort field and order."}), 400
        filtered = filtered.sort_values(
            sort_options[sort_key],
            ascending=order == "asc",
            na_position="last",
            kind="mergesort",
        )
        total = len(filtered)
        total_pages = math.ceil(total / per_page)
        current_page = min(page, total_pages) if total_pages else 1
        start = (current_page - 1) * per_page
        return jsonify(
            {
                "records": _records(filtered.iloc[start : start + per_page]),
                "total": total,
                "page": current_page,
                "per_page": per_page,
                "total_pages": total_pages,
            }
        )

    @app.get("/api/dataset-info")
    def api_dataset_info():
        try:
            frame = load_active_dataset()
            path = active_dataset_path()
        except (DatasetValidationError, FileNotFoundError, OSError) as error:
            return api_dataset_error(error)
        return jsonify(_dataset_info(frame, path))

    @app.get("/api/dataset")
    def api_dataset_preview():
        try:
            frame = load_active_dataset()
        except (DatasetValidationError, FileNotFoundError, OSError) as error:
            return api_dataset_error(error)
        return jsonify(
            {"rows": _records(frame.head(100)), "total_rows": int(len(frame))}
        )

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send the student details as a JSON object."}), 400
        student_data, errors = _validate_prediction(payload)
        if errors:
            return jsonify({"error": " ".join(errors)}), 400
        try:
            model = model_bundle()
        except ModelArtifactError as error:
            return jsonify({"error": str(error)}), 503
        input_frame = pd.DataFrame([student_data])
        prediction = str(model.predict(input_frame)[0])
        probabilities = {
            str(label): float(probability)
            for label, probability in zip(model.classes_, model.predict_proba(input_frame)[0])
        }
        return jsonify(
            {
                "prediction": prediction,
                "class_probabilities": probabilities,
                "probability_notice": (
                    "These are model output scores, not calibrated probabilities or guarantees."
                ),
                "estimate_notice": (
                    "This is a model estimate for learning and analysis, not a guarantee or "
                    "a substitute for human review."
                ),
            }
        )

    @app.get("/api/model-metrics")
    @app.get("/api/status")
    def api_model_metrics():
        try:
            metrics = load_model_metrics()
        except ModelArtifactError as error:
            return jsonify({"error": str(error)}), 503
        return jsonify(metrics)

    @app.post("/api/upload-dataset")
    def api_upload_dataset():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Choose a CSV file to upload."}), 400
        if not uploaded.filename.lower().endswith(".csv"):
            return jsonify({"error": "Please upload a file with a .csv extension."}), 400
        if request.form.get("confirm_replace") != "true":
            return jsonify(
                {"error": "Confirm that this upload should become the active dataset."}
            ), 400
        content = uploaded.read()
        try:
            candidate = pd.read_csv(io.BytesIO(content), dtype={"student_id": "string"})
            validated = validate_dataframe(candidate)
        except (DatasetValidationError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
            return jsonify({"error": str(error)}), 422

        destination = Path(app.config["UPLOAD_DATASET_PATH"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix="dataset-",
                suffix=".csv",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                validated.to_csv(temporary_file, index=False)
            os.replace(temporary_path, destination)
        except OSError as error:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
            return jsonify({"error": f"The dataset could not be saved: {error}"}), 500
        return jsonify(
            {
                "message": "Dataset validated and activated.",
                "rows": int(len(validated)),
            }
        ), 201

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(error: RequestEntityTooLarge):
        message = "The uploaded file exceeds the 10 MB limit."
        if request.path.startswith("/api/"):
            return jsonify({"error": message}), error.code or 413
        return (
            render_template("error.html", status="Upload too large", message=message),
            error.code or 413,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
