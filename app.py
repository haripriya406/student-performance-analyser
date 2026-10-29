
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, jsonify, request
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Find the dataset relative to this app.py file
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "student_performance_dataset.csv"

TARGET_COLUMN = "academic_risk"

# These columns should not be used as prediction features
EXCLUDED_COLUMNS = [
    "student_id",
    "final_score",
    "academic_risk",
]

df = None
model = None
feature_columns = []
model_error = None


def load_and_train():
    """Load the CSV and train the academic-risk prediction model."""
    global df, model, feature_columns, model_error

    try:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATASET_PATH}"
            )

        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.strip()

        if df.empty:
            raise ValueError("The dataset is empty.")

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Missing '{TARGET_COLUMN}' column in the CSV."
            )

        # Remove rows without a known target label
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("string").str.strip()
        training_data = df.dropna(subset=[TARGET_COLUMN]).copy()
        training_data = training_data[
            training_data[TARGET_COLUMN] != ""
        ]

        if training_data[TARGET_COLUMN].nunique() < 2:
            raise ValueError(
                "The academic_risk column must contain at least "
                "two different classes, such as At Risk and Not At Risk."
            )

        feature_columns = [
            column for column in df.columns
            if column not in EXCLUDED_COLUMNS
        ]

        if not feature_columns:
            raise ValueError("No usable prediction features were found.")

        X = training_data[feature_columns].copy()
        y = training_data[TARGET_COLUMN].astype(str)

        # Fill missing values before encoding
        for column in X.columns:
            if pd.api.types.is_numeric_dtype(X[column]):
                X[column] = X[column].fillna(X[column].median())
            else:
                X[column] = X[column].fillna("Unknown").astype(str)

        # Convert categorical values into numeric columns
        X = pd.get_dummies(X)
        feature_columns = X.columns.tolist()

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X, y)

        model_error = None
        print(f"Dataset loaded: {DATASET_PATH}")
        print(f"Student records: {len(df)}")
        print("Machine learning model trained successfully.")

    except Exception as error:
        df = None
        model = None
        feature_columns = []
        model_error = str(error)
        print("Dataset/model error:", model_error)


def get_dataset_records():
    """Return CSV records in a JSON-friendly format."""
    if df is None:
        return []
    return df.to_json(orient="records", date_format="iso")


def prepare_features(student_data):
    """Prepare submitted student details for model prediction."""
    input_df = pd.DataFrame([student_data])

    # Keep only the features used during training
    input_df = input_df.reindex(
        columns=[
            column for column in df.columns
            if column not in EXCLUDED_COLUMNS
        ]
    )

    for column in input_df.columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            input_df[column] = pd.to_numeric(
                input_df[column], errors="coerce"
            )
            median_value = df[column].median()
            input_df[column] = input_df[column].fillna(
                median_value if pd.notna(median_value) else 0
            )
        else:
            input_df[column] = input_df[column].fillna("Unknown").astype(str)

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return input_df


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/students")
def students():
    return render_template("students.html")


@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "dataset_loaded": df is not None,
        "dataset_path": str(DATASET_PATH),
        "total_students": len(df) if df is not None else 0,
        "total_columns": len(df.columns) if df is not None else 0,
        "model_ready": model is not None,
        "target_column": TARGET_COLUMN,
        "prediction_features": len(feature_columns),
        "error": model_error
    })


@app.route("/api/dataset")
def api_dataset():
    if df is None:
        return jsonify({"error": model_error or "Dataset is not loaded."}), 500

    return jsonify({
        "columns": df.columns.tolist(),
        "total": len(df),
        "rows": __import__("json").loads(get_dataset_records())
    })


@app.route("/api/students")
def api_students():
    if df is None:
        return jsonify({"error": model_error or "Dataset is not loaded."}), 500

    return jsonify({
        "students": __import__("json").loads(get_dataset_records()),
        "total": len(df)
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if df is None or model is None:
        return jsonify({
            "error": model_error or "The prediction model is not ready."
        }), 500

    student_data = request.get_json(silent=True)

    if not isinstance(student_data, dict):
        return jsonify({"error": "Please submit valid student details."}), 400

    try:
        X_new = prepare_features(student_data)
        prediction = str(model.predict(X_new)[0])

        probabilities = model.predict_proba(X_new)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

        return jsonify({
            "prediction": prediction,
            "academic_risk": prediction,
            "confidence": confidence,
            "message": "Prediction completed successfully."
        })

    except Exception as error:
        return jsonify({
            "error": f"Prediction failed: {str(error)}"
        }), 400


# Load the dataset and train the model when the application starts
load_and_train()


if __name__ == "__main__":
    app.run(debug=True)
