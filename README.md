# Student Performance Analysis and Prediction System

A beginner-friendly Flask application for exploring student records in a CSV and evaluating a Random Forest classifier. Predictions are educational estimates only; they are not definitive judgments, diagnoses, or a basis for consequential decisions about students.

## Important: add your dataset

The project folder currently does not contain `student_performance_dataset.csv`. The application does **not** create an artificial replacement. Add your original CSV to this folder before training or viewing analytics. The required headers are:

```text
student_id,age,gender,study_hours_per_day,attendance_percentage,previous_score,assignments_completed_out_of_10,sleep_hours_per_day,internet_access,parental_support,extracurricular_hours_per_week,family_income_level,final_score,academic_risk
```

Column order may differ, but missing or additional columns are rejected. Numeric values must be within reasonable bounds (including scores and attendance from 0 to 100), and `academic_risk` must contain `At Risk` or `Not At Risk`. Category choices in the prediction form are derived from the active dataset.

## Folder structure

```text
StudentPerformanceProject/
├── app.py
├── data_utils.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── student_performance_dataset.csv       # Add your original CSV; intentionally not generated
├── models/
│   └── .gitkeep                           # Trained pipeline is generated here
├── reports/
│   └── .gitkeep                           # Evaluation JSON is generated here
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── students.html
│   ├── evaluation.html
│   ├── dataset.html
│   ├── about.html
│   └── error.html
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── js/dashboard.js
│   ├── js/predict.js
│   ├── js/students.js
│   ├── js/evaluation.js
│   ├── js/dataset.js
│   └── images/.gitkeep
├── instance/.gitkeep
└── tests/test_app.py
```

`data_utils.py` centralizes the schema and validation so that uploads and model training apply the same checks. The trained model and metrics are generated files and are excluded from source control.

## Run on Windows with VS Code

Open PowerShell in VS Code, then run these commands from the project directory:

Use Python 3.10 or later.

1. **Open the project** (if VS Code is not already open):

   ```powershell
   cd "D:\student performance analyser"
   code .
   ```

2. **Add the dataset:** copy your original `student_performance_dataset.csv` into `D:\student performance analyser`. Keep a separate backup of your original file.

3. **Create and activate a virtual environment:**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation for this terminal, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

4. **Install dependencies:**

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. **Train and evaluate the model once:**

   ```powershell
   python train_model.py
   ```

   This validates the active CSV (a confirmed replacement upload, when present; otherwise the project-root CSV), removes exact duplicate records before splitting, trains a pipeline on prediction-time features, and writes `models\student_performance_model.joblib` and `reports\model_metrics.json`. The test set is held out from fitting and is not used for tuning.

6. **Start Flask:**

   ```powershell
   python app.py
   ```

7. **Open the local website** in a browser: <http://127.0.0.1:5000>

8. **Run automated tests** (open a second terminal and activate the environment):

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m unittest discover -s tests -v
   ```

The dashboard, students, and dataset views read the active CSV through Flask API endpoints. The model evaluation page reads the last saved test metrics. The prediction page requires both a valid active dataset (for categorical choices and validation) and a trained model.

## Upload behavior

The Dataset page accepts CSV files up to 10 MB. The server validates the schema, values, and target labels before writing to `instance\replacement_dataset.csv`. You must confirm that the upload should become active. The original project CSV is never overwritten. The validated upload becomes the active source for dashboard, student, and dataset analytics. Uploading **does not retrain** the saved model; predictions continue using the previously trained model until you explicitly run `python train_model.py`, which then trains from the active replacement CSV. Back up files you need before manually changing datasets.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/dashboard` | Dataset-backed summary and chart data |
| GET | `/api/students` | Searchable, filterable, paginated student records |
| POST | `/api/predict` | Validates inputs and returns a model estimate |
| GET | `/api/model-metrics` | Most recent held-out evaluation results |
| GET | `/api/dataset-info` | Schema, quality summary, and category choices |
| POST | `/api/upload-dataset` | Validates and activates a replacement CSV |

`/api/students` accepts `page`, `per_page` (maximum 100), `search`, `risk`, `min_attendance`, `max_attendance`, `sort` (`student_id`, `score`, or `attendance`), and `order` (`asc` or `desc`).

## Troubleshooting

- **Dataset not found:** ensure the CSV is in the project root and named exactly `student_performance_dataset.csv`, or set `STUDENT_DATASET_PATH` to its path before starting Flask/training.
- **Incorrect CSV headers or invalid values:** compare against the listed schema. Empty cells, unsupported risk labels, invalid number formats, and out-of-range values are rejected with a validation message.
- **Model not trained:** activate the virtual environment and run `python train_model.py`; then restart or refresh the application.
- **PowerShell activation is blocked:** use the process-only execution-policy command above; it does not change the machine-wide policy.
- **Port 5000 is already in use:** set a different port in the current PowerShell session, then run the app: `$env:FLASK_PORT = "5001"; python app.py`. Open `http://127.0.0.1:5001`.
- **Charts do not appear:** Chart.js is loaded from a CDN, so check the browser's internet access and refresh. Data tables and API endpoints do not depend on the chart library.
- **Upload succeeds but predictions stay the same:** uploads update analytics only. Retrain explicitly using the intended training CSV; review the training output and metrics after doing so.

## Verification checklist

- [ ] Dashboard values and charts populate from the active CSV; missing data displays an explicit error.
- [ ] Sample records and the Students page show CSV records, and search, risk/attendance filters, sorting, and pagination work.
- [ ] Prediction rejects missing, unexpected, out-of-range, or unsupported category inputs; a valid request returns a model estimate and, when supported, model output scores.
- [ ] Evaluation displays test-set counts, accuracy, macro precision/recall/F1, confusion matrix, classification report, and class distribution from the saved training report.
- [ ] Dataset page displays dimensions, duplicate count, missing-value summary, descriptions, and risk distribution.
- [ ] Invalid uploads are rejected; a confirmed valid upload becomes active while the original file remains unchanged.
- [ ] About page explains the methodology, limitations, and need for external validation.
- [ ] Run `python -m unittest discover -s tests -v` and confirm all tests pass.

## Responsible-use note

The application is a technical and educational demonstration, not a validated intervention system. Synthetic data and artificially generated target labels can produce misleadingly strong test scores. Even real-data evaluation needs representative external validation, fairness and privacy review, clear consent and governance, and educator/student judgment. Never treat model output as the student's actual probability of failure or as a definitive assessment of ability.
