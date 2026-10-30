# Student Performance Analyser

A Flask dashboard for exploring student records and demonstrating an academic-risk classifier. Dataset summaries and charts use the CSV in `data\student_performance_dataset.csv`. Predictions are educational estimates only; they are not guarantees, diagnoses, or a basis for consequential decisions about students.

## Features

- Dashboard with dataset-derived student counts, average attendance and study hours, and three interactive charts.
- Searchable, risk-filtered student records with pagination.
- Dataset explorer with a preview, column descriptions, missing-value counts, and optional confirmed CSV replacement.
- Logistic Regression predictions using a scikit-learn pipeline with numeric imputation/scaling and categorical imputation/one-hot encoding.
- Held-out evaluation metrics, including accuracy, macro precision, macro recall, macro F1-score, and a confusion matrix.

The model uses age, gender, study hours, attendance, previous score, assignments completed, sleep hours, internet access, parental support, extracurricular hours, and family income. It excludes `student_id` and `final_score` to avoid target leakage. Category choices are read from the active dataset, and unknown categories are safely handled by the encoder.

## Dataset

The expected CSV path is `data\student_performance_dataset.csv`. It must contain these columns:

```text
student_id,age,gender,study_hours_per_day,attendance_percentage,previous_score,assignments_completed_out_of_10,sleep_hours_per_day,internet_access,parental_support,extracurricular_hours_per_week,family_income_level,final_score,academic_risk
```

Predictor values can be missing; the model pipeline imputes them. Missing `academic_risk` labels are excluded from model training and evaluation. The dataset explorer reports missing values. If a replacement dataset is uploaded, it becomes active without overwriting the original CSV; predictions and evaluation retrain from the active file when requested.

## Run on Windows

Open the VS Code terminal in the project folder and run:

```powershell
cd "D:\student performance analyser"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate the environment again. If `.venv` already exists, you can skip `py -m venv .venv`.

Open <http://127.0.0.1:5000> in your browser. The home URL redirects to the dashboard. Model training happens automatically the first time you open Prediction or Evaluation after starting the app. You can also train and save the model and held-out metrics from the terminal:

```powershell
python train_model.py
```

If port 5000 is already in use, start the app on port 5001 and open <http://127.0.0.1:5001>:

```powershell
$env:FLASK_PORT = "5001"
python app.py
```

Run the test suite with:

```powershell
python -m unittest discover -s tests -v
```

## Pages

| Page | URL | Purpose |
|---|---|---|
| Dashboard | `/dashboard` | CSV-based summary cards, risk charts, and recent records |
| Student records | `/students` | Search by student ID, filter by risk category, and paginate |
| Dataset explorer | `/dataset` | Inspect columns, missing values, a data preview, and upload an optional replacement |
| Risk prediction | `/predict` | Submit student attributes and view the result page |
| Model evaluation | `/evaluation` | Inspect fixed-seed, held-out test metrics and confusion matrix |
| About | `/about` | Review methodology and responsible-use limitations |

Chart.js is loaded from a CDN, so charts require an internet connection. The rest of the app runs locally.

## Responsible-use note

This application is a technical and educational demonstration, not a validated student assessment system. A model score may reflect biases, missing context, data-quality issues, or artificial patterns in the dataset. Human judgment and student context must remain central; do not use this tool by itself to make consequential decisions.
