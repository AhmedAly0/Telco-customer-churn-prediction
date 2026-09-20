# Telco Customer Churn Prediction

An end-to-end machine learning project: raw telecom data in, a containerized churn-prediction API out.

The scope is the full data science workflow, implemented in code rather than notebooks — data validation,
preprocessing, feature engineering, model training and tuning, evaluation, and serving through a REST API
and a web UI — with supporting engineering practices (experiment tracking, CI, containerization) integrated
along the way.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"/>
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-3.2.0-EB6E4B"/>
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white"/>
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-3.16.1-0194E2?logo=mlflow&logoColor=white"/>
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.141.1-009688?logo=fastapi&logoColor=white"/>
  <img alt="Docker Hub" src="https://img.shields.io/badge/Docker%20Hub-ahmed1py%2Ftelco--fastapi-2496ED?logo=docker&logoColor=white"/>
  <img alt="CI" src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white"/>
</p>

## Contents

- [Why recall-first](#why-recall-first)
- [Results](#results)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [API](#api)
- [Docker and CI](#docker-and-ci)
- [Engineering notes](#engineering-notes)
- [Roadmap](#roadmap)

## Why recall-first

Missing a real churner means losing a customer; flagging a loyal customer only costs a retention offer.
The problem is asymmetric, so everything downstream — the class weighting, the decision threshold, the
evaluation view — is optimised for recall on the churn class rather than accuracy.

## Results

The champion model below is the one actually served: trained by `scripts/run_pipeline.py`, tracked in
MLflow, and promoted into `src/serving/model/` by `scripts/promote_model.py`. Metrics are measured on a
held-out 20% test set, with churn as the positive class.

| Metric | Value |
|---|---|
| Recall | **0.818** (306 of 374 churners caught) |
| Precision | 0.489 |
| F1 | 0.612 |
| ROC-AUC | 0.838 |
| Accuracy | 0.725 |
| Decision threshold | 0.35 (tuned, logged as a run param) |
| Training time | 1.5 s |

- A separate Optuna study (see `notebooks/EDA.ipynb`) reached recall 0.922 / ROC-AUC 0.847; its best
  hyperparameters are baked into the pipeline:
  `n_estimators=301, learning_rate=0.034, max_depth=7, subsample=0.95, colsample_bytree=0.98`.
- Single-row inference, including the feature transform, takes ~16 ms.

## Dataset

The public Telco Customer Churn sample dataset (IBM, via Kaggle): 7,043 customers x 21 columns with a
26.5% churn rate. The CSV is not committed to this repo — download it into
`data/raw/Telco-Customer-Churn.csv` (see Quickstart).

Main churn signals found in EDA: month-to-month contracts, fiber-optic internet, electronic-check
payments, low tenure, and missing online security or tech support.

## Pipeline

One command runs the whole workflow (`scripts/run_pipeline.py`):

1. **Ingest** — load the raw CSV.
2. **Validate** — a Great Expectations gate on the raw data: required columns exist and are non-null,
   categorical domains (contract type, internet service, payment method, ...), numeric ranges
   (`tenure` 0-120, `MonthlyCharges` 0-200, `TotalCharges` >= 0), and `TotalCharges > MonthlyCharges`
   (>= 95% of rows). Training aborts if the gate fails.
3. **Preprocess** — type coercion and cleaning (e.g. blank `TotalCharges`).
4. **Features** — deterministic binary mapping plus one-hot encoding produce 30 model features. The
   exact serving schema is exported (`feature_columns.txt`, `preprocessing.pkl`) and logged to MLflow.
5. **Train** — XGBoost with a class-imbalance weight (`scale_pos_weight` ~ 2.77) on a stratified
   80/20 split.
6. **Evaluate** — recall / precision / F1 / ROC-AUC plus confusion matrix, all logged to MLflow
   together with params, the model, and the feature schema.
7. **Promote** — `scripts/promote_model.py` copies the champion run's artifacts into
   `src/serving/model/` (tracked in git, so the Docker build always ships a model).

## Project structure

```
├── data/raw/                       # Telco-Customer-Churn.csv (downloaded, git-ignored)
├── notebooks/EDA.ipynb             # EDA, model comparison (RF / LightGBM / XGBoost), Optuna study
├── src/
│   ├── data/                       # loading + preprocessing
│   ├── features/build_features.py  # encoding + feature schema export
│   ├── models/                     # train.py, tune.py (Optuna), evaluate.py
│   ├── utils/validate_data.py      # Great Expectations 1.x data gate
│   ├── serving/                    # inference.py + model/ (promoted model, tracked)
│   └── app/main.py                 # FastAPI + Gradio app
├── scripts/                        # run_pipeline.py, promote_model.py, smoke-test scripts
├── .github/workflows/ci.yml        # build + push the image to Docker Hub
├── dockerfile
└── requirements.txt
```

## Quickstart

Requires Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset (IBM / Kaggle) into `data/raw/Telco-Customer-Churn.csv`, then train and promote:

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
python scripts/promote_model.py
```

Serve locally:

```bash
export MODEL_DIR="$PWD/src/serving/model"     # Windows PowerShell: $env:MODEL_DIR = "$PWD\src\serving\model"
uvicorn src.app.main:app --port 8000
```

- Interactive API docs: `http://127.0.0.1:8000/docs`
- Web UI: `http://127.0.0.1:8000/ui`

On some Windows machines port 8000 falls inside a reserved port range (bind fails with
`WinError 10013`); check with `netsh interface ipv4 show excludedportrange protocol=tcp` and pick a free
port, e.g. `--port 8139`.

Or run the container — the promoted model ships inside it, no training needed:

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

Predict:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","PhoneService":"Yes",
       "MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes",
       "Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "tenure":1,"MonthlyCharges":85.0,"TotalCharges":85.0}'
```

Response: `{"prediction": "Likely to churn"}` or `{"prediction": "Not likely to churn"}`.


## API

`POST /predict` accepts the 19 raw dataset fields — the 18 features plus `SeniorCitizen`, which the
model was trained on (omitting it would silently skew every prediction). The full request/response
schema is auto-documented at `/docs` (OpenAPI). `GET /` is a health check for load balancers, and the
Gradio UI is mounted at `/ui`. `GET /monitoring` returns an Evidently summary (dataset drift, drifted
columns, prediction/target shifts); `GET /monitoring/report` serves the full interactive HTML dashboard.

## Docker and CI

- The `dockerfile` uses a slim Python 3.12 base, installs the pinned requirements, copies the promoted
  model and feature schema into `/app/model`, and runs uvicorn on port 8000.
- GitHub Actions (`.github/workflows/ci.yml`) builds the image and pushes it to Docker Hub as
  `ahmed1py/telco-fastapi`.

## Engineering notes

- **Train/serve parity.** Serving re-implements the training-time transform deterministically. One
  subtlety: one-hot encoding a single row with `drop_first=True` would drop that row's only category
  and collapse every categorical feature to its reference level. Serving therefore encodes with
  `drop_first=False` and aligns to the exported schema — equivalent to training and correct for
  single rows.
- **Data quality gate.** The Great Expectations suite runs on the raw CSV before anything else; a
  failed expectation stops the pipeline and reports the failing expectation types.
- **Recall-first threshold.** The 0.35 cut-off is read from the `CHURN_THRESHOLD` env var, so it can
  be changed per deployment without touching code.
- **Modern APIs.** Validation uses the Great Expectations 1.x fluent API with an ephemeral in-memory
  context; promotion and serving target the MLflow 3 logged-model layout.

## Roadmap

- Turn the smoke-test scripts into a real `pytest` suite and run it in CI before pushing the image.
- Wire the Optuna study into `run_pipeline.py` (pull best params from the tracking store) instead of
  using fixed values.
- Remove the legacy duplicate `src/app/app.py` (the app entry point is `src/app/main.py`).
- Swap the monitoring current-window from the CSV tail to logged `/predict` traffic with joined
  labels; stage the promoted model in the MLflow registry with a `champion` alias.
- Add a LICENSE.

## Credits

Author: [Ahmed Aly](https://github.com/AhmedAly0).
Dataset: Telco Customer Churn sample dataset (IBM, via Kaggle).

---

All metrics in this README come from actual runs of this repository — the MLflow-tracked pipeline run,
the promoted model, and the served API.

