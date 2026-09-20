"""MODEL MONITORING WITH EVIDENTLY - drift and prediction-stability checks.

Simple monitoring workflow for the deployed churn model:

* Reference window = the first rows of the raw training CSV (baseline behaviour).
* Current window   = the latest rows (simulating recent production traffic).
* Both windows are scored with the *deployed* model: the same serving transform
  plus the tuned decision threshold used by ``/predict``.

Evidently then compares the windows and reports feature drift, target drift
(changed churn-rate mix) and prediction behaviour (positive-rate plus
precision/recall/F1 drift).

Outputs: ``GET /monitoring`` (JSON summary) and ``GET /monitoring/report``
(full interactive Evidently HTML dashboard).

Production wiring: point the current window at logged ``/predict`` requests
(with delayed ground-truth labels joined in) instead of the CSV tail used
here for the demo.
"""

import json
import os
import re

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "Telco-Customer-Churn.csv")
REPORT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "monitoring")
REPORT_PATH = os.path.join(REPORT_DIR, "evidently_report.html")

# Window sizes: small enough to run in seconds, large enough for stable tests.
N_REFERENCE = 2000
N_CURRENT = 500

_CACHE = {"summary": None}

# Evidently 0.7.x names per-column metrics like
# "ValueDrift(column=tenure,method=...,threshold=0.1)" with a numeric value.
_VALUE_DRIFT_RE = re.compile(r"ValueDrift\(column=(.*?),method=.*?threshold=([\d.]+)\)")
_DRIFT_SHARE_RE = re.compile(r"DriftedColumnsCount\(drift_share=([\d.]+)\)")


def _load_windows(n_reference=N_REFERENCE, n_current=N_CURRENT):
    """Load reference/current raw windows, normalised like production input."""
    from src.data.preprocess import preprocess_data

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Reference data not found at {RAW_CSV}. "
            "Mount ./data into the container to enable monitoring."
        )
    raw = pd.read_csv(RAW_CSV)
    ref = preprocess_data(raw.head(n_reference).copy(), target_col="Churn")
    cur = preprocess_data(raw.tail(n_current).copy(), target_col="Churn")
    for df in (ref, cur):
        if df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype(int)
        df["Churn"] = df["Churn"].astype(int)
    return ref, cur


def _score_with_deployed_model(df):
    """Score rows with the deployed model (same transform + threshold as /predict)."""
    from src.serving.inference import THRESHOLD, _serve_transform, model

    features = df.drop(columns=["Churn"])
    proba = model.predict_proba(_serve_transform(features))[:, 1]
    return (proba >= THRESHOLD).astype(int)


def _quality_numbers(y_true, y_pred):
    """Plain precision/recall/F1/accuracy (avoids a second heavy dependency)."""
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "accuracy": round((tp + tn) / max(len(y_true), 1), 4),
        "precision": round(tp / (tp + fp) if tp + fp else 0.0, 4),
        "recall": round(tp / (tp + fn) if tp + fn else 0.0, 4),
        "f1": round(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0, 4),
    }


def _parse_run_dict(d):
    """Extract dataset drift flag, drift share and drifted columns (Evidently 0.7.x)."""
    drift_share = None
    share_cutoff = 0.5
    columns = {}
    for m in d.get("metrics", []):
        name = m.get("metric_name", "") or ""
        value = m.get("value")
        dm = _DRIFT_SHARE_RE.search(name)
        if dm and isinstance(value, dict):
            share_cutoff = float(dm.group(1))
            drift_share = float(value.get("share", 0.0))
            continue
        vm = _VALUE_DRIFT_RE.search(name)
        if vm and isinstance(value, (int, float)):
            columns[vm.group(1)] = {
                "score": float(value),
                "threshold": float(vm.group(2)),
            }
    drifted = sorted(c for c, v in columns.items() if v["score"] >= v["threshold"])
    dataset_drift = (drift_share is not None) and (drift_share >= share_cutoff)
    return dataset_drift, drift_share, drifted


def run_monitoring(force=False):
    """Run Evidently (reference vs current), save HTML report, return summary."""
    try:
        from evidently import BinaryClassification, DataDefinition, Dataset, Report
        from evidently.presets.classification import ClassificationPreset
        from evidently.presets.drift import DataDriftPreset
    except ImportError as e:
        raise RuntimeError(
            "Install monitoring dependency: pip install evidently==0.7.23"
        ) from e

    if not force and _CACHE["summary"] is not None and os.path.exists(REPORT_PATH):
        return _CACHE["summary"]

    ref, cur = _load_windows()
    ref["prediction"] = _score_with_deployed_model(ref)
    cur["prediction"] = _score_with_deployed_model(cur)

    definition = DataDefinition(
        classification=[
            BinaryClassification(target="Churn", prediction_labels="prediction")
        ]
    )
    datasets = {
        "reference": Dataset.from_pandas(ref, data_definition=definition),
        "current": Dataset.from_pandas(cur, data_definition=definition),
    }
    run = Report(metrics=[DataDriftPreset(), ClassificationPreset()]).run(
        current_data=datasets["current"], reference_data=datasets["reference"]
    )
    os.makedirs(REPORT_DIR, exist_ok=True)
    run.save_html(REPORT_PATH)

    dataset_drift, drift_share, drifted = _parse_run_dict(run.dict())
    ref_pos, cur_pos = float(ref["prediction"].mean()), float(cur["prediction"].mean())
    ref_rate, cur_rate = float(ref["Churn"].mean()), float(cur["Churn"].mean())
    summary = {
        "reference_rows": len(ref),
        "current_rows": len(cur),
        "dataset_drift": dataset_drift,
        "drift_share": drift_share,
        "drifted_columns": drifted,
        "prediction_positive_rate": {
            "reference": round(ref_pos, 4),
            "current": round(cur_pos, 4),
            "delta": round(cur_pos - ref_pos, 4),
        },
        "target_churn_rate": {
            "reference": round(ref_rate, 4),
            "current": round(cur_rate, 4),
            "delta": round(cur_rate - ref_rate, 4),
        },
        "quality_reference_vs_current": {
            "reference": _quality_numbers(ref["Churn"], ref["prediction"]),
            "current": _quality_numbers(cur["Churn"], cur["prediction"]),
        },
        "report": "artifacts/monitoring/evidently_report.html",
        "report_abs_path": REPORT_PATH,
    }
    _CACHE["summary"] = summary
    return summary


def register_monitoring(app):
    """Attach GET /monitoring and GET /monitoring/report to the FastAPI app."""
    try:
        import evidently  # noqa: F401 - version check only
        enabled = True
    except ImportError:
        enabled = False

    if not enabled:

        @app.get("/monitoring", tags=["monitoring"])
        def monitoring_disabled():
            return {
                "status": "disabled",
                "hint": "pip install evidently==0.7.23, then GET /monitoring?refresh=true",
            }

        return app

    from fastapi.responses import FileResponse

    @app.get("/monitoring", tags=["monitoring"])
    def monitoring_summary(refresh: bool = False):
        """JSON summary: dataset drift, drifted columns, prediction/target shifts."""
        try:
            return run_monitoring(force=refresh)
        except (FileNotFoundError, RuntimeError) as e:
            return {"status": "unavailable", "reason": str(e)}

    @app.get("/monitoring/report", tags=["monitoring"])
    def monitoring_report(refresh: bool = False):
        """Full interactive Evidently HTML dashboard."""
        try:
            summary = run_monitoring(force=refresh)
        except (FileNotFoundError, RuntimeError) as e:
            return {"status": "unavailable", "reason": str(e)}
        return FileResponse(
            summary["report_abs_path"],
            media_type="text/html",
            filename="evidently_report.html",
        )

    return app


if __name__ == "__main__":
    print(json.dumps(run_monitoring(force=True), indent=2))
