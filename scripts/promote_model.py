#!/usr/bin/env python3
"""
Promote the newest finished MLflow run into the serving model directory.

Why this exists
---------------
The training pipeline (`scripts/run_pipeline.py`) writes its model, feature
schema and preprocessing contract to MLflow. The serving layer
(`src/serving/inference.py`) and the Docker image expect a *flat* model folder:

    <dest>/MLmodel
    <dest>/model.ubj
    <dest>/feature_columns.txt
    <dest>/preprocessing.pkl

This script resolves the newest FINISHED run in an experiment, downloads the
logged model plus the two serving artifacts, copies them into that flat layout
and writes a `model_info.json` for traceability. The Dockerfile then simply does
`COPY src/serving/model /app/model`, so no run id is ever hardcoded.

Usage:
    python scripts/promote_model.py
    python scripts/promote_model.py --experiment "Telco Churn" --dest src/serving/model
    python scripts/promote_model.py --tracking-uri sqlite:///mlflow.db
"""

import argparse
import json
import os
import shutil
import tempfile
import time

import mlflow
from mlflow.tracking import MlflowClient

# Artifacts the training pipeline logs next to the model and that serving needs
SERVING_ARTIFACTS = ["feature_columns.txt", "preprocessing.pkl"]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _pick_run_and_model(client: MlflowClient, experiment_name: str):
    """Return (run, logged_model) for the newest finished run that logged a model."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"❌ MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    if not runs:
        raise SystemExit(f"❌ No FINISHED runs in experiment: {experiment_name}")

    # MLflow 3.x keeps logged models in their own store -> group them by source run
    models_by_run = {}
    for model in client.search_logged_models(experiment_ids=[experiment.experiment_id]):
        if str(model.status).upper().endswith("READY"):
            models_by_run.setdefault(model.source_run_id, []).append(model)

    for run in runs:
        candidates = sorted(
            models_by_run.get(run.info.run_id, []),
            key=lambda m: m.creation_timestamp,
            reverse=True,
        )
        if candidates:
            return run, candidates[0]

    raise SystemExit(
        f"❌ No FINISHED run with a logged model in experiment: {experiment_name}"
    )


def _copy_files(src_dir: str, dest_dir: str) -> list:
    """Copy every file from src_dir into dest_dir (flat) and return their names."""
    copied = []
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest_dir, name))
            copied.append(name)
    return copied


def promote(client: MlflowClient, experiment_name: str, dest: str) -> dict:
    """Download the newest model + serving artifacts into `dest` (flat layout)."""
    run, logged_model = _pick_run_and_model(client, experiment_name)
    os.makedirs(dest, exist_ok=True)

    print(f"🏆 Promoting run {run.info.run_id} ({run.data.tags.get('mlflow.runName', '-')})")
    print(f"   model_id: {logged_model.model_id}")

    with tempfile.TemporaryDirectory() as tmp:
        # 1) the logged model itself (MLmodel, model.ubj, conda.yaml, ...)
        model_dir = mlflow.artifacts.download_artifacts(
            f"models:/{logged_model.model_id}", dst_path=os.path.join(tmp, "model")
        )
        model_files = _copy_files(model_dir, dest)

        # 2) the serving artifacts logged by the training pipeline
        extras_dir = os.path.join(tmp, "extras")
        artifact_files = []
        for artifact in SERVING_ARTIFACTS:
            try:
                downloaded = client.download_artifacts(
                    run.info.run_id, artifact, dst_path=extras_dir
                )
                shutil.copy2(downloaded, os.path.join(dest, artifact))
                artifact_files.append(artifact)
            except Exception as e:  # keep promoting even if an extra is missing
                print(f"⚠️  Could not download {artifact}: {e}")

    # 3) traceability record shared with the serving team / README
    info = {
        "experiment": experiment_name,
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "model_id": logged_model.model_id,
        "source": run.data.tags.get("mlflow.source.name"),
        "git_commit": run.data.tags.get("mlflow.source.git.commit"),
        "params": run.data.params,
        "metrics": run.data.metrics,
        "model_files": model_files,
        "serving_artifacts": artifact_files,
        "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(dest, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    missing = [a for a in SERVING_ARTIFACTS if a not in artifact_files]
    if missing:
        print(f"⚠️  Missing serving artifacts in {dest}: {missing}")
    else:
        print(f"✅ Promoted model + artifacts to {dest}")
    print(f"   metrics: { {k: round(v, 4) for k, v in run.data.metrics.items()} }")
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Promote the newest finished MLflow run into the serving model dir"
    )
    parser.add_argument("--experiment", type=str, default="Telco Churn",
                        help="MLflow experiment name")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db",
                        help="MLflow tracking URI (default: sqlite:///mlflow.db)")
    parser.add_argument("--dest", type=str, default=os.path.join("src", "serving", "model"),
                        help="destination directory (relative paths resolve from the project root)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    dest = args.dest if os.path.isabs(args.dest) else os.path.join(PROJECT_ROOT, args.dest)
    promote(MlflowClient(), args.experiment, dest)


if __name__ == "__main__":
    main()
