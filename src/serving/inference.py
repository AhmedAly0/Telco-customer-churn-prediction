"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies the same one-hot encoding as training, aligned to the exported feature schema
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import glob
import pandas as pd
import mlflow
import mlflow.xgboost

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: MODEL_DIR is set during Docker container build (/app/model)
# In development it can be overridden with the MODEL_DIR environment variable
# (e.g. the promoted model in src/serving/model) or resolved from local mlruns.
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")

# === DECISION THRESHOLD ===
# Training/tuning optimised recall-first thresholds (0.30 - 0.35) instead of the
# naive 0.5 cut-off, because a missed churner (FN) costs far more than a wasted
# retention offer (FP). Serving must use the same threshold to stay aligned with
# the business objective that was logged to MLflow as a run parameter.
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.35"))


def _load_model(model_dir: str):
    """
    Load the trained model from an MLflow model directory.

    The native xgboost flavour is preferred because it exposes `predict_proba`,
    which lets us apply the tuned THRESHOLD. Any other flavour falls back to the
    generic pyfunc loader (which predicts with the default 0.5 cut-off).
    """
    try:
        return mlflow.xgboost.load_model(model_dir)
    except Exception:
        return mlflow.pyfunc.load_model(model_dir)


try:
    model = _load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
    print(f"✅ Decision threshold: {THRESHOLD}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development: newest MLflow run artifacts under <repo>/mlruns
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        mlruns_root = os.path.join(repo_root, "mlruns")
        # MLflow 3.x stores logged models in <root>/<exp_id>/models/<model_id>/artifacts,
        # older versions used <root>/<exp_id>/<run_id>/artifacts/model - support both.
        candidates = glob.glob(os.path.join(mlruns_root, "*", "models", "*", "artifacts"))
        candidates += glob.glob(os.path.join(mlruns_root, "*", "*", "artifacts", "model"))
        local_model_paths = [
            p for p in candidates if os.path.exists(os.path.join(p, "MLmodel"))
        ]
        if not local_model_paths:
            raise Exception(
                f"No model found under {mlruns_root}. Run scripts/promote_model.py and "
                f"point MODEL_DIR at src/serving/model instead."
            )
        latest_model = max(local_model_paths, key=os.path.getmtime)
        model = _load_model(latest_model)
        MODEL_DIR = latest_model
        print(f"✅ Fallback: Loaded model from {latest_model}")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === FEATURE SCHEMA LOADING ===
# CRITICAL: Load the exact feature column order used during training
# This ensures the model receives features in the expected order
# Two layouts are supported:
#   1. flat container layout  -> <model_dir>/feature_columns.txt
#   2. MLflow run layout      -> <model_dir>/../feature_columns.txt
FEATURE_COLS = None
for candidate in (
    os.path.join(MODEL_DIR, "feature_columns.txt"),
    os.path.join(MODEL_DIR, os.pardir, "feature_columns.txt"),
):
    if os.path.exists(candidate):
        with open(candidate) as f:
            FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
        print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from {candidate}")
        break

if not FEATURE_COLS:
    raise Exception(f"Failed to load feature_columns.txt next to {MODEL_DIR}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features  
    4. Convert boolean columns to integers
    5. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw customer data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # IMPORTANT: drop_first=False here (NOT True like in training).
        # Training dropped the first category because it saw ALL categories at
        # once; a single serving row contains only ONE category per feature, so
        # drop_first=True would drop that row's only category and collapse every
        # multi-category feature to the reference category (all zeros) - e.g. a
        # "Two year" customer encoded as "Month-to-month".
        # The training reference columns do not exist in FEATURE_COLS, so the
        # reindex() in STEP 5 removes them anyway -> identical encoding.
        df = pd.get_dummies(df, columns=obj_cols, drop_first=False)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.
    
    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string
    
    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)
                   
    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)
        
    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """
    
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        # Prefer probability output so the tuned recall-first THRESHOLD is honoured
        # (the native xgboost flavour exposes predict_proba; pyfunc does not).
        churn_probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_enc)
            churn_probability = float(proba[0][1])
            print(f"   📈 Churn probability: {churn_probability:.4f} (threshold={THRESHOLD})")

        preds = (
            model.predict(df_enc)
            if churn_probability is None
            else int(churn_probability >= THRESHOLD)
        )
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to churn"      # High risk - needs intervention
    else:
        return "Not likely to churn"  # Low risk - maintain normal service
    