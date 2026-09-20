FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/* 

COPY . .

# Promoted model artifacts (flat layout):
#   MLmodel, model.ubj, feature_columns.txt, preprocessing.pkl
# This is the exact layout src/serving/inference.py expects via MODEL_DIR=/app/model,
# so no run id needs to be hardcoded here anymore.
COPY src/serving/model /app/model


# Ensure logs are shown in time without buffering
# Lets you import moduels without using the src prefix using from app... instead of from src.app...
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]