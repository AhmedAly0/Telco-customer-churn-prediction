FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/* 

COPY . .

COPY src/serving/model /app/src/serving/model

# Copy MLFlow run (artifacts+metadata) to the flat /app/model convenience path
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/preprocessing.pkl /app/model/preprocessing.pkl


# Ensure logs are shown in time without buffering
# Lets you import moduels without using the src prefix using from app... instead of from src.app...
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]