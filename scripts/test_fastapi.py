"""Quick smoke test for the local FastAPI service.

Start the API first:
    python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000

Then run:
    python scripts/test_fastapi.py
    # or point it somewhere else:
    $env:API_URL = "http://127.0.0.1:8137/predict"; python scripts/test_fastapi.py
"""

import os

import requests

url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

sample_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 350.75
}

response = requests.post(url, json=sample_data, timeout=30)
print(f"POST {url}")
print("Status Code:", response.status_code)
print("Response:", response.json())
