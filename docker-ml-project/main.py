from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import joblib

app = FastAPI(title="Titanic Survival Predictor")

# Load model and scaler
model = joblib.load("model.pkl")
model.eval()
scaler = joblib.load("scaler.pkl")


class PassengerInput(BaseModel):
    Age: float
    Sex: int          # 0 = male, 1 = female
    Pclass: int       # 1, 2, or 3
    Fare: float
    Embarked: int     # S=0, Q=1, C=2


@app.get("/")
def root():
    return {"message": "Titanic Survival Predictor API is running!"}


@app.post("/predict")
def predict(passenger: PassengerInput):
    features = np.array([[
        passenger.Age,
        passenger.Sex,
        passenger.Pclass,
        passenger.Fare,
        passenger.Embarked
    ]])

    features_scaled = scaler.transform(features)
    tensor_input = torch.FloatTensor(features_scaled)

    with torch.no_grad():
        output = model(tensor_input)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0

    return {
        "survived": prediction,
        "probability": round(probability, 4),
        "result": "Survived ✅" if prediction == 1 else "Did not survive ❌"
    }