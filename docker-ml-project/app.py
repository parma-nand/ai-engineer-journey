from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

# Define request schema
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)

        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}