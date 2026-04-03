from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI + Docker 🚀"}

@app.get("/predict")
def predict():
    # Dummy ML logic
    return {"prediction": "This is a demo output"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}