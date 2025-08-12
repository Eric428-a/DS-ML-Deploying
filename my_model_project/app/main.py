from fastapi import FastAPI
import joblib
import pandas as pd

# Load your trained model
model = joblib.load("model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML Model API is running!"}

@app.post("/predict")
def predict(features: dict):
    try:
        # Convert incoming dict to DataFrame
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
