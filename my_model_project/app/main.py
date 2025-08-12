from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Path to model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# Create FastAPI app
app = FastAPI(title="Salary Prediction API", version="1.0")

# Define request body
class Features(BaseModel):
    Age: int
    Years_Experience: int
    Education_Level: int
    Department: int
    Performance_Score: float

@app.get("/")
def home():
    return {"message": "Salary Prediction API is running!"}

@app.post("/predict")
def predict(features: Features):
    try:
        # Convert input to numpy array
        data = np.array([[features.Age,
                          features.Years_Experience,
                          features.Education_Level,
                          features.Department,
                          features.Performance_Score]])
        
        # Make prediction
        prediction = model.predict(data)[0]
        return {"predicted_salary": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
