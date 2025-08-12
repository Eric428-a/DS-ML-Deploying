from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load the model
with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict(np.array(features).reshape(1, -1))
    return {"prediction": prediction.tolist()}
