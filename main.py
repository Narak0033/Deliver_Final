from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Food Delivery Delay Prediction API")

model = joblib.load("delivery_model.pkl")

class DeliveryInput(BaseModel):
    Distance_km: float
    Preparation_Time_min: int
    Courier_Experience_yrs: float
    Weather: str
    Traffic_Level: str
    Time_of_Day: str
    Vehicle_Type: str

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: DeliveryInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].max()
    return {
        "prediction": prediction,
        "probability": round(float(probability), 4)
    }