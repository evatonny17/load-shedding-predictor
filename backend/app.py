from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
area_encoder = joblib.load("area_encoder.pkl")
day_encoder = joblib.load("day_encoder.pkl")

@app.get("/")
def home():
    return {"message": "Smart Load Shedding Predictor API is running"}

@app.get("/predict")
def predict(area: str, hour: int, day: str, temperature: int, demand: int, generation: int, previous_outage: int):
    area_encoded = area_encoder.transform([area])[0]
    day_encoded = day_encoder.transform([day])[0]

    input_data = pd.DataFrame([{
        "area": area_encoded,
        "hour": hour,
        "day": day_encoded,
        "temperature": temperature,
        "demand": demand,
        "generation": generation,
        "previous_outage": previous_outage
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    return {
        "load_shedding": int(prediction),
        "probability": round(probability, 2)
    }