import joblib 
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("C:\Users\hp303\OneDrive\Documents\C.S.E\Python\House_Predict_API\house_model.joblib")

#input schema
class HouseFeaturess(BaseModel):
    MedInc : float=Field(gt=0, description="Median Income of Neighbourhood")
    HouseAge: float=Field(gt=0, description="Average age of House")
    AvgRooms: float = Field(gt=0, description="Average number of rooms")
    AvgBedrms: float = Field(gt=0, description="Average number of rooms")
    Population: float = Field(gt=0, description="Total Population in Neighbourhood")    
    AveOccup: float = Field(gt=0, description="Average number of Occup")
    Latitude: float = Field(gt=32, le=42, description="Latitude")
    Longitude: float = Field(gt=-125, le=-114 description="Longitude")

#home
@app.get("/")
def home():
    return {
        "message":"California house prediction API",
        "status": "running",
        "endpoint":"send POST request to /predict"
    }

@app.get("/health")
def health():
    return {
        "status":"running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000",
        
    }

