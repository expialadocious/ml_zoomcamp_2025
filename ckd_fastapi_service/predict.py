import pickle
from typing import Literal
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field



### PYDANTIC INPUT SCHEMA (matches df_train columns)
class Features(BaseModel):
    # Numeric features (float64 in df_train)
    age: float = Field(..., ge=0, le=120)   # years
    bp: float = Field(..., ge=0)            # blood pressure
    sg: float = Field(..., ge=0)            # specific gravity
    al: float = Field(..., ge=0)            # albumin
    su: float = Field(..., ge=0)            # sugar
    bgr: float = Field(..., ge=0)           # blood glucose random
    bu: float = Field(..., ge=0)            # blood urea
    sc: float = Field(..., ge=0)            # serum creatinine
    sod: float = Field(..., ge=0)           # sodium
    pot: float = Field(..., ge=0)           # potassium
    hemo: float = Field(..., ge=0)          # hemoglobin
    pcv: float = Field(..., ge=0)           # packed cell volume
    wbcc: float = Field(..., ge=0)          # white blood cell count
    rbcc: float = Field(..., ge=0)          # red blood cell count

    # Categorical features
    pcc: Literal["notpresent", "present"]   # pus cell clumps
    ba: Literal["notpresent", "present"]    # bacterial
    htn: Literal["yes", "no"]               # hypertension
    dm: Literal["yes", "no"]                # diabetes
    cad: Literal["no", "yes"]               # coronary artery disease
    appet: Literal["good", "poor"]          # appetite
    pe: Literal["yes", "no"]                # pedal edema
    ane: Literal["no", "yes"]               # anemia


### PYDANTIC response Schema
class PredictResponse(BaseModel):
    ckd_probability: float  # probability of positive class = CKD
    ckd: bool               # True if probability >= threshold


### FAST API 
app = FastAPI(title="ckd-xgboost-predictions")


### LOAD XGBOOST MODEL WITH PICKLE
with open("pipeline_v2.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


### PREDICT WITH XGBOOST MODEL IN FAST API
def predict_single(features: dict) -> float:
    """
    features: dict from Pydantic (feature_name -> value)
    returns: probability of positive class (e.g., CKD = 1)
    """
    # pipeline expects a list of dicts similiar to df_train.to_dict(orient='records')
    proba = pipeline.predict_proba([features])[0, 1]
    return float(proba)

### CREATE APP ENDPOINT
@app.post("/predict", response_model=PredictResponse)
def predict(payload: Features) -> PredictResponse:
    # Convert Pydantic model -> plain dict for DictVectorizer
    features_dict = payload.model_dump()

    prob = predict_single(features_dict)

    # choose threshold; 0.5 is standard
    has_ckd = prob >= 0.5

    return PredictResponse(
        ckd_probability=prob,
        ckd=has_ckd
    )


#### LAUNCH LOCALLY
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

