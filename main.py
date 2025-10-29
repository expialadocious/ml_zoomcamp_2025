
import pickle
from sklearn.pipeline import make_pipeline


# ## Question 4, 5, and 6
# ## Also built Dockerfile
# #Now let's serve this model as a web service

# #Install FastAPI
#Write FastAPI code for serving the model
#Now score this client using requests:

### NOTE: Docker Image "FROM" contains pipeline_v2.bin model

import uvicorn
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI


class Customer(BaseModel):
    lead_source: Literal["organic_search", 
                         "paid_ads", "referral", 
                         'social_media', 'events']
    number_of_courses_viewed: int = Field(...)
    annual_income: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title='customer-churn-predictions')

with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn = prob >= 0.5
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
