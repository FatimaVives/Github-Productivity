from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()
MODEL = joblib.load("models/rf.joblib")

class RepoFeatures(BaseModel):
    commits_per_week: float
    avg_issue_close_days: float
    # add other features...

@app.post("/predict")
def predict(feat: RepoFeatures):
    df = [feat.dict()]
    pred = MODEL.predict(df)[0]
    proba = MODEL.predict_proba(df)[0].tolist()
    return {"prediction": pred, "probability": proba}
