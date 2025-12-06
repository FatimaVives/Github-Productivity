import joblib
import pandas as pd
from utils import setup_logging
logger = setup_logging()

def predict_for_repo(model_path, repo_features: dict):
    model = joblib.load(model_path)
    df = pd.DataFrame([repo_features])
    probs = model.predict_proba(df)
    pred = model.predict(df)
    return pred[0], probs[0]

# Example usage:
# pred, probs = predict_for_repo("models/rf.joblib", {"commits_per_week": 3.2, "avg_issue_close_days": 2.1, ...})
