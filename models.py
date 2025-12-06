# models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_baseline_pipeline():
    # simple baseline: logistic regression with scaling
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    return pipe

def get_rf_pipeline(rf_params=None):
    rf_params = rf_params or {}
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(**rf_params))
    ])
    return pipe
