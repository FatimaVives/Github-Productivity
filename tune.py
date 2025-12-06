# tune.py
from sklearn.model_selection import GridSearchCV
from models import get_rf_pipeline
from config import RF_PARAMS
import joblib

def tune_rf(X_train, y_train):
    pipe = get_rf_pipeline()
    param_grid = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [None, 10, 20]
    }
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs

# usage: call tune_rf from a small script or notebook after loading prepared train data
