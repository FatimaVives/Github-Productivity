import joblib
from sklearn.metrics import classification_report, confusion_matrix
from data import load_raw, basic_clean, create_features, train_val_test_split
from utils import setup_logging
logger = setup_logging()

def evaluate_model(model_path, dataset_split="test"):
    model = joblib.load(model_path)
    df = load_raw(); df = basic_clean(df); df = create_features(df)
    _, _, test = train_val_test_split(df, target_col="productivity")
    features = [c for c in test.columns if c not in ("productivity", "repo_name")]
    X_test, y_test = test[features], test["productivity"]
    yhat = model.predict(X_test)
    logger.info(classification_report(y_test, yhat))
    print(confusion_matrix(y_test, yhat))

if __name__ == "__main__":
    evaluate_model("models/rf.joblib")
