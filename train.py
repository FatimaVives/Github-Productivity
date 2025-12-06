# train.py
import joblib
from utils import setup_logging
from data import load_raw, basic_clean, create_features, train_val_test_split
from models import get_baseline_pipeline, get_rf_pipeline
from config import MODEL_DIR, RF_PARAMS, RANDOM_SEED
from sklearn.metrics import f1_score, accuracy_score

logger = setup_logging()


def prepare_data():
    df = load_raw()
    df = basic_clean(df)
    df = create_features(df)
    logger.info("Prepared dataframe with shape %s", df.shape)
    return df


def fit_and_save(model, X_train, y_train, model_name="model"):
    model.fit(X_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    outpath = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, outpath)
    logger.info("Saved model to %s", outpath)


def main():
    df = prepare_data()
    train, val, test = train_val_test_split(df, target_col="productivity")  # ensure target exists

    # Select numeric features only (exclude the target)
    numeric_feats = train.select_dtypes(include=["number"]).columns.tolist()
    if "productivity" in numeric_feats:
        numeric_feats.remove("productivity")

    if not numeric_feats:
        raise RuntimeError("No numeric features found for training. Check your preprocessing.")

    logger.info("Using %d numeric features for training", len(numeric_feats))

    X_train, y_train = train[numeric_feats], train["productivity"]
    X_val, y_val = val[numeric_feats], val["productivity"]

    try:
        # baseline
        baseline = get_baseline_pipeline()
        baseline.fit(X_train, y_train)
        yhat = baseline.predict(X_val)
        logger.info("Baseline acc %.4f f1 %.4f", accuracy_score(y_val, yhat), f1_score(y_val, yhat, average="macro"))
        fit_and_save(baseline, X_train, y_train, model_name="baseline")

        # random forest
        rf = get_rf_pipeline(RF_PARAMS)
        rf.fit(X_train, y_train)
        yhat_rf = rf.predict(X_val)
        logger.info("RF acc %.4f f1 %.4f", accuracy_score(y_val, yhat_rf), f1_score(y_val, yhat_rf, average="macro"))
        fit_and_save(rf, X_train, y_train, model_name="rf")
    except Exception:
        logger.exception("Training failed")
        raise


if __name__ == "__main__":
    main()
