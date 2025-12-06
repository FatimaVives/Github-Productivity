import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from data import load_raw, basic_clean, create_features, train_val_test_split
from utils import setup_logging

logger = setup_logging()


def _ensure_outputs_dir(path: str = "outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def evaluate_model(model_path, dataset_split="test"):
    model = joblib.load(model_path)
    df = load_raw()
    df = basic_clean(df)
    df = create_features(df)
    _, _, test = train_val_test_split(df, target_col="productivity")

    # Select numeric features only (same logic as training)
    numeric_feats = test.select_dtypes(include=["number"]).columns.tolist()
    if "productivity" in numeric_feats:
        numeric_feats.remove("productivity")

    if not numeric_feats:
        raise RuntimeError("No numeric features found in test set. Check preprocessing.")

    X_test = test[numeric_feats].copy()
    y_test = test["productivity"].copy()

    # Align columns with model if it exposes expected feature names
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        expected = list(expected)
        missing = [c for c in expected if c not in X_test.columns]
        if missing:
            raise RuntimeError(f"Model expects features not present in test set: {missing}")
        # Reorder and drop any extra columns
        X_test = X_test[expected]

    # Run prediction and report
    yhat = model.predict(X_test)
    # Log textual report
    logger.info(classification_report(y_test, yhat))
    cm = confusion_matrix(y_test, yhat)
    print(cm)

    # Prepare outputs directory and save artifacts for the Streamlit app
    outdir = _ensure_outputs_dir("outputs")

    # Metrics to save
    metrics = {
        "accuracy": float(accuracy_score(y_test, yhat)),
        "f1_macro": float(f1_score(y_test, yhat, average="macro")),
        "f1_weighted": float(f1_score(y_test, yhat, average="weighted")),
    }
    metrics_path = os.path.join(outdir, "metrics.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        logger.info("Wrote metrics to %s", metrics_path)
    except Exception:
        logger.exception("Failed to write metrics.json")

    # Save confusion matrix image
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = os.path.join(outdir, "conf_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        logger.info("Saved confusion matrix to %s", cm_path)
    except Exception:
        logger.exception("Failed to save confusion matrix image")

    # Save feature importances if available
    try:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            fi_df = None
            try:
                fi_df = (
                    X_test.columns.to_series().reset_index(drop=True).to_frame(name="feature")
                )
                fi_df["importance"] = fi
            except Exception:
                # fallback: create from model.feature_importances_
                fi_df = [
                    {"feature": f, "importance": float(v)}
                    for f, v in zip(X_test.columns.tolist(), fi)
                ]
            fi_df = fi_df if isinstance(fi_df, (list,)) else fi_df
            fi_out = os.path.join(outdir, "feature_importances.csv")
            # If fi_df is a list, convert to DataFrame
            if isinstance(fi_df, list):
                import pandas as pd

                fi_df = pd.DataFrame(fi_df)
            fi_df.to_csv(fi_out, index=False)
            logger.info("Saved feature importances to %s", fi_out)
        else:
            logger.info("Model has no feature_importances_ attribute; skipping feature importances output")
    except Exception:
        logger.exception("Failed to save feature importances")

if __name__ == "__main__":
    evaluate_model("models/rf.joblib")
