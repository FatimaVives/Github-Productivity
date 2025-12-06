# GitHub Repository Productivity Predictor

A small toolkit and demo app that analyzes GitHub repositories and predicts their "productivity" using a combination of simple heuristics (commits/month) and reference machine learning models.

## Overview

- Purpose: Provide an easy-to-run pipeline that loads a GitHub repositories dataset, engineers features, trains simple classification models, and exposes an interactive Streamlit app that reports repository productivity.
- Approach: The app computes an interpretable productivity metric based on commits per month (a human-friendly heuristic) and also loads previously-trained models for comparison and reference.

## Features

- Robust CSV loader and feature engineering pipeline (`data.py`).
- Two simple ML pipelines (baseline logistic regression and a random forest) saved under `models/`.
- Training and evaluation utilities (`train.py`, `evaluate.py`).
- Lightweight exploratory analysis script that saves plots to `outputs/` (`eda.py`).
- Streamlit app (`app.py`) that accepts a GitHub repo URL, fetches public metadata, estimates commit activity, and shows a clear, interpretable productivity badge plus model reference predictions.

## Project Structure

Top-level files and their roles:

- `config.py` : Configuration constants and default paths used across scripts.
- `data.py` : Data loading, cleaning, feature engineering, and train/validation/test splitting.
- `models.py` : Scikit-learn pipeline definitions for baseline and random-forest models.
- `train.py` : Script to prepare data, train pipelines, evaluate, and save model artifacts to `models/`.
- `evaluate.py` : Helper utilities to evaluate and report model metrics.
- `predict.py` : Small helper for running predictions against saved model artifacts.
- `serve.py` : Minimal API server (if included) for exposing model predictions programmatically.
- `eda.py` : Exploratory Data Analysis script that generates and saves plots to `outputs/`.
- `app.py` : Streamlit application that fetches GitHub repository data, computes commits/month, displays a productivity badge, and exposes the trained model's reference prediction.
- `tune.py` : hyperparameter tuning utilities if present.
- `requirements.txt` : Python dependency list for creating a reproducible environment.
- `data/` : Directory containing input CSVs (e.g. `repositories.csv`).
- `models/` : Directory where trained model artifacts are written (e.g. `baseline.joblib`, `rf.joblib`).
- `outputs/` : Generated outputs like EDA plots: `outputs/plot1.png`, `outputs/plot2.png`.
- `tests/` : Unit tests for data logic (e.g. `tests/test_data.py`).

Basic workflow diagram:

```
CSV (data/repositories.csv)
        |
     data.py  -> feature engineering
        |
   train.py -> models/ (joblib)
        |
  app.py (Streamlit) -> uses heuristics + models/ for reference
```

## Dataset

This project uses a GitHub Repositories dataset (commonly available on sources like Kaggle). The example dataset included is `data/repositories.csv`. Typical columns used include repository name, owner, created/updated dates, stars, forks, open issues, and an optional `commits_count` column. The codebase contains defensive parsing so missing `commits_count` values fall back to a heuristic or GitHub API estimate.


## Productivity Metric

We define a clear, interpretable productivity metric based on commits per month (commits/month):

- Compute: commits_per_month = commits_count / max(1, repo_age_days / 30)
- Thresholds used by the app:
  - High productivity: > 20 commits/month
  - Medium productivity: 5–20 commits/month
  - Low productivity: < 5 commits/month

These thresholds are intentionally simple and conservative for demonstration and teaching purposes. The Streamlit UI displays the computed commits/month value and a colored badge so users can quickly understand where a repository falls.

Note: `commits_count` is estimated by either reading the dataset column or by querying the GitHub commits endpoint when a live repository URL is provided. The estimation method is pragmatic and may undercount in some histories — see Limitations below.

## Installation

Clone the repository and create a Python virtual environment. Example commands for Windows PowerShell:

```powershell
git clone <your-repo-url>
cd "Github-Productivity"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux replace the activation line with:

```bash
source .venv/bin/activate
```

If you see any missing packages during development, ensure your virtual environment is activated and run `pip install -r requirements.txt` again.

## How to train the model

Training is automated through `train.py`. This prepares the dataset, fits baseline and random-forest pipelines, evaluates them, and saves artifacts to `models/`.

Run:

```powershell
python train.py
```

Outputs:
- `models/baseline.joblib` — baseline logistic regression pipeline.
- `models/rf.joblib` — trained random forest pipeline.

The script logs metrics and will raise helpful errors if required columns are missing.

## How to run EDA

The repository includes a simple EDA script `eda.py` that generates exploratory plots and saves them to `outputs/`.

Run:

```powershell
python eda.py
```

Generated files (examples):
- `outputs/plot1.png` — scatter / relationship plot
- `outputs/plot2.png` — distribution / histogram

Add or modify the EDA script to explore additional features or produce publication-ready figures.

## How to launch the Streamlit app

Start the app with Streamlit. Make sure your virtual environment is activated.

```powershell
streamlit run app.py
```

App behavior:
- Paste a GitHub repository URL (https://github.com/owner/repo) into the input field.
- The app fetches public repo metadata from the GitHub API, estimates commits (if needed), computes commits/month, and shows a productivity badge (High / Medium / Low) plus supporting metrics.
- The app also exposes the trained ML model's prediction and probabilities in an expandable section for reference.


## Technologies used

- Python 3.x
- pandas for data processing
- scikit-learn for modeling
- joblib for model persistence
- Streamlit for the web UI
- matplotlib / seaborn for plotting (EDA)
- pytest for basic unit tests
