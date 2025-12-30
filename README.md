# GitHub Repository Productivity Predictor

A small toolkit and demo app that analyzes GitHub repositories and predicts their "productivity" using a combination of simple heuristics (commits/month) and reference machine learning models.

## Overview

- Purpose: Provide an easy-to-run pipeline that loads a GitHub repositories dataset, engineers features, trains simple classification models, and exposes an interactive Streamlit app that reports repository productivity.
- Approach: The app computes an interpretable productivity metric based on commits per month (a human-friendly heuristic) and also loads previously-trained models for comparison and reference.

## Features

- Robust CSV loader and feature engineering pipeline (`data.py`).
- Two simple ML pipelines (baseline logistic regression and a random forest) saved under `models/`.
- Training and evaluation utilities (`train.py`, `evaluate.py`).
- Streamlit app (`app.py`) that accepts a GitHub repo URL, fetches public metadata, estimates commit activity, and shows a productivity badge (High/Medium/Low) with supporting metrics.

## Project Structure

Top-level files and their roles:

- `app.py` : Streamlit web application (entry point). Accepts GitHub URLs, fetches repo data, analyzes productivity.
- `train.py` : Script to train ML pipelines from `data/repositories.csv`.
- `evaluate.py` : Script to evaluate trained models and save metrics to `outputs/metrics.json`.
- `data.py` : Data loading, cleaning, and feature engineering utilities.
- `models.py` : Scikit-learn pipeline definitions (baseline logistic regression, random forest).
- `predict.py` : Prediction utilities for trained models.
- `config.py` : Configuration constants and paths.
- `utils.py` : Logging and utility functions.
- `requirements.txt` : Python dependencies (streamlit, scikit-learn, pandas, joblib).
- `vercel.json` : Vercel deployment config (optional; app deployed on Streamlit Cloud).
- `models/` : Pre-trained model artifacts (`baseline.joblib`, `rf.joblib`).
- `outputs/` : Generated evaluation metrics (`metrics.json`).
- `data/` : Input datasets (note: `repositories.csv` excluded from deployment).
- `tests/` : Unit tests for data validation.

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

## How to launch the Streamlit app

Start the app with Streamlit. Make sure your virtual environment is activated.

```powershell
streamlit run app.py
```

App behavior:
- Paste a GitHub repository URL (https://github.com/owner/repo) into the input field.
- The app fetches public repo metadata from the GitHub API, estimates commits (if needed), computes commits/month, and shows a productivity badge (High / Medium / Low) plus supporting metrics.
- The app also exposes the trained ML model's prediction and probabilities in an expandable section for reference.

## Deployment

This app is deployed on **Streamlit Cloud** (free tier):

```
https://app-appuctivity-rt5uaqf59jpkvzme2gsn86.streamlit.app/
```

**Git-based deploys**: Push to `main` on GitHub and Streamlit Cloud auto-redeploys.

Quick steps:
1) Go to https://streamlit.io/cloud
2) Click "New app" and connect your GitHub repo
3) Select main branch and `app.py` as entry point
4) Deploy

Every push to `main` triggers automatic redeployment. No Docker, no configuration files needed (Streamlit Cloud detects `requirements.txt` automatically).

## Evaluation Outputs

Run:

```powershell
python evaluate.py
```

Outputs (written to `outputs/`):
- `metrics.json` with `accuracy`, `f1_macro`, `f1_weighted`

These metrics appear in the Streamlit app (accuracy shown as a progress bar). If the file is missing, rerun the command and check the console logs.

## Technologies used

- **Python 3.x**
- **Streamlit** — web UI framework
- **scikit-learn** — machine learning models
- **pandas** — data processing
- **joblib** — model serialization
