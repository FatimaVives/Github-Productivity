import os
import json
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import joblib
import pandas as pd
import streamlit as st


MODEL_PATHS = ["models/rf.joblib", "models/baseline.joblib"]


def classify_productivity(commits_count, repo_age_days):
    """Classify productivity based on commits per month."""
    if repo_age_days <= 0:
        return "low"
    commits_per_day = commits_count / repo_age_days
    commits_per_month = commits_per_day * 30
    if commits_per_month > 20:
        return "high"
    elif commits_per_month >= 5:
        return "medium"
    else:
        return "low"


def parse_github_url(url: str):
    """Return (owner, repo) or raise ValueError."""
    if url is None:
        raise ValueError("No URL provided")
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    # expect github.com/owner/repo
    parts = url.split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL")
    # owner is second last, repo is last
    owner = parts[-2]
    repo = parts[-1]
    if not owner or not repo:
        raise ValueError("Invalid GitHub URL")
    return owner, repo


def fetch_repo(owner: str, repo: str):
    """Fetch repository JSON from GitHub API. Honor GITHUB_TOKEN if present."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=15) as resp:
            data = resp.read().decode()
            return json.loads(data)
    except HTTPError as e:
        raise RuntimeError(f"GitHub API error: {e.code} {e.reason}")
    except Exception as e:
        raise RuntimeError(f"Error fetching repo: {e}")


def fetch_commits_count(owner: str, repo: str):
    """Estimate commits count by requesting 1 commit per page and reading Link header."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=1"
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=15) as resp:
            # Link header contains last page information when multiple pages exist
            link = resp.headers.get("Link")
            if link:
                # parse rel="last" page number
                parts = [p.strip() for p in link.split(",")]
                for p in parts:
                    if 'rel="last"' in p:
                        # find page number
                        start = p.find("<") + 1
                        end = p.find(">")
                        last_url = p[start:end]
                        # parse page param
                        from urllib.parse import parse_qs, urlparse

                        q = parse_qs(urlparse(last_url).query)
                        if "page" in q:
                            return int(q["page"][0])
            # No Link header: read body to determine 0 or 1
            data = resp.read().decode()
            arr = json.loads(data)
            if isinstance(arr, list):
                return len(arr)
            return 0
    except Exception:
        return 0


def preprocess_repo(repo_json: dict, owner: str = None, repo_name: str = None):
    """Build feature DataFrame row matching training features.

    We create numeric and boolean features used in training and engineered ones.
    """
    # map api fields to expected names
    stars = int(repo_json.get("stargazers_count", 0) or 0)
    forks = int(repo_json.get("forks_count", 0) or 0)
    issues = int(repo_json.get("open_issues_count", 0) or 0)
    size = float(repo_json.get("size", 0) or 0)
    watchers = int(repo_json.get("watchers_count", 0) or 0)

    # booleans
    bools = {
        "Has Issues": int(bool(repo_json.get("has_issues", False))),
        "Has Projects": int(bool(repo_json.get("has_projects", False))),
        "Has Downloads": int(bool(repo_json.get("has_downloads", False))),
        "Has Wiki": int(bool(repo_json.get("has_wiki", False))),
        "Has Pages": int(bool(repo_json.get("has_pages", False))),
        "Has Discussions": int(bool(repo_json.get("has_discussions", False))),
        "Is Fork": int(bool(repo_json.get("fork", False))),
        "Is Archived": int(bool(repo_json.get("archived", False))),
        "Is Template": int(bool(repo_json.get("is_template", False))),
    }

    # commits_count: try to fetch a count via commits API if owner/repo provided
    commits_count = 0
    if owner and repo_name:
        try:
            commits_count = int(fetch_commits_count(owner, repo_name) or 0)
        except Exception:
            commits_count = 0

    # dates
    created_at = repo_json.get("created_at")
    updated_at = repo_json.get("updated_at")
    pushed_at = repo_json.get("pushed_at")
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00")) if created_at else None
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00")) if updated_at else None
        pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00")) if pushed_at else None
    except Exception:
        created = None
        updated = None
        pushed = None

    today = datetime.now(timezone.utc)
    if created and updated:
        repo_age_days = max(0, (updated - created).days)
    elif created:
        repo_age_days = max(0, (today - created).days)
    else:
        repo_age_days = 0

    # last push days
    if pushed:
        days_since_push = max(0, (today - pushed).days)
    else:
        days_since_push = None

    # safe ratios
    stars_per_fork = stars / forks if forks > 0 else 0.0
    issues_per_star = issues / stars if stars > 0 else 0.0

    # Build row in a stable order. This order should roughly match training features.
    feature_order = [
        "Size",
        "Stars",
        "Forks",
        "Issues",
        "Watchers",
        "Has Issues",
        "Has Projects",
        "Has Downloads",
        "Has Wiki",
        "Has Pages",
        "Has Discussions",
        "Is Fork",
        "Is Archived",
        "Is Template",
        "commits_count",
        "repo_age_days",
        "stars_per_fork",
        "issues_per_star",
    ]

    row = {
        "Size": size,
        "Stars": stars,
        "Forks": forks,
        "Issues": issues,
        "Watchers": watchers,
        **bools,
        "commits_count": commits_count,
        "repo_age_days": float(repo_age_days),
        "stars_per_fork": float(stars_per_fork),
        "issues_per_star": float(issues_per_star),
    }

    df = pd.DataFrame([row], columns=feature_order)
    meta = {"days_since_push": days_since_push, "created": created, "updated": updated}
    return df, meta


def load_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception:
                continue
    return None


def load_model_performance(output_dir: str = "outputs"):
    """Load model evaluation artifacts: metrics.json, conf_matrix.png, feature_importances.csv.

    Returns a dict with keys: metrics (or None), conf_matrix_path (or None), feature_importances_df (or None)
    """
    out = {"metrics": None, "conf_matrix_path": None, "feature_importances": None}
    metrics_path = os.path.join(output_dir, "metrics.json")
    conf_matrix_path = os.path.join(output_dir, "conf_matrix.png")
    fi_path = os.path.join(output_dir, "feature_importances.csv")

    # metrics
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                out["metrics"] = json.load(f)
    except Exception:
        out["metrics"] = None

    # confusion matrix image
    if os.path.exists(conf_matrix_path):
        out["conf_matrix_path"] = conf_matrix_path

    # feature importances
    try:
        if os.path.exists(fi_path):
            df_fi = pd.read_csv(fi_path)
            # expect columns like 'feature' and 'importance' or similar
            if "feature" in df_fi.columns and "importance" in df_fi.columns:
                out["feature_importances"] = df_fi
            else:
                # try to handle alternate column names
                if df_fi.shape[1] >= 2:
                    df_fi = df_fi.iloc[:, :2]
                    df_fi.columns = ["feature", "importance"]
                    out["feature_importances"] = df_fi
    except Exception:
        out["feature_importances"] = None

    return out


def predict(model, features_df: pd.DataFrame):
    # model expects numeric feature order used during training; pass DataFrame
    preds = model.predict(features_df)
    return preds[0]


def generate_feedback(repo_json: dict, features_df: pd.DataFrame, meta: dict, pred: str):
    """Return a short message and list of reasons or suggestions depending on prediction."""
    reasons = []
    suggestions = []

    stars = int(repo_json.get("stargazers_count", 0) or 0)
    forks = int(repo_json.get("forks_count", 0) or 0)
    issues = int(repo_json.get("open_issues_count", 0) or 0)
    commits = int(features_df.iloc[0].get("commits_count", 0) or 0)
    archived = bool(repo_json.get("archived", False))

    issues_per_star = float(features_df.iloc[0]["issues_per_star"])
    stars_per_fork = float(features_df.iloc[0]["stars_per_fork"])
    days_since_push = meta.get("days_since_push")

    # Positive signals
    if days_since_push is not None and days_since_push <= 90:
        reasons.append("recent activity (recent pushes) \U0001F525")
    if not archived:
        reasons.append("project is active (not archived) \U0001F44D")
    if 0 < stars_per_fork < 100:
        reasons.append("healthy star-to-fork ratio \U0001F4C8")
    if issues_per_star < 0.1:
        reasons.append("low open-issues relative to stars \U0001F44C")

    # Negative signals / suggestions
    if days_since_push is None or (days_since_push is not None and days_since_push > 180):
        suggestions.append("Make more frequent commits/pushes to show activity \U0001F4AA")
    if stars < 50:
        suggestions.append("Improve README, docs and examples to attract users \U0001F4D6")
    if issues_per_star > 0.5:
        suggestions.append("Triage and close issues faster to improve contributor experience \U0001F6A7")
    if forks == 0:
        suggestions.append("Encourage contributions (PRs) to increase forks and engagement \U0001F91D")
    if commits < 10:
        suggestions.append("Make more commits to show ongoing maintenance (small, frequent commits are great) \U0001F4DD")

    # Build messages
    if pred == "high":
        header = "🎉 Nice project — this repo shows strong signals!"
        details = [f"{r}" for r in reasons] if reasons else ["multiple positive signals found \U0001F389"]
    else:
        header = "💪 Keep going — a few ideas to improve this repo"
        details = [f"{s}" for s in suggestions] if suggestions else ["Consider improving documentation and activity \U0001F4AA"]

    return header, details


def main():
    st.set_page_config(page_title="GitHub Repo Success Predictor", layout="wide")
    st.title("GitHub Repo Success Predictor")
    st.write("Paste a GitHub repository URL and click 'Check Repository' to predict productivity (low/medium/high).")

    url = st.text_input("GitHub repository URL")
    if st.button("Check Repository"):
        if not url:
            st.error("Please enter a GitHub repository URL.")
            return

        with st.spinner("Fetching repository..."):
            try:
                owner, repo = parse_github_url(url)
                repo_json = fetch_repo(owner, repo)
            except Exception as e:
                st.error(f"Failed to fetch repository: {e}")
                return

        features, meta = preprocess_repo(repo_json, owner, repo)

        model = load_model()  # keep model loading for compatibility (not used for label)
        # compute commits-based productivity and ignore model prediction
        commits = int(features.iloc[0].get("commits_count", 0) or 0)
        repo_age_days = float(features.iloc[0].get("repo_age_days", 0) or 0)
        pred = classify_productivity(commits, repo_age_days)

        # Compute commits-per-month and show a friendly, colored badge
        commits = int(features.iloc[0].get("commits_count", 0) or 0)
        repo_age_days = float(features.iloc[0].get("repo_age_days", 0) or 0)
        commits_per_month = (commits / repo_age_days) * 30 if repo_age_days > 0 else 0.0

        # Badge + thresholds explanation
        if pred == "high":
            st.success(f"🎉 Productivity (by commits): **{pred.upper()}** — {commits_per_month:.1f} commits/month")
        elif pred == "medium":
            st.warning(f"⚡ Productivity (by commits): **{pred.capitalize()}** — {commits_per_month:.1f} commits/month")
        else:
            st.info(f"🔧 Productivity (by commits): **{pred.capitalize()}** — {commits_per_month:.1f} commits/month")

        st.caption("Thresholds: high > 20 commits/month, medium 5–20 commits/month, low < 5 commits/month")

        # (Model prediction expander removed by user request)

        # Model Performance (informational only)
        perf = load_model_performance(output_dir="outputs")
        with st.expander("🔍 Model Performance"):
            st.subheader("Model evaluation summary")
            metrics = perf.get("metrics")
            if metrics is None:
                st.warning("Model evaluation files not found in `outputs/`. Skipping performance display.")
            else:
                # Display core metrics with explanations
                acc = metrics.get("accuracy")
                macro_f1 = metrics.get("f1_macro") or metrics.get("macro_f1") or metrics.get("f1_macro_score")
                weighted_f1 = metrics.get("f1_weighted") or metrics.get("weighted_f1") or metrics.get("f1_weighted_score")

                cols = st.columns(3)
                with cols[0]:
                    if acc is not None:
                        st.metric("Accuracy", f"{float(acc):.2f}")
                    else:
                        st.write("Accuracy: N/A")
                with cols[1]:
                    if macro_f1 is not None:
                        st.metric("Macro F1", f"{float(macro_f1):.2f}")
                    else:
                        st.write("Macro F1: N/A")
                with cols[2]:
                    if weighted_f1 is not None:
                        st.metric("Weighted F1", f"{float(weighted_f1):.2f}")
                    else:
                        st.write("Weighted F1: N/A")

                st.markdown("**Explanation:** Accuracy is the percentage of correct predictions. F1 Score combines precision and recall into a single metric (0-1, higher is better).")

                # Simple progress bar showing model accuracy
                if acc is not None:
                    acc_float = float(acc)
                    st.subheader("📈 Model Accuracy")
                    st.progress(min(acc_float, 1.0))
                    st.write(f"**{acc_float*100:.1f}%** of predictions are correct")

                # Feature importances
                fi_df = perf.get("feature_importances")
                if fi_df is not None and not fi_df.empty:
                    try:
                        st.subheader("Feature Importances")
                        # ensure proper types
                        fi_df = fi_df.copy()
                        fi_df["importance"] = fi_df["importance"].astype(float)
                        fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
                        # show a bar chart
                        st.bar_chart(fi_df.set_index("feature")["importance"])
                        # show top table
                        st.table(fi_df.head(10).assign(importance=lambda d: d["importance"].map(lambda x: f"{x:.2f}")))
                    except Exception as e:
                        st.warning(f"Failed to display feature importances: {e}")
                else:
                    st.info("No feature importances file found at `outputs/feature_importances.csv`.")

        # Show key metrics - simplified
        commits = int(features.iloc[0].get("commits_count", 0) or 0)
        st.subheader("📊 Repository Statistics")
        
        # Display key metrics in large, easy-to-read format
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("⭐ Stars", f"{int(repo_json.get('stargazers_count', 0) or 0):,}")
        col2.metric("🍴 Forks", f"{int(repo_json.get('forks_count', 0) or 0):,}")
        col3.metric("📝 Commits", f"{commits:,}")
        col4.metric("⚠️ Issues", f"{int(repo_json.get('open_issues_count', 0) or 0):,}")
        col5.metric("📅 Age (days)", f"{int(features.iloc[0]['repo_age_days'])}")

        # Friendly feedback (positive reasons or actionable suggestions)
        header, items = generate_feedback(repo_json, features, meta, pred)
        if pred == "high":
            st.success(header)
            for it in items:
                st.write(f"- {it}")
        else:
            st.info(header)
            for it in items:
                st.write(f"- {it}")


if __name__ == "__main__":
    main()
