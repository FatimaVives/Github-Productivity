from pathlib import Path
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_outputs_dir(path: Path = Path("outputs")) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}. Please provide the dataset at this path.")

    # Read CSV and parse dates
    df = pd.read_csv(csv_path, parse_dates=["Created At", "Updated At"], dayfirst=False)

    # Basic sanity: strip column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Ensure numeric-ish columns exist, add defaults if missing
    for c in ["Size", "Stars", "Forks", "Issues", "Watchers"]:
        if c not in df.columns:
            warnings.warn(f"Column '{c}' missing from CSV; filling with 0")
            df[c] = 0

    # Coerce to numeric and fill NaNs
    df["Stars"] = pd.to_numeric(df["Stars"], errors="coerce").fillna(0).astype(float)
    df["Forks"] = pd.to_numeric(df["Forks"], errors="coerce").fillna(0).astype(float)
    df["Issues"] = pd.to_numeric(df["Issues"], errors="coerce").fillna(0).astype(float)

    # Ensure date columns are datetime (coerce bad values to NaT)
    for d in ["Created At", "Updated At"]:
        if d not in df.columns:
            raise ValueError(f"Required date column '{d}' not found in CSV")
        df[d] = pd.to_datetime(df[d], errors="coerce")

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # repo_age_days (non-negative)
    df["repo_age_days"] = (df["Updated At"] - df["Created At"]).dt.days
    df["repo_age_days"] = df["repo_age_days"].clip(lower=0).fillna(0).astype(float)

    # stars_per_fork and issues_per_star with safe denominators
    df["stars_per_fork"] = df["Stars"] / (df["Forks"] + 1)
    df["issues_per_star"] = df["Issues"] / (df["Stars"] + 1)

    # guard against inf/nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[["stars_per_fork", "issues_per_star"]] = df[["stars_per_fork", "issues_per_star"]].fillna(0)

    return df


def plot_stars_vs_forks(df: pd.DataFrame, outpath: Path):
    # Scatter plot of Stars vs Forks
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # use a small jitter and alpha for readability; avoid annotations/tooltips
    x = df["Forks"].astype(float) + 1  # +1 to avoid log(0)
    y = df["Stars"].astype(float) + 1

    ax = sns.scatterplot(x=x, y=y, edgecolor=None, alpha=0.5, s=20)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Forks (log scale, +1)")
    ax.set_ylabel("Stars (log scale, +1)")
    ax.set_title("Stars vs Forks (log-log scatter)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_stars_histogram(df: pd.DataFrame, outpath: Path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Histogram of Stars; log-scale x-axis recommended because distribution is heavily skewed
    stars = df["Stars"].astype(float) + 1
    ax = sns.histplot(stars, bins=100, kde=False, color="C0")
    ax.set_xscale("log")
    ax.set_xlabel("Stars (log scale, +1)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Stars (log-scaled)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def print_insights(df: pd.DataFrame):
    # Insight 1: correlation between Stars and Forks
    if "Stars" in df.columns and "Forks" in df.columns:
        corr = df["Stars"].corr(df["Forks"])
        print("Insight 1: Correlation between Stars and Forks:")
        print(f"  Pearson correlation (Stars vs Forks): {corr:.3f}")
        if abs(corr) > 0.7:
            print("  Interpretation: Strong correlation — repositories with more forks tend to have more stars.")
        elif abs(corr) > 0.3:
            print("  Interpretation: Moderate correlation between forks and stars.")
        else:
            print("  Interpretation: Weak correlation between forks and stars.")
    else:
        print("Insight 1: Stars or Forks column missing; cannot compute correlation.")

    # Insight 2: typical star distribution / skew
    stars = df["Stars"].astype(float)
    median = stars.median()
    mean = stars.mean()
    skew = stars.skew()
    p90 = stars.quantile(0.9)
    p99 = stars.quantile(0.99)

    print("Insight 2: Star distribution summary:")
    print(f"  Count: {len(stars):,}, Mean: {mean:.1f}, Median: {median:.1f}, Skewness: {skew:.3f}")
    print(f"  90th percentile: {p90:.0f}, 99th percentile: {p99:.0f}")
    if skew > 1:
        print("  Interpretation: Distribution is highly right-skewed — a few repositories have very large star counts compared to typical repos.")
    else:
        print("  Interpretation: Distribution is not strongly skewed.")


def main():
    csv_path = Path("data") / "repositories.csv"
    outdir = ensure_outputs_dir(Path("outputs"))

    try:
        df = load_and_prepare(csv_path)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    df = add_engineered_features(df)

    # Produce plots
    p1 = outdir / "plot1.png"
    p2 = outdir / "plot2.png"

    print("Saving plot 1 (Stars vs Forks) to", p1)
    plot_stars_vs_forks(df, p1)

    print("Saving plot 2 (Stars distribution) to", p2)
    plot_stars_histogram(df, p2)

    # Print insights
    print("\n--- Insights ---")
    print_insights(df)


if __name__ == "__main__":
    main()
