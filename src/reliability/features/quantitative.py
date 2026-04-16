import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def collapse_to_int(x):
    if isinstance(x, (list, np.ndarray)):
        return int(np.sum(x))
    if pd.isna(x):
        return 0
    try:
        return int(x)
    except Exception:
        return 0

def collapse_to_sum(x):
    if isinstance(x, (list, np.ndarray)):
        return float(np.sum(x))
    if pd.isna(x):
        return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0

def safe_skew(arr: np.ndarray) -> float:
    if arr.size <= 1 or np.allclose(arr, arr[0]):
        return 0.0
    return float(skew(arr, bias=False))

def safe_kurtosis(arr: np.ndarray) -> float:
    if arr.size <= 1 or np.allclose(arr, arr[0]):
        return 0.0
    return float(kurtosis(arr, bias=False))

def flag_outliers(series: pd.Series, method: str = "zscore", threshold: float = 3.0) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    if method == "zscore":
        z = (series - series.mean()) / series.std(ddof=0)
        return (abs(z) > threshold)
    elif method == "iqr":
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        return (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))
    else:
        return pd.Series(False, index=series.index)

class QuantFeatures:
    """
    Quantitative feature engineering for product-level review aggregates.
    Includes rating stats, outliers, temporal patterns, and user credibility.
    """

    @staticmethod
    def run(df: pd.DataFrame) -> pd.DataFrame:
        # --- Review count ---
        if "review_text" in df.columns:
            df["review_count"] = df["review_text"].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )

        # --- Verified purchase ratio ---
        if "verified_purchases" in df.columns:
            df["verified_purchases"] = df["verified_purchases"].apply(collapse_to_int)
            df["verified_ratio"] = df["verified_purchases"] / df["review_count"].replace(0, np.nan)

        # --- Helpful votes total ---
        if "helpful_votes" in df.columns:
            df["helpful_votes"] = df["helpful_votes"].apply(collapse_to_sum)

        # --- Rating statistics ---
        if "user_rating" in df.columns:
            df["user_rating"] = df["user_rating"].apply(
                lambda x: np.array(x, dtype=float) if isinstance(x, (list, np.ndarray)) else np.array([x], dtype=float)
            )

            df["avg_rating"] = df["user_rating"].apply(
                lambda arr: float(np.nanmean(arr)) if arr.size > 0 else np.nan
            )
            df["rating_std"] = df["user_rating"].apply(
                lambda arr: float(np.nanstd(arr)) if arr.size > 1 else 0.0
            )
            df["rating_skew"] = df["user_rating"].apply(safe_skew)
            df["rating_kurtosis"] = df["user_rating"].apply(safe_kurtosis)

        # --- Drop intermediate arrays ---
        if "user_rating" in df.columns:
            df = df.drop(columns=["user_rating"], errors="ignore")

        # --- Outlier flags ---
        if "avg_rating" in df.columns:
            df["rating_outlier"] = flag_outliers(df["avg_rating"], method="zscore", threshold=3.0)
        if "helpful_votes" in df.columns:
            df["helpful_outlier"] = flag_outliers(df["helpful_votes"], method="iqr")

        # --- Temporal features ---
        if "date_reviewed" in df.columns:
            def parse_dates(dates):
                if isinstance(dates, (list, np.ndarray)) and len(dates) > 0:
                    return pd.to_datetime(dates, errors="coerce")
                elif pd.notna(dates):
                    return pd.to_datetime([dates], errors="coerce")
                else:
                    return pd.to_datetime([], errors="coerce")

            df["parsed_dates"] = df["date_reviewed"].apply(parse_dates)

            df["review_age_days"] = df["parsed_dates"].apply(
                lambda dates: (pd.Timestamp.now() - dates.min()).days if len(dates) > 0 else np.nan
            )
            df["recent_review_ratio"] = df["parsed_dates"].apply(
                lambda dates: (dates >= pd.Timestamp.now() - pd.Timedelta(days=90)).sum() / len(dates)
                if len(dates) > 0 else 0
            )
            df["review_velocity"] = df.apply(
                lambda row: row["review_count"] / (row["review_age_days"]/30)
                if pd.notna(row["review_age_days"]) and row["review_age_days"] > 0 else np.nan,
                axis=1
            )

            df = df.drop(columns=["parsed_dates"], errors="ignore")

        # --- User credibility features ---
        if "user_id" in df.columns:
            df["unique_users"] = df["user_id"].apply(
                lambda ids: len(set(ids)) if isinstance(ids, (list, np.ndarray)) else 0
            )
            df["repeat_reviewer_ratio"] = df["user_id"].apply(
                lambda ids: sum(pd.Series(ids).value_counts() > 1) / len(set(ids))
                if isinstance(ids, (list, np.ndarray)) and len(ids) > 0 else 0
            )
            df["user_products_reviewed"] = df["user_id"].apply(
                lambda ids: len(set(ids)) if isinstance(ids, (list, np.ndarray)) else 0
            )

        return df
