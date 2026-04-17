import numpy as np
import pandas as pd

def add_confidence_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a confidence score for each product/review group.
    Handles list-aggregated columns safely by collapsing them to scalars.
    """

    # --- Helpful votes ---
    if "helpful_votes" in df.columns:
        df["helpful_votes"] = df["helpful_votes"].apply(
            lambda x: sum(x) if isinstance(x, list) else (x if pd.notna(x) else 0)
        )
        df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0)

    # --- Verified purchases ---
    if "verified_purchases" in df.columns:
        df["verified_purchases"] = df["verified_purchases"].apply(
            lambda x: sum(x) if isinstance(x, list) else (int(x) if pd.notna(x) else 0)
        )
        df["verified_purchases"] = pd.to_numeric(df["verified_purchases"], errors="coerce").fillna(0)

    # --- Average rating ---
    if "avg_rating" not in df.columns and "user_rating" in df.columns:
        df["user_rating"] = pd.to_numeric(df["user_rating"], errors="coerce")
        avg_df = df.groupby("parent_asin")["user_rating"].mean().reset_index()
        avg_df.rename(columns={"user_rating": "avg_rating"}, inplace=True)
        df = df.merge(avg_df, on="parent_asin", how="left")

    # --- Base score ---
    base = df.get("avg_rating", 0).fillna(0)

    # --- Bonuses ---
    helpful_bonus = np.log1p(df.get("helpful_votes", 0))
    verified_bonus = np.log1p(df.get("verified_purchases", 0))

    # --- Penalties ---
    low_review_penalty = df.get("review_count", 0).apply(lambda x: 1.0 if x < 5 else 0)
    repeat_penalty = df.get("repeat_reviewer_ratio", 0) * 2.0
    burst_penalty = df.apply(
        lambda row: 0.5 if row.get("recent_review_ratio", 0) > 0.8 and row.get("review_velocity", 0) > 10 else 0,
        axis=1
    )
    outlier_penalty = df.get("rating_outlier", False).astype(float) * 0.5 + \
                      df.get("helpful_outlier", False).astype(float) * 0.5

    # --- Rewards ---
    diversity_reward = df.get("unique_users", 0).apply(lambda x: 0.5 if x > 10 else 0)
    verified_reward = df.get("verified_ratio", 0).apply(lambda x: 0.5 if x > 0.7 else 0)
    longevity_reward = df.get("review_age_days", 0).apply(lambda x: 0.5 if x > 365 else 0)

    # --- Final confidence score ---
    df["confidence_score"] = (
        base
        + 0.1 * helpful_bonus
        + 0.2 * verified_bonus
        + diversity_reward
        + verified_reward
        + longevity_reward
        - low_review_penalty
        - repeat_penalty
        - burst_penalty
        - outlier_penalty
    )

    # Clamp between 0–5
    df["confidence_score"] = df["confidence_score"].clip(lower=0, upper=5)

    return df

def add_reliability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a reliability score based on confidence and credibility features.
    Applies penalties and a floor for single-review products.
    """
    if "confidence_score" not in df.columns:
        df["reliability_score"] = 0
        return df

    reliability = df["confidence_score"].copy()

    # Penalize repeat reviewers
    if "repeat_reviewer_ratio" in df.columns:
        reliability *= (1 - df["repeat_reviewer_ratio"].clip(0, 1))

    # Penalize helpful vote outliers
    if "helpful_outlier" in df.columns:
        reliability *= (~df["helpful_outlier"]).astype(int)

    # Penalize rating outliers
    if "rating_outlier" in df.columns:
        reliability *= (~df["rating_outlier"]).astype(int)

    # Penalize very low unique user counts
    if "unique_users" in df.columns:
        reliability *= (df["unique_users"] >= 3).astype(int)

    # Apply a floor for single-review products
    if "review_count" in df.columns:
        single_mask = df["review_count"] == 1
        reliability[single_mask] = df.loc[single_mask, "confidence_score"] * 0.25

    df["reliability_score"] = reliability.clip(0, 5)
    return df