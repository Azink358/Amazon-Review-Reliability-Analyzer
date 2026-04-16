import pandas as pd

class ReviewCleaning:
    """
    Cleaning utilities for review dataset (normalized schema).
    """

    @staticmethod
    def fill_na(df: pd.DataFrame, fill_map: dict = None) -> pd.DataFrame:
        if not fill_map:
            return df.fillna("")
        for col, val in fill_map.items():
            if col in df.columns:
                if isinstance(val, (str, int, float)):
                    df[col] = df[col].fillna(val)
                else:
                    df[col] = df[col].apply(
                        lambda x: val if (x is None or (isinstance(x, float) and pd.isna(x))) else x
                    )
        return df

    @staticmethod
    def drop_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
        return df.drop_duplicates(subset=subset)

    @staticmethod
    def convert_timestamp(df: pd.DataFrame, column: str = "date_reviewed") -> pd.DataFrame:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        return df

    @staticmethod
    def refine_review_text(df: pd.DataFrame, column: str = "review_text") -> pd.DataFrame:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
        return df

    @staticmethod
    def compute_review_count(df: pd.DataFrame) -> pd.DataFrame:
        if "parent_asin" in df.columns:
            df["review_count"] = df.groupby("parent_asin")["parent_asin"].transform("count")
        return df

    @staticmethod
    def compute_avg_rating(df: pd.DataFrame) -> pd.DataFrame:
        if "user_rating" in df.columns and "parent_asin" in df.columns:
            df["user_rating"] = pd.to_numeric(df["user_rating"], errors="coerce")
            avg_df = df.groupby("parent_asin")["user_rating"].mean().reset_index()
            avg_df.rename(columns={"user_rating": "avg_rating"}, inplace=True)
            df = df.merge(avg_df, on="parent_asin", how="left")
        return df

    @staticmethod
    def aggregate_reviews(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate reviews by parent_asin with normalized column names.
        """
        df["user_rating"] = pd.to_numeric(df.get("user_rating"), errors="coerce")
        agg_df = df.groupby("parent_asin").agg(
            user_id=("user_id", list),
            user_rating=("user_rating", list),
            review_text=("review_text", list),
            images_link=("images_link", list),
            date_reviewed=("date_reviewed", list),
            helpful_votes=("helpful_votes", list),
            verified_purchases=("verified_purchases", list),
        ).reset_index()
        return agg_df


class MetaCleaning:
    """
    Cleaning utilities for meta dataset (normalized schema).
    """

    @staticmethod
    def drop_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
        return df.drop_duplicates(subset=subset)

    @staticmethod
    def fill_na(df: pd.DataFrame, fill_map: dict = None) -> pd.DataFrame:
        if not fill_map:
            return df.fillna("")
        for col, val in fill_map.items():
            if col in df.columns:
                if isinstance(val, (str, int, float)):
                    df[col] = df[col].fillna(val)
                else:
                    df[col] = df[col].apply(
                        lambda x: val if (x is None or (isinstance(x, float) and pd.isna(x))) else x
                    )
        return df

    @staticmethod
    def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
        if "parent_asin" not in df.columns:
            df["parent_asin"] = None
        return df

    @staticmethod
    def refine_meta_text(df: pd.DataFrame, column: str = "product_title") -> pd.DataFrame:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
        return df

    @staticmethod
    def compute_meta_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if "product_images" in df.columns:
            df["image_count"] = df["product_images"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        if "total_ratings" in df.columns:
            df["total_ratings"] = pd.to_numeric(df["total_ratings"], errors="coerce")
        if "product_price" in df.columns:
            df["product_price"] = pd.to_numeric(df["product_price"], errors="coerce")
        return df

    @staticmethod
    def aggregate_meta(df: pd.DataFrame) -> pd.DataFrame:
        agg_df = df.groupby("parent_asin").agg(
            product_title=("product_title", "first"),
            product_features=("product_features", list),
            product_images=("product_images", list),
            product_store=("product_store", "first"),
            product_price=("product_price", "first"),
            total_ratings=("total_ratings", "sum"),
        ).reset_index()
        return agg_df


class DataIntegrationUtils:
    """
    Utilities to prepare review and meta datasets for merging.
    """

    @staticmethod
    def prepare_for_merge(df_reviews: pd.DataFrame, df_meta: pd.DataFrame) -> tuple:
        df_reviews = ReviewCleaning.refine_review_text(df_reviews)
        df_reviews = ReviewCleaning.compute_review_count(df_reviews)
        df_reviews = ReviewCleaning.compute_avg_rating(df_reviews)
        reviews_clean = ReviewCleaning.aggregate_reviews(df_reviews)

        df_meta = MetaCleaning.compute_meta_metrics(df_meta)
        meta_clean = MetaCleaning.aggregate_meta(df_meta)

        return reviews_clean, meta_clean
