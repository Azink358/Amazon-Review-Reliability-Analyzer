import pandas as pd

class ReviewCleaning:
    """
    Cleaning utilities for review dataset.
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
    def convert_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        return df

    @staticmethod
    def normalize_rating_column(df: pd.DataFrame, column: str = "rating") -> pd.DataFrame:
        """
        Normalize the rating column:
        - If values are lists, take the first element.
        - If values are strings, extract numeric part.
        - Coerce everything to numeric (float).
        """
        if column not in df.columns:
            return df

        def extract_rating(x):
            if isinstance(x, list):
                return x[0] if x else None
            if isinstance(x, (int, float)):
                return x
            if isinstance(x, str):
                # Extract digits and decimals from string
                import re
                match = re.search(r"(\d+\.?\d*)", x)
                return float(match.group(1)) if match else None
            return None

        df[column] = df[column].apply(extract_rating)
        df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    @staticmethod
    def refine_review_text(df: pd.DataFrame, column: str = "text") -> pd.DataFrame:
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
        if "rating" in df.columns and "parent_asin" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            avg_df = df.groupby("parent_asin")["rating"].mean().reset_index()
            avg_df.rename(columns={"rating": "avg_rating"}, inplace=True)
            df = df.merge(avg_df, on="parent_asin", how="left")
        return df

    @staticmethod
    def aggregate_reviews(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate reviews by parent_asin with renamed columns.
        """
        df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce")
        agg_df = df.groupby("parent_asin").agg(
            user_id=("user_id", list),
            user_rating=("rating", list),
            review_text=("text", list),
            images_link=("images", list),
            date_reviewed=("timestamp", list),
            helpful_votes=("helpful_vote", list),
            verified_purchases=("verified_purchase", list),
        ).reset_index()
        return agg_df


class MetaCleaning:
    """
    Cleaning utilities for meta dataset.
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
    def normalize_images(df: pd.DataFrame, column: str = "images") -> pd.DataFrame:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: x if isinstance(x, list) else [])
        return df

    @staticmethod
    def normalize_specs(df: pd.DataFrame, column: str = "details") -> pd.DataFrame:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: x if isinstance(x, dict) else {})
        return df

    @staticmethod
    def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
        if "parent_asin" not in df.columns:
            df["parent_asin"] = None
        return df

    @staticmethod
    def refine_meta_text(df: pd.DataFrame, column: str = "description") -> pd.DataFrame:
        """
        Strip whitespace and ensure description is string.
        """
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
        return df

    @staticmethod
    def compute_meta_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if "images" in df.columns:
            df["image_count"] = df["images"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        if "rating_number" in df.columns:
            df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce")
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df

    @staticmethod
    def aggregate_meta(df: pd.DataFrame) -> pd.DataFrame:
        agg_df = df.groupby("parent_asin").agg(
            product_title=("title", list),
            product_features=("features", list),
            product_images=("images", list),
            product_store=("store", list),
            product_price=("price", list),
            total_ratings=("rating_number", "sum"),
        ).reset_index()
        return agg_df


class DataIntegrationUtils:
    """
    Utilities to prepare review and meta datasets for merging.
    """

    @staticmethod
    def prepare_for_merge(df_reviews: pd.DataFrame, df_meta: pd.DataFrame) -> tuple:
        """
        Apply cleaning and aggregation to both datasets,
        returning trimmed DataFrames ready for merge.
        """
        # Apply review cleaning steps before aggregation
        df_reviews = ReviewCleaning.refine_review_text(df_reviews)
        df_reviews = ReviewCleaning.compute_review_count(df_reviews)
        df_reviews = ReviewCleaning.compute_avg_rating(df_reviews)
        reviews_clean = ReviewCleaning.aggregate_reviews(df_reviews)

        # Apply meta cleaning steps before aggregation
        df_meta = MetaCleaning.compute_meta_metrics(df_meta)
        meta_clean = MetaCleaning.aggregate_meta(df_meta)

        return reviews_clean, meta_clean
