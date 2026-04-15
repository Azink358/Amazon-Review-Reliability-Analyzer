import re
from bs4 import BeautifulSoup
import pandas as pd

class TextUtils:
    """
    Shared text refinement utilities for reviews and meta datasets.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize text: lowercase, strip whitespace, remove HTML tags."""
        if not isinstance(text, str):
            return ""
        text = BeautifulSoup(text, "html.parser").get_text()
        return text.lower().strip()

    @staticmethod
    def remove_special_chars(text: str) -> str:
        """Remove non-alphanumeric characters except spaces."""
        if not isinstance(text, str):
            return ""
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    @staticmethod
    def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply text cleaning to a specific column in a DataFrame."""
        if column in df.columns:
            df[column] = df[column].apply(TextUtils.clean_text)
        return df

    @staticmethod
    def normalize_list_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply text cleaning to each element of a list stored in a Series cell.
        Example: review_text column with lists of strings.
        """
        if column in df.columns:
            df[column] = df[column].apply(
                lambda lst: [TextUtils.clean_text(x) for x in lst] if isinstance(lst, list) else []
            )
        return df



class DataIntegrationUtils:
    """
    Utility class for aligning list columns and merging
    cleaned review + meta datasets on parent_asin.
    """

    # -----------------------------
    # List Alignment
    # -----------------------------
    @staticmethod
    def align_list_column(
        df: pd.DataFrame,
        column: str,
        target_length: int,
        pad_value: str = ""
    ) -> pd.DataFrame:
        """
        Align lists in a Series column to a target length.
        """
        if column not in df.columns:
            return df

        def align_list(lst):
            if not isinstance(lst, list):
                return [pad_value] * target_length
            if len(lst) > target_length:
                return lst[:target_length]
            return lst + [pad_value] * (target_length - len(lst))

        df[column] = df[column].apply(align_list)
        return df

    @staticmethod
    def align_multiple_list_columns(
        df: pd.DataFrame,
        columns: list,
        target_length: int,
        pad_value: str = ""
    ) -> pd.DataFrame:
        """
        Align multiple list columns to the same target length.
        """
        for col in columns:
            df = DataIntegrationUtils.align_list_column(df, col, target_length, pad_value)
        return df

    # -----------------------------
    # Merge Reviews + Meta
    # -----------------------------
    @staticmethod
    def merge_reviews_meta(
        df_reviews: pd.DataFrame,
        df_meta: pd.DataFrame,
        how: str = "inner"
    ) -> pd.DataFrame:
        """
        Merge reviews and meta datasets on parent_asin.
        """
        if df_reviews.empty or df_meta.empty:
            return pd.DataFrame()

        if "parent_asin" not in df_reviews.columns or "parent_asin" not in df_meta.columns:
            raise KeyError("Both DataFrames must contain 'parent_asin' column.")

        merged = pd.merge(df_reviews, df_meta, on="parent_asin", how="left")
        return merged

    @staticmethod
    def merge_and_align(
        df_reviews: pd.DataFrame,
        df_meta: pd.DataFrame,
        list_columns: list,
        target_length: int,
        pad_value: str = "",
        how: str = "inner"
    ) -> pd.DataFrame:
        """
        Merge reviews + meta datasets and align specified list columns.
        """
        merged = DataIntegrationUtils.merge_reviews_meta(df_reviews, df_meta, how="left")
        merged = DataIntegrationUtils.align_multiple_list_columns(
            merged, list_columns, target_length, pad_value
        )
        return merged