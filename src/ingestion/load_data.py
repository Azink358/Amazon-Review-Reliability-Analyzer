import pandas as pd
import json
import gzip
from pathlib import Path

class DataIngestion:
    """
    Static ingestion utilities for Amazon Review Reliability Analyzer.
    Supports JSONL and JSONL.GZ files with error handling.
    """

    @staticmethod
    def load_jsonl(file_path) -> pd.DataFrame:
        """
        Load a single JSONL or JSONL.GZ file into a DataFrame.
        Returns empty DataFrame if file is missing or unreadable.
        """
        file_path = Path(file_path)  # ensure Path object

        if not file_path.exists():
            return pd.DataFrame()

        records = []
        try:
            # Choose correct opener based on extension
            if file_path.suffix == ".gz":
                opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
            else:
                opener = lambda p: open(p, "r", encoding="utf-8")

            with opener(file_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue
        except Exception:
            return pd.DataFrame()

        return pd.DataFrame(records) if records else pd.DataFrame()


    @staticmethod
    def load_review_meta_pair(data_dir: str, review_file: str, meta_file: str) -> tuple:
        """
        Load review and meta files separately.
        Returns (review_df, meta_df).
        """
        data_dir = Path(data_dir)
        review_df = DataIngestion.load_jsonl(data_dir / review_file)
        meta_df = DataIngestion.load_jsonl(data_dir / meta_file)

        return review_df, meta_df