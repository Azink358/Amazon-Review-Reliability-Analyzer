import pandas as pd
import time

from ingestion.load_data import DataIngestion
from cleaning.clean_data import ReviewCleaning, MetaCleaning, DataIntegrationUtils
from reliability.features.quantitative import QuantFeatures
from reliability.scoring import add_confidence_score


class CleaningPipeline:
    """
    Stage 1: Ingest raw review + meta files, clean, and save processed parquet.
    """

    @staticmethod
    def run(df_reviews: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
        start = time.time()

        # --- Review cleaning ---
        df_reviews = ReviewCleaning.fill_na(df_reviews, fill_map={"images_link": [], "review_text": ""})
        df_reviews = ReviewCleaning.drop_duplicates(df_reviews, subset=["user_id", "review_text"])
        df_reviews = ReviewCleaning.convert_timestamp(df_reviews, column="date_reviewed")

        # --- Meta cleaning ---
        df_meta = MetaCleaning.drop_duplicates(df_meta, subset=["parent_asin"])
        # Apply product title cleaning
        df_meta= MetaCleaning.clean_product_titles(df_meta)
        df_meta = MetaCleaning.fill_na(df_meta, fill_map={"product_images": [], "product_features": []})
        df_meta = MetaCleaning.ensure_schema(df_meta)

        # --- Aggregation + merge ---
        reviews_clean, meta_clean = DataIntegrationUtils.prepare_for_merge(df_reviews, df_meta)
        df_merged = reviews_clean.merge(meta_clean, on="parent_asin", how="inner")

        print(f"⏱ CleaningPipeline finished in {time.time() - start:.2f} seconds")
        return df_merged


class ReliabilityPipeline:
    """
    Stage 2: Load processed parquet, compute quantitative features + confidence scores.
    """

    @staticmethod
    def run(df: pd.DataFrame) -> pd.DataFrame:
        start = time.time()

        # Quantitative features
        df = QuantFeatures.run(df)
        print(f"⏱ QuantFeatures finished in {time.time() - start:.2f} seconds")

        # Confidence scoring
        scoring_start = time.time()
        df = add_confidence_score(df)
        print(f"⏱ Confidence scoring finished in {time.time() - scoring_start:.2f} seconds")

        # Drop intermediate arrays if any slipped through
        drop_cols = [c for c in df.columns if c.endswith("_array")]
        df = df.drop(columns=drop_cols, errors="ignore")

        print(f"⏱ ReliabilityPipeline finished in {time.time() - start:.2f} seconds")
        return df


if __name__ == "__main__":
    # --- Stage 1: Raw ingestion + cleaning ---
    data_dir = "data/raw"
    review_file = "All_Beauty.jsonl.gz"
    meta_file = "meta_All_Beauty.jsonl.gz"

    load_start = time.time()
    df_reviews_raw, df_meta_raw = DataIngestion.load_review_meta_pair(data_dir, review_file, meta_file)
    print(f"⏱ Data loading finished in {time.time() - load_start:.2f} seconds")

    df_processed = CleaningPipeline.run(df_reviews_raw, df_meta_raw)

    processed_path = "data/processed/all_beauty_cleaned.parquet"
    df_processed.to_parquet(processed_path, index=False)
    print(f"✅ Saved processed parquet to {processed_path}")

    # --- Stage 2: Feature engineering + scoring ---
    df_loaded = pd.read_parquet(processed_path)
    df_final = ReliabilityPipeline.run(df_loaded)

    scored_path = "data/processed/all_beauty_scored.parquet"
    df_final.to_parquet(scored_path, index=False)
    print(f"✅ Saved scored parquet to {scored_path}")
