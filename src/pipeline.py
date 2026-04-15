import pandas as pd
from ingestion.load_data import DataIngestion
from cleaning.clean_data import ReviewCleaning, MetaCleaning, DataIntegrationUtils

class Pipeline:
    """
    End-to-end pipeline for cleaning, aggregating, and merging review + meta datasets.
    """

    @staticmethod
    def run_pipeline(df_reviews: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full cleaning pipeline and return merged dataset.
        """

        # --- Review cleaning ---
        df_reviews = ReviewCleaning.fill_na(df_reviews, fill_map={"images": [], "text": ""})
        df_reviews = ReviewCleaning.drop_duplicates(df_reviews, subset=["user_id", "text"])
        df_reviews = ReviewCleaning.convert_timestamp(df_reviews, column="timestamp")
        df_reviews = ReviewCleaning.refine_review_text(df_reviews)
        df_reviews = ReviewCleaning.compute_review_count(df_reviews)
        df_reviews = ReviewCleaning.compute_avg_rating(df_reviews)

        # --- Meta cleaning ---
        df_meta = MetaCleaning.drop_duplicates(df_meta, subset=["parent_asin"])
        df_meta = MetaCleaning.fill_na(df_meta, fill_map={"images": [], "features": []})
        df_meta = MetaCleaning.normalize_images(df_meta, column="images")
        df_meta = MetaCleaning.normalize_specs(df_meta, column="details")
        df_meta = MetaCleaning.ensure_schema(df_meta)
        df_meta = MetaCleaning.refine_meta_text(df_meta)
        df_meta = MetaCleaning.compute_meta_metrics(df_meta)

        # --- Aggregation + merge ---
        reviews_clean, meta_clean = DataIntegrationUtils.prepare_for_merge(df_reviews, df_meta)
        df_merged = reviews_clean.merge(meta_clean, on="parent_asin", how="inner")

        return df_merged


if __name__ == "__main__":
    # Example usage when running pipeline.py directly
    data_dir = "data/raw"
    review_file = "All_Beauty.jsonl.gz"
    meta_file = "meta_All_Beauty.jsonl.gz"

    # Load raw data
    df_reviews_raw, df_meta_raw = DataIngestion.load_review_meta_pair(data_dir, review_file, meta_file)

    # Run pipeline
    pipeline = Pipeline()
    df_final = pipeline.run_pipeline(df_reviews_raw, df_meta_raw)

    print("Final dataset preview:")
    print(df_final.head())

    # --- Save to processed folder as Parquet ---
    output_path = "../data/processed/all_beauty.parquet"
    df_final.to_parquet(output_path, index=False)

    print(f"✅ Saved cleaned dataset to {output_path}")

