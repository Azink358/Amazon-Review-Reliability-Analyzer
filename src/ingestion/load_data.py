import pandas as pd


class DataIngestion:
    """
    Utilities for loading raw Amazon review + meta datasets.
    """

    @staticmethod
    def load_review_meta_pair(data_dir: str, review_file: str, meta_file: str):
        """
        Load raw review and meta JSONL.GZ files, normalize schema for downstream cleaning.
        """
        # Load reviews
        df_reviews = pd.read_json(f"{data_dir}/{review_file}", lines=True, compression="gzip")

        # Normalize schema
        df_reviews = df_reviews.rename(
            columns={
                "rating": "user_rating",
                "text": "review_text",
                "images": "images_link",
                "timestamp": "date_reviewed",
                "verified_purchase": "verified_purchases",
                "helpful_vote": "helpful_votes"
            }
        )

        # Load meta
        df_meta = pd.read_json(f"{data_dir}/{meta_file}", lines=True, compression="gzip")

        # Normalize meta schema
        df_meta = df_meta.rename(
            columns={
                "title": "product_title",
                "features": "product_features",
                "images": "product_images",
                "store": "product_store",
                "price": "product_price",
                "rating_number": "total_ratings"
            }
        )

        return df_reviews, df_meta
