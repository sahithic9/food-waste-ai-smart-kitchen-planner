"""Feature engineering shared by training and the app."""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "food_prepared_kg" in df:
        df["prep_per_customer"] = df["food_prepared_kg"] / df["expected_customers"].clip(lower=1)
    return df