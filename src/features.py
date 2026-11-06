"""Data cleaning + feature engineering shared by training and the dashboard."""
import pandas as pd

ADDON_COLS = ["online_security", "tech_support", "streaming_tv", "online_backup"]

CATEGORICAL = [
    "gender", "partner", "dependents", "contract", "internet_service",
    "payment_method", "paperless_billing", *ADDON_COLS, "tenure_group",
]
NUMERIC = [
    "senior_citizen", "tenure", "monthly_charges", "total_charges",
    "avg_monthly_spend", "num_services",
]
FEATURES = CATEGORICAL + NUMERIC


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, fix types, handle missing values, encode target."""
    df = df.copy()
    df = df.drop_duplicates(subset="customer_id")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    # blank total_charges belong to brand-new customers (tenure 0)
    df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"] * df["tenure"])
    if "churn" in df.columns:
        df = df.dropna(subset=["churn"])
        df["churn"] = (df["churn"] == "Yes").astype(int)
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering."""
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12m", "13-24m", "25-48m", "49-72m"]
    ).astype(str)
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure"].clip(lower=1)
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in ADDON_COLS)
    return df
