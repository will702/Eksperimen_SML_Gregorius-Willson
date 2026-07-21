"""Automated preprocessing for Telco Customer Churn (Gregorius Willson).

Steps mirror Eksperimen_Gregorius-Willson.ipynb:
load -> clean -> encode -> save ready-to-train dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "telco_raw" / "Telco-Customer-Churn.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "telco_preprocessing"
OUTPUT_PATH = OUTPUT_DIR / "telco_preprocessing.csv"


def load_raw_data(path: Path | str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Return a train-ready dataframe (same steps as the experiment notebook)."""
    data = df.copy()

    # Drop identifier
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    # TotalCharges may contain blanks
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    # Binary target
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    # Encode remaining object columns
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))

    return data


def run_preprocessing(
    raw_path: Path | str = RAW_PATH,
    output_path: Path | str = OUTPUT_PATH,
) -> pd.DataFrame:
    raw = load_raw_data(raw_path)
    processed = preprocess_telco(raw)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output, index=False)
    print(f"Saved preprocessed data to {output} shape={processed.shape}")
    return processed


if __name__ == "__main__":
    run_preprocessing()
