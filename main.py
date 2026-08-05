#!/usr/bin/env python3
"""Pipeline completo: fetch → features → train → predict."""
import os
import pandas as pd
from src.data_fetch import build_historical
from src.features import build_features
from src.models import train_ensemble
from src.predict import predict_today

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("1. Fetching historical data...")
    raw = build_historical(2023, 2025)
    raw.to_parquet("data/raw_games.parquet", index=False)

    print("2. Feature engineering...")
    feat = build_features(raw)
    feat.to_parquet("data/features.parquet", index=False)

    print("3. Training ensemble...")
    metrics = train_ensemble(feat)
    print("Metrics:", metrics)

    print("4. Predicting today...")
    preds = predict_today(bankroll=1000.0)
    print("Done. predictions.csv generated.")

if __name__ == "__main__":
    main()
