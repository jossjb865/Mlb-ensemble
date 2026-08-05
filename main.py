#!/usr/bin/env python3
"""Orquestador completo StatsAPI → Features → Ensemble → Predicción."""
import os
from data_fetch import build_historical
from features import build_features
from models import train_ensemble
from predict import predict_today

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("1/4 Fetching StatsAPI historical...")
    raw = build_historical(2023, 2025)
    raw.to_parquet("data/raw_games.parquet", index=False)

    print("2/4 Feature engineering...")
    feat = build_features(raw)
    feat.to_parquet("data/features.parquet", index=False)

    print("3/4 Training ensemble...")
    metrics = train_ensemble(feat)
    print("Metrics:", metrics)

    print("4/4 Live prediction...")
    predict_today(bankroll=1000.0)
    print("Done → predictions.csv + models/")

if __name__ == "__main__":
    main()
