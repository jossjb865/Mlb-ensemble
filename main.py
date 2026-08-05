#!/usr/bin/env python3
"""Orquestador StatsAPI → Features → Ensemble → Predicción."""
import os
import sys
import logging

# asegurar src como paquete (si trabajas con installs, preferir pip editable)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_fetch import build_historical
from features import build_features
from models import train_ensemble
from predict import predict_today

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    try:
        logger.info("1/4 Fetching StatsAPI historical...")
        raw = build_historical(2023, 2025)
        raw.to_parquet("data/raw_games.parquet", index=False)

        logger.info("2/4 Feature engineering...")
        feat = build_features(raw)
        feat.to_parquet("data/features.parquet", index=False)

        logger.info("3/4 Training ensemble...")
        metrics = train_ensemble(feat)
        logger.info("Metrics: %s", metrics)

        logger.info("4/4 Live prediction...")
        predict_today(bankroll=1000.0)
        logger.info("Done → predictions.csv + models/")
    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()
