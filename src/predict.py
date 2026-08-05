#!/usr/bin/env python3
"""Inferencia live + Kelly. Usa StatsAPI para el schedule de hoy."""
from __future__ import annotations
import os
from typing import Any, Dict
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from data_fetch import fetch_schedule, fetch_today_schedule, build_historical
from features import build_features, FEATURE_COLS

def kelly(prob: float, odds: float = 1.91, fraction: float = 0.25) -> float:
    if odds <= 1.0 or prob <= 0:
        return 0.0
    b = odds - 1.0
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f * fraction)

def load_models(target: str) -> Dict[str, Any]:
    models = {
        "xgb": joblib.load(f"models/{target}_xgb.joblib"),
        "cat": joblib.load(f"models/{target}_cat.joblib"),
        "nn": (
            tf.keras.models.load_model(f"models/{target}_nn.keras"),
            joblib.load(f"models/{target}_nn_scaler.joblib"),
        ),
    }
    for name in ["lstm_momentum", "lstm_result", "lstm_model"]:
        path = f"models/{target}_{name}.keras"
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path)
    return models

def predict_today(bankroll: float = 1000.0) -> pd.DataFrame:
    # Histórico reciente para rolling
    hist = fetch_schedule(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
        pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    )
    today_df = fetch_today_schedule()
    if today_df.empty:
        print("No games scheduled today")
        return pd.DataFrame()

    full = pd.concat([hist, today_df], ignore_index=True)
    full = build_features(full)
    today_feat = full[full["game_pk"].isin(today_df["game_pk"])].copy()
    if today_feat.empty:
        print("Insufficient history for today’s games")
        return pd.DataFrame()

    X = today_feat[FEATURE_COLS].astype(np.float32).values
    models_ml = load_models("home_win")
    models_tot = load_models("total_runs")

    p_home = []
    p_tot = []

    p_home.append(models_ml["xgb"].predict_proba(X)[:, 1])
    p_home.append(models_ml["cat"].predict_proba(X)[:, 1])
    p_tot.append(models_tot["xgb"].predict(X))
    p_tot.append(models_tot["cat"].predict(X))

    nn, sc = models_ml["nn"]
    p_home.append(nn.predict(sc.transform(X), verbose=0).ravel())
    nn_t, sc_t = models_tot["nn"]
    p_tot.append(nn_t.predict(sc_t.transform(X), verbose=0).ravel())

    today_feat["prob_home_win"] = np.nanmean(p_home, axis=0)
    today_feat["pred_total"] = np.nanmean(p_tot, axis=0)
    today_feat["kelly_home"] = today_feat["prob_home_win"].apply(lambda p: kelly(p) * bankroll)
    today_feat["kelly_over"] = today_feat["pred_total"].apply(
        lambda t: kelly(0.55 if t > 8.5 else 0.45) * bankroll
    )

    cols = ["game_date", "home_name", "away_name",
            "home_pitcher_name", "away_pitcher_name",
            "prob_home_win", "pred_total", "kelly_home", "kelly_over"]
    out = today_feat[cols].round(4)
    out.to_csv("predictions.csv", index=False)
    print(out.to_string(index=False))
    return out

if __name__ == "__main__":
    predict_today()
