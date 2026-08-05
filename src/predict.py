#!/usr/bin/env python3
"""Inferencia en tiempo real + Kelly Criterion."""
from __future__ import annotations
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Dict, List
from data_fetch import fetch_schedule
from features import build_features, FEATURE_COLS  # type: ignore
import os

def load_models(target: str) -> Dict:
    models = {}
    models["xgb"] = joblib.load(f"models/{target}_xgb.joblib")
    models["cat"] = joblib.load(f"models/{target}_cat.joblib")
    models["nn"] = (
        tf.keras.models.load_model(f"models/{target}_nn.keras"),
        joblib.load(f"models/{target}_nn_scaler.joblib"),
    )
    for name in ["lstm_momentum", "lstm_result", "lstm_model"]:
        path = f"models/{target}_{name}.keras"
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path)
    return models

def kelly(prob: float, odds_decimal: float, fraction: float = 0.25) -> float:
    """Kelly fraccional. odds_decimal = 1/implied."""
    if odds_decimal <= 1.0 or prob <= 0:
        return 0.0
    b = odds_decimal - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(0.0, f * fraction)

def predict_today(bankroll: float = 1000.0) -> pd.DataFrame:
    today = datetime.utcnow().date()
    start = (today - timedelta(days=45)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    # Histórico reciente para features
    hist = fetch_schedule(start, end)
    if hist.empty:
        raise ValueError("No historical data")
    feat = build_features(hist)

    # Juegos de hoy (pueden no estar Final)
    url_params_today = fetch_schedule(today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
    # Si no hay finalizados, usamos schedule live
    from data_fetch import _get, BASE
    data = _get(f"{BASE}/schedule", {
        "sportId": 1, "date": today.strftime("%Y-%m-%d"),
        "hydrate": "probablePitcher,team", "gameType": "R"
    })
    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            pp_h = home.get("probablePitcher", {}) or {}
            pp_a = away.get("probablePitcher", {}) or {}
            rows.append({
                "game_pk": g["gamePk"],
                "game_date": g["officialDate"],
                "home_id": home["team"]["id"],
                "away_id": away["team"]["id"],
                "home_name": home["team"]["name"],
                "away_name": away["team"]["name"],
                "home_runs": 0, "away_runs": 0, "total_runs": 0, "home_win": 0,
                "home_pitcher_id": pp_h.get("id"),
                "home_pitcher_name": pp_h.get("fullName", "TBD"),
                "away_pitcher_id": pp_a.get("id"),
                "away_pitcher_name": pp_a.get("fullName", "TBD"),
            })
    today_df = pd.DataFrame(rows)
    if today_df.empty:
        print("No games today")
        return pd.DataFrame()

    # Features usando el histórico + hoy (rolling se calcula solo con pasado)
    full = pd.concat([hist, today_df], ignore_index=True)
    full = build_features(full)
    today_feat = full[full["game_pk"].isin(today_df["game_pk"])].copy()

    # Predicciones
    models_ml = load_models("home_win")
    models_tot = load_models("total_runs")

    X = today_feat[FEATURE_COLS].astype(float).values
    proba_home = []
    total_pred = []

    # XGB + Cat
    proba_home.append(models_ml["xgb"].predict_proba(X)[:, 1])
    proba_home.append(models_ml["cat"].predict_proba(X)[:, 1])
    total_pred.append(models_tot["xgb"].predict(X))
    total_pred.append(models_tot["cat"].predict(X))

    # NN
    nn, sc = models_ml["nn"]
    proba_home.append(nn.predict(sc.transform(X), verbose=0).ravel())
    nn_t, sc_t = models_tot["nn"]
    total_pred.append(nn_t.predict(sc_t.transform(X), verbose=0).ravel())

    # Ensemble
    p_home = np.nanmean(proba_home, axis=0)
    p_total = np.nanmean(total_pred, axis=0)

    today_feat["prob_home_win"] = p_home
    today_feat["pred_total_runs"] = p_total
    today_feat["prob_away_win"] = 1 - p_home

    # Kelly ejemplo (cuotas dummy 1.90 / Over 1.91)
    today_feat["kelly_home"] = today_feat["prob_home_win"].apply(lambda p: kelly(p, 1.90) * bankroll)
    today_feat["kelly_over"] = today_feat.apply(
        lambda r: kelly(0.55 if r["pred_total_runs"] > 8.5 else 0.45, 1.91) * bankroll, axis=1
    )

    cols = ["game_date", "home_name", "away_name", "home_pitcher_name", "away_pitcher_name",
            "prob_home_win", "pred_total_runs", "kelly_home", "kelly_over"]
    out = today_feat[cols].round(4)
    out.to_csv("predictions.csv", index=False)
    print(out.to_string(index=False))
    return out

if __name__ == "__main__":
    predict_today()
