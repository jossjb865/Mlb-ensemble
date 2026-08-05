#!/usr/bin/env python3
"""Feature engineering cronológico + Gematria. Sin data leakage."""
from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np

FEATURE_COLS: List[str] = [
    "home_rf_5", "home_ra_5", "home_win_5",
    "away_rf_5", "away_ra_5", "away_win_5",
    "home_rf_10", "home_ra_10", "home_win_10",
    "away_rf_10", "away_ra_10", "away_win_10",
    "home_rf_20", "home_ra_20", "home_win_20",
    "away_rf_20", "away_ra_20", "away_win_20",
    "rf_diff_5", "ra_diff_5", "win_diff_5",
    "rf_diff_10", "ra_diff_10", "win_diff_10",
    "rf_diff_20", "ra_diff_20", "win_diff_20",
    "home_gem", "away_gem", "gem_diff",
    "pitcher_home_gem", "pitcher_away_gem", "pitcher_gem_diff",
]

def simple_gematria(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return sum(max(0, ord(c.upper()) - 64) for c in text if c.isalpha())

def add_gematria(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_gem"] = df["home_name"].map(simple_gematria)
    df["away_gem"] = df["away_name"].map(simple_gematria)
    df["pitcher_home_gem"] = df["home_pitcher_name"].map(simple_gematria)
    df["pitcher_away_gem"] = df["away_pitcher_name"].map(simple_gematria)
    df["gem_diff"] = df["home_gem"] - df["away_gem"]
    df["pitcher_gem_diff"] = df["pitcher_home_gem"] - df["pitcher_away_gem"]
    return df

def rolling_team_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    df = df.sort_values("game_date").reset_index(drop=True)

    # Long format (home + away)
    home = df[["game_pk", "game_date", "home_id", "home_runs", "away_runs", "home_win"]].copy()
    home.columns = ["game_pk", "game_date", "team_id", "runs_for", "runs_against", "win"]

    away = df[["game_pk", "game_date", "away_id", "away_runs", "home_runs", "home_win"]].copy()
    away["win"] = 1 - away["home_win"]
    away = away.drop(columns=["home_win"])
    away.columns = ["game_pk", "game_date", "team_id", "runs_for", "runs_against", "win"]

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    for w in windows:
        long[f"rf_{w}"] = (
            long.groupby("team_id")["runs_for"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
        )
        long[f"ra_{w}"] = (
            long.groupby("team_id")["runs_against"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
        )
        long[f"win_{w}"] = (
            long.groupby("team_id")["win"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
        )

    # Merge de vuelta a perspectiva home/away
    feat_rows = []
    for _, row in df.iterrows():
        h = long[(long["game_pk"] == row["game_pk"]) & (long["team_id"] == row["home_id"])]
        a = long[(long["game_pk"] == row["game_pk"]) & (long["team_id"] == row["away_id"])]
        rec = {"game_pk": row["game_pk"]}
        if not h.empty:
            for w in windows:
                rec[f"home_rf_{w}"] = h.iloc[0][f"rf_{w}"]
                rec[f"home_ra_{w}"] = h.iloc[0][f"ra_{w}"]
                rec[f"home_win_{w}"] = h.iloc[0][f"win_{w}"]
        if not a.empty:
            for w in windows:
                rec[f"away_rf_{w}"] = a.iloc[0][f"rf_{w}"]
                rec[f"away_ra_{w}"] = a.iloc[0][f"ra_{w}"]
                rec[f"away_win_{w}"] = a.iloc[0][f"win_{w}"]
        feat_rows.append(rec)

    feat_df = pd.DataFrame(feat_rows)
    out = df.merge(feat_df, on="game_pk", how="left")

    for w in windows:
        out[f"rf_diff_{w}"] = out[f"home_rf_{w}"] - out[f"away_rf_{w}"]
        out[f"ra_diff_{w}"] = out[f"home_ra_{w}"] - out[f"away_ra_{w}"]
        out[f"win_diff_{w}"] = out[f"home_win_{w}"] - out[f"away_win_{w}"]

    return out

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_gematria(raw)
    df = rolling_team_features(df)
    # Solo filas con history suficiente
    df = df.dropna(subset=["home_rf_10", "away_rf_10"]).reset_index(drop=True)
    return df

if __name__ == "__main__":
    raw = pd.read_parquet("data/raw_games.parquet")
    feat = build_features(raw)
    feat.to_parquet("data/features.parquet", index=False)
    print(feat.shape)
    print(feat[FEATURE_COLS].describe().T[["mean", "std"]].head(12))
