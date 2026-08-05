#!/usr/bin/env python3
"""Feature engineering estricto + Gematria + rolling sin leakage."""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List

def simple_gematria(text: str) -> int:
    """English Ordinal: A=1 ... Z=26."""
    if not isinstance(text, str):
        return 0
    return sum(ord(c.upper()) - 64 for c in text if c.isalpha())

def add_gematria(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_gem"] = df["home_name"].apply(simple_gematria)
    df["away_gem"] = df["away_name"].apply(simple_gematria)
    df["pitcher_home_gem"] = df["home_pitcher_name"].apply(simple_gematria)
    df["pitcher_away_gem"] = df["away_pitcher_name"].apply(simple_gematria)
    df["gem_diff"] = df["home_gem"] - df["away_gem"]
    df["pitcher_gem_diff"] = df["pitcher_home_gem"] - df["pitcher_away_gem"]
    return df

def rolling_team_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Rolling de runs y win% por equipo (solo pasado)."""
    df = df.sort_values("game_date").reset_index(drop=True)
    # Expandimos a perspectiva home/away
    home = df[["game_pk", "game_date", "home_id", "home_runs", "away_runs", "home_win"]].copy()
    home.columns = ["game_pk", "game_date", "team_id", "runs_for", "runs_against", "win"]
    away = df[["game_pk", "game_date", "away_id", "away_runs", "home_runs", "home_win"]].copy()
    away["win"] = 1 - away["home_win"]
    away = away.drop(columns=["home_win"])
    away.columns = ["game_pk", "game_date", "team_id", "runs_for", "runs_against", "win"]
    long = pd.concat([home, away], ignore_index=True).sort_values(["team_id", "game_date"])

    for w in windows:
        long[f"rf_{w}"] = long.groupby("team_id")["runs_for"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=3).mean()
        )
        long[f"ra_{w}"] = long.groupby("team_id")["runs_against"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=3).mean()
        )
        long[f"win_{w}"] = long.groupby("team_id")["win"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=3).mean()
        )

    # Merge back
    home_feats = long[long["game_pk"].isin(df["game_pk"])].copy()
    # Home
    h = home_feats.rename(columns={c: f"home_{c}" for c in home_feats.columns if c not in ["game_pk", "game_date", "team_id"]})
    h = h.drop(columns=["team_id"], errors="ignore")
    # Away
    a = long.rename(columns={c: f"away_{c}" for c in long.columns if c not in ["game_pk", "game_date", "team_id"]})
    a = a.drop(columns=["team_id"], errors="ignore")

    # Re-merge limpio
    feats = []
    for _, row in df.iterrows():
        hrow = long[(long["game_pk"] == row["game_pk"]) & (long["team_id"] == row["home_id"])]
        arow = long[(long["game_pk"] == row["game_pk"]) & (long["team_id"] == row["away_id"])]
        rec = {"game_pk": row["game_pk"]}
        if not hrow.empty:
            for c in [f"rf_{w}" for w in windows] + [f"ra_{w}" for w in windows] + [f"win_{w}" for w in windows]:
                rec[f"home_{c}"] = hrow.iloc[0][c]
        if not arow.empty:
            for c in [f"rf_{w}" for w in windows] + [f"ra_{w}" for w in windows] + [f"win_{w}" for w in windows]:
                rec[f"away_{c}"] = arow.iloc[0][c]
        feats.append(rec)
    feat_df = pd.DataFrame(feats)
    return df.merge(feat_df, on="game_pk", how="left")

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_gematria(raw)
    df = rolling_team_features(df)
    # Diferenciales
    for w in [5, 10, 20]:
        df[f"rf_diff_{w}"] = df[f"home_rf_{w}"] - df[f"away_rf_{w}"]
        df[f"ra_diff_{w}"] = df[f"home_ra_{w}"] - df[f"away_ra_{w}"]
        df[f"win_diff_{w}"] = df[f"home_win_{w}"] - df[f"away_win_{w}"]
    df = df.dropna(subset=["home_rf_10", "away_rf_10"]).reset_index(drop=True)
    return df

if __name__ == "__main__":
    raw = pd.read_parquet("data/raw_games.parquet")
    feat = build_features(raw)
    feat.to_parquet("data/features.parquet", index=False)
    print(feat.shape)
