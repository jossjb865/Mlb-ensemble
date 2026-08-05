#!/usr/bin/env python3
"""Extracción end-to-end desde MLB StatsAPI (schedule + boxscore + probablePitcher)."""
from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import pandas as pd
import numpy as np

BASE = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "MLB-Ensemble/1.0"}

def _get(url: str, params: Optional[Dict] = None, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Failed after retries: {url}")

def fetch_schedule(start: str, end: str) -> pd.DataFrame:
    """Devuelve games finalizados con scores y probable pitchers."""
    url = f"{BASE}/schedule"
    params = {
        "sportId": 1,
        "startDate": start,
        "endDate": end,
        "hydrate": "probablePitcher,linescore,team",
        "gameType": "R",
    }
    data = _get(url, params)
    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("detailedState") != "Final":
                continue
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            lines = g.get("linescore", {})
            home_runs = lines.get("teams", {}).get("home", {}).get("runs")
            away_runs = lines.get("teams", {}).get("away", {}).get("runs")
            if home_runs is None or away_runs is None:
                continue
            pp_home = home.get("probablePitcher", {}) or {}
            pp_away = away.get("probablePitcher", {}) or {}
            rows.append({
                "game_pk": g["gamePk"],
                "game_date": g["officialDate"],
                "home_id": home["team"]["id"],
                "away_id": away["team"]["id"],
                "home_name": home["team"]["name"],
                "away_name": away["team"]["name"],
                "home_runs": int(home_runs),
                "away_runs": int(away_runs),
                "total_runs": int(home_runs) + int(away_runs),
                "home_win": 1 if home_runs > away_runs else 0,
                "home_pitcher_id": pp_home.get("id"),
                "home_pitcher_name": pp_home.get("fullName", "TBD"),
                "away_pitcher_id": pp_away.get("id"),
                "away_pitcher_name": pp_away.get("fullName", "TBD"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values("game_date").reset_index(drop=True)

def fetch_team_stats_season(season: int) -> pd.DataFrame:
    """Stats de equipo por temporada (hitting + pitching)."""
    url = f"{BASE}/teams/stats"
    params = {
        "season": season,
        "group": "hitting,pitching",
        "stats": "season",
        "sportIds": 1,
        "gameType": "R",
    }
    data = _get(url, params)
    records = []
    for s in data.get("stats", []):
        group = s.get("group", {}).get("displayName", "")
        for split in s.get("splits", []):
            team = split.get("team", {})
            st = split.get("stat", {})
            rec = {
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "season": season,
                "group": group,
            }
            rec.update({k: st.get(k) for k in st})
            records.append(rec)
    return pd.DataFrame(records)

def build_historical(start_year: int = 2022, end_year: int = 2025) -> pd.DataFrame:
    """Descarga multi-temporada y construye dataset base."""
    frames = []
    for y in range(start_year, end_year + 1):
        s = f"{y}-03-20"
        e = f"{y}-10-05"
        print(f"Fetching {y}...")
        df = fetch_schedule(s, e)
        if not df.empty:
            frames.append(df)
        time.sleep(1.0)
    if not frames:
        raise ValueError("No data retrieved")
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    df = build_historical(2023, 2025)
    df.to_parquet("data/raw_games.parquet", index=False)
    print(df.shape)
    print(df.head())
