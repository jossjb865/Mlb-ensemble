#!/usr/bin/env python3
"""Extracción robusta desde MLB StatsAPI oficial (statsapi.mlb.com).
Endpoints: /schedule (hydrate=probablePitcher,linescore,team) + fallback boxscore."""
from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import requests
import pandas as pd
import numpy as np

BASE = "https://statsapi.mlb.com/api/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MLB-Ensemble/2.0)",
    "Accept": "application/json",
}

def _get(url: str, params: Optional[Dict] = None, retries: int = 4) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=35)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
        except requests.exceptions.RequestException:
            time.sleep(1.8 * (attempt + 1))
    raise RuntimeError(f"StatsAPI failed after {retries} retries → {url}")

def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default

def fetch_schedule(start: str, end: str, game_type: str = "R") -> pd.DataFrame:
    """
    Descarga juegos finalizados entre start y end (YYYY-MM-DD).
    Incluye scores, total_runs, home_win y probable pitchers.
    """
    url = f"{BASE}/schedule"
    params = {
        "sportId": 1,
        "startDate": start,
        "endDate": end,
        "hydrate": "probablePitcher,linescore,team",
        "gameType": game_type,
    }
    data = _get(url, params)
    rows: List[Dict[str, Any]] = []

    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if status not in ("Final", "Completed Early"):
                continue

            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            lines = g.get("linescore", {})
            home_ls = lines.get("teams", {}).get("home", {})
            away_ls = lines.get("teams", {}).get("away", {})

            home_runs = _safe_int(home_ls.get("runs"))
            away_runs = _safe_int(away_ls.get("runs"))
            if home_runs == 0 and away_runs == 0 and status == "Final":
                # fallback por si linescore llega vacío
                home_runs = _safe_int(home.get("score"))
                away_runs = _safe_int(away.get("score"))

            pp_home = home.get("probablePitcher") or {}
            pp_away = away.get("probablePitcher") or {}

            rows.append({
                "game_pk": g.get("gamePk"),
                "game_date": g.get("officialDate"),
                "home_id": home.get("team", {}).get("id"),
                "away_id": away.get("team", {}).get("id"),
                "home_name": home.get("team", {}).get("name", "Unknown"),
                "away_name": away.get("team", {}).get("name", "Unknown"),
                "home_abbr": home.get("team", {}).get("abbreviation", ""),
                "away_abbr": away.get("team", {}).get("abbreviation", ""),
                "home_runs": home_runs,
                "away_runs": away_runs,
                "total_runs": home_runs + away_runs,
                "home_win": 1 if home_runs > away_runs else 0,
                "home_pitcher_id": pp_home.get("id"),
                "home_pitcher_name": pp_home.get("fullName", "TBD"),
                "away_pitcher_id": pp_away.get("id"),
                "away_pitcher_name": pp_away.get("fullName", "TBD"),
                "venue": g.get("venue", {}).get("name", ""),
                "game_type": g.get("gameType", "R"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Aseguramos dtype datetime homogéneo (coerce para evitar valores mixtos que rompan sort)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    # Eliminamos filas sin fecha válida
    df = df.dropna(subset=["game_date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["game_pk"]).sort_values("game_date").reset_index(drop=True)
    return df

def fetch_today_schedule() -> pd.DataFrame:
    """Juegos del día (incluye Scheduled / Pre-Game) con probable pitchers."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"{BASE}/schedule"
    params = {
        "sportId": 1,
        "date": today,
        "hydrate": "probablePitcher,team,venue",
        "gameType": "R",
    }
    data = _get(url, params)
    rows: List[Dict[str, Any]] = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            pp_home = home.get("probablePitcher") or {}
            pp_away = away.get("probablePitcher") or {}
            rows.append({
                "game_pk": g.get("gamePk"),
                "game_date": g.get("officialDate"),
                "status": g.get("status", {}).get("detailedState", ""),
                "home_id": home.get("team", {}).get("id"),
                "away_id": away.get("team", {}).get("id"),
                "home_name": home.get("team", {}).get("name", "Unknown"),
                "away_name": away.get("team", {}).get("name", "Unknown"),
                "home_pitcher_id": pp_home.get("id"),
                "home_pitcher_name": pp_home.get("fullName", "TBD"),
                "away_pitcher_id": pp_away.get("id"),
                "away_pitcher_name": pp_away.get("fullName", "TBD"),
                "venue": g.get("venue", {}).get("name", ""),
                "home_runs": 0,
                "away_runs": 0,
                "total_runs": 0,
                "home_win": 0,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    return out

def build_historical(start_year: int = 2023, end_year: int = 2025) -> pd.DataFrame:
    """Multi-temporada regular season."""
    frames: List[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        start = f"{year}-03-20"
        end = f"{year}-10-06"
        print(f"[StatsAPI] Fetching {year} ({start} → {end}) ...")
        df = fetch_schedule(start, end)
        if not df.empty:
            print(f"  → {len(df)} games")
            frames.append(df)
        time.sleep(1.2)
    if not frames:
        raise ValueError("No games retrieved from StatsAPI")
    full = pd.concat(frames, ignore_index=True)
    # Garantizar dtype datetime uniforme tras concatenar
    full["game_date"] = pd.to_datetime(full["game_date"], errors="coerce")
    full = full.dropna(subset=["game_date"]).drop_duplicates(subset=["game_pk"]).sort_values("game_date").reset_index(drop=True)
    return full

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    hist = build_historical(2023, 2025)
    hist.to_parquet("data/raw_games.parquet", index=False)
    print(hist.shape)
    print(hist[["game_date", "home_name", "away_name", "home_runs", "away_runs", "home_pitcher_name"]].tail(8))
