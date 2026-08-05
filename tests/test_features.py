import pandas as pd
from src.features import build_features


def make_dummy():
    # construir dataset pequeño incrementando fechas y equipos
    rows = []
    for i, d in enumerate(pd.date_range("2024-04-01", periods=12)):
        rows.append({
            "game_pk": f"{d.strftime('%Y%m%d')}-1",
            "game_date": d,
            "home_id": 1,
            "away_id": 2,
            "home_name": "TeamA",
            "away_name": "TeamB",
            "home_pitcher_name": "PitcherA",
            "away_pitcher_name": "PitcherB",
            "home_runs": 4 + (i % 3),
            "away_runs": 3 + (i % 2),
            "home_win": 1,
        })
    return pd.DataFrame(rows)


def test_build_features_vectorized():
    raw = make_dummy()
    feat = build_features(raw)
    assert "home_rf_5" in feat.columns
    assert "rf_diff_10" in feat.columns
