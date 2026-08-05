import pandas as pd
from src.data_fetch import fetch_schedule, build_historical

def test_fetch_schedule_dates_coerced():
    # This test assumes there is at least some historical data; if none, the test is skipped
    try:
        df = build_historical(2023, 2023)
    except ValueError:
        # no data available in StatsAPI during test run environment
        return
    assert not df["game_date"].dtype == object
    assert pd.api.types.is_datetime64_any_dtype(df["game_date"]) 
