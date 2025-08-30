from __future__ import annotations
import os
import json
import pandas as pd
from typing import List, Optional
from ..config import RAW_DIR
from ..data_sources.cfbd_client import CFBDClient

def fetch_and_cache(seasons: List[int]) -> pd.DataFrame:
    client = CFBDClient()
    frames = []
    for yr in seasons:
        games = client.get_games_df(yr, season_type='regular', classification='fbs')
        bowls = client.get_games_df(yr, season_type='postseason', classification='fbs')
        df = pd.concat([games, bowls], ignore_index=True)
        outp = os.path.join(RAW_DIR, f"games_{yr}.parquet")
        df.to_parquet(outp, index=False)
        frames.append(df)
        # Lines per week (optional)
        # We'll fetch season aggregate too
        lines = client.get_lines_df(yr, season_type='regular')
        lines_post = client.get_lines_df(yr, season_type='postseason')
        ln = pd.concat([lines, lines_post], ignore_index=True)
        ln.to_parquet(os.path.join(RAW_DIR, f"lines_{yr}.parquet"), index=False)
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_df
