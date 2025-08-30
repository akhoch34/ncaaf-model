from __future__ import annotations
import os, pickle
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from .config import RAW_DIR, PROCESSED_DIR, MODELS_DIR
from .features.feature_builder import build_features, join_lines

def _load_model(name: str):
    p = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing model {name}. Train first.")
    with open(p, 'rb') as f:
        return pickle.load(f)

def _load_season(season: int) -> pd.DataFrame:
    p = os.path.join(RAW_DIR, f"games_{season}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing games for {season}. Run fetch-data.")
    return pd.read_parquet(p)

def _load_lines(season: int) -> pd.DataFrame:
    p = os.path.join(RAW_DIR, f"lines_{season}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()

def _auto_week(df: pd.DataFrame) -> int:
    # pick the next week with any game missing scores
    played = df[pd.notna(df['home_points']) & pd.notna(df['away_points'])]
    if played.empty:
        return int(df['week'].min())
    last_week = int(played['week'].max())
    return last_week + 1

def predict(season: int, week: Optional[int] = None, book: str = 'consensus', min_edge: float = 0.5) -> pd.DataFrame:
    games = _load_season(season)
    lines = _load_lines(season)
    feats = build_features(games)
    feats = join_lines(feats, lines, book=book)

    if week is None or (isinstance(week, str) and week.lower() == 'auto'):
        wk = _auto_week(games)
    else:
        wk = int(week)

    wk_df = feats[feats['week'] == wk].copy()

    win_m = _load_model('win')
    margin_m = _load_model('margin')
    total_m = _load_model('total')

    # Predict
    Xw = wk_df[win_m['features']].fillna(0.0).values
    wk_df['win_prob_home'] = win_m['model'].predict_proba(Xw)[:,1]

    Xm = wk_df[margin_m['features']].fillna(0.0).values
    wk_df['pred_margin'] = margin_m['model'].predict(Xm)

    Xt = wk_df[total_m['features']].fillna(0.0).values
    wk_df['pred_total'] = total_m['model'].predict(Xt)

    # Moneyline pick
    wk_df['pick_ml'] = np.where(wk_df['win_prob_home'] >= 0.5, wk_df['home_team'], wk_df['away_team'])

    # ATS pick
    wk_df['edge_spread'] = wk_df['pred_margin'] - wk_df['spread_line']
    wk_df['pick_spread'] = np.where(
        wk_df['spread_line'].notna() & (wk_df['edge_spread'].abs() >= min_edge),
        np.where(wk_df['edge_spread'] > 0, f"{wk_df['home_team']} - ATS", f"{wk_df['away_team']} + ATS"),
        'no bet'
    )

    # Total pick
    wk_df['edge_total'] = wk_df['pred_total'] - wk_df['total_line']
    wk_df['pick_total'] = np.where(
        wk_df['total_line'].notna() & (wk_df['edge_total'].abs() >= min_edge),
        np.where(wk_df['edge_total'] > 0, 'Over', 'Under'),
        'no bet'
    )

    # Output
    out_cols = ['season','week','start_date','home_team','away_team','is_neutral','win_prob_home','pred_margin','spread_line','edge_spread','pick_spread','pred_total','total_line','edge_total','pick_total']
    out = wk_df[out_cols].sort_values('start_date')
    os.makedirs(os.path.join(PROCESSED_DIR, 'predictions'), exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, 'predictions', f'predictions_{season}_wk{wk}.csv')
    out.to_csv(out_path, index=False)
    return out
