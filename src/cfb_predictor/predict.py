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
    # Load historical seasons to build continuous ELO ratings
    historical_seasons = [2020, 2021, 2022, 2023, 2024]
    all_games = []
    
    # Load all historical seasons for ELO continuity
    for hist_season in historical_seasons:
        try:
            hist_games = _load_season(hist_season)
            all_games.append(hist_games)
        except FileNotFoundError:
            print(f"Warning: Historical season {hist_season} not found, ELO ratings may be less accurate")
    
    # Load current season
    current_games = _load_season(season)
    all_games.append(current_games)
    
    # Combine all seasons for continuous ELO calculation
    combined_games = pd.concat(all_games, ignore_index=True).sort_values(['season', 'week', 'start_date'])
    
    # Build features across all seasons to get proper ELO ratings
    all_feats = build_features(combined_games)
    
    # Extract features for the current season only
    feats = all_feats[all_feats['season'] == season].copy()
    
    # Load and join lines for current season
    lines = _load_lines(season)
    feats = join_lines(feats, lines, book=book)

    if week is None or (isinstance(week, str) and week.lower() == 'auto'):
        wk = _auto_week(current_games)
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
    wk_df['pred_margin'] = margin_m['model'].predict(Xm)  # Positive = home team wins by this much

    Xt = wk_df[total_m['features']].fillna(0.0).values
    wk_df['pred_total'] = total_m['model'].predict(Xt)

    # Moneyline pick
    wk_df['pick_ml'] = np.where(wk_df['win_prob_home'] >= 0.5, wk_df['home_team'], wk_df['away_team'])

    # ATS pick  
    # Edge calculation: how much better/worse than covering the spread
    # pred_margin convention: positive = home wins by that much, negative = home loses by that much
    # spread_line convention: negative = home favored, positive = home underdog  
    # 
    # Home favored: Penn State -45.5 → need to win by >45.5
    #   pred_margin +43.7 → edge = 43.7 - (-45.5) = 89.2 (would cover by 89.2)
    # 
    # Home underdog: Florida State +13.5 → can lose by <13.5 and still cover  
    #   pred_margin -14.28 → edge = -14.28 - 13.5 = -27.78 (don't cover by 27.78)
    # Add predicted line for clarity (negative means home favored)
    # Convert pred_margin to betting line convention: negative pred_line = home favored
    wk_df['pred_line'] = -wk_df['pred_margin']
    
    # Edge = predicted line - spread line (always)
    # For ATS: we want the absolute value of the edge to determine bet size/confidence
    # The sign of (pred_line - spread_line) tells us who covers
    wk_df['edge_spread'] = wk_df['pred_line'] - wk_df['spread_line']
    
    def make_ats_pick(row):
        if pd.notna(row['spread_line']) and abs(row['edge_spread']) >= min_edge:
            spread_line = row['spread_line']
            pred_line = row['pred_line']
            
            # Compare our predicted line to market line
            # If our pred_line is more favorable to the away team, take away team
            # If our pred_line is more favorable to the home team, take home team
            
            # Example: Oklahoma vs Michigan
            # pred_line = +1.51 (Oklahoma +1.51), spread_line = -3.0 (Oklahoma -3.0)
            # Our model thinks Oklahoma should get 1.51 points, market gives them -3.0
            # Market line is much more favorable to away team (Michigan), so take Michigan
            
            if pred_line > spread_line:
                # Our line is higher (worse for home team) - take away team
                if spread_line > 0:
                    return f"{row['away_team']} - ATS"  # Away is favorite
                else:
                    return f"{row['away_team']} + ATS"  # Away is underdog
            else:
                # Our line is lower (better for home team) - take home team
                if spread_line < 0:
                    return f"{row['home_team']} - ATS"  # Home is favorite
                else:
                    return f"{row['home_team']} + ATS"  # Home is underdog
        return 'no bet'
    
    wk_df['pick_spread'] = wk_df.apply(make_ats_pick, axis=1)

    # Total pick
    wk_df['edge_total'] = wk_df['pred_total'] - wk_df['total_line']
    wk_df['pick_total'] = np.where(
        wk_df['total_line'].notna() & (wk_df['edge_total'].abs() >= min_edge),
        np.where(wk_df['edge_total'] > 0, 'Over', 'Under'),
        'no bet'
    )

    # Output
    out_cols = ['season','week','start_date','home_team','away_team','is_neutral','win_prob_home','pred_margin','pred_line','spread_line','edge_spread','pick_spread','pred_total','total_line','edge_total','pick_total']
    out = wk_df[out_cols].sort_values('start_date')
    os.makedirs(os.path.join(PROCESSED_DIR, 'predictions'), exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, 'predictions', f'predictions_{season}_wk{wk}.csv')
    out.to_csv(out_path, index=False)
    return out
