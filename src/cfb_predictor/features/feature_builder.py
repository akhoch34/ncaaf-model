from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from .elo import EloRatings, EloParams

ROLL_N = 3  # rolling window for points features

def build_features(games: pd.DataFrame, cutoff_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Build per-game features:
    - elo_home, elo_away, elo_diff
    - rolling last-N points for/against (per team at game time)
    - is_neutral, is_home
    - target variables: home_win (1/0), margin (home - away), total_points
    Assumes games for a single season OR multiple seasons; sorted by date/week.
    
    Args:
        games: DataFrame of games
        cutoff_date: If provided, only use completed games before this date for features
    """
    df = games.copy().sort_values(['season','week','start_date']).reset_index(drop=True)

    # Clean numeric
    for col in ['home_points','away_points']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['margin'] = df['home_points'] - df['away_points']
    df['total_points'] = df['home_points'] + df['away_points']
    df['home_win'] = (df['margin'] > 0).astype(float)

    # Elo before game - only update using games before cutoff
    elo = EloRatings(EloParams())
    prev_season = None
    elo_home_list, elo_away_list = [], []
    for i, row in df.iterrows():
        season = row['season']
        if prev_season is None:
            prev_season = season
        if season != prev_season:
            # offseason regression
            elo.regress_offseason()
            prev_season = season

        home = row['home_team']; away = row['away_team']
        elo_home_list.append(elo.get(home))
        elo_away_list.append(elo.get(away))

        # After game, update - but only if this game happened before cutoff
        game_date = pd.to_datetime(row['start_date'])
        should_update = (cutoff_date is None or game_date < cutoff_date)
        
        if (should_update and 
            pd.notna(row['home_points']) and pd.notna(row['away_points'])):
            elo.update_game(home, away, int(row['home_points']), int(row['away_points']), bool(row['neutral_site']))

    df['elo_home'] = elo_home_list
    df['elo_away'] = elo_away_list
    df['elo_diff'] = df['elo_home'] - df['elo_away'] + (~df['neutral_site']).astype(float) * elo.params.home_field

    # Rolling features per team based on *past* games (respecting cutoff)
    def add_rolling(prefix: str, team_col: str, points_for_col: str, points_against_col: str):
        # Build team-level time series
        tmp = df[[team_col,'season','week','start_date',points_for_col, points_against_col]].copy()
        tmp = tmp.rename(columns={team_col:'team', points_for_col:'pf', points_against_col:'pa'})
        tmp = tmp.reset_index(drop=True)
        
        # Only use completed games before cutoff for rolling calculations
        if cutoff_date is not None:
            tmp['game_date'] = pd.to_datetime(tmp['start_date'])
            # Mark games that should be excluded from rolling calculations
            tmp['valid_for_rolling'] = (tmp['game_date'] < cutoff_date) & tmp['pf'].notna() & tmp['pa'].notna()
            # Set points to NaN for games that shouldn't be used in rolling calculations
            tmp.loc[~tmp['valid_for_rolling'], ['pf', 'pa']] = np.nan
        
        # For rolling, we need history PRIOR to each game. We'll compute expanding means and shift by 1.
        tmp['pf_roll'] = tmp.groupby('team')['pf'].transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean())
        tmp['pa_roll'] = tmp.groupby('team')['pa'].transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean())
        return tmp[['pf_roll','pa_roll']].rename(columns={'pf_roll': f'{prefix}_pf_roll{ROLL_N}','pa_roll': f'{prefix}_pa_roll{ROLL_N}'})

    home_roll = add_rolling('home','home_team','home_points','away_points')
    away_roll = add_rolling('away','away_team','away_points','home_points')
    df = pd.concat([df, home_roll, away_roll], axis=1)

    # Basic flags
    df['is_neutral'] = df['neutral_site'].astype(bool).astype(int)
    df['is_home'] = 1 - df['is_neutral']  # treat non-neutral as 'home has HFA'
    return df

def join_lines(features: pd.DataFrame, lines: pd.DataFrame, book: str = 'consensus') -> pd.DataFrame:
    if lines is None or lines.empty:
        features['spread_line'] = np.nan
        features['total_line'] = np.nan
        return features
    lines = lines.copy()
    # Prefer the chosen book; if multiple entries, take the latest by last_updated
    lines = lines[lines['provider'].str.lower() == book.lower()] if 'provider' in lines.columns else lines
    lines = lines.sort_values('last_updated').drop_duplicates(subset=['game_id','provider'], keep='last')
    # CFBD game_id is `id` in games; ensure type alignment
    features = features.copy()
    # Some seasons may mismatch id types; coerce to numeric where possible
    for c in ['id']:
        if c in features.columns:
            features[c] = pd.to_numeric(features[c], errors='coerce')
    for c in ['game_id']:
        if c in lines.columns:
            lines[c] = pd.to_numeric(lines[c], errors='coerce')

    merged = features.merge(lines[['game_id','provider','spread','over_under']], left_on='id', right_on='game_id', how='left')
    merged = merged.rename(columns={'spread':'spread_line','over_under':'total_line'})
    merged.drop(columns=['game_id'], inplace=True)
    return merged
