from __future__ import annotations
import os, pickle
from typing import List
import pandas as pd
import numpy as np
from .config import RAW_DIR, PROCESSED_DIR, MODELS_DIR
from .features.feature_builder import build_features, join_lines

def _load_model(name: str):
    p = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    with open(p, 'rb') as f:
        return pickle.load(f)

def backtest(seasons: List[int], book: str = 'consensus', min_edge: float = 0.5) -> pd.DataFrame:
    win_m = _load_model('win')
    margin_m = _load_model('margin')
    total_m = _load_model('total')

    frames = []
    for yr in seasons:
        games_p = os.path.join(RAW_DIR, f"games_{yr}.parquet")
        lines_p = os.path.join(RAW_DIR, f"lines_{yr}.parquet")
        if not os.path.exists(games_p):
            continue
        games = pd.read_parquet(games_p)
        lines = pd.read_parquet(lines_p) if os.path.exists(lines_p) else pd.DataFrame()
        feats = build_features(games)
        feats = join_lines(feats, lines, book=book)

        # Only completed games
        df = feats.dropna(subset=['home_points','away_points']).copy()

        Xw = df[win_m['features']].fillna(0.0).values
        df['win_prob_home'] = win_m['model'].predict_proba(Xw)[:,1]
        df['pred_margin'] = margin_m['model'].predict(df[margin_m['features']].fillna(0.0).values)
        df['pred_total'] = total_m['model'].predict(df[total_m['features']].fillna(0.0).values)

        # Results
        df['actual_margin'] = df['home_points'] - df['away_points']
        df['actual_total'] = df['home_points'] + df['away_points']

        # ATS evaluation
        df['edge_spread'] = df['pred_margin'] - df['spread_line']
        df['bet_spread'] = (df['spread_line'].notna()) & (df['edge_spread'].abs() >= min_edge)
        df['ats_pick_home'] = df['edge_spread'] > 0  # True = home cover pick
        def ats_result(row):
            if not row['bet_spread']:
                return np.nan
            # Spread line is from home perspective: home favored if negative
            actual_margin = row['actual_margin']
            line = row['spread_line']
            # If we picked home (laying points when line<0), we win if actual_margin > line
            pick_home = row['ats_pick_home']
            if pick_home:
                return 1.0 if actual_margin > line else (0.5 if actual_margin == line else 0.0)
            else:
                # Pick away (taking points)
                return 1.0 if actual_margin < line else (0.5 if actual_margin == line else 0.0)
        df['ats_outcome'] = df.apply(ats_result, axis=1)

        # O/U evaluation
        df['edge_total'] = df['pred_total'] - df['total_line']
        df['bet_total'] = (df['total_line'].notna()) & (df['edge_total'].abs() >= min_edge)
        df['pick_over'] = df['edge_total'] > 0
        def ou_result(row):
            if not row['bet_total']:
                return np.nan
            total = row['actual_total']
            line = row['total_line']
            pick_over = row['pick_over']
            if pick_over:
                return 1.0 if total > line else (0.5 if total == line else 0.0)
            else:
                return 1.0 if total < line else (0.5 if total == line else 0.0)
        df['ou_outcome'] = df.apply(ou_result, axis=1)

        frames.append(df[['season','week','home_team','away_team','bet_spread','ats_outcome','bet_total','ou_outcome','edge_spread','edge_total']])

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    os.makedirs(os.path.join(PROCESSED_DIR,'backtests'), exist_ok=True)
    out.to_csv(os.path.join(PROCESSED_DIR,'backtests','results.csv'), index=False)

    # Summary
    def summarize(col_bet, col_outcome):
        sub = out[out[col_bet]].copy()
        games = len(sub)
        wins = (sub[col_outcome] == 1.0).sum()
        pushes = (sub[col_outcome] == 0.5).sum()
        losses = (sub[col_outcome] == 0.0).sum()
        win_pct = wins / max(1, (wins+losses))
        return games, wins, pushes, losses, win_pct

    ats_games, ats_w, ats_p, ats_l, ats_win = summarize('bet_spread','ats_outcome')
    ou_games, ou_w, ou_p, ou_l, ou_win = summarize('bet_total','ou_outcome')
    summ = pd.DataFrame({
        'market':['ATS','O/U'],
        'bets':[ats_games, ou_games],
        'wins':[ats_w, ou_w],
        'pushes':[ats_p, ou_p],
        'losses':[ats_l, ou_l],
        'win_pct':[ats_win, ou_win],
    })
    summ.to_csv(os.path.join(PROCESSED_DIR,'backtests','summary.csv'), index=False)
    return summ

def backtest_week_time_aware(season: int, week: int, book: str = 'consensus', min_edge: float = 0.5) -> pd.DataFrame:
    """
    Backtest a specific week using only information available before that week.
    This prevents look-ahead bias by:
    1. Training models only on data before the week being evaluated
    2. Building features using only historical information
    """
    from .models.train import train_time_aware
    
    # Load all available seasons for training - check which ones exist
    potential_seasons = [2020, 2021, 2022, 2023, 2024, season]
    training_seasons = []
    for s in potential_seasons:
        games_path = os.path.join(RAW_DIR, f"games_{s}.parquet")
        if os.path.exists(games_path):
            training_seasons.append(s)
    
    if not training_seasons:
        return pd.DataFrame({'market':['ATS','O/U'], 'bets':[0,0], 'wins':[0,0], 'pushes':[0,0], 'losses':[0,0], 'win_pct':[0.0,0.0]})
    
    games_p = os.path.join(RAW_DIR, f"games_{season}.parquet")
    lines_p = os.path.join(RAW_DIR, f"lines_{season}.parquet")
    if not os.path.exists(games_p):
        return pd.DataFrame({'market':['ATS','O/U'], 'bets':[0,0], 'wins':[0,0], 'pushes':[0,0], 'losses':[0,0], 'win_pct':[0.0,0.0]})
    
    games = pd.read_parquet(games_p)
    lines = pd.read_parquet(lines_p) if os.path.exists(lines_p) else pd.DataFrame()
    
    # Get games for the specific week we want to evaluate
    week_games = games[games['week'] == week].copy()
    if week_games.empty:
        return pd.DataFrame({'market':['ATS','O/U'], 'bets':[0,0], 'wins':[0,0], 'pushes':[0,0], 'losses':[0,0], 'win_pct':[0.0,0.0]})
    
    # Find the earliest game start date for this week
    week_games['start_date'] = pd.to_datetime(week_games['start_date'])
    cutoff_date = week_games['start_date'].min()
    
    # Train models using only data before this week
    models = train_time_aware(training_seasons, cutoff_date=cutoff_date)
    
    # Build features for prediction (using cutoff to prevent look-ahead)
    feats = build_features(games, cutoff_date=cutoff_date)
    feats = join_lines(feats, lines, book=book)
    
    # Only evaluate completed games from this specific week
    df = feats[(feats['week'] == week)].dropna(subset=['home_points','away_points']).copy()
    
    if df.empty:
        return pd.DataFrame({'market':['ATS','O/U'], 'bets':[0,0], 'wins':[0,0], 'pushes':[0,0], 'losses':[0,0], 'win_pct':[0.0,0.0]})

    # Make predictions using time-aware models
    try:
        # Check if model was actually trained (has non-zero coefficients)
        if (hasattr(models['win']['model'], 'coef_') and 
            models['win']['model'].coef_ is not None and 
            models['win']['model'].coef_.size > 0):
            model_trained = True
        else:
            model_trained = False
    except:
        model_trained = False
        
    if not model_trained:
        # Not enough training data, return zeros
        return pd.DataFrame({'market':['ATS','O/U'], 'bets':[0,0], 'wins':[0,0], 'pushes':[0,0], 'losses':[0,0], 'win_pct':[0.0,0.0]})
        
    Xw = df[models['win']['features']].fillna(0.0).values
    df['win_prob_home'] = models['win']['model'].predict_proba(Xw)[:,1]
    df['pred_margin'] = models['margin']['model'].predict(df[models['margin']['features']].fillna(0.0).values)
    df['pred_total'] = models['total']['model'].predict(df[models['total']['features']].fillna(0.0).values)

    # Results
    df['actual_margin'] = df['home_points'] - df['away_points']
    df['actual_total'] = df['home_points'] + df['away_points']

    # ATS evaluation
    df['edge_spread'] = df['pred_margin'] - df['spread_line']
    df['bet_spread'] = (df['spread_line'].notna()) & (df['edge_spread'].abs() >= min_edge)
    # Determine ATS pick: if predicted margin beats the spread, pick home, otherwise pick away
    df['ats_pick_home'] = df.apply(lambda row: 
        row['pred_margin'] > abs(row['spread_line']) if row['spread_line'] < 0 else 
        row['pred_margin'] > row['spread_line'], axis=1)
    def ats_result(row):
        if not row['bet_spread']:
            return np.nan
        actual_margin = row['actual_margin']
        line = row['spread_line'] 
        pick_home = row['ats_pick_home']
        
        if pick_home:
            # We picked home team - they need to cover the spread
            if line < 0:  # Home favored
                covers = actual_margin > abs(line)
            else:  # Home underdog  
                covers = actual_margin > line
        else:
            # We picked away team - home team must NOT cover
            if line < 0:  # Home favored
                covers = actual_margin < abs(line)
            else:  # Home underdog
                covers = actual_margin < line
                
        if line < 0:
            return 1.0 if covers else (0.5 if actual_margin == abs(line) else 0.0)
        else:
            return 1.0 if covers else (0.5 if actual_margin == line else 0.0)
    df['ats_outcome'] = df.apply(ats_result, axis=1)

    # O/U evaluation
    df['edge_total'] = df['pred_total'] - df['total_line']
    df['bet_total'] = (df['total_line'].notna()) & (df['edge_total'].abs() >= min_edge)
    df['pick_over'] = df['edge_total'] > 0
    def ou_result(row):
        if not row['bet_total']:
            return np.nan
        total = row['actual_total']
        line = row['total_line']
        pick_over = row['pick_over']
        if pick_over:
            return 1.0 if total > line else (0.5 if total == line else 0.0)
        else:
            return 1.0 if total < line else (0.5 if total == line else 0.0)
    df['ou_outcome'] = df.apply(ou_result, axis=1)

    # Add detailed pick information for clarity
    df['ats_pick_team'] = df.apply(lambda row: 
        row['home_team'] if (row['bet_spread'] and row['ats_pick_home']) else 
        row['away_team'] if (row['bet_spread'] and not row['ats_pick_home']) else 
        'no bet', axis=1)
    
    df['ats_pick_description'] = df.apply(lambda row:
        f"{row['home_team']} {row['spread_line']:+.1f}" if (row['bet_spread'] and row['ats_pick_home']) else
        f"{row['away_team']} {-row['spread_line']:+.1f}" if (row['bet_spread'] and not row['ats_pick_home']) else
        'no bet' if pd.notna(row['spread_line']) else 'no line', axis=1)
    
    df['actual_margin'] = df['home_points'] - df['away_points']  
    df['spread_result'] = df.apply(lambda row:
        'WIN' if row['ats_outcome'] == 1.0 else 
        'PUSH' if row['ats_outcome'] == 0.5 else
        'LOSS' if row['ats_outcome'] == 0.0 else
        'no bet', axis=1)

    # Save week-specific results for reference
    os.makedirs(os.path.join(PROCESSED_DIR,'backtests'), exist_ok=True)
    df[['season','week','home_team','away_team','home_points','away_points','actual_margin',
        'spread_line','ats_pick_team','ats_pick_description','spread_result',
        'bet_spread','ats_outcome','bet_total','ou_outcome','edge_spread','edge_total']].to_csv(
        os.path.join(PROCESSED_DIR,'backtests',f'week_{season}_{week}_time_aware.csv'), index=False)

    # Summary
    def summarize(col_bet, col_outcome):
        sub = df[df[col_bet]].copy()
        games = len(sub)
        wins = (sub[col_outcome] == 1.0).sum()
        pushes = (sub[col_outcome] == 0.5).sum()
        losses = (sub[col_outcome] == 0.0).sum()
        win_pct = wins / max(1, (wins+losses))
        return games, wins, pushes, losses, win_pct

    ats_games, ats_w, ats_p, ats_l, ats_win = summarize('bet_spread','ats_outcome')
    ou_games, ou_w, ou_p, ou_l, ou_win = summarize('bet_total','ou_outcome')
    summ = pd.DataFrame({
        'market':['ATS','O/U'],
        'bets':[ats_games, ou_games],
        'wins':[ats_w, ou_w],
        'pushes':[ats_p, ou_p],
        'losses':[ats_l, ou_l],
        'win_pct':[ats_win, ou_win],
    })
    return summ

# Keep the old function for compatibility but mark as deprecated
def backtest_week(season: int, week: int, book: str = 'consensus', min_edge: float = 0.5) -> pd.DataFrame:
    """DEPRECATED: Use backtest_week_time_aware instead to prevent data leakage"""
    return backtest_week_time_aware(season, week, book, min_edge)
