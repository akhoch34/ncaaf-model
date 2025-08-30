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
