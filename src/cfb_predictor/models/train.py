from __future__ import annotations
import os
import pickle
from typing import List
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, mean_absolute_error
from ..config import RAW_DIR, PROCESSED_DIR, MODELS_DIR
from ..features.feature_builder import build_features, join_lines

FEATURES_WIN = ['elo_diff','is_neutral','home_pf_roll3','home_pa_roll3','away_pf_roll3','away_pa_roll3']
FEATURES_MARGIN = [
    # Original features (model was 58% with just these!)
    'elo_diff', 'is_neutral',
    # Postseason indicator (critical - postseason is 42.9% vs 58% regular season)
    'is_postseason',
    # Recent margin history (ATS momentum)
    'home_margin_roll3', 'away_margin_roll3',
    # Conference strength differential
    'conf_strength_diff',
    # Recent form (point differential trends)
    'form_differential',
    # Scoring trends
    'home_pf_roll3', 'away_pf_roll3', 'home_pa_roll3', 'away_pa_roll3',
]
FEATURES_TOTAL = [
    # Original features
    'home_pf_roll3', 'home_pa_roll3', 'away_pf_roll3', 'away_pa_roll3', 'is_neutral',
    # Conference features
    'home_conf_scoring_factor', 'away_conf_scoring_factor', 'is_conference_game',
    # Combined team stats
    'expected_combined_total', 'expected_combined_defense',
    'home_off_vs_away_def', 'away_off_vs_home_def',
    # Game total history
    'home_game_total_roll3', 'away_game_total_roll3', 'combined_game_total_history',
    # Timing features
    'week_num', 'is_early_season', 'is_late_season',
    # ELO for context
    'elo_diff'
]

def load_season(season: int) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"games_{season}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cached games for {season}. Run fetch-data first.")
    df = pd.read_parquet(path)
    df['season'] = season  # ensure present
    return df

def load_lines(season: int) -> pd.DataFrame:
    p = os.path.join(RAW_DIR, f"lines_{season}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return pd.DataFrame()

def train(seasons: List[int]) -> None:
    frames = []
    for yr in seasons:
        games = load_season(yr)
        feats = build_features(games)
        lines = load_lines(yr)
        feats = join_lines(feats, lines)
        frames.append(feats)
    data = pd.concat(frames, ignore_index=True)

    # Drop rows with missing outcomes
    train_df = data.dropna(subset=['home_points','away_points']).copy()

    # Win model (logistic): predict home_win
    X_win = train_df[FEATURES_WIN].fillna(0.0).values
    y_win = train_df['home_win'].values
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_win, y_win)
    p = clf.predict_proba(X_win)[:,1]
    ll = log_loss(y_win, p)

    # Margin model (linear): predict margin (home - away)
    X_margin = train_df[FEATURES_MARGIN].fillna(0.0).values
    y_margin = train_df['margin'].values
    reg_m = LinearRegression().fit(X_margin, y_margin)
    mae_m = mean_absolute_error(y_margin, reg_m.predict(X_margin))

    # Total model (linear): predict total points
    X_total = train_df[FEATURES_TOTAL].fillna(0.0).values
    y_total = train_df['total_points'].values
    reg_t = LinearRegression().fit(X_total, y_total)
    mae_t = mean_absolute_error(y_total, reg_t.predict(X_total))

    # Save models + metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "win_model.pkl"), "wb") as f:
        pickle.dump({"model": clf, "features": FEATURES_WIN, "log_loss": ll}, f)
    with open(os.path.join(MODELS_DIR, "margin_model.pkl"), "wb") as f:
        pickle.dump({"model": reg_m, "features": FEATURES_MARGIN, "mae": mae_m}, f)
    with open(os.path.join(MODELS_DIR, "total_model.pkl"), "wb") as f:
        pickle.dump({"model": reg_t, "features": FEATURES_TOTAL, "mae": mae_t}, f)

    # Save a training summary
    summ = pd.DataFrame({
        "model": ["win","margin","total"],
        "metric": ["log_loss","MAE","MAE"],
        "value": [ll, mae_m, mae_t],
    })
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    summ.to_csv(os.path.join(PROCESSED_DIR, "training_summary.csv"), index=False)

def train_time_aware(seasons: List[int], cutoff_date: pd.Timestamp = None) -> dict:
    """
    Train models using only data available before cutoff_date.
    Returns dictionary of trained models instead of saving to disk.
    """
    frames = []
    for yr in seasons:
        games = load_season(yr)
        # Use time-aware feature builder
        feats = build_features(games, cutoff_date=cutoff_date)
        lines = load_lines(yr)
        feats = join_lines(feats, lines)
        frames.append(feats)
    data = pd.concat(frames, ignore_index=True)

    # Only use completed games before cutoff for training
    if cutoff_date is not None:
        data = data[pd.to_datetime(data['start_date']) < cutoff_date]
    
    # Drop rows with missing outcomes
    train_df = data.dropna(subset=['home_points','away_points']).copy()
    
    if train_df.empty:
        # Return dummy models if no training data
        return {
            'win': {'model': LogisticRegression(), 'features': FEATURES_WIN},
            'margin': {'model': LinearRegression(), 'features': FEATURES_MARGIN},
            'total': {'model': LinearRegression(), 'features': FEATURES_TOTAL}
        }

    # Win model (logistic): predict home_win
    X_win = train_df[FEATURES_WIN].fillna(0.0).values
    y_win = train_df['home_win'].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_win, y_win)

    # Margin model (linear): predict margin (home - away)  
    X_margin = train_df[FEATURES_MARGIN].fillna(0.0).values
    y_margin = train_df['margin'].values
    reg_m = LinearRegression().fit(X_margin, y_margin)

    # Total model (linear): predict total points
    X_total = train_df[FEATURES_TOTAL].fillna(0.0).values
    y_total = train_df['total_points'].values
    reg_t = LinearRegression().fit(X_total, y_total)

    return {
        'win': {'model': clf, 'features': FEATURES_WIN},
        'margin': {'model': reg_m, 'features': FEATURES_MARGIN}, 
        'total': {'model': reg_t, 'features': FEATURES_TOTAL}
    }
