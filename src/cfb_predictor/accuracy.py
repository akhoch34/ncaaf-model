"""
Accuracy tracking module for NCAAF predictions
"""
from __future__ import annotations
import os
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from .config import PROCESSED_DIR

def update_weekly_accuracy(season: int, week: int, book: str = 'DraftKings') -> Dict[str, Any]:
    """
    Calculate and store weekly accuracy for completed games.
    
    Args:
        season: Season year
        week: Week number to evaluate
        book: Sportsbook used for lines
    
    Returns:
        Dictionary with accuracy metrics
    """
    from .backtest import backtest
    
    # Run backtest for this specific week
    weekly_results = backtest([season], book=book, min_edge=0.5)
    
    # Load existing accuracy tracking file
    accuracy_file = os.path.join(PROCESSED_DIR, 'weekly_accuracy.csv')
    
    if os.path.exists(accuracy_file):
        accuracy_df = pd.read_csv(accuracy_file)
    else:
        accuracy_df = pd.DataFrame(columns=[
            'season', 'week', 'date_updated', 'book',
            'ats_bets', 'ats_wins', 'ats_losses', 'ats_pushes', 'ats_win_pct',
            'ou_bets', 'ou_wins', 'ou_losses', 'ou_pushes', 'ou_win_pct'
        ])
    
    # Extract accuracy metrics
    ats_row = weekly_results[weekly_results['market'] == 'ATS'].iloc[0] if len(weekly_results) > 0 else None
    ou_row = weekly_results[weekly_results['market'] == 'O/U'].iloc[0] if len(weekly_results) > 1 else None
    
    # Create new row
    new_row = {
        'season': season,
        'week': week,
        'date_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'book': book,
        'ats_bets': ats_row['bets'] if ats_row is not None else 0,
        'ats_wins': ats_row['wins'] if ats_row is not None else 0,
        'ats_losses': ats_row['losses'] if ats_row is not None else 0,
        'ats_pushes': ats_row['pushes'] if ats_row is not None else 0,
        'ats_win_pct': ats_row['win_pct'] if ats_row is not None else 0.0,
        'ou_bets': ou_row['bets'] if ou_row is not None else 0,
        'ou_wins': ou_row['wins'] if ou_row is not None else 0,
        'ou_losses': ou_row['losses'] if ou_row is not None else 0,
        'ou_pushes': ou_row['pushes'] if ou_row is not None else 0,
        'ou_win_pct': ou_row['win_pct'] if ou_row is not None else 0.0,
    }
    
    # Remove existing entry for this season/week if it exists
    accuracy_df = accuracy_df[~((accuracy_df['season'] == season) & (accuracy_df['week'] == week))]
    
    # Add new row
    accuracy_df = pd.concat([accuracy_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Sort by season and week
    accuracy_df = accuracy_df.sort_values(['season', 'week']).reset_index(drop=True)
    
    # Save updated accuracy tracking
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    accuracy_df.to_csv(accuracy_file, index=False)
    
    return new_row

def get_season_accuracy(season: int) -> pd.DataFrame:
    """Get accuracy summary for entire season"""
    accuracy_file = os.path.join(PROCESSED_DIR, 'weekly_accuracy.csv')
    
    if not os.path.exists(accuracy_file):
        return pd.DataFrame()
    
    df = pd.read_csv(accuracy_file)
    season_data = df[df['season'] == season]
    
    if season_data.empty:
        return pd.DataFrame()
    
    # Calculate cumulative season totals
    season_summary = {
        'season': season,
        'total_weeks': len(season_data),
        'ats_total_bets': season_data['ats_bets'].sum(),
        'ats_total_wins': season_data['ats_wins'].sum(),
        'ats_total_losses': season_data['ats_losses'].sum(),
        'ats_total_pushes': season_data['ats_pushes'].sum(),
        'ou_total_bets': season_data['ou_bets'].sum(),
        'ou_total_wins': season_data['ou_wins'].sum(),
        'ou_total_losses': season_data['ou_losses'].sum(),
        'ou_total_pushes': season_data['ou_pushes'].sum(),
    }
    
    # Calculate overall win percentages
    ats_decisions = season_summary['ats_total_wins'] + season_summary['ats_total_losses']
    ou_decisions = season_summary['ou_total_wins'] + season_summary['ou_total_losses']
    
    season_summary['ats_season_win_pct'] = (
        season_summary['ats_total_wins'] / ats_decisions if ats_decisions > 0 else 0.0
    )
    season_summary['ou_season_win_pct'] = (
        season_summary['ou_total_wins'] / ou_decisions if ou_decisions > 0 else 0.0
    )
    
    return pd.DataFrame([season_summary])

def print_accuracy_summary(season: Optional[int] = None):
    """Print accuracy summary to console"""
    accuracy_file = os.path.join(PROCESSED_DIR, 'weekly_accuracy.csv')
    
    if not os.path.exists(accuracy_file):
        print("No accuracy data found. Run some backtests first.")
        return
    
    df = pd.read_csv(accuracy_file)
    
    if season:
        df = df[df['season'] == season]
        print(f"=== {season} Season Accuracy ===")
    else:
        print("=== All-Time Accuracy ===")
    
    if df.empty:
        print("No data available for the specified criteria.")
        return
    
    # Recent weeks
    print("\nRecent Weekly Results:")
    recent = df.tail(10)[['season', 'week', 'ats_bets', 'ats_wins', 'ats_win_pct', 'ou_bets', 'ou_wins', 'ou_win_pct']]
    print(recent.to_string(index=False))
    
    # Season summaries
    if not season:
        print("\nSeason Summaries:")
        for s in sorted(df['season'].unique()):
            season_summary = get_season_accuracy(s)
            if not season_summary.empty:
                row = season_summary.iloc[0]
                print(f"{s}: ATS {row['ats_total_wins']}-{row['ats_total_losses']}-{row['ats_total_pushes']} ({row['ats_season_win_pct']:.1%}), "
                      f"O/U {row['ou_total_wins']}-{row['ou_total_losses']}-{row['ou_total_pushes']} ({row['ou_season_win_pct']:.1%})")