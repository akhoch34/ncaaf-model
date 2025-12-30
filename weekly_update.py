
#!/usr/bin/env python3
"""
Weekly NCAAF Model Update Script
================================

This script automates the weekly process of:
1. Fetching the latest game data for the current season (2025)
2. Retraining models on all available data (2020-2024 + completed 2025 games)
3. Generating predictions for upcoming games
4. Sending a modernized weekly picks email (logic refactored to email_renderer.py)

Usage:
    python weekly_update.py [--week WEEK] [--book BOOK] [--min-edge EDGE]

The script should be run weekly, ideally on Tuesday mornings after Monday's games
are finalized in the CFBD database.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
import logging
import pandas as pd

from typing import Iterable, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weekly_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CURRENT_SEASON = 2025  # Production season
TRAINING_SEASONS = [2020, 2021, 2022, 2023, 2024]  # Historical data for training
SRC_DIR = "src"


def _coerce_list(recipients: Optional[Iterable[str] | str]) -> List[str]:
    if recipients is None:
        return []
    if isinstance(recipients, str):
        return [r.strip() for r in recipients.split(",") if r.strip()]
    return [str(r).strip() for r in recipients if str(r).strip()]


def run_command(cmd, cwd=None):
    """Run a shell command and return success status"""
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            logger.info(f"Output: {result.stdout.strip()}")
        if result.stderr.strip():
            logger.debug(f"Stderr: {result.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def run_python_command(python_args, cwd=None):
    """Run a Python command with appropriate environment setup"""
    # In GitHub Actions, Python is already available globally
    if os.getenv('GITHUB_ACTIONS'):
        cmd = ['python'] + python_args
        return run_command(cmd, cwd)

    # If we're in an activated virtual environment, use python directly
    if os.getenv('VIRTUAL_ENV'):
        cmd = ['python'] + python_args
        return run_command(cmd, cwd)

    # For local development, assume user activated the venv manually
    cmd = ['python'] + python_args
    return run_command(cmd, cwd)


def fetch_latest_data():
    """Fetch the latest game data for current season"""
    logger.info(f"Fetching latest data for {CURRENT_SEASON} season...")
    python_args = ["-m", "cfb_predictor.cli", "fetch-data", str(CURRENT_SEASON)]
    return run_python_command(python_args, cwd=SRC_DIR)


def retrain_models():
    """Retrain models on all available data"""
    logger.info("Retraining models on all available data...")
    # Include current season in training if it has completed games
    all_seasons = TRAINING_SEASONS + [CURRENT_SEASON]
    season_args = [str(s) for s in all_seasons]
    python_args = ["-m", "cfb_predictor.cli", "train"] + season_args
    return run_python_command(python_args, cwd=SRC_DIR)


def get_current_week():
    """
    Determine the current week/date for predictions based on game schedule.

    Logic:
    1. For regular season: Return week number (1-16)
    2. For postseason: Return special string "postseason" to trigger date-based predictions
    3. Checks if games are in progress or finds earliest upcoming games
    """
    try:
        games_file = f"{SRC_DIR}/data/raw/games_{CURRENT_SEASON}.parquet"
        if not os.path.exists(games_file):
            logger.warning(f"Games file not found: {games_file}")
            return None

        games = pd.read_parquet(games_file)

        games['start_date'] = pd.to_datetime(games['start_date'])
        now = pd.Timestamp.now(tz='UTC')

        # Prioritize regular season, then postseason
        regular_season = games[games['season_type'] == 'regular']
        postseason = games[games['season_type'] == 'postseason']

        # First, check if any REGULAR SEASON week is "in progress" (some games played, some not)
        for week in sorted(regular_season['week'].unique()):
            week_games = regular_season[regular_season['week'] == week]
            completed = len(week_games[pd.notna(week_games['home_points'])])
            total = len(week_games)

            # If this week has started but isn't finished, this is the current week
            if 0 < completed < total:
                logger.info(f"Week {week} is in progress ({completed}/{total} regular season games completed)")
                return int(week)

        # No regular season week in progress, find the week with the earliest upcoming REGULAR SEASON games
        earliest_week = None
        earliest_time = None

        for week in sorted(regular_season['week'].unique()):
            week_games = regular_season[regular_season['week'] == week]
            upcoming = week_games[
                pd.isna(week_games['home_points']) &
                (week_games['start_date'] > now)
            ]

            if len(upcoming) > 0:
                week_earliest = upcoming['start_date'].min()
                if earliest_time is None or week_earliest < earliest_time:
                    earliest_time = week_earliest
                    earliest_week = int(week)

        if earliest_week is not None:
            logger.info(f"Week {earliest_week} has earliest upcoming regular season games (starts {earliest_time})")
            return earliest_week

        # All regular season done, check for postseason games
        # For postseason, we use date-based predictions instead of week numbers
        if not postseason.empty:
            upcoming_postseason = postseason[
                pd.isna(postseason['home_points']) &
                (postseason['start_date'] > now)
            ]

            if not upcoming_postseason.empty:
                logger.info(f"Postseason in progress - using date-based predictions")
                return "postseason"

            # Check if there are any postseason games in progress
            in_progress = postseason[
                pd.notna(postseason['home_points']) &
                (postseason['start_date'] <= now)
            ]

            if not in_progress.empty:
                logger.info(f"Postseason games still being played - using date-based predictions")
                return "postseason"

        # Fallback: find last completed week + 1
        reg_completed = regular_season[pd.notna(regular_season['home_points'])]
        if not reg_completed.empty:
            return int(reg_completed['week'].max()) + 1

        return 1

    except Exception as e:
        logger.error(f"Error determining current week: {e}")
        return None


def update_accuracy(week=None, book="DraftKings"):
    """
    Update accuracy tracking for completed games.

    For regular season: Updates accuracy for week-1, but only if that week is fully completed.
    For postseason: Updates accuracy for all completed postseason games in the prediction file.
    """
    if week is None:
        week = get_current_week()
        if week is None:
            logger.info("Could not determine current week, skipping accuracy update")
            return True

    # Handle postseason separately
    if isinstance(week, str) and week.lower() == 'postseason':
        logger.info("Updating postseason accuracy for completed games...")
        python_args = ["-m", "cfb_predictor.cli", "update-accuracy",
                       "--season", str(CURRENT_SEASON),
                       "--week", "postseason",
                       "--book", book]
        return run_python_command(python_args, cwd=SRC_DIR)

    if week <= 1:
        logger.info("Skipping accuracy update - week <= 1, no previous week to evaluate")
        return True

    prev_week = week - 1

    # Verify the previous week is fully completed before updating accuracy
    try:
        games_file = f"{SRC_DIR}/data/raw/games_{CURRENT_SEASON}.parquet"
        if os.path.exists(games_file):
            games = pd.read_parquet(games_file)
            prev_week_games = games[games['week'] == prev_week]

            if len(prev_week_games) > 0:
                completed = prev_week_games[pd.notna(prev_week_games['home_points'])]
                completion_pct = len(completed) / len(prev_week_games)

                if completion_pct < 1.0:
                    logger.info(f"Week {prev_week} is not fully completed ({len(completed)}/{len(prev_week_games)} games). Skipping accuracy update.")
                    return True

                logger.info(f"Week {prev_week} is fully completed ({len(completed)} games). Updating accuracy...")
    except Exception as e:
        logger.warning(f"Could not verify week completion: {e}. Proceeding with accuracy update...")

    logger.info(f"Updating accuracy for completed week {prev_week} of {CURRENT_SEASON} season...")
    python_args = ["-m", "cfb_predictor.cli", "update-accuracy",
                   "--season", str(CURRENT_SEASON),
                   "--week", str(prev_week),
                   "--book", book]
    return run_python_command(python_args, cwd=SRC_DIR)


def generate_predictions(week=None, book="DraftKings", min_edge=0.5):
    """Generate predictions for upcoming games"""
    week_display = week if week else 'auto'
    logger.info(f"Generating predictions for {week_display} of {CURRENT_SEASON} season...")

    # Convert week to string for CLI - handles both int and "postseason"
    if week is None:
        week_arg = "auto"
    elif isinstance(week, str):
        week_arg = week
    else:
        week_arg = str(week)

    python_args = ["-m", "cfb_predictor.cli", "predict",
                   "--season", str(CURRENT_SEASON),
                   "--week", week_arg,
                   "--book", book,
                   "--min-edge", str(min_edge)]
    return run_python_command(python_args, cwd=SRC_DIR)


def send_email_picks(week=None, book="DraftKings"):
    """
    Thin wrapper that delegates the email generation/sending to email_renderer.py
    """
    try:
        from email_renderer import send_email_picks as _send
    except Exception as e:
        logger.error(f"Failed to import email_renderer.send_email_picks: {e}")
        return False

    return _send(week=week, book=book, CURRENT_SEASON=CURRENT_SEASON, SRC_DIR=SRC_DIR)


def main():
    parser = argparse.ArgumentParser(description="Weekly NCAAF model update and prediction")
    parser.add_argument("--week", type=int, help="Specific week to predict (default: auto)")
    parser.add_argument("--book", default="DraftKings", help="Sportsbook for lines (default: ESPN Bet)")
    parser.add_argument("--min-edge", type=float, default=0.5, help="Minimum edge for betting (default: 0.5)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching latest data")
    parser.add_argument("--skip-train", action="store_true", help="Skip retraining models")
    parser.add_argument("--skip-predict", action="store_true", help="Skip generating predictions")
    parser.add_argument("--skip-accuracy", action="store_true", help="Skip accuracy update")
    parser.add_argument("--skip-email", action="store_true", help="Skip email notification")

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info(f"Weekly NCAAF Model Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    success = True

    # Step 1: Fetch latest data
    if not args.skip_fetch:
        if not fetch_latest_data():
            logger.error("Failed to fetch latest data")
            success = False
    else:
        logger.info("Skipping data fetch")

    # Determine the week if not specified
    week = args.week
    if week is None and success:
        week = get_current_week()
        if week is not None:
            logger.info(f"Auto-determined current week: {week}")
        else:
            logger.warning("Could not auto-determine week, will use 'auto' mode for predictions")

    # Step 2: Update accuracy for previous week (if week specified and > 1)
    if not args.skip_accuracy and success:
        if not update_accuracy(week, args.book):
            logger.warning("Failed to update accuracy (non-fatal)")
    else:
        logger.info("Skipping accuracy update")

    # Step 3: Retrain models
    if not args.skip_train and success:
        if not retrain_models():
            logger.error("Failed to retrain models")
            success = False
    else:
        if args.skip_train:
            logger.info("Skipping model training")

    # Step 4: Generate predictions
    if not args.skip_predict and success:
        if not generate_predictions(week, args.book, args.min_edge):
            logger.error("Failed to generate predictions")
            success = False
    else:
        if args.skip_predict:
            logger.info("Skipping prediction generation")

    # Step 5: Send email notification (refactored)
    if not args.skip_email and success:
        if not send_email_picks(week, args.book):
            logger.warning("Failed to send email notification (non-fatal)")
    else:
        if args.skip_email:
            logger.info("Skipping email notification")

    if success:
        logger.info("Weekly update completed successfully!")
        logger.info("Check data/processed/predictions/ for the latest predictions")
        logger.info("Check data/processed/weekly_accuracy.csv for accuracy tracking")
    else:
        logger.error("Weekly update failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
