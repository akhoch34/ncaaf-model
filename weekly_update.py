#!/usr/bin/env python3
"""
Weekly NCAAF Model Update Script
================================

This script automates the weekly process of:
1. Fetching the latest game data for the current season (2025)
2. Retraining models on all available data (2020-2024 + completed 2025 games)
3. Generating predictions for upcoming games

Usage:
    python weekly_update.py [--week WEEK] [--book BOOK] [--min-edge EDGE]

The script should be run weekly, ideally on Tuesday mornings after Monday's games
are finalized in the CFBD database.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime, timedelta
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd

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
        logger.info(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def activate_venv_cmd():
    """Return command prefix to activate virtual environment"""
    if os.name == 'nt':  # Windows
        return ["cmd", "/c", ".venv\\Scripts\\activate", "&&"]
    else:  # Unix/Linux
        return ["bash", "-c", "source .venv/bin/activate &&"]

def fetch_latest_data():
    """Fetch the latest game data for current season"""
    logger.info(f"Fetching latest data for {CURRENT_SEASON} season...")
    
    # Activate venv and fetch current season data
    if os.name == 'nt':
        cmd = ["cmd", "/c", ".venv\\Scripts\\activate", "&&", "cd", SRC_DIR, "&&", 
               "python", "-m", "cfb_predictor.cli", "fetch-data", str(CURRENT_SEASON)]
    else:
        cmd = ["bash", "-c", f"cd {os.getcwd()} && source .venv/bin/activate && cd {SRC_DIR} && python -m cfb_predictor.cli fetch-data {CURRENT_SEASON}"]
    
    return run_command(cmd)

def retrain_models():
    """Retrain models on all available data"""
    logger.info("Retraining models on all available data...")
    
    # Include current season in training if it has completed games
    all_seasons = TRAINING_SEASONS + [CURRENT_SEASON]
    season_args = [str(s) for s in all_seasons]
    
    if os.name == 'nt':
        cmd = ["cmd", "/c", ".venv\\Scripts\\activate", "&&", "cd", SRC_DIR, "&&", 
               "python", "-m", "cfb_predictor.cli", "train"] + season_args
    else:
        cmd = ["bash", "-c", f"cd {os.getcwd()} && source .venv/bin/activate && cd {SRC_DIR} && python -m cfb_predictor.cli train {' '.join(season_args)}"]
    
    return run_command(cmd)

def update_accuracy(week=None, book="DraftKings"):
    """Update accuracy tracking for the previous completed week"""
    if week is None or week <= 1:
        logger.info("Skipping accuracy update - no previous week or week <= 1")
        return True
        
    prev_week = week - 1
    logger.info(f"Updating accuracy for completed week {prev_week} of {CURRENT_SEASON} season...")
    
    if os.name == 'nt':
        cmd = ["cmd", "/c", ".venv\\Scripts\\activate", "&&", "cd", SRC_DIR, "&&", 
               "python", "-m", "cfb_predictor.cli", "update-accuracy", 
               "--season", str(CURRENT_SEASON),
               "--week", str(prev_week),
               "--book", book]
    else:
        cmd = ["bash", "-c", f"cd {os.getcwd()} && source .venv/bin/activate && cd {SRC_DIR} && python -m cfb_predictor.cli update-accuracy --season {CURRENT_SEASON} --week {prev_week} --book {book}"]
    
    return run_command(cmd)

def generate_predictions(week=None, book="DraftKings", min_edge=0.5):
    """Generate predictions for upcoming games"""
    logger.info(f"Generating predictions for week {week or 'auto'} of {CURRENT_SEASON} season...")
    
    week_arg = str(week) if week else "auto"
    
    if os.name == 'nt':
        cmd = ["cmd", "/c", ".venv\\Scripts\\activate", "&&", "cd", SRC_DIR, "&&", 
               "python", "-m", "cfb_predictor.cli", "predict", 
               "--season", str(CURRENT_SEASON),
               "--week", week_arg,
               "--book", book,
               "--min-edge", str(min_edge)]
    else:
        cmd = ["bash", "-c", f"cd {os.getcwd()} && source .venv/bin/activate && cd {SRC_DIR} && python -m cfb_predictor.cli predict --season {CURRENT_SEASON} --week {week_arg} --book {book} --min-edge {min_edge}"]
    
    return run_command(cmd)

def send_email_picks(week=None, book="DraftKings"):
    """Send email with weekly picks and performance summary"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        email_from = os.getenv("EMAIL_FROM", "akhoch54@gmail.com")
        email_to = os.getenv("EMAIL_TO")
        
        if not gmail_password or not email_to:
            logger.warning("Email credentials not found. Skipping email notification.")
            return True
            
        # Find prediction file for specific week
        predictions_dir = "src/data/processed/predictions"  # Check src directory first
        if not os.path.exists(predictions_dir):
            predictions_dir = "data/processed/predictions"
        
        if not os.path.exists(predictions_dir):
            logger.warning("No predictions directory found for email.")
            return True
        
        # If week is specified, look for that specific week's file
        if week:
            target_file = f"predictions_{CURRENT_SEASON}_wk{week}.csv"
            pred_path = os.path.join(predictions_dir, target_file)
            if os.path.exists(pred_path):
                latest_pred_file = target_file
            else:
                logger.warning(f"No prediction file found for week {week}.")
                return True
        else:
            # Find latest prediction file by modification time
            pred_files = [f for f in os.listdir(predictions_dir) if f.startswith(f"predictions_{CURRENT_SEASON}")]
            if not pred_files:
                logger.warning("No prediction files found for email.")
                return True
            # Sort by modification time, not alphabetically    
            pred_files_with_time = [(f, os.path.getmtime(os.path.join(predictions_dir, f))) for f in pred_files]
            latest_pred_file = sorted(pred_files_with_time, key=lambda x: x[1])[-1][0]
            pred_path = os.path.join(predictions_dir, latest_pred_file)
        
        if not 'pred_path' in locals():
            pred_path = os.path.join(predictions_dir, latest_pred_file)
        
        # Load predictions
        predictions = pd.read_csv(pred_path)
        week_num = predictions['week'].iloc[0] if not predictions.empty else week or "auto"
        
        # Get accuracy data
        accuracy_file = "src/data/processed/weekly_accuracy.csv"  # Check src directory first
        if not os.path.exists(accuracy_file):
            accuracy_file = "data/processed/weekly_accuracy.csv"
            
        accuracy_summary = ""
        if os.path.exists(accuracy_file):
            acc_df = pd.read_csv(accuracy_file)
            if not acc_df.empty:
                recent_acc = acc_df.tail(1).iloc[0]
                accuracy_summary = f"""
📊 RECENT PERFORMANCE:
- ATS Record: {recent_acc['ats_wins']}-{recent_acc['ats_losses']}-{recent_acc['ats_pushes']} ({recent_acc['ats_win_pct']:.1%})
- O/U Record: {recent_acc['ou_wins']}-{recent_acc['ou_losses']}-{recent_acc['ou_pushes']} ({recent_acc['ou_win_pct']:.1%})
"""
        
        # Filter for actual picks (not "no bet")
        ats_picks = predictions[predictions['pick_spread'] != 'no bet']
        ou_picks = predictions[predictions['pick_total'] != 'no bet']
        
        # Create email content
        subject = f"🏈 NCAAF Week {week_num} Picks - {CURRENT_SEASON} Season"
        
        body = f"""🏈 NCAAF WEEKLY PICKS - Week {week_num}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

{accuracy_summary}

🎯 THIS WEEK'S TOP PICKS:

AGAINST THE SPREAD ({len(ats_picks)} picks):
"""
        
        # Add top ATS picks
        for _, game in ats_picks.head(10).iterrows():
            edge = abs(game['edge_spread'])
            body += f"• {game['pick_spread']} (Edge: {edge:.1f})\n"
        
        body += f"\nOVER/UNDER ({len(ou_picks)} picks):\n"
        
        # Add top O/U picks  
        for _, game in ou_picks.head(10).iterrows():
            teams = f"{game['away_team']} @ {game['home_team']}"
            edge = abs(game['edge_total'])
            body += f"• {teams} {game['pick_total']} {game['total_line']} (Edge: {edge:.1f})\n"
        
        body += f"""
📁 Full picks available in: {latest_pred_file}

🤖 MODEL INFO:
- Training Data: 2020-{CURRENT_SEASON-1} seasons + completed {CURRENT_SEASON} games
- Sportsbook: {book}
- Minimum Edge: 0.5 points

---
Generated by NCAAF Predictor
https://github.com/your-username/ncaaf-model
"""
        
        # Send email
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email_to  
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_from, gmail_password)
        server.sendmail(msg['From'], email_to, msg.as_string())
        server.quit()
        
        logger.info(f"Email sent successfully to {email_to}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Weekly NCAAF model update and prediction")
    parser.add_argument("--week", type=int, help="Specific week to predict (default: auto)")
    parser.add_argument("--book", default="DraftKings", help="Sportsbook for lines (default: DraftKings)")
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
    
    # Step 2: Update accuracy for previous week
    if not args.skip_accuracy and success:
        if not update_accuracy(args.week, args.book):
            logger.warning("Failed to update accuracy (non-fatal)")
            # Don't mark as failure since this is informational
    else:
        if args.skip_accuracy:
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
        if not generate_predictions(args.week, args.book, args.min_edge):
            logger.error("Failed to generate predictions")
            success = False
    else:
        if args.skip_predict:
            logger.info("Skipping prediction generation")
    
    # Step 5: Send email notification
    if not args.skip_email and success:
        if not send_email_picks(args.week, args.book):
            logger.warning("Failed to send email notification (non-fatal)")
            # Don't mark as failure since this is optional
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