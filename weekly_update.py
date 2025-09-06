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
from datetime import datetime
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

def update_accuracy(week=None, book="DraftKings"):
    """Update accuracy tracking for the previous completed week"""
    if week is None or week <= 1:
        logger.info("Skipping accuracy update - no previous week or week <= 1")
        return True

    prev_week = week - 1
    logger.info(f"Updating accuracy for completed week {prev_week} of {CURRENT_SEASON} season...")
    python_args = ["-m", "cfb_predictor.cli", "update-accuracy",
                   "--season", str(CURRENT_SEASON),
                   "--week", str(prev_week),
                   "--book", book]
    return run_python_command(python_args, cwd=SRC_DIR)

def generate_predictions(week=None, book="DraftKings", min_edge=0.5):
    """Generate predictions for upcoming games"""
    logger.info(f"Generating predictions for week {week or 'auto'} of {CURRENT_SEASON} season...")
    week_arg = str(week) if week else "auto"
    python_args = ["-m", "cfb_predictor.cli", "predict",
                   "--season", str(CURRENT_SEASON),
                   "--week", week_arg,
                   "--book", book,
                   "--min-edge", str(min_edge)]
    return run_python_command(python_args, cwd=SRC_DIR)

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

        # Locate predictions directory/file
        predictions_dir = "src/data/processed/predictions"
        if not os.path.exists(predictions_dir):
            predictions_dir = "data/processed/predictions"
        if not os.path.exists(predictions_dir):
            logger.warning("No predictions directory found for email.")
            return True

        if week:
            target_file = f"predictions_{CURRENT_SEASON}_wk{week}.csv"
            pred_path = os.path.join(predictions_dir, target_file)
            if os.path.exists(pred_path):
                latest_pred_file = target_file
            else:
                logger.warning(f"No prediction file found for week {week}.")
                return True
        else:
            pred_files = [f for f in os.listdir(predictions_dir)
                          if f.startswith(f"predictions_{CURRENT_SEASON}")]
            if not pred_files:
                logger.warning("No prediction files found for email.")
                return True
            pred_files_with_time = [(f, os.path.getmtime(os.path.join(predictions_dir, f)))
                                    for f in pred_files]
            latest_pred_file = sorted(pred_files_with_time, key=lambda x: x[1])[-1][0]
            pred_path = os.path.join(predictions_dir, latest_pred_file)

        if 'pred_path' not in locals():
            pred_path = os.path.join(predictions_dir, latest_pred_file)

        # Load predictions
        predictions = pd.read_csv(pred_path)
        week_num = int(predictions['week'].iloc[0]) if not predictions.empty else (week or 0)

        # Accuracy summary (if available)
        accuracy_file = "src/data/processed/weekly_accuracy.csv"
        if not os.path.exists(accuracy_file):
            accuracy_file = "data/processed/weekly_accuracy.csv"

        accuracy_summary = ""
        if os.path.exists(accuracy_file):
            acc_df = pd.read_csv(accuracy_file)
            if not acc_df.empty:
                current_season_data = acc_df[acc_df['season'] == CURRENT_SEASON]
                if not current_season_data.empty:
                    total_ats_wins = current_season_data['ats_wins'].sum()
                    total_ats_losses = current_season_data['ats_losses'].sum()
                    total_ats_pushes = current_season_data['ats_pushes'].sum()
                    total_ou_wins = current_season_data['ou_wins'].sum()
                    total_ou_losses = current_season_data['ou_losses'].sum()
                    total_ou_pushes = current_season_data['ou_pushes'].sum()

                    total_ats_pct = (total_ats_wins / (total_ats_wins + total_ats_losses)
                                     if (total_ats_wins + total_ats_losses) > 0 else 0)
                    total_ou_pct = (total_ou_wins / (total_ou_wins + total_ou_losses)
                                    if (total_ou_wins + total_ou_losses) > 0 else 0)

                    last_week_summary = ""
                    if week_num > 1:
                        last_week_data = current_season_data[current_season_data['week'] == week_num - 1]
                        if not last_week_data.empty:
                            last_week = last_week_data.iloc[0]
                            last_week_summary = f"""
📈 LAST WEEK PERFORMANCE (Week {week_num - 1}):
- ATS: {last_week['ats_wins']}-{last_week['ats_losses']}-{last_week['ats_pushes']} ({last_week['ats_win_pct']:.1%})
- O/U: {last_week['ou_wins']}-{last_week['ou_losses']}-{last_week['ou_pushes']} ({last_week['ou_win_pct']:.1%})

"""

                    accuracy_summary = f"""{last_week_summary}📊 {CURRENT_SEASON} SEASON TOTALS:
- ATS Record: {total_ats_wins}-{total_ats_losses}-{total_ats_pushes} ({total_ats_pct:.1%})
- O/U Record: {total_ou_wins}-{total_ou_losses}-{total_ou_pushes} ({total_ou_pct:.1%})
"""
                else:
                    recent_acc = acc_df.tail(1).iloc[0]
                    accuracy_summary = f"""
📊 RECENT PERFORMANCE:
- ATS Record: {recent_acc['ats_wins']}-{recent_acc['ats_losses']}-{recent_acc['ats_pushes']} ({recent_acc['ats_win_pct']:.1%})
- O/U Record: {recent_acc['ou_wins']}-{recent_acc['ou_losses']}-{recent_acc['ou_pushes']} ({recent_acc['ou_win_pct']:.1%})
"""

        # Filter to actual picks
        ats_picks = predictions[predictions['pick_spread'] != 'no bet'].copy()
        ats_picks['abs_edge_spread'] = ats_picks['edge_spread'].abs()
        ats_picks = ats_picks.sort_values('abs_edge_spread', ascending=False)

        ou_picks = predictions[predictions['pick_total'] != 'no bet'].copy()
        ou_picks['abs_edge_total'] = ou_picks['edge_total'].abs()
        ou_picks = ou_picks.sort_values('abs_edge_total', ascending=False)

        subject = f"🏈 NCAAF Week {week_num} Picks - {CURRENT_SEASON} Season"

        # ---------------- Plain text version ----------------
        body = f"""
═══════════════════════════════════════════════════════════════
🏈 NCAAF WEEKLY PICKS - Week {week_num} ({CURRENT_SEASON} Season)
═══════════════════════════════════════════════════════════════

📅 Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
📊 Sportsbook: {book}
📈 Minimum Edge: 0.5 points

{accuracy_summary}

🎯 TOP AGAINST THE SPREAD PICKS ({len(ats_picks)} total):
{"="*65}
"""
        for i, (_, game) in enumerate(ats_picks.iterrows(), 1):
            edge = abs(game['edge_spread'])
            game_time = pd.to_datetime(game['start_date']).strftime('%a %m/%d %I:%M%p ET')
            pred_margin = game.get('pred_margin', 0.0)
            spread = game['spread_line']

            if "+" in game['pick_spread']:
                pick_team = game['pick_spread'].replace(' + ATS', '')
                spread_display = f"+{abs(spread)}"
            else:
                pick_team = game['pick_spread'].replace(' - ATS', '')
                spread_display = f"{spread}"

            body += f"""
{i:2d}. {game['away_team']} @ {game['home_team']}
    🕒 {game_time}
    🎯 PICK: {pick_team} ({spread_display})
    📊 Predicted Margin (home): {pred_margin:+.1f}
    💰 Edge: {edge:.1f} points
"""

        body += f"""
{"="*65}

🎲 TOP OVER/UNDER PICKS ({len(ou_picks)} total):
{"="*65}
"""
        for i, (_, game) in enumerate(ou_picks.iterrows(), 1):
            edge = abs(game['edge_total'])
            game_time = pd.to_datetime(game['start_date']).strftime('%a %m/%d %I:%M%p ET')
            pred_total = float(game['pred_total'])
            total_line = game['total_line']
            pick = game['pick_total']

            body += f"""
{i:2d}. {game['away_team']} @ {game['home_team']}
    🕒 {game_time}
    🎯 PICK: {pick} {total_line}
    📊 Predicted Total: {pred_total:.1f}
    💰 Edge: {edge:.1f} points
"""

        body += f"""
{"="*65}

📂 ADDITIONAL INFO:
• Full picks file: {latest_pred_file}
• Training Data: 2020-{CURRENT_SEASON-1} seasons + completed {CURRENT_SEASON} games
• All times shown in Eastern Time

💡 BETTING NOTES:
• Predicted Line is from the HOME team's perspective.
• Edge = |Predicted Value - Book Line|
• Only picks with minimum 0.5 point edge are shown
• ATS = Against The Spread

═══════════════════════════════════════════════════════════════
🤖 Generated by NCAAF Predictor
═══════════════════════════════════════════════════════════════
"""

        # ---------------- HTML version ----------------
        # Helpers for ET conversion and windowed grouping
        import pytz
        eastern = pytz.timezone("US/Eastern")

        def to_et(df: pd.DataFrame, start_col: str = "start_date") -> pd.DataFrame:
            out = df.copy()
            # robust to naive/aware datetimes; treat as UTC then convert
            start_dt = pd.to_datetime(out[start_col], utc=True)
            out["start_date_et"] = start_dt.dt.tz_convert(eastern)
            out["date_only"] = out["start_date_et"].dt.date
            out["game_time"] = out["start_date_et"].dt.strftime("%I:%M %p ET")
            return out

        def window_group(df: pd.DataFrame, minutes: int = 60):
            """
            Groups by date, then by floor(window) of ET time.
            Returns list of tuples: (date_str, time_range, block_df_sorted)
            """
            df = df.sort_values("start_date_et")
            window_start = df["start_date_et"].dt.floor(f"{minutes}min")
            df = df.assign(_window_start=window_start)

            groups = []
            for (d, wstart), block in df.groupby(["date_only", "_window_start"]):
                block = block.copy().sort_values(["start_date_et"])
                first_time = block.iloc[0]["game_time"]
                last_time = block.iloc[-1]["game_time"]
                time_range = first_time if first_time == last_time else f"{first_time} - {last_time}"
                date_str = pd.Timestamp(d).strftime("%a, %b %d")
                groups.append((date_str, time_range, block))
            return groups

        def section_header(title: str, count: int) -> str:
            return f"""
                <h2 style="color:#2c5aa0;border-bottom:3px solid #2c5aa0;padding-bottom:10px;margin:30px 0 25px 0;font-size:24px;">
                    {title} ({count} total)
                </h2>
            """

        def block_header(date_str: str, time_range: str) -> str:
            return f"""
                <h3 style="color:#2c5aa0;font-size:20px;margin:25px 0 15px 0;border-left:4px solid #2c5aa0;padding-left:15px;">
                    {date_str} • {time_range}
                </h3>
            """

        def game_card_row(left_label: str, matchup: str,
                          mid_label: str, right_label: str,
                          right_value: str, kickoff_html: str, edge: float,
                          highlight_threshold: float = 2.0) -> str:
            highlight = edge >= highlight_threshold
            pick_bg    = "#28a745" if highlight else "#f8f9fa"
            pick_color = "white"   if highlight else "#333333"
            edge_bg    = pick_bg
            edge_color = pick_color
            return f"""
                <table role="presentation" style="width:100%;border-collapse:collapse;background-color:#ffffff;border:1px solid #e9ecef;border-left:4px solid #2c5aa0;margin:15px 0;">
                    <tr>
                        <td style="padding:20px;color:#333333;">
                            <table role="presentation" style="width:100%;border-collapse:collapse;margin-bottom:15px;">
                                <tr>
                                    <td style="font-size:18px;font-weight:bold;color:#2c5aa0;">{matchup}</td>
                                    <td style="text-align:right;color:#6c757d;font-size:14px;">{kickoff_html}</td>
                                </tr>
                            </table>
                            <table role="presentation" style="width:100%;border-collapse:collapse;">
                                <tr>
                                    <td style="text-align:center;padding:10px;background-color:{pick_bg};color:{pick_color};width:33.33%;">
                                        <div style="font-size:13px;margin-bottom:5px;">{left_label}</div>
                                        <div style="font-weight:bold;">{right_label}</div>
                                    </td>
                                    <td style="text-align:center;padding:10px;background-color:#f8f9fa;color:#333333;width:33.33%;">
                                        <div style="font-size:13px;color:#6c757d;margin-bottom:5px;">{mid_label}</div>
                                        <div style="font-weight:bold;color:#2c5aa0;">{right_value}</div>
                                    </td>
                                    <td style="text-align:center;padding:10px;background-color:{edge_bg};color:{edge_color};width:33.33%;">
                                        <div style="font-size:13px;margin-bottom:5px;">💰 Edge</div>
                                        <div style="font-weight:bold;">{edge:.1f} pts</div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                </table>
            """

        # Header + Info boxes
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NCAAF Week {week_num} Picks</title>
    <!--[if mso]>
    <style type="text/css">
        table {{ border-collapse: collapse; }}
        .header-table {{ width: 100%; }}
        .content-table {{ width: 100%; }}
    </style>
    <![endif]-->
</head>
<body style="margin:0;padding:0;font-family:Arial,sans-serif;line-height:1.6;color:#333333;background-color:#f8f9fa;">
    <table role="presentation" style="width:100%;max-width:800px;margin:0 auto;border-collapse:collapse;">
        <tr>
            <td style="background-color:#2c5aa0;color:white;text-align:center;padding:30px;">
                <h1 style="margin:0;font-size:28px;font-weight:bold;">🏈 NCAAF Weekly Picks</h1>
                <div style="margin-top:10px;font-size:18px;">Week {week_num} • {CURRENT_SEASON} Season</div>
            </td>
        </tr>
    </table>

    <table role="presentation" style="width:100%;max-width:800px;margin:0 auto;border-collapse:collapse;background-color:#ffffff;">
        <tr>
            <td style="padding:30px;color:#333333;">

                <table role="presentation" style="width:100%;border-collapse:collapse;background-color:#f8f9fa;border-left:4px solid #2c5aa0;margin:20px 0;">
                    <tr>
                        <td style="padding:20px;">
                            <table role="presentation" style="width:100%;border-collapse:collapse;">
                                <tr>
                                    <td style="text-align:center;padding:10px;width:33.33%;">
                                        <strong style="display:block;color:#2c5aa0;font-size:16px;">📅 Generated</strong>
                                        <span style="color:#333333;">{datetime.now().strftime('%B %d, %Y')}</span>
                                    </td>
                                    <td style="text-align:center;padding:10px;width:33.33%;">
                                        <strong style="display:block;color:#2c5aa0;font-size:16px;">📊 Sportsbook</strong>
                                        <span style="color:#333333;">{book}</span>
                                    </td>
                                    <td style="text-align:center;padding:10px;width:33.33%;">
                                        <strong style="display:block;color:#2c5aa0;font-size:16px;">📈 Min Edge</strong>
                                        <span style="color:#333333;">0.5 points</span>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                </table>
"""

        # Performance section
        if accuracy_summary.strip():
            html_body += f"""
                <table role="presentation" style="width:100%;border-collapse:collapse;background-color:#28a745;margin:25px 0;">
                    <tr>
                        <td style="color:white;padding:25px;">
                            <h3 style="margin:0 0 15px 0;font-size:20px;color:white;">📊 Performance Summary</h3>
                            <div style="white-space:pre-line;font-family:monospace;color:white;">{accuracy_summary.strip()}</div>
                        </td>
                    </tr>
                </table>
"""

        # ---------- ATS SECTION ----------
        html_body += section_header("🎯 Against The Spread Picks", len(ats_picks))
        ats_et = to_et(ats_picks)
        ats_et["edge_abs"] = ats_et["edge_spread"].abs()

        for d_str, time_range, block in window_group(ats_et, minutes=60):
            block_sorted = block.sort_values(["start_date_et", "edge_abs"], ascending=[True, False])
            html_body += block_header(d_str, time_range)
            for _, game in block_sorted.iterrows():
                edge = float(abs(game["edge_spread"]))
                spread = game["spread_line"]
                # Predicted line from HOME perspective; fallback to pred_margin if needed
                pred_home = float(game.get("pred_line", game.get("pred_margin", 0.0)))
                pred_home_disp = f"{game['home_team']} {pred_home:+.1f}"

                if "+" in game["pick_spread"]:
                    pick_team = game["pick_spread"].replace(" + ATS", "")
                    spread_disp = f"+{abs(spread)}"
                else:
                    pick_team = game["pick_spread"].replace(" - ATS", "")
                    spread_disp = f"{spread}"

                html_body += game_card_row(
                    left_label="🎯 PICK",
                    matchup=f"{game['away_team']} @ {game['home_team']}",
                    mid_label="📊 Predicted Line (home)",
                    right_label=f"{pick_team} ({spread_disp})",
                    right_value=pred_home_disp,
                    kickoff_html=game["game_time"],
                    edge=edge
                )

        # ---------- O/U SECTION ----------
        html_body += section_header("🎲 Over/Under Picks", len(ou_picks))
        ou_et = to_et(ou_picks)
        ou_et["edge_abs"] = ou_et["edge_total"].abs()

        for d_str, time_range, block in window_group(ou_et, minutes=60):
            block_sorted = block.sort_values(["start_date_et", "edge_abs"], ascending=[True, False])
            html_body += block_header(d_str, time_range)
            for _, game in block_sorted.iterrows():
                edge = float(abs(game["edge_total"]))
                pick = game["pick_total"]
                total_line = game["total_line"]
                pred_total = float(game["pred_total"])
                html_body += game_card_row(
                    left_label="🎯 PICK",
                    matchup=f"{game['away_team']} @ {game['home_team']}",
                    mid_label="📊 Predicted Total",
                    right_label=f"{pick} {total_line}",
                    right_value=f"{pred_total:.1f}",
                    kickoff_html=game["game_time"],
                    edge=edge
                )

        # ---------- Notes / legend ----------
        html_body += f"""
                <table role="presentation" style="width:100%;border-collapse:collapse;background-color:#fff3cd;border:1px solid #ffeaa7;margin:25px 0;">
                    <tr>
                        <td style="color:#856404;padding:20px;">
                            <h4 style="margin:0 0 10px 0;color:#856404;">💡 Betting Notes</h4>
                            <ul style="margin:10px 0;padding-left:20px;color:#856404;">
                                <li><strong>Predicted Line (home)</strong> is from the <em>home team's</em> perspective (e.g., "HomeTeam +3.5").</li>
                                <li><strong>Edge</strong> = |Model Prediction − Book Line|.</li>
                                <li>Only picks with minimum 0.5 point edge are shown.</li>
                                <li>ATS = Against The Spread. All times Eastern.</li>
                                <li>Training data: 2020–{CURRENT_SEASON-1} + completed {CURRENT_SEASON} games.</li>
                            </ul>
                        </td>
                    </tr>
                </table>

            </td>
        </tr>
    </table>

    <table role="presentation" style="width:100%;max-width:800px;margin:0 auto;border-collapse:collapse;">
        <tr>
            <td style="background-color:#2c5aa0;color:white;text-align:center;padding:25px;">
                <div>🤖 Generated by NCAAF Predictor</div>
            </td>
        </tr>
    </table>
</body>
</html>
"""

        # Send email
        msg = MIMEMultipart('alternative')
        msg['From'] = email_from
        msg['To'] = email_to
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

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
    parser.add_argument("--book", default="ESPN Bet", help="Sportsbook for lines (default: ESPN Bet)")
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
