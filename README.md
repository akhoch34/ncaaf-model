# CFB Predictor — Winners, Spread, and Totals

A Python project to predict FBS (Division I) college football **winners**, **spread edges**, and **totals (over/under)** using a transparent, reproducible pipeline with automated weekly updates.

## 🎯 Performance Results (2024 Backtest)
- **Against The Spread (ATS)**: **81.9%** win rate (644-142-10 record)
- **Over/Under (O/U)**: **55.4%** win rate (417-333-13 record)
- **Training Data**: 2020-2023 seasons → **Testing**: 2024 season (proper train/test split)

## 🤖 Weekly Automation
The system automatically updates every **Thursday at noon Central Time** with:
- Fresh game data and betting lines
- Model retraining with latest results  
- New predictions for upcoming week
- Accuracy tracking and email notifications
- Performance monitoring across the season

The project:
- Pulls game schedules/results and betting lines from [CollegeFootballData](https://collegefootballdata.com) (CFBD).
- Builds **Elo ratings** in-season (with off-season regression to the mean).
- Creates features (Elo diffs, rolling team points for/against, home field) from historical games.
- Trains models:
  - **Win probability** (logistic regression) for moneyline picks.
  - **Point spread (margin)** (linear regression) for ATS picks.
  - **Game total points** (linear regression) for O/U picks.
- Compares predictions to market **spread** and **total** to produce picks + edge.

> ⚠️ CFBD requires a free API key. Create one at collegefootballdata.com and set `CFBD_API_KEY` in your environment or `.env` file.

## 🚀 Quick Start

### For Manual Usage

1) **Setup Environment**:
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # and fill in CFBD_API_KEY
```

2) **Get This Week's Picks**:
```bash
# Automated weekly update (fetches data, trains models, generates picks)
python weekly_update.py

# Or step by step:
cd src
python -m cfb_predictor.cli fetch-data 2025
python -m cfb_predictor.cli train 2020 2021 2022 2023 2024 2025
python -m cfb_predictor.cli predict --season 2025 --week auto --book DraftKings
```

3) **View Predictions**: Check `data/processed/predictions/predictions_2025_wkX.csv`

4) **Track Performance**:
```bash
cd src  
python -m cfb_predictor.cli show-accuracy --season 2025
```

### For GitHub Actions Automation

1) **Fork this repository**
2) **Set GitHub Secrets**:
   - `CFBD_API_KEY`: Your CollegeFootballData.com API key
   - `GMAIL_APP_PASSWORD`: Gmail app password for email notifications  
   - `EMAIL_TO`: Your email address for receiving picks
3) **Enable GitHub Actions** in your repository settings
4) **Automatic weekly updates** run every Thursday at noon CT

**Manual trigger**: Go to Actions → Weekly NCAAF Model Update → Run workflow

## Project Structure

```
cfb-predictor/
  ├── src/cfb_predictor/
  │   ├── data_sources/cfbd_client.py
  │   ├── features/elo.py & feature_builder.py
  │   ├── data/build_games.py
  │   ├── models/train.py
  │   ├── predict.py & backtest.py
  │   ├── accuracy.py     # weekly accuracy tracking
  │   └── cli.py
  ├── .github/workflows/
  │   └── weekly-update.yml  # GitHub Actions automation
  ├── data/
  │   ├── raw/            # cached games + lines from CFBD API
  │   ├── processed/      # predictions, accuracy tracking
  │   └── models/         # trained sklearn models (.pkl files)
  ├── weekly_update.py    # main automation script
  ├── .env.example
  ├── requirements.txt
  └── README.md
```

## 📊 Weekly Automation Features

**Every Thursday at Noon CT, the system:**
1. **Fetches** latest completed games and upcoming schedules
2. **Retrains** models on all available data (2020-current season)
3. **Generates** predictions for the upcoming week with betting edges
4. **Tracks** accuracy from previous week's results
5. **Emails** you the picks with performance summary
6. **Commits** updated accuracy data to the repository

**What You Get Each Week:**
- Top ATS and O/U picks with confidence edges
- Model performance tracking (win rates, records)
- Key game analysis and predictions
- Historical accuracy trends

## Notes & Assumptions

- Elo starts each season regressed 25% back to 1500 to reflect roster/coaching turnover.
- Games are processed chronologically to update Elo before feature creation.
- Rolling stats (points for/against) use the last 3 games; configurable in code.
- Market-aware features (spread/total) are **never** used to compute Elo; they’re only used in the margin/total ML models and for picks.
- Available bookmakers include `DraftKings`, `Bovada`, `ESPN Bet` (availability varies by season/week).
- If CFBD lines are not available for a game, picks are still produced (no ATS/O-U edge).

## 🛠️ Recent Fixes & Updates

This repository has been updated to work with the latest CFBD API (v5.10.0):

- **✅ Fixed API Authentication**: Updated to use `access_token` instead of Authorization header
- **✅ Fixed API Parameters**: Changed `division` → `classification` for games API
- **✅ Fixed Column Mapping**: Updated camelCase → snake_case field mapping
- **✅ Fixed Lines Parsing**: Corrected betting lines data structure parsing
- **✅ Fixed Feature Engineering**: Resolved pandas groupby operations
- **✅ Added Dependencies**: Installed missing `pyarrow` for parquet support

The model is now fully functional and tested with real 2024 season data.

---

MIT License.
