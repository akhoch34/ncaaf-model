# CFB Predictor — Winners, Spread, and Totals

A Python project to predict FBS (Division I) college football **winners**, **spread edges**, and **totals (over/under)** using a transparent, reproducible pipeline with automated weekly updates and email notifications.

## 🎯 Performance Results (2025 Season - Live)
- **Against The Spread (ATS)**: **57.4%** win rate (507-377-0 record)
- **Over/Under (O/U)**: **53.3%** win rate (447-391-0 record)
- **ROI**: +9.8% (ATS) | +1.8% (O/U) on flat betting
- **Above Breakeven**: +5.1pp (ATS) | +1.0pp (O/U)
- **Training Data**: 2020-2025 seasons with continuous model updates

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
- Engineers **19 features for totals** and **11 features for spreads** (see Model Architecture below).
- Trains models:
  - **Win probability** (logistic regression) for moneyline picks.
  - **Point spread (margin)** (linear regression) for ATS picks.
  - **Game total points** (linear regression) for O/U picks.
- Compares predictions to market **spread** and **total** to produce picks with calculated edges.
- Automatically handles **regular season** and **postseason** games differently.
- Sends **HTML email notifications** with picks, performance tracking, and team logos.

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

# For postseason/bowl games:
python -m cfb_predictor.cli predict --season 2025 --week postseason --book DraftKings
```

3) **View Predictions**:
   - Regular season: `data/processed/predictions/predictions_2025_wkX.csv`
   - Postseason: `data/processed/predictions/predictions_2025_postseason.csv`

4) **Track Performance**:
```bash
cd src
python -m cfb_predictor.cli show-accuracy --season 2025
python -m cfb_predictor.cli update-accuracy --season 2025 --week postseason --book DraftKings
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

## 🏗️ Model Architecture

### Total (O/U) Model - 19 Features
**Linear Regression** predicting total points scored in a game.

**Conference Features** (Most Important):
- `home_conf_scoring_factor` - Conference scoring tendency (Big 12: 1.15, Big Ten: 0.90)
- `away_conf_scoring_factor` - Away team conference scoring factor
- `is_conference_game` - In-conference games tend to score less

**Combined Team Statistics:**
- `expected_combined_total` - Sum of recent team scoring averages
- `expected_combined_defense` - Sum of recent points allowed
- `home_off_vs_away_def` - Home offense vs away defense matchup
- `away_off_vs_home_def` - Away offense vs home defense matchup

**Game Total History:**
- `home_game_total_roll3` - Rolling 3-game average of total points in home team's games
- `away_game_total_roll3` - Rolling 3-game average of total points in away team's games
- `combined_game_total_history` - Average of both teams' recent game totals

**Scoring Trends:**
- `home_pf_roll3` - Home team points scored (last 3 games)
- `home_pa_roll3` - Home team points allowed (last 3 games)
- `away_pf_roll3` - Away team points scored (last 3 games)
- `away_pa_roll3` - Away team points allowed (last 3 games)

**Timing & Context:**
- `week_num` - Week number in season
- `is_early_season` - Early season indicator (weeks 1-4, higher variance)
- `is_late_season` - Late season indicator (weeks 11+)
- `elo_diff` - Team strength differential
- `is_neutral` - Neutral site indicator

**Performance:** Training MAE = 13.43 points

### ATS (Spread) Model - 11 Features
**Linear Regression** predicting point differential (home - away).

**Conference Strength** (Most Important):
- `conf_strength_diff` - Difference in conference strength (SEC: 1.10, MAC: 0.70)

**Postseason Handling:**
- `is_postseason` - Critical flag for bowl games (opt-outs, motivation issues)

**Recent Margin Performance:**
- `home_margin_roll3` - Rolling 3-game average of margins for home team
- `away_margin_roll3` - Rolling 3-game average of margins for away team

**Momentum & Form:**
- `form_differential` - Difference in recent point differentials
- `home_pf_roll3` - Home team recent scoring
- `away_pf_roll3` - Away team recent scoring
- `home_pa_roll3` - Home team recent defense
- `away_pa_roll3` - Away team recent defense

**Core Features:**
- `elo_diff` - Team strength differential
- `is_neutral` - Neutral site adjustment

**Performance:** Training MAE = 14.52 points (regular season) | 13.75 points (postseason)

### Win Probability Model - 6 Features
**Logistic Regression** predicting home team win probability.

- `elo_diff` - Primary predictor of win probability
- `is_neutral` - Home field advantage adjustment
- `home_pf_roll3` - Recent offensive performance
- `home_pa_roll3` - Recent defensive performance
- `away_pf_roll3` - Opponent offensive performance
- `away_pa_roll3` - Opponent defensive performance

## 📧 Email Notifications

The system sends beautiful HTML emails with:
- **Performance Overview**: Season-to-date and recent week statistics
- **All Picks**: Games sorted by date and edge, with team logos
- **High-Value Picks**: Games with ATS edge ≥ 7.0 points highlighted with gold borders
- **Accuracy Tracking**: ATS and O/U win rates, units won/lost
- **Postseason Stats**: Separate tracking for bowl games

Regular season and postseason predictions are handled separately, with postseason games predicted for all remaining bowl games.

## Notes & Assumptions

- Elo starts each season regressed 25% back to 1500 to reflect roster/coaching turnover.
- Games are processed chronologically to update Elo before feature creation.
- Rolling stats (points for/against, margins, totals) use the last 3 games; configurable in code.
- Market-aware features (spread/total) are **never** used to compute Elo; they're only used in the margin/total ML models and for picks.
- Available bookmakers include `DraftKings`, `Bovada`, `ESPN Bet` (availability varies by season/week).
- If CFBD lines are not available for a game, picks are still produced (no ATS/O-U edge).
- **Postseason Handling**: Bowl games are predicted differently due to opt-outs, long layoffs, and motivation factors.

## 🛠️ Recent Updates & Improvements

### December 2025 - Model Feature Engineering Overhaul
- **✅ Total Model Enhancement**: Expanded from 5 to 19 features
  - Added conference scoring factors (Big 12 vs Big Ten context)
  - Implemented game total history tracking
  - Added combined team statistics and matchup analysis
  - Training MAE improved significantly
- **✅ ATS Model Enhancement**: Expanded from 2 to 11 features
  - Added postseason indicator for bowl game handling
  - Implemented conference strength differential
  - Added recent margin history and form indicators
  - Regular season MAE improved by 1.43 points
- **✅ Postseason Support**: Separate handling for bowl games
  - Date-based predictions instead of week-based
  - Postseason-aware features (opt-outs, motivation)
  - Separate accuracy tracking for regular vs postseason
- **✅ Email Improvements**:
  - Removed duplicate "Featured Games" section
  - High-value picks (edge ≥ 7.0) highlighted with gold borders
  - Added postseason performance tracking
  - Fixed badge color contrast

### Previous Updates (2024)
- **✅ CFBD API v5.10.0 Compatibility**
  - Fixed API authentication using `access_token`
  - Updated parameter names (`division` → `classification`)
  - Fixed column mapping (camelCase → snake_case)
  - Corrected betting lines parsing
- **✅ Feature Engineering Fixes**
  - Resolved pandas groupby operations
  - Added missing `pyarrow` dependency
- **✅ Weekly Automation**
  - GitHub Actions workflow for automated updates
  - Email notification system with HTML formatting
  - Accuracy tracking and performance monitoring

The model is fully functional with validated improvements on 2025 season data.

---

MIT License.
