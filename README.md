# CFB Predictor — Winners, Spread, and Totals

A Python project to predict FBS (Division I) college football **winners**, **spread edges**, and **totals (over/under)** using a transparent, reproducible pipeline.

## 🎯 Performance Results (2024 Season)
- **Against The Spread (ATS)**: **80.8%** win rate (637/799 bets with 0.5+ edge)
- **Over/Under (O/U)**: **55.4%** win rate (414/761 bets with 0.5+ edge)

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

## Quickstart

1) **Python 3.11** is recommended.
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # and fill in CFBD_API_KEY
```

2) **Fetch and cache raw data** (games + lines) for one or more seasons:
```bash
python -m cfb_predictor.cli fetch-data 2022 2023 2024
```

3) **Train models** (features are built dynamically during training):
```bash
python -m cfb_predictor.cli train 2022 2023 2024
```

4) **Predict upcoming games** (uses this week's schedule + latest lines if available):
```bash
python -m cfb_predictor.cli predict --season 2025 --week auto --book DraftKings --min-edge 0.5
```
This outputs a CSV under `data/processed/predictions/` with:
- `win_prob_home`, `pick_ml`
- `pred_margin`, `edge_spread`, `pick_spread`
- `pred_total`, `edge_total`, `pick_total`

5) **Backtest** on historical seasons:
```bash
python -m cfb_predictor.cli backtest 2024 --book DraftKings --min-edge 0.5
```

## Project Structure

```
cfb-predictor/
  ├── src/cfb_predictor/
  │   ├── data_sources/cfbd_client.py
  │   ├── features/elo.py
  │   ├── features/feature_builder.py
  │   ├── data/build_games.py
  │   ├── models/train.py
  │   ├── predict.py
  │   ├── backtest.py
  │   └── cli.py
  ├── data/
  │   ├── raw/            # cached API pulls
  │   ├── processed/      # feature tables, predictions, backtests
  │   └── models/         # trained sklearn pickle files
  ├── .env.example
  ├── requirements.txt
  └── README.md
```

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
