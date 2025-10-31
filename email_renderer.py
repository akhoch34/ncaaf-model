
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
email_renderer.py (clean, dark-mode-safe)
-----------------------------------------
- Explicit dark background & text colors (safe in dark-mode clients)
- Unified "Performance Overview" with optional Week 8 examples
- "Featured Games" (Saturday-first), "Weekly Picks (All)" unified ATS + O/U
- Team logos via CSV mapping (team,logo_url)
- Correct ATS display from picked team perspective
- Pred label shows home team: "Pred (<home_team>): +3.5"
"""

import os
import logging
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


def _coerce_list(recipients: Optional[Iterable[str] | str]) -> List[str]:
    if recipients is None:
        return []
    if isinstance(recipients, str):
        return [r.strip() for r in recipients.split(",") if r.strip()]
    return [str(r).strip() for r in recipients if str(r).strip()]


# ----------------------- Time helpers -----------------------
_eastern = pytz.timezone("US/Eastern")


def to_et(df: pd.DataFrame, start_col: str = "start_date") -> pd.DataFrame:
    out = df.copy()
    start_dt = pd.to_datetime(out[start_col], utc=True)
    out["start_date_et"] = start_dt.dt.tz_convert(_eastern)
    out["date_only"] = out["start_date_et"].dt.date
    out["game_time"] = out["start_date_et"].dt.strftime("%I:%M %p ET")
    return out


def window_group(df: pd.DataFrame, minutes: int = 60):
    """Group by date and time window."""
    df = df.sort_values("start_date_et")
    window_start = df["start_date_et"].dt.floor(f"{minutes}min")
    df = df.assign(_window_start=window_start)

    groups = []
    for (d, _wstart), block in df.groupby(["date_only", "_window_start"]):
        block = block.copy().sort_values(["start_date_et"])
        first_time = block.iloc[0]["game_time"]
        last_time = block.iloc[-1]["game_time"]
        time_range = first_time if first_time == last_time else f"{first_time} - {last_time}"
        date_str = pd.Timestamp(d).strftime("%a, %b %d")
        groups.append((date_str, time_range, block))
    return groups


# ----------------------- Logo helpers -----------------------

def _load_logo_map() -> Optional[pd.DataFrame]:
    """Load mapping of team -> logo_url from CSV if available."""
    candidates = []
    env_path = os.getenv("TEAM_LOGO_CSV")
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        "src/data/static/team_logos.csv",
        "data/static/team_logos.csv",
    ])
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if {"team", "logo_url"}.issubset(df.columns):
                    return df[["team", "logo_url"]]
            except Exception as e:
                logger.warning(f"Failed to read logo CSV at {p}: {e}")
    return None


def _attach_logos(df: pd.DataFrame) -> pd.DataFrame:
    """Add home_logo_url and away_logo_url columns if logo map exists."""
    logo_df = _load_logo_map()
    if logo_df is None:
        return df

    out = df.copy()
    out = out.merge(logo_df.rename(columns={"team": "home_team", "logo_url": "home_logo_url"}),
                    on="home_team", how="left")
    out = out.merge(logo_df.rename(columns={"team": "away_team", "logo_url": "away_logo_url"}),
                    on="away_team", how="left")
    return out


# ----------------------- Picks shaping -----------------------

def _unified_picks_df(preds: pd.DataFrame) -> pd.DataFrame:
    """Create one row per game with ATS and/or O/U picks."""
    df = preds.copy()

    df["ats_edge_abs"] = df["edge_spread"].abs()
    df["ou_edge_abs"] = df["edge_total"].abs()
    df["has_ats"] = df["pick_spread"].astype(str).str.lower().ne("no bet")
    df["has_ou"] = df["pick_total"].astype(str).str.lower().ne("no bet")

    df = df[df["has_ats"] | df["has_ou"]].copy()

    df = df.sort_values(
        by=["ats_edge_abs", "ou_edge_abs", "start_date"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return df


def _featured_games(df: pd.DataFrame, max_items: int = 8) -> pd.DataFrame:
    """Saturday-first by edge; tiebreak by earliest kickoff (no deprecated .view)."""
    if df.empty:
        return df

    sdt = pd.to_datetime(df["start_date"], utc=True)
    dow = sdt.dt.weekday  # Monday=0, Saturday=5
    is_sat = (dow == 5).astype(int)

    out = df.assign(__is_sat=is_sat)
    out = out.sort_values(
        by=["__is_sat", "ats_edge_abs", "ou_edge_abs", "start_date"],
        ascending=[False, False, False, True]
    )
    return out.head(max_items).drop(columns=["__is_sat"])


# ----------------------- ATS helpers -----------------------

def _format_signed(val: float) -> str:
    try:
        return f"{float(val):+0.1f}"
    except Exception:
        return str(val)


def _pick_side_and_team(ps: str, home: str, away: str) -> Tuple[str, str]:
    """Return ('home'|'away', team_name) inferred from pick_spread text."""
    ps_l = (ps or "").lower()
    if home and home.lower() in ps_l:
        return "home", home
    if away and away.lower() in ps_l:
        return "away", away
    # fallback
    return "home", home


def _ats_strings(row: pd.Series) -> Tuple[str, str, float]:
    """
    Build ATS strings from a prediction row:
      - label: "<Picked Team> (+/-X.X)"
      - pred_label: "Pred (<home_team>): +Y.Y"
      - edge: float
    """
    if not bool(row.get("has_ats")):
        return "—", "—", 0.0

    side, team = _pick_side_and_team(str(row.get("pick_spread", "")),
                                     str(row.get("home_team", "")),
                                     str(row.get("away_team", "")))
    spread_line_home = float(row.get("spread_line", 0.0))
    # display spread from PICKED team perspective
    disp_line = spread_line_home if side == "home" else -spread_line_home
    ats_label = f"{team} ({_format_signed(disp_line)})"

    pred_home = float(row.get("pred_line", row.get("pred_margin", 0.0)))
    home_team = str(row.get("home_team", "Home"))
    ats_pred_label = f"Pred ({home_team}): {_format_signed(pred_home)}"

    ats_edge = float(row.get("ats_edge_abs", 0.0))
    return ats_label, ats_pred_label, ats_edge


# ----------------------- Week examples (rnd hits/misses) -----------------------

def _infer_pick_team_from_text(ps: str, home: str, away: str):
    ps_l = (ps or "").lower()
    if home and home.lower() in ps_l:
        return "home", home
    if away and away.lower() in ps_l:
        return "away", away
    return None, None


def _evaluate_week_examples(season: int, predictions_dir: str, week: int = 8, max_each: int = 3):
    """
    Load predictions_{season}_wk{week}.csv and compute a few randomly selected
    'right' and 'wrong' examples (ATS and O/U). Requires final scores.
    """
    import numpy as np

    fname = f"predictions_{season}_wk{week}.csv"
    path = os.path.join(predictions_dir, fname)
    if not os.path.exists(path):
        return ""

    try:
        df = pd.read_csv(path)
    except Exception:
        return ""

    # Load actual game results from raw games data
    games_candidates = [
        f"src/data/raw/games_{season}.parquet",
        f"data/raw/games_{season}.parquet",
    ]
    games_path = None
    for gp in games_candidates:
        if os.path.exists(gp):
            games_path = gp
            break

    if not games_path:
        return ""

    try:
        games = pd.read_parquet(games_path)
        # Filter to the specific week
        games_week = games[games['week'] == week].copy()

        # Join predictions with actual results
        # Match on home_team, away_team, and week
        df = df.merge(
            games_week[['home_team', 'away_team', 'week', 'home_points', 'away_points']],
            on=['home_team', 'away_team', 'week'],
            how='left'
        )
    except Exception as e:
        logger.warning(f"Failed to load game results for week {week}: {e}")
        return ""

    if not {"home_points", "away_points"}.issubset(df.columns):
        return ""

    # Filter to only completed games
    df = df[df['home_points'].notna() & df['away_points'].notna()].copy()
    if df.empty:
        return ""

    df["has_ats"] = df["pick_spread"].astype(str).str.lower().ne("no bet") if "pick_spread" in df.columns else False
    df["has_ou"] = df["pick_total"].astype(str).str.lower().ne("no bet") if "pick_total" in df.columns else False

    lines = []

    # ATS
    if df["has_ats"].any() and {"spread_line","pick_spread","home_team","away_team"}.issubset(df.columns):
        ats = df[df["has_ats"]].copy()
        ex_rows = []
        for _, r in ats.iterrows():
            side, team = _infer_pick_team_from_text(str(r.get("pick_spread","")), str(r.get("home_team","")), str(r.get("away_team","")))
            if side is None:
                continue
            hp, ap = r.get("home_points"), r.get("away_points")
            if pd.isna(hp) or pd.isna(ap):
                continue
            line_home = float(r.get("spread_line", 0.0))
            handicap = line_home if side=="home" else -line_home
            margin = (hp - ap) if side=="home" else (ap - hp)
            outcome_val = margin + handicap
            result = "right" if outcome_val > 0 else ("push" if abs(outcome_val) < 1e-9 else "wrong")
            disp_line = _format_signed(handicap)
            opp = r["away_team"] if side=="home" else r["home_team"]
            ex_rows.append((result, f"{team} ({disp_line}) vs {opp}: won by {margin:+.0f} vs spread"))

        if ex_rows:
            rights = [t for res,t in ex_rows if res=="right"]
            wrongs = [t for res,t in ex_rows if res=="wrong"]
            if rights:
                sample = rights if len(rights) <= max_each else list(pd.Series(rights).sample(max_each, random_state=None))
                lines.append(f"✅ Week {week} ATS hits:")
                lines += [f"  • {s}" for s in sample]
            if wrongs:
                sample = wrongs if len(wrongs) <= max_each else list(pd.Series(wrongs).sample(max_each, random_state=None))
                lines.append(f"❌ Week {week} ATS misses:")
                lines += [f"  • {s}" for s in sample]

    # O/U
    if df["has_ou"].any() and {"total_line","pick_total"}.issubset(df.columns):
        ou = df[df["has_ou"]].copy()
        ex_rows = []
        for _, r in ou.iterrows():
            hp, ap = r.get("home_points"), r.get("away_points")
            if pd.isna(hp) or pd.isna(ap):
                continue
            total = float(hp) + float(ap)
            line = float(r.get("total_line", 0.0))
            pick = str(r.get("pick_total","")).lower()
            if "over" in pick:
                result = "right" if total > line else ("push" if abs(total - line) < 1e-9 else "wrong")
                tag = "Over"
            elif "under" in pick:
                result = "right" if total < line else ("push" if abs(total - line) < 1e-9 else "wrong")
                tag = "Under"
            else:
                continue
            ex_rows.append((result, f"{tag} {line:.1f} ({r['away_team']} @ {r['home_team']}) final total {total:.1f}"))

        if ex_rows:
            rights = [t for res,t in ex_rows if res=="right"]
            wrongs = [t for res,t in ex_rows if res=="wrong"]
            if rights:
                sample = rights if len(rights) <= max_each else list(pd.Series(rights).sample(max_each, random_state=None))
                lines.append(f"✅ Week {week} O/U hits:")
                lines += [f"  • {s}" for s in sample]
            if wrongs:
                sample = wrongs if len(wrongs) <= max_each else list(pd.Series(wrongs).sample(max_each, random_state=None))
                lines.append(f"❌ Week {week} O/U misses:")
                lines += [f"  • {s}" for s in sample]

    extra_text = "\n".join(lines) if lines else ""
    return extra_text


# ----------------------- HTML helpers -----------------------

def _pill(label: str) -> str:
    return f"<span style=\"display:inline-block;padding:4px 10px;border-radius:999px;background:#1f2937;color:#cbd5e1;font-size:12px;font-weight:700;\">{label}</span>"


def _header_html(week_num: int, season: int, book: str) -> str:
    return f"""
    <tr>
      <td style="background:linear-gradient(90deg,#1f3b73,#2c5aa0);color:#ffffff;text-align:center;padding:36px;border-top-left-radius:14px;border-top-right-radius:14px;box-shadow:0 6px 18px rgba(0,0,0,0.12);">
        <h1 style="margin:0;font-size:28px;font-weight:800;letter-spacing:.3px;color:#ffffff;">NCAAF Weekly Picks</h1>
        <div style="margin-top:10px;font-size:17px;opacity:.95;color:#ffffff;">Week {week_num} • {season} Season • {book}</div>
      </td>
    </tr>
    """


def _info_row(book: str) -> str:
    return f"""
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" bgcolor="#111827" style="background:#111827;border-left:4px solid #60a5fa;border-radius:10px;">
      <tr>
        <td style="padding:16px 18px;color:#e5e7eb;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border-collapse:collapse;">
            <tr>
              <td align="center" style="padding:8px;color:#e5e7eb;">
                <strong style="display:block;color:#93c5fd;font-size:14px;">Generated</strong>
                <span style="color:#e5e7eb;">{datetime.now().strftime('%B %d, %Y')}</span>
              </td>
              <td align="center" style="padding:8px;color:#e5e7eb;">
                <strong style="display:block;color:#93c5fd;font-size:14px;">Sportsbook</strong>
                <span style="color:#e5e7eb;">{book}</span>
              </td>
              <td align="center" style="padding:8px;color:#e5e7eb;">
                <strong style="display:block;color:#93c5fd;font-size:14px;">Min Edge</strong>
                <span style="color:#e5e7eb;">0.5 points</span>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
    """


def _section_header(title: str, count: int) -> str:
    return f"<h2 style=\"color:#93c5fd;border-bottom:2px solid #1f2937;padding-bottom:8px;margin:18px 0 12px 0;font-size:20px;\">{title} <span style=\"color:#94a3b8;font-size:14px;\">({count})</span></h2>"


def _block_header(date_str: str, time_range: str) -> str:
    return f"<h3 style=\"color:#cbd5e1;font-size:16px;margin:12px 0 8px 0;border-left:4px solid #60a5fa;padding-left:10px;\">{date_str} • {time_range}</h3>"


def _card_open() -> str:
    return "<table role=\"presentation\" width=\"100%\" cellspacing=\"0\" cellpadding=\"0\" border=\"0\" bgcolor=\"#111827\" style=\"background:#111827;border:1px solid #1f2937;border-left:4px solid #60a5fa;margin:12px 0;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.25);\"><tr><td style=\"padding:14px 14px 12px 14px;color:#e5e7eb;\">"


def _card_close() -> str:
    return "</td></tr></table>"


def _matchup_row(matchup: str, kickoff: str, home_logo: Optional[str], away_logo: Optional[str]) -> str:
    def _logo(url: Optional[str]) -> str:
        if not isinstance(url, str) or not url.startswith("http"):
            return ""
        return f'<img src="{url}" width="20" height="20" style="vertical-align:middle;border-radius:3px;margin-right:6px;" alt="logo" />'
    away, home = matchup.split(" @ ")
    return f"""
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border-collapse:collapse;margin-bottom:6px;">
      <tr>
        <td style="font-size:16px;font-weight:800;color:#e5e7eb;">
          { _logo(away_logo) }{away}
          <span style="color:#9ca3af;font-weight:600;"> @ </span>
          { _logo(home_logo) }{home}
        </td>
        <td align="right" style="color:#cbd5e1;font-size:13px;">{_pill(kickoff)}</td>
      </tr>
    </table>
    """


def _two_col(label_left: str, value_left: str, label_right: str, value_right: str, hl_left: bool=False, hl_right: bool=False) -> str:
    left_bg  = "#052e16" if hl_left else "#0b1220"
    right_bg = "#052e16" if hl_right else "#0b1220"
    left_col = "#86efac" if hl_left else "#e5e7eb"
    right_col= "#86efac" if hl_right else "#e5e7eb"
    return f"""
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border-collapse:collapse;">
      <tr>
        <td align="center" style="padding:10px;background-color:{left_bg};color:{left_col};width:50%;border-radius:10px;">
          <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">{label_left}</div>
          <div style="font-weight:800;color:{left_col};">{value_left}</div>
        </td>
        <td align="center" style="padding:10px;background-color:{right_bg};color:{right_col};width:50%;border-radius:10px;">
          <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">{label_right}</div>
          <div style="font-weight:800;color:{right_col};">{value_right}</div>
        </td>
      </tr>
    </table>
    """


def _edge_badges(ats_edge: float, ou_edge: float) -> str:
    return f"""
    <div style="margin-top:8px;">
      <span style="display:inline-block;background:#2563eb;color:#ffffff;border-radius:999px;padding:4px 10px;font-size:12px;font-weight:700;margin-right:6px;">ATS Edge {ats_edge:.1f}</span>
      <span style="display:inline-block;background:#10b981;color:#ffffff;border-radius:999px;padding:4px 10px;font-size:12px;font-weight:700;">O/U Edge {ou_edge:.1f}</span>
    </div>
    """


# ----------------------- Main sender -----------------------

def send_email_picks(*, week: int | None, book: str, CURRENT_SEASON: int, SRC_DIR: str) -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv()

        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        email_from = os.getenv("EMAIL_FROM", "akhoch54@gmail.com")
        email_to = os.getenv("EMAIL_TO")
        rcpts = _coerce_list(email_to)

        if not gmail_password or not rcpts:
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

        # Accuracy summary
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
Last Week (Week {week_num - 1}):
- ATS: {last_week['ats_wins']}-{last_week['ats_losses']}-{last_week['ats_pushes']} ({last_week['ats_win_pct']:.1%})
- O/U: {last_week['ou_wins']}-{last_week['ou_losses']}-{last_week['ou_pushes']} ({last_week['ou_win_pct']:.1%})

"""

                    accuracy_summary = f"""{last_week_summary}Season Totals {CURRENT_SEASON}:
- ATS Record: {total_ats_wins}-{total_ats_losses}-{total_ats_pushes} ({total_ats_pct:.1%})
- O/U Record: {total_ou_wins}-{total_ou_losses}-{total_ou_pushes} ({total_ou_pct:.1%})
"""
                else:
                    recent_acc = acc_df.tail(1).iloc[0]
                    accuracy_summary = f"""
Recent Performance:
- ATS Record: {recent_acc['ats_wins']}-{recent_acc['ats_losses']}-{recent_acc['ats_pushes']} ({recent_acc['ats_win_pct']:.1%})
- O/U Record: {recent_acc['ou_wins']}-{recent_acc['ou_losses']}-{recent_acc['ou_pushes']} ({recent_acc['ou_win_pct']:.1%})
"""

        # Add previous week examples (if available and completed)
        if week_num > 1:
            examples_text = _evaluate_week_examples(CURRENT_SEASON, predictions_dir, week=week_num - 1, max_each=3)
            if examples_text:
                accuracy_summary = (accuracy_summary or "") + "\n" + examples_text + "\n"

        # ---------- Build unified picks ----------
        unified = _unified_picks_df(predictions)
        unified_et = to_et(unified)
        unified_et = _attach_logos(unified_et)
        unified_et["ats_edge_abs"] = unified_et["ats_edge_abs"].fillna(0.0)
        unified_et["ou_edge_abs"]  = unified_et["ou_edge_abs"].fillna(0.0)
        unified_et["primary_edge"] = unified_et["ats_edge_abs"]

        # ---------- Featured games ----------
        featured = _featured_games(unified_et, max_items=8)

        subject = f"NCAAF Week {week_num} Picks - {CURRENT_SEASON} Season"

        # ---------------- Plain text ----------------
        body = f"""
===============================================================
NCAAF WEEKLY PICKS - Week {week_num} ({CURRENT_SEASON} Season)
===============================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
Sportsbook: {book}
Minimum Edge: 0.5 points

{accuracy_summary}

Featured Games (Saturday-first)
{('='*65)}
"""

        if not featured.empty:
            for i, (_, g) in enumerate(featured.iterrows(), 1):
                game_time = pd.to_datetime(g['start_date']).strftime('%a %m/%d %I:%M%p ET')
                ats_label, ats_pred_label, _ats_edge = _ats_strings(g)
                if bool(g["has_ou"]):
                    ou_label = f"{g['pick_total']} {g['total_line']} | PredTot: {float(g['pred_total']):.1f} | Edge {float(g['ou_edge_abs']):.1f}"
                else:
                    ou_label = "—"

                body += f"""
{i:2d}. {g['away_team']} @ {g['home_team']} • {game_time}
    ATS: {ats_label} | {ats_pred_label} | Edge {float(g['ats_edge_abs']):.1f}
    O/U: {ou_label}
"""

        body += f"""
{('='*65)}

WEEKLY PICKS (ALL) — Sorted by ATS Edge
{('='*65)}
"""

        for i, (_, g) in enumerate(unified_et.sort_values(["start_date_et","primary_edge"], ascending=[True, False]).iterrows(), 1):
            game_time = pd.to_datetime(g['start_date']).strftime('%a %m/%d %I:%M%p ET')
            ats_label, ats_pred_label, _ats_edge = _ats_strings(g)
            if bool(g["has_ou"]):
                ou_label = f"{g['pick_total']} {g['total_line']} | PredTot: {float(g['pred_total']):.1f} | Edge {float(g['ou_edge_abs']):.1f}"
            else:
                ou_label = "—"

            body += f"""
{i:2d}. {g['away_team']} @ {g['home_team']} • {game_time}
    ATS: {ats_label} | {ats_pred_label} | Edge {float(g['ats_edge_abs']):.1f}
    O/U: {ou_label}
"""

        body += f"""
{('='*65)}

Notes:
- Predicted Line is from the HOME team's perspective.
- Edge = |Predicted Value - Book Line|
- Only picks with minimum 0.5 point edge are shown
- ATS = Against The Spread
- All times Eastern

===============================================================
Generated by NCAAF Predictor
===============================================================
"""

        # ---------------- HTML (dark-mode-safe) ----------------
        html_parts = []

        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="color-scheme" content="light">
  <meta name="supported-color-schemes" content="light">
  <title>NCAAF Picks</title>
  <style>
    :root, html, body { color-scheme: light; supported-color-schemes: light; }
  </style>
</head>
<body style="margin:0;padding:0;background:#0b1220;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" bgcolor="#0b1220" style="background:#0b1220;">
    <tr>
      <td align="center" style="padding:24px;">
        <table role="presentation" width="860" cellspacing="0" cellpadding="0" border="0" style="width:100%;max-width:860px;border-collapse:separate;border-spacing:0;">
""")

        html_parts.append(_header_html(week_num, CURRENT_SEASON, book))

        html_parts.append("""
          <tr>
            <td bgcolor="#0f172a" style="background:#0f172a;color:#e5e7eb;padding:26px 26px 8px 26px;border-bottom-left-radius:14px;border-bottom-right-radius:14px;box-shadow:0 6px 18px rgba(0,0,0,0.12);">
""")

        html_parts.append(_info_row(book))

        
        if accuracy_summary.strip():
            # Improved Performance Overview with badges and logos
            examples_dict = {}
            if week_num > 1:
                try:
                    examples_dict = _evaluate_week_examples_struct(CURRENT_SEASON, predictions_dir, week=week_num - 1, max_each=3) or {}
                except Exception:
                    examples_dict = {}
            perf_struct = _compute_perf_struct(accuracy_file, CURRENT_SEASON, week_num)
            html_parts.append(_perf_overview_html(perf_struct, examples_dict))


        # Featured
        if not featured.empty:
            html_parts.append(_section_header("Featured Games", len(featured)))
            for _, g in featured.iterrows():
                matchup = f"{g['away_team']} @ {g['home_team']}"
                kickoff = g["game_time"]

                ats_label, ats_pred_label, ats_edge = _ats_strings(g)

                if bool(g["has_ou"]):
                    ou_label = f"{g['pick_total']} {g['total_line']}"
                    ou_pred = f"{float(g['pred_total']):.1f}"
                    ou_edge = float(g["ou_edge_abs"])
                else:
                    ou_label, ou_pred, ou_edge = "—", "—", 0.0

                html_parts.append(_card_open())
                html_parts.append(_matchup_row(matchup, kickoff, g.get("home_logo_url"), g.get("away_logo_url")))
                html_parts.append(_two_col("ATS Pick", f"{ats_label}<div style='font-size:12px;color:#9ca3af;margin-top:2px;'>{ats_pred_label}</div>",
                                           "O/U Pick", f"{ou_label}<div style='font-size:12px;color:#9ca3af;margin-top:2px;'>Pred Total: {ou_pred}</div>",
                                           hl_left=ats_edge>=2.0, hl_right=ou_edge>=2.0))
                html_parts.append(_edge_badges(ats_edge, ou_edge))
                html_parts.append(_card_close())

        # Weekly picks
        html_parts.append(_section_header("Weekly Picks (All)", len(unified_et)))
        for d_str, time_range, block in window_group(unified_et, minutes=60):
            block = block.sort_values(["start_date_et", "primary_edge"], ascending=[True, False])
            html_parts.append(_block_header(d_str, time_range))
            for _, g in block.iterrows():
                matchup = f"{g['away_team']} @ {g['home_team']}"
                kickoff = g["game_time"]

                ats_label, ats_pred_label, ats_edge = _ats_strings(g)

                if bool(g["has_ou"]):
                    ou_label = f"{g['pick_total']} {g['total_line']}"
                    ou_pred = f"{float(g['pred_total']):.1f}"
                    ou_edge = float(g["ou_edge_abs"])
                else:
                    ou_label, ou_pred, ou_edge = "—", "—", 0.0

                html_parts.append(_card_open())
                html_parts.append(_matchup_row(matchup, kickoff, g.get("home_logo_url"), g.get("away_logo_url")))
                html_parts.append(_two_col("ATS Pick", f"{ats_label}<div style='font-size:12px;color:#9ca3af;margin-top:2px;'>{ats_pred_label}</div>",
                                           "O/U Pick", f"{ou_label}<div style='font-size:12px;color:#9ca3af;margin-top:2px;'>Pred: {ou_pred}</div>",
                                           hl_left=ats_edge>=2.0, hl_right=ou_edge>=2.0))
                html_parts.append(_edge_badges(ats_edge, ou_edge))
                html_parts.append(_card_close())

        # Notes
        html_parts.append(f"""
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" bgcolor="#1f2937" style="background:#1f2937;border:1px solid #374151;margin:20px 0;border-radius:10px;">
                <tr>
                  <td style="color:#e5e7eb;padding:16px 18px;">
                    <h4 style="margin:0 0 8px 0;color:#93c5fd;">Betting Notes</h4>
                    <ul style="margin:8px 0;padding-left:18px;color:#d1d5db;">
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
          <tr><td height="8"></td></tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
""")

        html_body = "".join(html_parts)

        # Send email
        msg = MIMEMultipart('alternative')
        msg['From'] = email_from
        msg['To'] = ", ".join(rcpts)
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_from, gmail_password)
        server.sendmail(msg['From'], rcpts, msg.as_string())
        server.quit()

        logger.info(f"Email sent successfully to {rcpts}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

# ===================== Performance Overview helpers =====================

def _badge(label: str, tone: str = "blue") -> str:
    tones = {
        "blue": ("#1d4ed8", "#dbeafe"),
        "green": ("#065f46", "#d1fae5"),
        "slate": ("#374151", "#e5e7eb"),
        "red": ("#7f1d1d", "#fee2e2"),
        "amber": ("#92400e", "#fef3c7"),
    }
    bg, fg = tones.get(tone, tones["blue"])
    return (f'<span style="display:inline-block;font-size:12px;font-weight:700;'
            f'border-radius:999px;padding:4px 10px;background:{bg};color:{fg};'
            f'margin-right:6px;">{label}</span>')


def _tiny_logo(url: Optional[str], alt: str = "logo") -> str:
    if not isinstance(url, str) or not url.startswith("http"):
        return ""
    return (f'<img src="{url}" width="16" height="16" '
            f'style="vertical-align:middle;border-radius:3px;margin-right:6px;" '
            f'alt="{alt}" />')


def _compute_perf_struct(accuracy_file: str, CURRENT_SEASON: int, week_num: int) -> dict:
    import pandas as pd
    perf_struct: dict = {}
    if os.path.exists(accuracy_file):
        acc_df = pd.read_csv(accuracy_file)
        if not acc_df.empty:
            cur = acc_df[acc_df["season"] == CURRENT_SEASON]
            if not cur.empty:
                ats_w = int(cur["ats_wins"].sum())
                ats_l = int(cur["ats_losses"].sum())
                ats_p = int(cur["ats_pushes"].sum())
                ou_w  = int(cur["ou_wins"].sum())
                ou_l  = int(cur["ou_losses"].sum())
                ou_p  = int(cur["ou_pushes"].sum())
                ats_pct = (ats_w / (ats_w + ats_l)) if (ats_w + ats_l) else 0.0
                ou_pct  = (ou_w  / (ou_w  + ou_l )) if (ou_w  + ou_l ) else 0.0

                last_week_struct = None
                if week_num and week_num > 1:
                    lw = cur[cur["week"] == week_num - 1]
                    if not lw.empty:
                        row = lw.iloc[0]
                        last_week_struct = {
                            "week": int(week_num - 1),
                            "ats_wins": int(row["ats_wins"]),
                            "ats_losses": int(row["ats_losses"]),
                            "ats_pushes": int(row["ats_pushes"]),
                            "ats_win_pct": float(row["ats_win_pct"]),
                            "ou_wins": int(row["ou_wins"]),
                            "ou_losses": int(row["ou_losses"]),
                            "ou_pushes": int(row["ou_pushes"]),
                            "ou_win_pct": float(row["ou_win_pct"]),
                        }

                perf_struct = {
                    "last_week": last_week_struct,
                    "season_totals": {
                        "season": int(CURRENT_SEASON),
                        "ats_wins": ats_w, "ats_losses": ats_l, "ats_pushes": ats_p,
                        "ats_win_pct": float(ats_pct),
                        "ou_wins": ou_w, "ou_losses": ou_l, "ou_pushes": ou_p,
                        "ou_win_pct": float(ou_pct),
                    }
                }
    return perf_struct



def _evaluate_week_examples_struct(season: int, predictions_dir: str, week: int = 8, max_each: int = 3) -> dict:
    """
    Build curated "Last Week Picks" sets with logos.
    Categories for ATS: nailed, barely_won, bad_beats (barely missed), terrible.
    Categories for O/U: barely_won, bad_beats, terrible; 'nailed' if predicted total is available.
    """
    import pandas as pd
    path = os.path.join(predictions_dir, f"predictions_{season}_wk{week}.csv")
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    games_candidates = [
        f"src/data/raw/games_{season}.parquet",
        f"data/raw/games_{season}.parquet",
    ]
    games_path = next((gp for gp in games_candidates if os.path.exists(gp)), None)
    if not games_path:
        return {}

    try:
        games = pd.read_parquet(games_path)
        games_week = games[games['week'] == week].copy()
        df = df.merge(
            games_week[['home_team','away_team','week','home_points','away_points']],
            on=['home_team','away_team','week'], how='left'
        )
    except Exception:
        return {}

    if not {'home_points','away_points'}.issubset(df.columns):
        return {}

    df = df[df['home_points'].notna() & df['away_points'].notna()].copy()
    if df.empty:
        return {}

    # Determine if we have a predicted margin for "nailed"
    pred_cols = [c for c in ['pred_home_margin','home_predicted_margin','pred_margin','model_home_margin'] if c in df.columns]
    pred_margin_col = pred_cols[0] if pred_cols else None

    df['has_ats'] = df.get('pick_spread','no bet').astype(str).str.lower().ne('no bet')
    df['has_ou']  = df.get('pick_total','no bet').astype(str).str.lower().ne('no bet')

    logo_df = _load_logo_map()

    # Output buckets
    out = {
        'ats_nailed': [], 'ats_barely_won': [], 'ats_bad_beats': [], 'ats_terrible': [],
        'ou_nailed': [],  'ou_barely_won': [],  'ou_bad_beats': [],  'ou_terrible': []
    }

    # Tunable thresholds
    BARELY = 1.5    # within 1.5 points of the number
    TERRIBLE = 10.0 # missed by 10+ vs number counts as terrible
    NAIL_DELTA = 1.0 # actual margin close to predicted margin

    # ============== ATS CURATION ==============
    if df['has_ats'].any() and {'spread_line','pick_spread','home_team','away_team'}.issubset(df.columns):
        ats = df[df['has_ats']].copy()
        for _, r in ats.iterrows():
            side, team = _infer_pick_team_from_text(str(r['pick_spread']), str(r['home_team']), str(r['away_team']))
            if not side:
                continue
            hp, ap = float(r['home_points']), float(r['away_points'])
            actual_home_margin = hp - ap
            # Number is from home perspective
            line_home = float(r.get('spread_line', 0.0))
            # Handicap applied to our picked side
            handicap  = line_home if side == 'home' else -line_home
            # Outcome vs the number from our pick's perspective
            outcome_val = (actual_home_margin if side == 'home' else -actual_home_margin) + handicap

            opp    = r['away_team'] if side == 'home' else r['home_team']
            detail = f'{team} ({_format_signed(handicap)}) vs {opp}: {"covered" if outcome_val>0 else ("push" if abs(outcome_val)<1e-9 else "did not cover")} by {outcome_val:+.1f}'

            # logos
            tl = ol = None
            if isinstance(logo_df, pd.DataFrame):
                sel_t = logo_df[logo_df['team'].eq(team)]
                sel_o = logo_df[logo_df['team'].eq(opp)]
                tl = sel_t['logo_url'].iloc[0] if not sel_t.empty else None
                ol = sel_o['logo_url'].iloc[0] if not sel_o.empty else None

            rec = {"team": team, "opp": opp, "detail": detail, "team_logo": tl, "opp_logo": ol}

            # Categorize
            if outcome_val > 0 and outcome_val <= BARELY:
                out['ats_barely_won'].append(rec)
            elif outcome_val < 0 and abs(outcome_val) <= BARELY:
                out['ats_bad_beats'].append(rec)
            elif outcome_val <= -TERRIBLE:
                out['ats_terrible'].append(rec)

            # Nailed: if we have predicted margin, and we picked the correct side AND actual margin ~ predicted
            if pred_margin_col is not None:
                pred_home_margin = float(r[pred_margin_col])
                picked_correct = outcome_val > 0
                if picked_correct and abs(actual_home_margin - pred_home_margin) <= NAIL_DELTA:
                    out['ats_nailed'].append(rec)

        # trim each bucket
        for k in ['ats_nailed','ats_barely_won','ats_bad_beats','ats_terrible']:
            if len(out[k]) > max_each:
                out[k] = out[k][:max_each]

    # ============== O/U CURATION ==============
    if df['has_ou'].any() and {'total_line','pick_total'}.issubset(df.columns):
        ou = df[df['has_ou']].copy()
        for _, r in ou.iterrows():
            hp, ap = float(r['home_points']), float(r['away_points'])
            total  = hp + ap
            line   = float(r.get('total_line', 0.0))
            pick_l = str(r.get('pick_total','')).lower()
            if 'over' in pick_l:
                diff = total - line
                outcome_correct = diff > 0
                label = 'Over'
            elif 'under' in pick_l:
                diff = line - total
                outcome_correct = diff > 0
                label = 'Under'
            else:
                continue

            home, away = r['home_team'], r['away_team']
            detail = f'{label} {line:.1f} ({away} @ {home}): {"hit" if outcome_correct else "missed"} by {diff:+.1f} (final {total:.1f})'

            hl = al = None
            if isinstance(logo_df, pd.DataFrame):
                sel_h = logo_df[logo_df['team'].eq(home)]
                sel_a = logo_df[logo_df['team'].eq(away)]
                hl = sel_h['logo_url'].iloc[0] if not sel_h.empty else None
                al = sel_a['logo_url'].iloc[0] if not sel_a.empty else None

            rec = {"team": home, "opp": away, "detail": detail, "team_logo": hl, "opp_logo": al}

            if outcome_correct and 0 < abs(diff) <= BARELY:
                out['ou_barely_won'].append(rec)
            elif (not outcome_correct) and abs(diff) <= BARELY:
                out['ou_bad_beats'].append(rec)
            elif (not outcome_correct) and abs(diff) >= TERRIBLE:
                out['ou_terrible'].append(rec)

            # O/U "nailed" only if we have a predicted total column
            pred_total_cols = [c for c in ['pred_total','predicted_total','model_total'] if c in ou.columns]
            if pred_total_cols:
                try:
                    pred_total = float(r[pred_total_cols[0]])
                    if outcome_correct and abs(total - pred_total) <= NAIL_DELTA:
                        out['ou_nailed'].append(rec)
                except Exception:
                    pass

        for k in ['ou_nailed','ou_barely_won','ou_bad_beats','ou_terrible']:
            if len(out[k]) > max_each:
                out[k] = out[k][:max_each]

    return out

def _perf_overview_html(perf: dict, examples: dict | None) -> str:
    def row_html(title: str, ats: str, ou: str) -> str:
        return (
            '<tr><td style="padding:8px 10px;border-bottom:1px solid #155e75;">'
            f'<div style="font-weight:800;color:#e6fffb;font-size:13px;margin-bottom:6px;">{title}</div>'
            f'<div>{_badge("ATS " + ats, "blue")}{_badge("O/U " + ou, "green")}</div>'
            '</td></tr>'
        )

    rows = []
    last_week = perf.get('last_week')
    season    = perf.get('season_totals')

    if last_week:
        rows.append(row_html(
            f"Last Week (Week {last_week['week']})",
            f"{last_week['ats_wins']}-{last_week['ats_losses']}-{last_week['ats_pushes']} ({last_week['ats_win_pct']:.1%})",
            f"{last_week['ou_wins']}-{last_week['ou_losses']}-{last_week['ou_pushes']} ({last_week['ou_win_pct']:.1%})",
        ))
    if season:
        rows.append(row_html(
            f"Season Totals {season['season']}",
            f"{season['ats_wins']}-{season['ats_losses']}-{season['ats_pushes']} ({season['ats_win_pct']:.1%})",
            f"{season['ou_wins']}-{season['ou_losses']}-{season['ou_pushes']} ({season['ou_win_pct']:.1%})",
        ))

    examples_html = ""
    if examples and any(examples.get(k) for k in (
        'ats_nailed','ats_barely_won','ats_bad_beats','ats_terrible',
        'ou_nailed','ou_barely_won','ou_bad_beats','ou_terrible'
    )):
        def sect(label: str, items: list, tone: str) -> str:
            if not items:
                return ""
            lis = []
            for it in items:
                lis.append(
                    '<li style="margin:6px 0;color:#e5e7eb;">'
                    f'{_tiny_logo(it.get("team_logo"))}{it.get("team","")}'
                    '<span style="color:#9ca3af;"> vs </span>'
                    f'{_tiny_logo(it.get("opp_logo"))}{it.get("opp","")}'
                    '<span style="color:#9ca3af;"> — </span>'
                    f'<span style="color:#cbd5e1;">{it.get("detail","")}</span>'
                    '</li>'
                )
            return (
                '<div style="margin-top:8px;">'
                f'<div style="margin-bottom:4px;font-weight:800;">{_badge(label, tone)}</div>'
                f'<ul style="margin:0;padding-left:18px;list-style:disc;">{"".join(lis)}</ul>'
                '</div>'
            )

        # Use "Last Week Picks" phrasing instead of "Examples"
        examples_html = "".join([
            sect("Last Week Picks · ATS — Nailed",       examples.get("ats_nailed")       or [], "green"),
            sect("Last Week Picks · ATS — Barely Won",   examples.get("ats_barely_won")   or [], "amber"),
            sect("Last Week Picks · ATS — Bad Beats",    examples.get("ats_bad_beats")    or [], "slate"),
            sect("Last Week Picks · ATS — Terrible",     examples.get("ats_terrible")     or [], "red"),
            sect("Last Week Picks · O/U — Nailed",       examples.get("ou_nailed")        or [], "green"),
            sect("Last Week Picks · O/U — Barely Won",   examples.get("ou_barely_won")    or [], "amber"),
            sect("Last Week Picks · O/U — Bad Beats",    examples.get("ou_bad_beats")     or [], "slate"),
            sect("Last Week Picks · O/U — Terrible",     examples.get("ou_terrible")      or [], "red"),
        ])

    return (
        '<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" bgcolor="#0b3b49" '
        'style="background:linear-gradient(90deg,#0f766e,#0e7490);border-radius:12px;margin:14px 0;'
        'box-shadow:0 2px 10px rgba(0,0,0,.25);">'
        '<tr><td style="color:#eafff4;padding:16px 18px;">'
        '<h3 style="margin:0 0 6px 0;font-size:18px;color:#ffffff;">Performance Overview</h3>'
        '<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border-collapse:collapse;">'
        f'{"".join(rows)}'
        '</table>'
        f'{examples_html}'
        '</td></tr></table>'
    )
# (duplicate _perf_overview_html removed)
