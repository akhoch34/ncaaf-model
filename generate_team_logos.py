#!/usr/bin/env python3
"""
Generate Team Logos CSV
=======================

This script creates a CSV mapping team names to their logo URLs.
Uses ESPN's publicly available team logo CDN.

Usage:
    python generate_team_logos.py
"""

import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ESPN team ID mapping (partial - common teams)
# Format: "Team Name": espn_team_id
ESPN_TEAM_IDS = {
    "Air Force": 2005,
    "Akron": 2006,
    "Alabama": 333,
    "App State": 2026,
    "Arizona": 12,
    "Arizona State": 9,
    "Arkansas": 8,
    "Arkansas State": 2032,
    "Army": 349,
    "Auburn": 2,
    "Ball State": 2050,
    "Baylor": 239,
    "Boise State": 68,
    "Boston College": 103,
    "Bowling Green": 189,
    "Buffalo": 2084,
    "BYU": 252,
    "California": 25,
    "UCF": 2116,
    "Central Michigan": 2117,
    "Charlotte": 2429,
    "Cincinnati": 2132,
    "Clemson": 228,
    "Coastal Carolina": 324,
    "Colorado": 38,
    "Colorado State": 36,
    "Duke": 150,
    "East Carolina": 151,
    "Eastern Michigan": 2199,
    "Florida": 57,
    "Florida Atlantic": 2226,
    "Florida International": 2229,
    "Florida State": 52,
    "Fresno State": 278,
    "Georgia": 61,
    "Georgia State": 2247,
    "Georgia Tech": 59,
    "Hawaii": 62,
    "Houston": 248,
    "Illinois": 356,
    "Indiana": 84,
    "Iowa": 2294,
    "Iowa State": 66,
    "Kansas": 2305,
    "Kansas State": 2306,
    "Kent State": 2309,
    "Kentucky": 96,
    "Liberty": 2335,
    "Louisiana": 309,
    "Louisiana Monroe": 2433,
    "Louisiana Tech": 2348,
    "Louisville": 97,
    "LSU": 99,
    "Marshall": 276,
    "Maryland": 120,
    "Memphis": 235,
    "Miami": 2390,
    "Miami (OH)": 193,
    "Michigan": 130,
    "Michigan State": 127,
    "Middle Tennessee": 2393,
    "Minnesota": 135,
    "Mississippi State": 344,
    "Missouri": 142,
    "Navy": 2426,
    "NC State": 152,
    "Nebraska": 158,
    "Nevada": 2440,
    "New Mexico": 167,
    "New Mexico State": 166,
    "North Carolina": 153,
    "North Texas": 249,
    "Northern Illinois": 2459,
    "Northwestern": 77,
    "Notre Dame": 87,
    "Ohio": 195,
    "Ohio State": 194,
    "Oklahoma": 201,
    "Oklahoma State": 197,
    "Old Dominion": 295,
    "Ole Miss": 145,
    "Oregon": 2483,
    "Oregon State": 204,
    "Penn State": 213,
    "Pittsburgh": 221,
    "Purdue": 2509,
    "Rice": 242,
    "Rutgers": 164,
    "San Diego State": 21,
    "San José State": 23,
    "SMU": 2567,
    "South Alabama": 6,
    "South Carolina": 2579,
    "South Florida": 58,
    "Southern Mississippi": 2572,
    "Stanford": 24,
    "Syracuse": 183,
    "TCU": 2628,
    "Temple": 218,
    "Tennessee": 2633,
    "Texas": 251,
    "Texas A&M": 245,
    "Texas State": 326,
    "Texas Tech": 2641,
    "Toledo": 2649,
    "Troy": 2653,
    "Tulane": 2655,
    "Tulsa": 202,
    "UAB": 5,
    "UCF": 2116,
    "UCLA": 26,
    "UConn": 41,
    "UNLV": 2439,
    "USC": 30,
    "Utah": 254,
    "Utah State": 328,
    "UTEP": 2638,
    "UTSA": 2636,
    "Vanderbilt": 238,
    "Virginia": 258,
    "Virginia Tech": 259,
    "Wake Forest": 154,
    "Washington": 264,
    "Washington State": 265,
    "West Virginia": 277,
    "Western Kentucky": 98,
    "Western Michigan": 2711,
    "Wisconsin": 275,
    "Wyoming": 2751,
    # Recently transitioned FBS / FCS schools with ESPN IDs
    "Delaware": 48,
    "James Madison": 256,
    "Jacksonville State": 55,
    "Kennesaw State": 338,
    "Sam Houston": 2929,
    "Georgia Southern": 290,
    "Massachusetts": 113,
    "Missouri State": 2623,
    "Montana State": 149,
    "North Dakota": 2449,
    "Sacramento State": 16,
    "Sam Houston State": 2929,
    "Tarleton State": 2913,
    "UT Martin": 2630,
}

# FCS/smaller schools - use generic placeholder
GENERIC_LOGO = "https://a.espncdn.com/i/teamlogos/ncaa/500/default-team-logo-500.png"


def get_all_teams_from_data():
    """Get unique team names from the games data"""
    games_file = 'src/data/raw/games_2025.parquet'
    if not os.path.exists(games_file):
        logger.error(f"Games file not found: {games_file}")
        return []

    games = pd.read_parquet(games_file)
    all_teams = set(games['home_team'].unique()) | set(games['away_team'].unique())
    return sorted(all_teams)


def get_logo_url(team_name):
    """
    Get logo URL for a team.

    Uses ESPN's CDN: https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png
    """
    # Check if we have ESPN team ID
    if team_name in ESPN_TEAM_IDS:
        espn_id = ESPN_TEAM_IDS[team_name]
        return f"https://a.espncdn.com/i/teamlogos/ncaa/500/{espn_id}.png"

    # For teams not in our mapping, use generic logo
    logger.warning(f"No ESPN ID found for: {team_name} - using generic logo")
    return GENERIC_LOGO


def generate_team_logos_csv():
    """Generate the team logos CSV file"""

    # Get all teams
    teams = get_all_teams_from_data()
    if not teams:
        logger.error("No teams found in data")
        return False

    logger.info(f"Found {len(teams)} teams")

    # Create DataFrame
    data = []
    for team in teams:
        logo_url = get_logo_url(team)
        data.append({
            'team': team,
            'logo_url': logo_url
        })

    df = pd.DataFrame(data)

    # Create output directory
    output_dir = 'src/data/static'
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    output_file = os.path.join(output_dir, 'team_logos.csv')
    df.to_csv(output_file, index=False)

    logger.info(f"✅ Generated {output_file}")
    logger.info(f"   Teams with ESPN logos: {len([t for t in teams if t in ESPN_TEAM_IDS])}")
    logger.info(f"   Teams with generic logos: {len([t for t in teams if t not in ESPN_TEAM_IDS])}")

    return True


if __name__ == "__main__":
    success = generate_team_logos_csv()
    exit(0 if success else 1)
