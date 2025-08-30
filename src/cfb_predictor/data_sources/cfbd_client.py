from __future__ import annotations
import time
from typing import Optional, List, Dict, Any
import pandas as pd

from ..config import CFBD_API_KEY
try:
    import cfbd
except ImportError:
    cfbd = None

class CFBDClient:
    def __init__(self):
        if cfbd is None:
            raise ImportError("cfbd is not installed. `pip install cfbd`")
        if not CFBD_API_KEY:
            raise RuntimeError("CFBD_API_KEY not set. Add to .env or environment.")
        configuration = cfbd.Configuration()
        configuration.access_token = CFBD_API_KEY
        self.api_client = cfbd.ApiClient(configuration)
        self.games_api = cfbd.GamesApi(self.api_client)
        self.lines_api = cfbd.BettingApi(self.api_client)

    def get_games_df(self, season: int, season_type: str = "regular", classification: str = "fbs") -> pd.DataFrame:
        games = self.games_api.get_games(year=season, season_type=season_type, classification=classification)
        # Convert to dict records safely
        def to_dict(g):
            d = g.to_dict()
            # Map camelCase to snake_case for consistency with rest of code
            return {
                'id': d.get('id'),
                'season': d.get('season'),
                'week': d.get('week'),
                'season_type': d.get('seasonType'),
                'start_date': d.get('startDate'),
                'neutral_site': d.get('neutralSite', False),
                'home_team': d.get('homeTeam'),
                'away_team': d.get('awayTeam'),
                'home_points': d.get('homePoints'),
                'away_points': d.get('awayPoints'),
                'home_conference': d.get('homeConference'),
                'away_conference': d.get('awayConference'),
                'venue': d.get('venue')
            }
        df = pd.DataFrame([to_dict(g) for g in games])
        if df.empty:
            return df
        # All columns are already normalized, no need to filter
        return df

    def get_lines_df(self, season: int, week: Optional[int] = None, season_type: str = "regular") -> pd.DataFrame:
        # CFBD returns a list of game line objects with nested 'lines' per provider
        recs = self.lines_api.get_lines(year=season, week=week, season_type=season_type) if week else self.lines_api.get_lines(year=season, season_type=season_type)
        rows = []
        for r in recs:
            base = r.to_dict()
            gid = base.get('id')
            home = base.get('homeTeam')
            away = base.get('awayTeam')
            if base.get('lines'):
                for ln in base['lines']:
                    rows.append({
                        'game_id': gid,
                        'home_team': home,
                        'away_team': away,
                        'provider': ln.get('provider'),
                        'spread': ln.get('spread'),
                        'over_under': ln.get('overUnder'),
                        'formatted_spread': ln.get('formattedSpread'),
                        'spread_open': ln.get('spreadOpen'),
                        'over_under_open': ln.get('overUnderOpen'),
                        'last_updated': base.get('startDate'),  # Use startDate from game since lines don't have lastUpdated
                    })
        return pd.DataFrame(rows)
