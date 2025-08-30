from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple

import math

@dataclass
class EloParams:
    k: float = 20.0
    home_field: float = 55.0  # Elo points
    regress: float = 0.25     # off-season regression to mean (1500)

@dataclass
class EloRatings:
    params: EloParams = field(default_factory=EloParams)
    ratings: Dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.get(team, 1500.0)

    def set(self, team: str, rating: float):
        self.ratings[team] = rating

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1 + 10 ** (-(ra - rb) / 400))

    def update_game(self, home: str, away: str, home_points: int, away_points: int, neutral: bool = False):
        ra = self.get(home)
        rb = self.get(away)

        # Home field
        hfa = 0.0 if neutral else self.params.home_field
        ea = self.expected(ra + hfa, rb)
        eb = 1.0 - ea

        # Actual results
        if home_points is None or away_points is None:
            return  # skip unfinished games
        if home_points > away_points:
            sa, sb = 1.0, 0.0
        elif home_points < away_points:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        margin = abs(home_points - away_points)
        # Margin multiplier (common Elo tweak)
        mov_mult = math.log(max(margin, 1) + 1) * (2.2 / ((abs(ra - rb) * 0.001) + 2.2))

        ra_new = ra + self.params.k * mov_mult * (sa - ea)
        rb_new = rb + self.params.k * mov_mult * (sb - eb)

        self.set(home, ra_new)
        self.set(away, rb_new)

    def regress_offseason(self):
        for t, r in list(self.ratings.items()):
            self.ratings[t] = 1500 + (r - 1500) * (1 - self.params.regress)
