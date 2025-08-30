from __future__ import annotations
import typer
from typing import List, Optional
import pandas as pd
from .data.build_games import fetch_and_cache
from .models.train import train as train_models
from .predict import predict as predict_week
from .backtest import backtest as run_backtest

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def fetch_data(seasons: List[int] = typer.Argument(..., help="One or more seasons, e.g. 2023 2024")):
    """Fetch and cache games and lines from CFBD"""
    df = fetch_and_cache(seasons)
    typer.echo(f"Fetched {len(df)} games across {len(seasons)} seasons.")

@app.command()
def build_features(seasons: List[int] = typer.Argument(..., help="Seasons used when training (ensures games cached).")):
    """No-op placeholder: features are built during training/predict dynamically."""
    typer.echo("Features are built on the fly during training/predict; no action required.")

@app.command()
def train(seasons: List[int] = typer.Argument(..., help="Seasons to train on, e.g. 2022 2023 2024")):
    train_models(seasons)
    typer.echo("Models trained and saved under data/models/.")

@app.command()
def predict(season: int = typer.Option(...), week: Optional[str] = typer.Option("auto"), book: str = typer.Option("consensus"), min_edge: float = typer.Option(0.5)):
    out = predict_week(season=season, week=week, book=book, min_edge=min_edge)
    typer.echo(out.head().to_string(index=False))

@app.command()
def backtest(seasons: List[int] = typer.Argument(...), book: str = typer.Option("consensus"), min_edge: float = typer.Option(0.5)):
    summ = run_backtest(seasons, book=book, min_edge=min_edge)
    typer.echo(summ.to_string(index=False))

if __name__ == "__main__":
    app()
