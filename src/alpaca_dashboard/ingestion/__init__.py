"""Ingestion pipeline: push backtest pulses from Turso → market_pulse."""
from .adapters import (
    HttpDestination,
    PulseDestination,
    SqliteDestination,
)
from .converter import to_market_pulse_row
from .pipeline import push

__all__ = [
    "HttpDestination",
    "PulseDestination",
    "SqliteDestination",
    "push",
    "to_market_pulse_row",
]
