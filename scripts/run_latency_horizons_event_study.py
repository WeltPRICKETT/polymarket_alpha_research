#!/usr/bin/env python3
"""Run sub-minute and short-horizon matched event study for latency-arbitrage falsification."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_full_data_matched_event_study as study  # noqa: E402


study.HORIZONS = {
    "10s": (10, 20),
    "30s": (30, 30),
    "60s": (60, 60),
    "5m": (5 * 60, 15 * 60),
    "15m": (15 * 60, 15 * 60),
    "60m": (60 * 60, 60 * 60),
}


if __name__ == "__main__":
    study.main()
