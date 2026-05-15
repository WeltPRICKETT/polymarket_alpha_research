"""
Feature-set registry with leakage semantics.

`live` contains only features that can be computed before market resolution.
`research` keeps post-resolution return features for explanatory analysis.
"""

from __future__ import annotations

from typing import Dict, List

LIVE_FEATURE_COLS = [
    "early_entry_score",
    "contrarian_score",
    "information_ratio",
    "cross_market_diversification",
    "avg_holding_period",
    "trading_frequency",
    "capital_flow_centrality",
]

RESEARCH_ONLY_FEATURE_COLS = [
    "total_roi",
    "max_drawdown",
    "win_rate",
    "profit_loss_ratio",
]

DERIVED_RESEARCH_FEATURE_COLS = [
    "Risk_Adjusted_Return",
]

FEATURE_SETS: Dict[str, List[str]] = {
    "live": LIVE_FEATURE_COLS,
    "research": RESEARCH_ONLY_FEATURE_COLS + LIVE_FEATURE_COLS,
    "research_with_rar": RESEARCH_ONLY_FEATURE_COLS + LIVE_FEATURE_COLS + DERIVED_RESEARCH_FEATURE_COLS,
}

FEATURE_LEAKAGE_CLASS = {
    **{col: "pre_resolution_live" for col in LIVE_FEATURE_COLS},
    **{col: "post_resolution_research_only" for col in RESEARCH_ONLY_FEATURE_COLS},
    **{col: "post_resolution_derived_research_only" for col in DERIVED_RESEARCH_FEATURE_COLS},
}


def get_feature_columns(feature_set: str = "live") -> List[str]:
    """Return columns for a named feature set."""
    try:
        return FEATURE_SETS[feature_set]
    except KeyError as exc:
        allowed = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(f"Unknown feature_set={feature_set!r}. Allowed: {allowed}") from exc
