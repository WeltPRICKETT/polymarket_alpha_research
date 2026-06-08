import json
from pathlib import Path

import pandas as pd

from src.data_ingestion.external_sources import (
    build_sii_market_event_map,
    build_wallet_event_features,
    build_resolution_coverage_summary,
    build_event_context_features,
    convert_sii_users_to_transactions,
    parse_clob_market_resolution,
    apply_secondary_terminal_price_fallback,
    summarize_event_metadata,
)


def test_convert_sii_users_prefers_condition_id_and_signed_amount():
    users = pd.DataFrame(
        [
            {
                "transaction_hash": "0xabc",
                "log_index": 7,
                "condition_id": "cond-1",
                "market_id": "market-1",
                "user": "0xwallet",
                "role": "maker",
                "token_amount": 10.0,
                "usd_amount": 4.0,
                "price": 0.4,
                "timestamp": 1_700_000_000,
            },
            {
                "transaction_hash": "0xdef",
                "log_index": 8,
                "condition_id": "cond-2",
                "market_id": "market-2",
                "user": "0xwallet",
                "role": "taker",
                "token_amount": -5.0,
                "usd_amount": 3.0,
                "price": 0.6,
                "timestamp": 1_700_000_060,
            },
        ]
    )

    converted = convert_sii_users_to_transactions(users)

    assert list(converted["market_id"]) == ["cond-1", "cond-2"]
    assert list(converted["side"]) == ["BUY", "SELL"]
    assert list(converted["amount"]) == [10.0, 5.0]
    assert converted["transaction_id"].tolist() == [
        "0xabc:7:maker:0xwallet",
        "0xdef:8:taker:0xwallet",
    ]
    assert converted["timestamp"].dt.tz is None


def test_convert_sii_users_accepts_actual_address_and_direction_schema():
    users = pd.DataFrame(
        [
            {
                "transaction_hash": "0xabc",
                "log_index": 7,
                "condition_id": "cond-1",
                "market_id": "market-1",
                "address": "0xwallet",
                "role": "maker",
                "direction": "SELL",
                "token_amount": 10.0,
                "usd_amount": 4.0,
                "price": 0.4,
                "timestamp": 1_700_000_000,
            }
        ]
    )

    converted = convert_sii_users_to_transactions(users)

    assert converted.loc[0, "address"] == "0xwallet"
    assert converted.loc[0, "side"] == "SELL"
    assert converted.loc[0, "amount"] == 10.0


def test_build_event_context_features_parses_tags_and_dates(tmp_path: Path):
    events = pd.DataFrame(
        [
            {
                "id": 101,
                "slug": "fed-decision",
                "title": "Fed decision?",
                "seriesSlug": "fed-rates",
                "tags": json.dumps(
                    [
                        {"label": "Politics", "slug": "politics"},
                        {"label": "Fed Rates", "slug": "fed-rates"},
                    ]
                ),
                "startDate": "2025-01-01T00:00:00Z",
                "endDate": "2025-01-10T00:00:00Z",
                "closedTime": "2025-01-11T00:00:00Z",
                "volume": "123.45",
                "liquidity": "9.5",
                "market_count": "4",
                "closed": True,
            }
        ]
    )
    path = tmp_path / "events.csv"
    events.to_csv(path, index=False)

    features = build_event_context_features(path)
    summary = summarize_event_metadata(features)

    assert features.loc[0, "event_id"] == "101"
    assert features.loc[0, "tag_slugs"] == "politics|fed-rates"
    assert features.loc[0, "duration_days"] == 9.0
    assert summary["rows"] == 1
    assert summary["closed_events"] == 1


def test_build_sii_market_event_map_and_wallet_event_features():
    transactions = pd.DataFrame(
        [
            {"address": "a1", "market_id": "cond-1", "amount": 10, "price": 0.5},
            {"address": "a1", "market_id": "cond-2", "amount": 20, "price": 0.4},
            {"address": "a2", "market_id": "cond-3", "amount": 5, "price": 0.2},
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "id": "m1",
                "condition_id": "cond-1",
                "question": "Q1",
                "slug": "q1",
                "event_id": "e1",
                "event_slug": "event-one",
                "event_title": "Event One",
                "neg_risk": 1,
                "volume": 100.0,
            },
            {
                "id": "m2",
                "condition_id": "cond-2",
                "question": "Q2",
                "slug": "q2",
                "event_id": "e1",
                "event_slug": "event-one",
                "event_title": "Event One",
                "neg_risk": 0,
                "volume": 200.0,
            },
            {
                "id": "m3",
                "condition_id": "cond-3",
                "question": "Q3",
                "slug": "q3",
                "event_id": "e2",
                "event_slug": "event-two",
                "event_title": "Event Two",
                "neg_risk": 0,
                "volume": 300.0,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "event_slug": "event-one",
                "event_title": "Event One Local",
                "category": "politics",
                "series_slug": "macro",
                "primary_tag_slug": "fed",
                "topic_slug": "rates",
                "market_count": 2,
                "volume": 1000.0,
                "liquidity": 100.0,
                "duration_days": 9.0,
                "enable_order_book": True,
            },
            {
                "event_id": "e2",
                "event_slug": "event-two",
                "event_title": "Event Two Local",
                "category": "sports",
                "series_slug": "sports",
                "primary_tag_slug": "nba",
                "topic_slug": "basketball",
                "market_count": 1,
                "volume": 500.0,
                "liquidity": 50.0,
                "duration_days": 1.0,
                "enable_order_book": True,
            },
        ]
    )

    market_map = build_sii_market_event_map(transactions, markets, events)
    wallet_features = build_wallet_event_features(transactions, market_map)

    assert market_map["sii_market_id"].notna().sum() == 3
    assert market_map.loc[market_map["market_id"] == "cond-1", "event_title"].iloc[0] == "Event One Local"
    a1 = wallet_features.set_index("address").loc["a1"]
    assert a1["unique_events"] == 1
    assert a1["unique_series"] == 1
    assert a1["unique_primary_tags"] == 1
    assert a1["unique_topics"] == 1
    assert a1["series_notional_hhi"] == 1.0
    assert a1["same_event_multi_market_share"] == 1.0
    assert a1["mean_markets_per_event_traded"] == 2.0
    assert a1["neg_risk_trade_share"] == 0.5
    assert a1["order_book_trade_share"] == 1.0


def test_parse_clob_market_resolution_uses_winner_field_only():
    payload = {
        "question": "Will X happen?",
        "closed": True,
        "market_slug": "will-x-happen",
        "end_date_iso": "2024-01-01T00:00:00Z",
        "tokens": [
            {"outcome": "Yes", "winner": False, "price": 0},
            {"outcome": "No", "winner": True, "price": 1},
        ],
    }

    result = parse_clob_market_resolution("0xabc", payload)

    assert result["resolution"] == "No"
    assert result["winning_outcome"] == "No"
    assert result["winning_outcome_index"] == 1
    assert result["resolution_source"] == "clob_winner_field"
    assert result["resolution_confidence"] == "primary"
    assert result["is_phase26_primary"] is True
    assert result["is_secondary_fallback"] is False


def test_parse_clob_market_resolution_does_not_argmax_without_winner():
    payload = {
        "question": "Will Y happen?",
        "closed": True,
        "tokens": [
            {"outcome": "Yes", "winner": False, "price": 1},
            {"outcome": "No", "winner": False, "price": 0},
        ],
    }

    result = parse_clob_market_resolution("0xdef", payload)

    assert result["resolution"] == "__OPEN__"
    assert result["winning_outcome"] == ""
    assert result["resolution_source"] == "clob_no_winner"


def test_secondary_fallback_recovers_zero_one_price():
    record = parse_clob_market_resolution(
        "0xsecondary1",
        {
            "closed": True,
            "tokens": [
                {"token_id": "yes-token", "outcome": "Yes", "winner": False, "price": 0},
                {"token_id": "no-token", "outcome": "No", "winner": False, "price": 1},
            ],
        },
    )

    result = apply_secondary_terminal_price_fallback(record)

    assert result["resolution_source"] == "clob_terminal_price_argmax_secondary"
    assert result["winner_token_id"] == "no-token"
    assert result["winner_outcome"] == "No"
    assert result["is_phase26_primary"] is False
    assert result["is_secondary_fallback"] is True
    assert result["terminal_price_gap"] == 1.0


def test_secondary_fallback_recovers_one_zero_price():
    record = parse_clob_market_resolution(
        "0xsecondary2",
        {
            "closed": True,
            "tokens": [
                {"token_id": "yes-token", "outcome": "Yes", "winner": False, "price": 1},
                {"token_id": "no-token", "outcome": "No", "winner": False, "price": 0},
            ],
        },
    )

    result = apply_secondary_terminal_price_fallback(record)

    assert result["resolution_source"] == "clob_terminal_price_argmax_secondary"
    assert result["winner_token_id"] == "yes-token"
    assert result["winner_outcome"] == "Yes"
    assert result["is_phase26_primary"] is False


def test_secondary_fallback_rejects_non_decisive_prices():
    record = parse_clob_market_resolution(
        "0xnondecisive",
        {
            "closed": True,
            "tokens": [
                {"token_id": "yes-token", "outcome": "Yes", "winner": False, "price": 0.7},
                {"token_id": "no-token", "outcome": "No", "winner": False, "price": 0.3},
            ],
        },
    )

    result = apply_secondary_terminal_price_fallback(record)

    assert result["resolution_source"] == "unresolved_non_decisive"
    assert result["winner_token_id"] is None
    assert result["is_secondary_fallback"] is False


def test_secondary_fallback_rejects_tied_high_prices():
    record = parse_clob_market_resolution(
        "0xtied",
        {
            "closed": True,
            "tokens": [
                {"token_id": "yes-token", "outcome": "Yes", "winner": False, "price": 0.99},
                {"token_id": "no-token", "outcome": "No", "winner": False, "price": 0.99},
            ],
        },
    )

    result = apply_secondary_terminal_price_fallback(record)

    assert result["resolution_source"] == "unresolved_non_decisive"
    assert result["winner_token_id"] is None


def test_phase26_coverage_ignores_secondary_fallback():
    summary = build_resolution_coverage_summary(
        pd.DataFrame(
            [
                {"market_id": f"primary-{i}", "resolution_source": "clob_winner_field"}
                for i in range(937)
            ]
            + [
                {"market_id": f"secondary-{i}", "resolution_source": "clob_terminal_price_argmax_secondary"}
                for i in range(172)
            ]
            + [
                {"market_id": f"unresolved-{i}", "resolution_source": "unresolved_non_decisive"}
                for i in range(125)
            ]
        )
    )

    assert summary["sample_markets"] == 1234
    assert summary["primary_clob_winner_field_coverage"] == 937 / 1234
    assert summary["combined_usable_resolution_coverage"] == (937 + 172) / 1234
    assert summary["phase26_primary_gate_passed"] is False
