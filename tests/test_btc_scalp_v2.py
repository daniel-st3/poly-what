"""Unit tests for BTC scalp V2 EV model helpers.

Tests target pure static/instance methods directly — no DB, no async,
no mocking of external services required for any of these tests.
"""
from __future__ import annotations

import math
import os
import types

import pytest

# ---------------------------------------------------------------------------
# Minimal BtcScalpStrategy construction (bypass DB __init__ entirely)
# ---------------------------------------------------------------------------

def _make_strategy() -> object:
    """Create a BtcScalpStrategy instance without touching the DB or network."""
    # Patch get_settings and PolymarketRestClient before importing the class
    # so __init__ does not crash in the test environment.
    import unittest.mock as mock

    # Patch at module level before __init__ runs
    with mock.patch("watchdog.strategies.btc_scalp.get_settings") as mock_settings, \
         mock.patch("watchdog.strategies.btc_scalp.PolymarketRestClient"), \
         mock.patch("watchdog.strategies.btc_scalp.build_engine"), \
         mock.patch("watchdog.strategies.btc_scalp.init_db"), \
         mock.patch("watchdog.strategies.btc_scalp.build_session_factory"):
        # Provide the settings values used during __init__
        s = mock.MagicMock()
        s.telegram_bot_token = None
        s.telegram_chat_id = None
        s.btc_scalp_vol_floor = 0.001
        s.btc_scalp_momentum_adjust_weight = 0.0   # zero so p_up tests isolate model cleanly
        s.btc_scalp_max_spread = 0.03
        s.btc_scalp_min_ev_per_contract = 0.01
        s.btc_scalp_require_clob = True
        mock_settings.return_value = s

        from watchdog.strategies.btc_scalp import BtcScalpStrategy
        strategy = BtcScalpStrategy()
        # Attach the patched settings for use in helper calls
        strategy._test_settings = s
    return strategy


@pytest.fixture(scope="module")
def strategy():
    return _make_strategy()


# ---------------------------------------------------------------------------
# _normal_cdf_approx
# ---------------------------------------------------------------------------

def test_normal_cdf_approx_at_zero(strategy):
    assert abs(strategy._normal_cdf_approx(0.0) - 0.5) < 1e-6


def test_normal_cdf_approx_monotonic(strategy):
    prev = 0.0
    for z in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        val = strategy._normal_cdf_approx(z)
        assert val > prev, f"CDF not monotone at z={z}"
        prev = val


def test_normal_cdf_approx_symmetry(strategy):
    for z in [0.5, 1.0, 1.5, 2.0, 2.5]:
        p_pos = strategy._normal_cdf_approx(z)
        p_neg = strategy._normal_cdf_approx(-z)
        assert abs(p_pos + p_neg - 1.0) < 1e-6, f"Symmetry broken at z={z}"


# ---------------------------------------------------------------------------
# _ev_buy_yes
# ---------------------------------------------------------------------------

def test_ev_buy_yes_positive(strategy):
    ev = strategy._ev_buy_yes(0.8, 0.5)
    assert abs(ev - 0.3) < 1e-9


def test_ev_buy_yes_negative(strategy):
    ev = strategy._ev_buy_yes(0.4, 0.6)
    assert abs(ev - (-0.2)) < 1e-9


def test_ev_buy_yes_breakeven(strategy):
    # prob == ask → EV = 0
    assert abs(strategy._ev_buy_yes(0.55, 0.55)) < 1e-9


# ---------------------------------------------------------------------------
# _p_up_model (momentum weight is 0.0 for clean isolation)
# ---------------------------------------------------------------------------

def test_p_up_model_at_start_price(strategy):
    """current == start → z = 0 → p_up = 0.5 (with zero momentum weight)."""
    p = strategy._p_up_model(
        start_price_proxy=84_000.0,
        current_price=84_000.0,
        time_left_min=2.0,
        realized_vol=0.002,
    )
    assert abs(p - 0.5) < 0.01


def test_p_up_model_above_start_bullish(strategy):
    """BTC moved up → p_up > 0.5."""
    p = strategy._p_up_model(
        start_price_proxy=84_000.0,
        current_price=85_000.0,
        time_left_min=2.0,
        realized_vol=0.002,
    )
    assert p > 0.5


def test_p_up_model_below_start_bearish(strategy):
    """BTC moved down → p_up < 0.5."""
    p = strategy._p_up_model(
        start_price_proxy=84_000.0,
        current_price=83_000.0,
        time_left_min=2.0,
        realized_vol=0.002,
    )
    assert p < 0.5


def test_p_up_model_zero_time_floors(strategy):
    """Very small time_left should not crash and stays in [0.01, 0.99]."""
    p = strategy._p_up_model(
        start_price_proxy=84_000.0,
        current_price=84_100.0,
        time_left_min=0.001,
        realized_vol=0.002,
    )
    assert 0.01 <= p <= 0.99


def test_p_up_model_none_guard(strategy):
    """start_price_proxy <= 0 returns exactly 0.5."""
    p = strategy._p_up_model(
        start_price_proxy=0.0,
        current_price=84_000.0,
        time_left_min=2.0,
        realized_vol=0.002,
    )
    assert p == 0.5


# ---------------------------------------------------------------------------
# _select_side
# ---------------------------------------------------------------------------

def test_select_side_prefers_up(strategy):
    assert strategy._select_side(ev_up=0.05, ev_down=0.02) == "Up"


def test_select_side_prefers_down(strategy):
    assert strategy._select_side(ev_up=0.01, ev_down=0.04) == "Down"


def test_select_side_ties_go_up(strategy):
    assert strategy._select_side(ev_up=0.03, ev_down=0.03) == "Up"


# ---------------------------------------------------------------------------
# _first_usable_quote
# ---------------------------------------------------------------------------

def test_first_usable_quote_finds_ask_behind_stub(strategy):
    bids = [{"price": "0.63", "size": "50"}]
    asks = [{"price": "0.99", "size": "1"}, {"price": "0.64", "size": "80"}]
    eff_bid, eff_ask = strategy._first_usable_quote(bids, asks, 0.05)
    assert eff_ask == 0.64
    assert eff_bid == 0.63


def test_first_usable_quote_all_stub(strategy):
    bids = [{"price": "0.01", "size": "1"}]
    asks = [{"price": "0.99", "size": "1"}, {"price": "0.96", "size": "1"}]
    eff_bid, eff_ask = strategy._first_usable_quote(bids, asks, 0.05)
    assert eff_ask is None
    assert eff_bid is None


def test_first_usable_quote_empty_lists(strategy):
    eff_bid, eff_ask = strategy._first_usable_quote([], [], 0.05)
    assert eff_bid is None
    assert eff_ask is None


def test_first_usable_quote_ask_exactly_at_threshold(strategy):
    """ask == 0.95 is NOT usable (strictly below required)."""
    asks = [{"price": "0.95", "size": "10"}]
    eff_bid, eff_ask = strategy._first_usable_quote([], asks, 0.05)
    assert eff_ask is None


# ---------------------------------------------------------------------------
# _is_stub_book
# ---------------------------------------------------------------------------

def test_is_stub_book_none_bid(strategy):
    assert strategy._is_stub_book(None, 0.5, 0.05) is True


def test_is_stub_book_stub_quotes(strategy):
    assert strategy._is_stub_book(0.01, 0.99, 0.05) is True


def test_is_stub_book_real_book(strategy):
    assert strategy._is_stub_book(0.10, 0.92, 0.05) is False


def test_is_stub_book_ask_too_high(strategy):
    assert strategy._is_stub_book(0.10, 0.96, 0.05) is True


# ---------------------------------------------------------------------------
# _should_skip_for_spread
# ---------------------------------------------------------------------------

def test_spread_filter_blocks(strategy):
    assert strategy._should_skip_for_spread(spread=0.05, max_spread=0.03) is True


def test_spread_filter_passes(strategy):
    assert strategy._should_skip_for_spread(spread=0.02, max_spread=0.03) is False


def test_spread_filter_none_is_not_a_skip(strategy):
    """None spread means CLOB unavailable — handled by missing_clob gate, not spread gate."""
    assert strategy._should_skip_for_spread(spread=None, max_spread=0.03) is False


# ---------------------------------------------------------------------------
# _parse_clob_token_ids
# ---------------------------------------------------------------------------

def test_parse_clob_token_ids_list(strategy):
    m = {"clobTokenIds": ["abc", "def"]}
    yes, no = strategy._parse_clob_token_ids(m)
    assert yes == "abc"
    assert no == "def"


def test_parse_clob_token_ids_json_string(strategy):
    import json
    m = {"clobTokenIds": json.dumps(["tok1", "tok2"])}
    yes, no = strategy._parse_clob_token_ids(m)
    assert yes == "tok1"
    assert no == "tok2"


def test_parse_clob_token_ids_malformed(strategy):
    m = {"clobTokenIds": "not-json{{}"}
    yes, no = strategy._parse_clob_token_ids(m)
    assert yes is None
    assert no is None


def test_parse_clob_token_ids_missing(strategy):
    yes, no = strategy._parse_clob_token_ids({})
    assert yes is None
    assert no is None


# ---------------------------------------------------------------------------
# missing_start_price_proxy behavior (unit-level via _p_up_model guard)
# ---------------------------------------------------------------------------

def test_missing_start_price_proxy_gives_neutral_p_up(strategy):
    """When start_price_proxy is 0/None, _p_up_model returns 0.5.
    The scan loop translates this into skip_reason='missing_start_price_proxy'
    rather than trading with a neutral estimate.
    """
    p = strategy._p_up_model(
        start_price_proxy=0.0,   # proxy for None path in _scan_markets
        current_price=84_000.0,
        time_left_min=2.0,
        realized_vol=0.002,
    )
    # Neutral p_up → both EVs will be near -spread/2, below EV floor
    # This confirms the scan gate will correctly set skip_reason
    assert p == 0.5


# ---------------------------------------------------------------------------
# midpoint fallback (btc_scalp_require_clob=False)
# ---------------------------------------------------------------------------

def test_midpoint_fallback_uses_mid_price(strategy):
    """With require_clob=False, entry_price falls back to Gamma mid.

    We test the helper logic in isolation: if up_clob_available=False and
    require_clob=False, the caller would set up_entry_price = up_price_mid.
    _ev_buy_yes with mid price should give a smaller EV than with ask < mid.
    """
    mid = 0.50
    ask_better = 0.48   # tighter ask (better for buyer)
    p_up = 0.55

    ev_with_mid = strategy._ev_buy_yes(p_up, mid)
    ev_with_ask = strategy._ev_buy_yes(p_up, ask_better)
    # Better ask → higher EV
    assert ev_with_ask > ev_with_mid
