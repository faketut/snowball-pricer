"""Tests for the yfinance delayed poll feed — everything mocked, no network.

Covers: normalization (incl. missing bid/ask -> None, never fabricated),
is_delayed hardcoded True, poll-cycle stamping, error/429 resilience
(feed survives, loop continues), and the OptionQuote.is_valid() hardening
for None/NaN quotes.
"""
import math

import pandas as pd
import pytest

from snowball_pricer.feeds.yfinance import YFinancePollFeed, IS_DELAYED
from snowball_pricer.tick import OptionQuote


# ------------------------------------------------------------------ fakes
def _frame(rows):
    return pd.DataFrame(rows)


class FakeTicker:
    """Mimics the yfinance.Ticker surface we use: options, option_chain,
    fast_info, history."""

    def __init__(self, symbol, *, expiries, chains, spot=100.0,
                 fail_on=(), raise_options=False):
        self.symbol = symbol
        self._expiries = expiries
        self._chains = chains  # {expiry: (calls_df, puts_df)}
        self._spot = spot
        self._fail_on = set(fail_on)
        self._raise_options = raise_options

    @property
    def options(self):
        if self._raise_options:
            raise RuntimeError("429 Too Many Requests")
        return self._expiries

    def option_chain(self, expiry):
        if expiry in self._fail_on:
            raise RuntimeError("429 Too Many Requests")
        return self._chains[expiry]

    @property
    def fast_info(self):
        return {"lastPrice": self._spot}

    def history(self, period="1d"):
        return pd.DataFrame({"Close": [self._spot]})


def _chain(strike=100.0, bid=2.5, ask=2.7, n=3):
    rows = [{"strike": strike + i, "bid": bid, "ask": ask,
             "volume": 10, "openInterest": 100} for i in range(n)]
    df = _frame(rows)
    return (df, df.copy())  # (calls, puts)


def _factory(tickers):
    def make(sym):
        if sym not in tickers:
            raise KeyError(sym)
        t = tickers[sym]
        if isinstance(t, Exception):
            raise t
        return t
    return make


def _feed(tickers, **kw):
    kw.setdefault("poll_interval_s", 0.01)
    kw.setdefault("request_pace_s", 0.0)
    kw.setdefault("sleep", lambda s: None)
    return YFinancePollFeed(["SPY"], ticker_factory=_factory(tickers), **kw)


# ------------------------------------------------------------------ tests
def test_is_delayed_hardcoded_true():
    tickers = {"SPY": FakeTicker("SPY", expiries=["2027-01-15"],
                                 chains={"2027-01-15": _chain()})}
    feed = _feed(tickers)
    assert IS_DELAYED is True
    quotes, _ = feed.poll_cycle()
    assert quotes, "expected quotes from the fake chain"
    assert all(q.is_delayed for q in quotes)
    assert all(q.underlying_id == "SPY" for q in quotes)
    assert all(q.expiry > 0 for q in quotes)
    calls = [q for q in quotes if q.is_call]
    puts = [q for q in quotes if not q.is_call]
    assert calls and puts


def test_missing_bid_ask_emitted_as_none_never_fabricated():
    rows = [{"strike": 100.0, "bid": float("nan"), "ask": float("nan")},
            {"strike": 101.0, "bid": 0.0, "ask": 0.0},       # zero quotes
            {"strike": 102.0, "bid": 2.5, "ask": 2.7}]        # good quote
    tickers = {"SPY": FakeTicker("SPY", expiries=["2027-01-15"],
                                 chains={"2027-01-15": (_frame(rows),
                                                        _frame(rows))})}
    feed = _feed(tickers)
    quotes, stats = feed.poll_cycle()
    n_rows = len(quotes)
    assert n_rows == 6  # 3 strikes x call/put, nothing hidden
    missing = [q for q in quotes if q.bid is None or q.ask is None]
    assert len(missing) == 2  # the NaN row, call+put
    assert all(not q.is_valid() for q in missing)
    # Zero-bid quotes are structurally invalid too (bid > 0 required).
    zero = [q for q in quotes
            if q.bid == 0.0 and q.strike == 101.0]
    assert zero and all(not q.is_valid() for q in zero)
    good = [q for q in quotes if q.strike == 102.0]
    assert good and all(q.is_valid() for q in good)
    # Completeness reflects reality: 2 valid of 6 rows.
    assert stats["n_rows"] == 6
    assert stats["n_valid"] == 2
    assert stats["completeness"] == pytest.approx(2 / 6)


def test_is_valid_hardening_none_nan():
    base = dict(ts=1.0, underlying_id="SPY", expiry=0.5, strike=100.0,
                is_call=True, underlying_price=100.0, rate=0.03,
                div_yield=0.0)
    assert OptionQuote(bid=None, ask=2.5, **base).is_valid() is False
    assert OptionQuote(bid=2.5, ask=None, **base).is_valid() is False
    assert OptionQuote(bid=float("nan"), ask=2.5, **base).is_valid() is False
    assert OptionQuote(bid=2.5, ask=float("nan"), **base).is_valid() is False
    assert OptionQuote(bid=2.5, ask=2.7, **base).is_valid() is True
    assert OptionQuote(bid=2.8, ask=2.7, **base).is_valid() is False  # crossed


def test_error_resilience_feed_survives_bad_symbol():
    good = FakeTicker("SPY", expiries=["2027-01-15"],
                      chains={"2027-01-15": _chain()})
    tickers = {"SPY": good, "BAD": RuntimeError("boom")}
    feed = YFinancePollFeed(["SPY", "BAD"],
                            ticker_factory=_factory(tickers),
                            poll_interval_s=0.01, request_pace_s=0.0,
                            sleep=lambda s: None)
    quotes, stats = feed.poll_cycle()
    assert quotes, "good symbol must still produce quotes"
    assert stats["underlyings"]["BAD"]["error"] is not None
    assert stats["underlyings"]["SPY"]["error"] is None
    # subscribe() keeps going across cycles despite the bad symbol
    feed2 = YFinancePollFeed(["BAD"], ticker_factory=_factory(tickers),
                             poll_interval_s=0.01, request_pace_s=0.0,
                             max_cycles=2, sleep=lambda s: None)
    assert list(feed2.subscribe()) == []
    assert len(feed2.cycle_stats) == 2
    assert all(s["n_rows"] == 0 for s in feed2.cycle_stats)


def test_429_on_options_list_skips_cycle_without_crash():
    tickers = {"SPY": FakeTicker("SPY", expiries=[], chains={},
                                 raise_options=True)}
    feed = _feed(tickers, max_cycles=3)
    quotes = list(feed.subscribe())
    assert quotes == []
    assert len(feed.cycle_stats) == 3
    # failures recorded, loop ran all cycles
    assert all(s["underlyings"]["SPY"]["error"] for s in feed.cycle_stats)


def test_cycle_stamping_monotonic():
    tickers = {"SPY": FakeTicker("SPY", expiries=["2027-01-15"],
                                 chains={"2027-01-15": _chain()})}
    feed = _feed(tickers, max_cycles=3)
    list(feed.subscribe())
    ids = [s["cycle_id"] for s in feed.cycle_stats]
    assert ids == [1, 2, 3]
    assert all("completeness" in s and "ts_start" in s
               for s in feed.cycle_stats)


def test_k_band_filter_opt_in():
    rows = [{"strike": 50.0, "bid": 2.5, "ask": 2.7},    # k = ln(0.5)
            {"strike": 100.0, "bid": 2.5, "ask": 2.7},   # k = 0
            {"strike": 300.0, "bid": 2.5, "ask": 2.7}]   # k = ln(3)
    tickers = {"SPY": FakeTicker("SPY", expiries=["2027-01-15"],
                                 chains={"2027-01-15": (_frame(rows),
                                                        _frame(rows))},
                                 spot=100.0)}
    feed = _feed(tickers, k_band=(-0.5, 0.3))
    quotes, _ = feed.poll_cycle()
    strikes = {q.strike for q in quotes}
    assert strikes == {100.0}, f"band should keep only ATM, got {strikes}"


def test_expired_expiries_skipped():
    tickers = {"SPY": FakeTicker("SPY", expiries=["2020-01-17", "2027-01-15"],
                                 chains={"2027-01-15": _chain()})}
    feed = _feed(tickers)
    quotes, stats = feed.poll_cycle()
    assert quotes
    assert all(q.expiry > 0 for q in quotes)
    assert stats["underlyings"]["SPY"]["expiries"] == 1


def test_min_expiry_days_skips_weeklies():
    tickers = {"SPY": FakeTicker("SPY",
                                 expiries=["2020-01-17", "2027-01-15"],
                                 chains={"2027-01-15": _chain()})}
    feed = _feed(tickers, min_expiry_days=7)
    quotes, stats = feed.poll_cycle()
    assert quotes
    assert all(q.expiry * 365.0 >= 7 for q in quotes)
    assert stats["underlyings"]["SPY"]["expiries"] == 1
