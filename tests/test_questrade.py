"""Tests for the Questrade adapter — everything mocked, no real credentials.

Covers: auth flow (incl. the credential-never-logged invariant), REST
retry semantics (401 re-auth, 429 backoff), option-chain parsing,
quote normalization, WS reconnect/resubscribe, and malformed-message
resilience.
"""
import itertools
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from snowball_pricer.feeds.questrade import (
    HttpResponse,
    QuestradeAuth,
    QuestradeAuthError,
    QuestradeError,
    QuestradeFeed,
    QuestradeRest,
    parse_option_chain,
)

TOKEN_URL_PREFIX = "https://login.questrade.com/oauth2/token"


# ---------------------------------------------------------------- helpers
class FakeHttp:
    """Scripted HTTP layer: script = list of (status, headers, body)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, method, url, headers, data):
        self.calls.append((method, url, dict(headers), data))
        status, headers_, body = self.script.pop(0)
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        return HttpResponse(status, dict(headers_), raw)


def token_ok(access="AT-1", refresh="RT-2"):
    return (200, {}, {
        "access_token": access,
        "token_type": "Bearer",
        "expires_in": 1800,
        "refresh_token": refresh,
        "api_server": "https://api01.iq.questrade.com/",
    })


class FakeAuth:
    def __init__(self):
        self.access_token = "AT"
        self.api_server = "https://api01.iq.questrade.com/"
        self.refreshes = 0

    def refresh(self):
        self.refreshes += 1

    def auth_headers(self):
        return {"Authorization": f"Bearer {self.access_token}"}


class FakeRest:
    """Canned REST surface for feed tests."""

    def __init__(self, chain, spot=600.0):
        self.auth = FakeAuth()
        self.chain = chain
        self.spot = spot
        self.stream_ports = 0

    def search_symbols(self, prefix):
        return [{"symbolId": 8049, "symbol": prefix.upper()}]

    def get_quote(self, symbol_id):
        return {"lastTradePrice": self.spot}

    def get_option_chain(self, symbol_id):
        return self.chain

    def get_stream_port(self, ids):
        self.stream_ports += 1
        return 443


def chain_json(days_ahead=60, strikes=(590.0, 600.0, 610.0), sid0=1000):
    expiry = (datetime.now(timezone.utc)
              + timedelta(days=days_ahead)).strftime("%Y-%m-%dT00:00:00.000000-04:00")
    legs, sid = [], sid0
    for k in strikes:
        legs.append({"strikePrice": k, "callSymbolId": sid,
                     "putSymbolId": sid + 1})
        sid += 2
    return [{"expiryDate": expiry,
             "chainPerRoot": [{"root": "SPY",
                               "chainPerStrikePrice": legs}],
             "multiplier": 100}]


class FakeWS:
    """Scripted websocket: yields messages, then raises ConnectionError."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []
        self.closed = False

    def send(self, data):
        self.sent.append(data)

    def recv(self):
        if not self._messages:
            raise ConnectionError("peer closed")
        return self._messages.pop(0)

    def close(self):
        self.closed = True


def make_feed(chain, messages_per_conn, **kw):
    """Feed whose ws_factory serves one FakeWS per connection."""
    conns = [FakeWS(list(m)) for m in messages_per_conn]
    created = []

    def factory(url):
        ws = conns.pop(0)
        created.append((url, ws))
        return ws

    sleeps = []
    feed = QuestradeFeed(
        ["SPY"], rate=0.03, div_yield=0.01, min_days_to_expiry=1,
        rest=FakeRest(chain), ws_factory=factory,
        sleeper=sleeps.append, max_reconnects=kw.pop("max_reconnects", 0),
        **kw,
    )
    return feed, created, sleeps


def drain_one(feed):
    """Next quote, then expect the stream to end (QuestradeError)."""
    it = feed.subscribe()
    q = next(it)
    with pytest.raises(QuestradeError):
        next(it)
    return q


# ---------------------------------------------------------------- auth
def test_auth_requires_token_source(monkeypatch):
    monkeypatch.delenv("QUESTRADE_REFRESH_TOKEN", raising=False)
    with pytest.raises(RuntimeError) as e:
        QuestradeAuth()
    assert "QUESTRADE_REFRESH_TOKEN" in str(e.value)


def test_auth_refresh_posts_and_parses():
    http = FakeHttp([token_ok()])
    auth = QuestradeAuth(refresh_token="RT-0", http=http)
    auth.refresh()
    method, url, headers, _ = http.calls[0]
    assert method == "POST"
    assert url.startswith(TOKEN_URL_PREFIX)
    assert "refresh_token=RT-0" in url
    assert auth.access_token == "AT-1"
    assert auth.api_server == "https://api01.iq.questrade.com/"
    assert auth.auth_headers() == {"Authorization": "Bearer AT-1"}


def test_auth_post_405_falls_back_to_get():
    http = FakeHttp([(405, {}, {}), token_ok()])
    auth = QuestradeAuth(refresh_token="RT-0", http=http)
    auth.refresh()  # must not raise
    assert [c[0] for c in http.calls] == ["POST", "GET"]
    assert auth.access_token == "AT-1"


def test_auth_bad_status_raises():
    http = FakeHttp([(401, {}, {"error": "bad"})])
    auth = QuestradeAuth(refresh_token="RT-0", http=http)
    with pytest.raises(QuestradeAuthError):
        auth.refresh()


def test_auth_rotates_refresh_token_in_memory_only(tmp_path):
    http = FakeHttp([token_ok(refresh="RT-NEW")])
    auth = QuestradeAuth(refresh_token="RT-0", http=http)
    auth.refresh()
    assert auth._refresh_token == "RT-NEW"
    # Nothing was written to disk by the adapter.
    assert list(tmp_path.iterdir()) == []


def test_credential_never_logged(caplog):
    secret_rt = "SECRET-RT-xyz-123"
    secret_at = "SECRET-AT-abc-456"
    http = FakeHttp([
        (401, {}, {}),                       # first refresh attempt fails
        token_ok(access=secret_at, refresh=secret_rt),
        (401, {}, {}),                       # rest call -> 401
        token_ok(access=secret_at, refresh=secret_rt),
        (200, {}, {"time": "now"}),
    ])
    auth = QuestradeAuth(refresh_token="RT-0", http=http)
    rest = QuestradeRest(auth, http=http)
    with caplog.at_level(logging.INFO, logger="snowball_pricer.feeds.questrade"):
        with pytest.raises(QuestradeAuthError):
            auth.refresh()                   # 401 on token endpoint
        auth2 = QuestradeAuth(refresh_token="RT-0", http=http)
        rest2 = QuestradeRest(auth2, http=http)
        auth2.refresh()
        assert rest2.get_time() == {"time": "now"}  # 401 -> re-auth -> retry
    assert secret_rt not in caplog.text
    assert secret_at not in caplog.text
    assert "RT-0" not in caplog.text


# ---------------------------------------------------------------- REST retry
def test_rest_429_backoff_with_retry_after():
    sleeps = []
    http = FakeHttp([
        (429, {}, {}),
        (429, {"Retry-After": "3"}, {}),
        (200, {}, {"ok": True}),
    ])
    auth = QuestradeAuth(refresh_token="RT-0", http=FakeHttp([token_ok()]))
    auth.refresh()
    rest = QuestradeRest(auth, http=http, sleeper=sleeps.append,
                         min_interval_s=0.0)
    assert rest.get_time() == {"ok": True}
    assert sleeps == [1.0, 3.0]  # exponential, then Retry-After honored


def test_rest_429_exhausted_raises():
    http = FakeHttp([(429, {}, {})] * 7)
    auth = QuestradeAuth(refresh_token="RT-0", http=FakeHttp([token_ok()]))
    auth.refresh()
    rest = QuestradeRest(auth, http=http, sleeper=lambda s: None,
                         min_interval_s=0.0)
    with pytest.raises(QuestradeError):
        rest.get_time()


def test_rest_401_reauthenticates_exactly_once():
    script = [token_ok(), (401, {}, {}), token_ok(access="AT-2"), (200, {}, {"t": 1})]
    http = FakeHttp(script)
    auth = QuestradeAuth(refresh_token="RT-0", http=http)
    auth.refresh()
    rest = QuestradeRest(auth, http=http, min_interval_s=0.0)
    assert rest.get_time() == {"t": 1}
    # token endpoint hit twice total (initial + one re-auth), then success.
    token_calls = [c for c in http.calls if c[1].startswith(TOKEN_URL_PREFIX)]
    assert len(token_calls) == 2


# ---------------------------------------------------------------- chain parsing
def test_parse_option_chain_basic():
    today = datetime.now(timezone.utc).date()
    contracts = parse_option_chain(chain_json(), 8049, "SPY", today)
    assert len(contracts) == 6  # 3 strikes x call/put
    calls = [c for c in contracts if c.is_call]
    assert {c.strike for c in calls} == {590.0, 600.0, 610.0}
    assert all(c.underlying_id == 8049 and c.underlying_symbol == "SPY"
               for c in contracts)
    assert all(abs(c.years_to_expiry - 60 / 365) < 0.01 for c in contracts)


def test_parse_option_chain_skips_bad_entries(caplog):
    today = datetime.now(timezone.utc).date()
    chain = [
        {"expiryDate": "not-a-date", "chainPerRoot": []},
        {"expiryDate": (datetime.now(timezone.utc) - timedelta(days=1))
         .strftime("%Y-%m-%dT00:00:00.000000-04:00"),
         "chainPerRoot": [{"chainPerStrikePrice":
                           [{"strikePrice": 600, "callSymbolId": 1,
                             "putSymbolId": 2}]}]},
    ]
    chain[1]["chainPerRoot"][0]["chainPerStrikePrice"].append(
        {"strikePrice": -5, "callSymbolId": 3, "putSymbolId": 4})
    chain[1]["chainPerRoot"][0]["chainPerStrikePrice"].append(
        {"strikePrice": 600, "callSymbolId": 5})  # missing put leg
    with caplog.at_level(logging.WARNING):
        contracts = parse_option_chain(chain, 8049, "SPY", today)
    # Only the valid call leg of the expired bucket is dropped entirely
    # (tte <= 0); nothing parseable remains.
    assert contracts == []


def test_parse_option_chain_partial_legs():
    today = datetime.now(timezone.utc).date()
    chain = chain_json()
    # Drop the put leg of the first strike: only its call survives.
    del chain[0]["chainPerRoot"][0]["chainPerStrikePrice"][0]["putSymbolId"]
    contracts = parse_option_chain(chain, 8049, "SPY", today)
    assert len(contracts) == 5
    assert sum(c.is_call for c in contracts) == 3
    assert sum(not c.is_call for c in contracts) == 2


# ---------------------------------------------------------------- normalization
def test_quote_normalization():
    chain = chain_json()
    call_sid = 1002  # strike 600 call (sids assigned in strike order)
    opt_msg = json.dumps([{"symbolId": call_sid, "bidPrice": 5.0,
                           "askPrice": 5.2, "bidSize": 10, "askSize": 12,
                           "delay": False}])
    und_msg = json.dumps([{"symbolId": 8049, "lastTradePrice": 601.5}])
    feed, created, _ = make_feed(chain, [[und_msg, opt_msg]])
    q = drain_one(feed)
    assert q.underlying_id == "SPY"
    assert q.strike == 600.0 and q.is_call is True
    assert q.bid == 5.0 and q.ask == 5.2
    assert q.underlying_price == 601.5
    assert q.is_delayed is False
    assert abs(q.expiry - 60 / 365) < 0.01
    assert q.rate == 0.03 and q.div_yield == 0.01
    assert q.is_valid()
    # Auth token was sent on socket open; URL is the WS endpoint.
    url, ws = created[0]
    assert url.startswith("wss://api01.iq.questrade.com:443/v1/markets/quotes")
    assert "stream=true" in url and "mode=WebSocket" in url
    assert ws.sent == ["AT"]


def test_delay_flag_defaults_true_when_absent():
    chain = chain_json()
    opt_msg = json.dumps([{"symbolId": 1000, "bidPrice": 5.0, "askPrice": 5.2}])
    feed, _, _ = make_feed(chain, [[opt_msg]])
    q = drain_one(feed)
    assert q.is_delayed is True  # unknown -> assume delayed (honest default)


def test_malformed_messages_skipped_not_fatal():
    chain = chain_json()
    good = json.dumps([{"symbolId": 1000, "bidPrice": 5.0, "askPrice": 5.2,
                        "delay": True}])
    feed, _, _ = make_feed(chain, [["{not json", "[1,2,3]", good]])
    q = drain_one(feed)
    assert q.bid == 5.0
    assert feed.health.malformed_skipped == 1
    assert feed.health.quotes_yielded == 1
    assert feed.health.delayed_quotes == 1


def test_missing_bid_ask_skipped():
    chain = chain_json()
    noquote = json.dumps([{"symbolId": 1000, "lastTradePrice": 5.1}])
    good = json.dumps([{"symbolId": 1001, "bidPrice": 4.0, "askPrice": 4.2}])
    feed, _, _ = make_feed(chain, [[noquote, good]])
    q = drain_one(feed)
    assert q.strike == 590.0 and q.is_call is False  # put leg survived


def test_crossed_quote_yielded_for_pipeline_to_drop():
    chain = chain_json()
    crossed = json.dumps([{"symbolId": 1000, "bidPrice": 5.5, "askPrice": 5.2}])
    feed, _, _ = make_feed(chain, [[crossed]])
    q = drain_one(feed)
    assert q.bid > q.ask and not q.is_valid()  # QuoteStore drops it


def test_unknown_symbol_id_ignored():
    chain = chain_json()
    stray = json.dumps([{"symbolId": 999999, "bidPrice": 1.0, "askPrice": 1.1}])
    good = json.dumps([{"symbolId": 1000, "bidPrice": 5.0, "askPrice": 5.2}])
    feed, _, _ = make_feed(chain, [[stray, good]])
    q = drain_one(feed)
    assert q.strike == 590.0


# ---------------------------------------------------------------- reconnect
def test_ws_reconnect_resubscribes_with_backoff():
    chain = chain_json()
    m1 = json.dumps([{"symbolId": 1000, "bidPrice": 5.0, "askPrice": 5.2}])
    m2 = json.dumps([{"symbolId": 1002, "bidPrice": 6.0, "askPrice": 6.3}])
    feed, created, sleeps = make_feed(chain, [[m1], [m2]], max_reconnects=2)
    it = feed.subscribe()
    q1, q2 = next(it), next(it)  # second forces a reconnect in between
    assert (q1.bid, q2.bid) == (5.0, 6.0)
    assert len(created) == 2  # resubscribed
    assert created[1][1].sent == ["AT"]  # re-authenticated on new socket
    assert feed.health.reconnects == 1
    assert sleeps == [1.0]  # exponential backoff, first attempt
    with pytest.raises(QuestradeError):
        next(it)  # budget exhausted on the next failure


def test_ws_url_uses_wss_and_port():
    url = QuestradeFeed._ws_url("https://api01.iq.questrade.com/", 7000, [1, 2])
    assert url == ("wss://api01.iq.questrade.com:7000/v1/markets/quotes"
                   "?ids=1,2&stream=true&mode=WebSocket")


def test_feed_health_tracks_staleness_hook():
    chain = chain_json()
    good = json.dumps([{"symbolId": 1000, "bidPrice": 5.0, "askPrice": 5.2}])
    feed, _, _ = make_feed(chain, [[good]])
    assert feed.health.last_msg_ts == 0.0
    drain_one(feed)
    assert feed.health.last_msg_ts > 0.0
    d = feed.health.as_dict()
    assert d["quotes_yielded"] == 1
    # The exhausted stream counted one reconnect attempt before the budget
    # check raised QuestradeError.
    assert d["reconnects"] == 1
