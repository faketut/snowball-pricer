"""Questrade broker adapter: real implementation of the P1 ``Feed`` contract.

Venue selected 2026-09-30 (replaces the P1 ``BrokerWsFeed`` stub): Questrade.
Official REST + WebSocket API, OAuth2 refresh-token auth, no local gateway
daemon required, Canadian accounts supported.

Endpoint/symbology verification status (2026-09-30): the official docs URL
(https://www.questrade.com/api/documentation) returned HTTP 404 at the time
of writing, so every path below was cross-checked against three independent
community API clients (lucaspanjer/questrade-client, leanderlee/questrade,
kaiyoux/kwess), which agree with each other. Re-verify against the official
docs before a production deployment; the one known divergence is the token
endpoint method (task spec: POST; community clients: GET) — this adapter
tries POST first and falls back to GET on HTTP 405.

Data-quality notes that affect the pricing pipeline
---------------------------------------------------
- Questrade streams **L1 quotes only** (bid/ask/size, no depth). US options
  L1 streaming requires the "Real-time streaming" market-data package
  ($9.95 CAD/mo); without it, stream messages carry ``"delay": true`` and
  this feed flags every tick ``is_delayed=True`` (P1's arbitrage gate and
  P5's monitors treat delayed input as lower-trust).
- US-listed equity options are **American-style**. P1's IV inverter assumes
  European exercise; early-exercise premium on short-dated deep-ITM puts is
  therefore folded into the inverted IV. De-Americanization is future work.
- The stream carries **no per-quote exchange timestamp**; ``ts`` is the
  local receive time (documented, not faked).
- Options stream only during market hours (09:30-16:00 ET). Outside hours
  the socket yields nothing — P5's STALENESS monitor is the right consumer
  of ``FeedHealth.last_msg_ts``.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence

from ..tick import Feed, OptionQuote

logger = logging.getLogger(__name__)

TOKEN_URL = "https://login.questrade.com/oauth2/token"
ENV_REFRESH_TOKEN = "QUESTRADE_REFRESH_TOKEN"

# Questrade market-data rate limits (documented): 20 req/s, 15,000 req/hour
# for market-data calls. We pace REST well below the per-second cap and let
# the 429 backoff handle the rest.
_MIN_REST_INTERVAL_S = 0.06
_MAX_429_RETRIES = 5
_BACKOFF_BASE_S = 1.0
_BACKOFF_MAX_S = 60.0


class QuestradeError(Exception):
    """Base error for the Questrade adapter."""


class QuestradeAuthError(QuestradeError):
    """Authentication failed (bad/expired refresh token, 401 loop)."""


class HttpResponse(NamedTuple):
    status: int
    headers: Dict[str, str]
    body: bytes


# Injectable HTTP layer: http(method, url, headers, data) -> HttpResponse.
# Default is stdlib urllib (no new dependency for REST). The token values
# are NEVER included in log messages — only statuses and paths.
def _urllib_http(
    method: str,
    url: str,
    headers: Dict[str, str],
    data: Optional[bytes],
) -> HttpResponse:
    req = urllib.request.Request(url, data=data, headers=headers or {},
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return HttpResponse(resp.status, dict(resp.headers), resp.read())
    except urllib.error.HTTPError as e:
        return HttpResponse(e.code, dict(e.headers or {}), e.read())


class QuestradeAuth:
    """OAuth2 refresh-token auth. The refresh token comes ONLY from the
    ``QUESTRADE_REFRESH_TOKEN`` env var (or an explicit constructor arg —
    never a default value, never from disk, never logged).

    Register a personal app with **read-only** scope at Questrade; this
    adapter never places orders, so the trade scope must not be granted.
    """

    def __init__(
        self,
        refresh_token: Optional[str] = None,
        http: Callable[..., HttpResponse] = _urllib_http,
        clock: Callable[[], float] = time.time,
    ) -> None:
        token = refresh_token or os.environ.get(ENV_REFRESH_TOKEN)
        if not token:
            raise RuntimeError(
                f"{ENV_REFRESH_TOKEN} is not set. Register a personal "
                "(read-only) app at Questrade, copy the refresh token, and "
                f"export {ENV_REFRESH_TOKEN}=... before running. "
                "See docs/questrade_setup.md."
            )
        self._refresh_token = token
        self._http = http
        self._clock = clock
        self.access_token: Optional[str] = None
        self.api_server: Optional[str] = None
        self.expires_at: float = 0.0

    # -- internals ----------------------------------------------------
    def _token_request(self, method: str) -> HttpResponse:
        # The refresh token travels in the query string; the URL is never
        # logged anywhere in this module.
        qs = urllib.parse.urlencode(
            {"grant_type": "refresh_token",
             "refresh_token": self._refresh_token}
        )
        return self._http(method, f"{TOKEN_URL}?{qs}", {}, None)

    def _parse_token_response(self, resp: HttpResponse) -> None:
        if resp.status != 200:
            raise QuestradeAuthError(
                f"token endpoint returned HTTP {resp.status}"
            )
        try:
            data = json.loads(resp.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise QuestradeAuthError(f"token endpoint returned bad JSON: {e}")
        for key in ("access_token", "api_server", "token_type"):
            if not data.get(key):
                raise QuestradeAuthError(
                    f"token response missing {key!r}"
                )
        self.access_token = data["access_token"]
        self.api_server = data["api_server"].rstrip("/") + "/"
        # Questrade rotates the refresh token on every use; keep the new one
        # in memory only (never written to disk per the credential rules).
        if data.get("refresh_token"):
            self._refresh_token = data["refresh_token"]
        self.expires_at = self._clock() + float(data.get("expires_in", 1800))

    # -- public -------------------------------------------------------
    def refresh(self) -> None:
        """Exchange the refresh token for a fresh access token."""
        resp = self._token_request("POST")
        if resp.status in (405, 501):
            # Method-not-allowed: fall back to the GET form used by the
            # community clients. Safe: a 405/501 never consumes the token.
            logger.info("token endpoint rejected POST; retrying with GET")
            resp = self._token_request("GET")
        self._parse_token_response(resp)
        logger.info("questrade auth refreshed; api_server set")

    def ensure_valid(self, skew_s: float = 60.0) -> None:
        if self.access_token is None or self._clock() > self.expires_at - skew_s:
            self.refresh()

    def auth_headers(self) -> Dict[str, str]:
        self.ensure_valid()
        # The header value is handed to urllib directly; never logged.
        return {"Authorization": f"Bearer {self.access_token}"}


class QuestradeRest:
    """Thin REST client over an authenticated session.

    429 -> exponential backoff (honoring Retry-After); 401 -> exactly one
    re-auth then retry; other 4xx/5xx -> QuestradeError.
    """

    def __init__(
        self,
        auth: QuestradeAuth,
        http: Callable[..., HttpResponse] = _urllib_http,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        min_interval_s: float = _MIN_REST_INTERVAL_S,
    ) -> None:
        self.auth = auth
        self._http = http
        self._clock = clock
        self._sleep = sleeper
        self._min_interval = min_interval_s
        self._last_call = 0.0

    # -- internals ----------------------------------------------------
    def _pace(self) -> None:
        dt = self._clock() - self._last_call
        if dt < self._min_interval:
            self._sleep(self._min_interval - dt)
        self._last_call = self._clock()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        assert self.auth.api_server, "call auth.refresh() first"
        url = self.auth.api_server + "v1/" + path.lstrip("/")
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        headers = dict(self.auth.auth_headers())
        if data is not None:
            headers["Content-Type"] = "application/json"

        retried_auth = False
        for attempt in range(_MAX_429_RETRIES + 1):
            self._pace()
            resp = self._http(method, url, headers, data)
            if resp.status == 429:
                wait = self._retry_after_s(resp.headers, attempt)
                logger.warning("questrade 429; backing off %.1fs", wait)
                self._sleep(wait)
                continue
            if resp.status == 401 and not retried_auth:
                logger.info("questrade 401; re-authenticating once")
                self.auth.refresh()
                headers = dict(self.auth.auth_headers())
                retried_auth = True
                continue
            if 200 <= resp.status < 300:
                return json.loads(resp.body.decode("utf-8")) if resp.body else {}
            # Log path only — the query string may carry ids but never tokens
            # (auth travels in the Authorization header).
            raise QuestradeError(
                f"{method} {path} -> HTTP {resp.status}"
            )
        raise QuestradeError(f"{method} {path} -> 429 persisted after retries")

    @staticmethod
    def _retry_after_s(headers: Dict[str, str], attempt: int) -> float:
        for key in ("Retry-After", "retry-after"):
            if key in headers:
                try:
                    return max(0.0, float(headers[key]))
                except ValueError:
                    break
        return min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * (2 ** attempt))

    # -- public endpoints ---------------------------------------------
    def get_time(self) -> Any:
        return self._request("GET", "time")

    def search_symbols(self, prefix: str) -> List[Dict[str, Any]]:
        data = self._request("GET", "symbols/search", {"prefix": prefix})
        return data.get("symbols", [])

    def get_symbol(self, symbol_id: int) -> Dict[str, Any]:
        data = self._request("GET", f"symbols/{symbol_id}")
        syms = data.get("symbols", [])
        return syms[0] if syms else {}

    def get_option_chain(self, symbol_id: int) -> List[Dict[str, Any]]:
        data = self._request("GET", f"symbols/{symbol_id}/options")
        return data.get("optionChain", [])

    def get_option_quotes(
        self, option_ids: Sequence[int], chunk: int = 100
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        ids = list(option_ids)
        for i in range(0, len(ids), chunk):
            data = self._request(
                "POST", "markets/quotes/options",
                body={"optionIds": ids[i:i + chunk]},
            )
            out.extend(data.get("optionQuotes", []))
        return out

    def get_quote(self, symbol_id: int) -> Dict[str, Any]:
        data = self._request("GET", f"markets/quotes/{symbol_id}")
        quotes = data.get("quotes", [])
        return quotes[0] if quotes else {}

    def get_stream_port(self, symbol_ids: Sequence[int]) -> int:
        data = self._request(
            "GET", "markets/quotes",
            {"ids": ",".join(str(i) for i in symbol_ids[:5]),
             "stream": "true", "mode": "WebSocket"},
        )
        port = data.get("streamPort")
        if not port:
            raise QuestradeError("stream handshake missing streamPort")
        return int(port)


@dataclass(frozen=True)
class OptionContract:
    """One option contract resolved from the chain structure."""

    symbol_id: int
    underlying_id: int
    underlying_symbol: str
    expiry: _dt.date
    strike: float
    is_call: bool
    years_to_expiry: float  # vs. the date the chain was fetched


def _parse_expiry_date(raw: str) -> Optional[_dt.date]:
    # Questrade: "2026-10-16T00:00:00.000000-04:00"
    try:
        return _dt.datetime.fromisoformat(raw).date()
    except (ValueError, TypeError):
        return None


def parse_option_chain(
    chain: Sequence[Dict[str, Any]],
    underlying_id: int,
    underlying_symbol: str,
    today: _dt.date,
) -> List[OptionContract]:
    """Flatten Questrade's option-chain structure into contracts.

    Malformed entries (bad expiry, missing strike, absent symbol ids) are
    skipped, never fatal — the chain is reference data, not the hot path.
    """
    contracts: List[OptionContract] = []
    for bucket in chain:
        expiry = _parse_expiry_date(bucket.get("expiryDate", ""))
        if expiry is None:
            logger.warning("skipping chain bucket with bad expiryDate")
            continue
        tte = (expiry - today).days / 365.0
        if tte <= 0:
            continue
        for root in bucket.get("chainPerRoot", []):
            for leg in root.get("chainPerStrikePrice", []):
                strike = leg.get("strikePrice")
                if not isinstance(strike, (int, float)) or strike <= 0:
                    continue
                for key, is_call in (("callSymbolId", True),
                                     ("putSymbolId", False)):
                    sid = leg.get(key)
                    if not isinstance(sid, int):
                        continue
                    contracts.append(OptionContract(
                        symbol_id=sid,
                        underlying_id=underlying_id,
                        underlying_symbol=underlying_symbol,
                        expiry=expiry,
                        strike=float(strike),
                        is_call=is_call,
                        years_to_expiry=tte,
                    ))
    return contracts


@dataclass
class FeedHealth:
    """Operational counters for monitoring (P5 STALENESS hook reads
    ``last_msg_ts``)."""

    last_msg_ts: float = 0.0
    reconnects: int = 0
    malformed_skipped: int = 0
    quotes_yielded: int = 0
    delayed_quotes: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {
            "last_msg_ts": self.last_msg_ts,
            "reconnects": self.reconnects,
            "malformed_skipped": self.malformed_skipped,
            "quotes_yielded": self.quotes_yielded,
            "delayed_quotes": self.delayed_quotes,
        }


def _default_ws_factory(url: str, timeout: float = 30.0):
    import websocket  # websocket-client (see requirements.txt)

    return websocket.create_connection(url, timeout=timeout)


class QuestradeFeed(Feed):
    """Live Questrade feed: option chains via REST, L1 quotes via WebSocket,
    normalized to P1 ``OptionQuote``.

    Parameters
    ----------
    underlyings : e.g. ["SPY", "QQQ"]. Resolved to symbol ids at subscribe().
    rate / div_yield : pricing inputs (not provided by Questrade).
    min_days_to_expiry : ignore weekly noise / expiry-week gamma blowups.
    max_otm : keep strikes within ±max_otm of spot (surface wings beyond
        this are extrapolation anyway; keeps the WS subscription small).
    """

    def __init__(
        self,
        underlyings: Sequence[str],
        *,
        rate: float = 0.03,
        div_yield: float = 0.0,
        min_days_to_expiry: float = 7.0,
        max_otm: float = 0.35,
        rest: Optional[QuestradeRest] = None,
        ws_factory: Callable[[str], Any] = _default_ws_factory,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        reconnect_base_s: float = 1.0,
        reconnect_max_s: float = 60.0,
        max_reconnects: Optional[int] = None,
    ) -> None:
        if not underlyings:
            raise ValueError("underlyings must be non-empty")
        self._underlyings = list(underlyings)
        self._rate = float(rate)
        self._div_yield = float(div_yield)
        self._min_tte = float(min_days_to_expiry) / 365.0
        self._max_otm = float(max_otm)
        self._rest = rest
        self._ws_factory = ws_factory
        self._clock = clock
        self._sleep = sleeper
        self._reconnect_base = reconnect_base_s
        self._reconnect_max = reconnect_max_s
        self._max_reconnects = max_reconnects
        self.health = FeedHealth()
        self._spots: Dict[int, float] = {}

    # -- setup --------------------------------------------------------
    def _ensure_rest(self) -> QuestradeRest:
        if self._rest is None:
            self._rest = QuestradeRest(QuestradeAuth())
        self._rest.auth.refresh()
        return self._rest

    def _resolve_underlying(self, rest: QuestradeRest, symbol: str) -> int:
        cands = rest.search_symbols(symbol)
        for c in cands:
            if str(c.get("symbol", "")).upper() == symbol.upper():
                return int(c["symbolId"])
        if not cands:
            raise QuestradeError(f"symbol not found: {symbol}")
        logger.warning("no exact match for %s; using %s", symbol,
                       cands[0].get("symbol"))
        return int(cands[0]["symbolId"])

    def _build_universe(
        self, rest: QuestradeRest, underlyings: Sequence[str]
    ) -> tuple[Dict[int, OptionContract], List[int], Dict[int, str]]:
        """Returns (contracts_by_symbol_id, underlying_ids, names)."""
        today = _dt.datetime.now(_dt.timezone.utc).date()
        contracts: Dict[int, OptionContract] = {}
        underlying_ids: List[int] = []
        names: Dict[int, str] = {}
        for sym in underlyings:
            uid = self._resolve_underlying(rest, sym)
            underlying_ids.append(uid)
            names[uid] = sym.upper()
            snap = rest.get_quote(uid)
            spot = snap.get("lastTradePrice") or snap.get("askPrice") or 0.0
            if spot and spot > 0:
                self._spots[uid] = float(spot)
            chain = rest.get_option_chain(uid)
            for c in parse_option_chain(chain, uid, sym.upper(), today):
                if c.years_to_expiry < self._min_tte:
                    continue
                spot = self._spots.get(uid, 0.0)
                if spot > 0 and abs(c.strike / spot - 1.0) > self._max_otm:
                    continue
                contracts[c.symbol_id] = c
        if not contracts:
            raise QuestradeError("no option contracts resolved; check filters")
        logger.info("questrade universe: %d contracts over %d underlyings",
                    len(contracts), len(underlying_ids))
        return contracts, underlying_ids, names

    # -- streaming ----------------------------------------------------
    @staticmethod
    def _ws_url(api_server: str, port: int, ids: Sequence[int]) -> str:
        base = api_server.rstrip("/")
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        id_list = ",".join(str(i) for i in ids)
        return (f"{base}:{port}/v1/markets/quotes"
                f"?ids={id_list}&stream=true&mode=WebSocket")

    def _normalize_option_message(
        self,
        msg: Dict[str, Any],
        contracts: Dict[int, OptionContract],
        names: Dict[int, str],
        recv_ts: float,
    ) -> Optional[OptionQuote]:
        try:
            sid = int(msg["symbolId"])
        except (KeyError, TypeError, ValueError):
            return None
        contract = contracts.get(sid)
        if contract is None:
            return None  # underlying quote or untracked symbol
        bid = msg.get("bidPrice")
        ask = msg.get("askPrice")
        if bid is None or ask is None:
            return None  # no tradeable quote on this message
        spot = self._spots.get(contract.underlying_id, 0.0)
        if spot <= 0:
            return None  # cannot price without the underlying
        return OptionQuote(
            ts=recv_ts,  # stream carries no exchange ts; receive time used
            underlying_id=names[contract.underlying_id],
            expiry=contract.years_to_expiry,
            strike=contract.strike,
            is_call=contract.is_call,
            bid=float(bid),
            ask=float(ask),
            underlying_price=float(spot),
            rate=self._rate,
            div_yield=self._div_yield,
            is_delayed=bool(msg.get("delay", True)),
        )

    def _handle_message(
        self,
        raw: Any,
        contracts: Dict[int, OptionContract],
        names: Dict[int, str],
    ) -> Optional[OptionQuote]:
        recv_ts = self._clock()
        try:
            msg = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except (ValueError, TypeError):
            self.health.malformed_skipped += 1
            logger.warning("skipping malformed stream message")
            return None
        items = msg if isinstance(msg, list) else [msg]
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                sid = int(item.get("symbolId", -1))
            except (TypeError, ValueError):
                continue
            if sid in contracts:
                quote = self._normalize_option_message(item, contracts,
                                                       names, recv_ts)
                if quote is not None:
                    self.health.last_msg_ts = recv_ts
                    self.health.quotes_yielded += 1
                    if quote.is_delayed:
                        self.health.delayed_quotes += 1
                    return quote
            else:
                # Underlying spot update (or untracked symbol — ignored).
                px = item.get("lastTradePrice") or item.get("askPrice")
                if isinstance(px, (int, float)) and px > 0:
                    self._spots[sid] = float(px)
                    self.health.last_msg_ts = recv_ts
        return None

    def subscribe(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Iterator[OptionQuote]:
        underlyings = list(symbols) if symbols else self._underlyings
        rest = self._ensure_rest()
        contracts, underlying_ids, names = self._build_universe(rest,
                                                                underlyings)
        all_ids = underlying_ids + sorted(contracts.keys())
        api_server = rest.auth.api_server or ""

        attempt = 0
        while True:
            ws = None
            try:
                port = rest.get_stream_port(all_ids)
                url = self._ws_url(api_server, port, all_ids)
                ws = self._ws_factory(url)
                ws.send(rest.auth.access_token)  # auth on open; never logged
                attempt = 0
                logger.info("questrade stream connected (%d symbols)",
                            len(all_ids))
                while True:
                    raw = ws.recv()
                    if raw is None or raw == "":
                        raise ConnectionError("stream closed by peer")
                    quote = self._handle_message(raw, contracts, names)
                    if quote is not None:
                        yield quote
            except (StopIteration, KeyboardInterrupt):
                raise
            except Exception as exc:  # reconnectable: network, 401, bad frame
                attempt += 1
                self.health.reconnects += 1
                logger.warning("questrade stream error (%s); reconnect #%d",
                               type(exc).__name__, attempt)
                if self._max_reconnects is not None and \
                        attempt > self._max_reconnects:
                    raise QuestradeError(
                        f"reconnect budget exhausted ({attempt})") from exc
                # Refresh the auth + universe so a new expiry listing or a
                # rotated api_server is picked up on reconnect.
                try:
                    rest.auth.refresh()
                    contracts, underlying_ids, names = self._build_universe(
                        rest, underlyings)
                    all_ids = underlying_ids + sorted(contracts.keys())
                    api_server = rest.auth.api_server or ""
                except Exception as re_exc:  # noqa: BLE001
                    logger.warning("universe refresh failed on reconnect: %s",
                                   type(re_exc).__name__)
                wait = min(self._reconnect_max,
                           self._reconnect_base * (2 ** (attempt - 1)))
                self._sleep(wait)
            finally:
                try:
                    if ws is not None:
                        ws.close()
                except Exception:  # noqa: BLE001
                    pass
