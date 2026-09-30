"""yfinance delayed poll feed: Phase 1.5 stepping stone (NOT real-time).

THIS FEED IS ALWAYS 15-MINUTES DELAYED. Every ``OptionQuote`` it emits has
``is_delayed=True`` hardcoded. It exists to exercise the full P1 pipeline
(poll -> IV inversion -> SSVI surface -> arbitrage gate) against real
market-data *shapes* for free — it does NOT validate the real-time thesis
(the delay is fixed, so the value of removing it cannot be measured here).

Known yfinance quirks handled explicitly:
- Poll, not push: Yahoo exposes REST snapshots, no WebSocket. Cadence is
  configurable (default 180 s per underlying); Yahoo throttles aggressively,
  so requests are paced and any 429/error only skips the cycle (log +
  backoff), never crashes the loop.
- Missing/NaN bid/ask: emitted as ``bid=None``/``ask=None`` (never
  fabricated). ``OptionQuote.is_valid()`` rejects them; the QuoteStore
  drops them. Completeness is measured, not hidden.
- Non-atomic chain: one poll cycle sweeps underlyings x expiries over
  several seconds. Each cycle is stamped with a ``cycle_id`` (tracked in
  ``feed.cycle_stats``); quotes carry local receive time as ``ts`` because
  Yahoo provides no per-quote exchange timestamp.
- After-hours: chains are still served (stale, from the close). Expect
  worse bid/ask completeness off-session; see docs/yfinance_notes.md.
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from ..tick import Feed, OptionQuote

logger = logging.getLogger(__name__)

# This feed is ALWAYS delayed. Hardcoded at emission; there is no code path
# that can produce is_delayed=False here.
IS_DELAYED = True

_DEFAULT_POLL_INTERVAL_S = 180.0
_DEFAULT_MAX_EXPIRIES = 4
_REQUEST_PACE_S = 1.0          # gentle pacing between Yahoo requests
_BACKOFF_BASE_S = 15.0
_BACKOFF_MAX_S = 300.0


class YFinanceError(Exception):
    """Base error for the yfinance adapter."""


def _as_float_or_none(v) -> Optional[float]:
    """NaN/None/NaT -> None; otherwise float(v). Never fabricates."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


class YFinancePollFeed(Feed):
    """Poll Yahoo option chains and emit ``OptionQuote`` ticks.

    Parameters
    ----------
    underlyings : e.g. ["SPY", "QQQ", "IWM"].
    rate / div_yield : passed through to every quote (P1 contract).
    poll_interval_s : seconds between poll cycles (per full sweep).
    max_expiries : only the first N expiries per underlying (limits
        request count; near-term expiries carry the surface anyway).
    min_expiry_days : skip expiries closer than this many days (0DTE/1DTE
        chains are noise for surface calibration; 0 = no filter).
    k_band : optional (k_min, k_max) log-moneyness filter vs spot;
        None disables. Production adapters filter wings; the raw feed
        does not hide data, so this is opt-in and recorded in stats.
    max_cycles : stop after N cycles (None = unbounded). Used by the
        quality experiment; live_loop bounds via --max-ticks instead.
    ticker_factory : injectable for tests (default: yfinance.Ticker).
    """

    def __init__(
        self,
        underlyings: Sequence[str],
        *,
        rate: float = 0.03,
        div_yield: float = 0.0,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        max_expiries: int = _DEFAULT_MAX_EXPIRIES,
        min_expiry_days: int = 0,
        request_pace_s: float = _REQUEST_PACE_S,
        max_cycles: Optional[int] = None,
        k_band: Optional[Tuple[float, float]] = None,
        ticker_factory: Optional[Callable[[str], object]] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if ticker_factory is None:
            try:
                import yfinance as _yf
            except ImportError as exc:
                raise RuntimeError(
                    "yfinance is not installed (pip install yfinance). "
                    "See requirements.txt."
                ) from exc
            ticker_factory = _yf.Ticker
        self._underlyings = [u.strip().upper() for u in underlyings if u.strip()]
        if not self._underlyings:
            raise ValueError("YFinancePollFeed: no underlyings given")
        self._rate = float(rate)
        self._div_yield = float(div_yield)
        self._poll_interval_s = float(poll_interval_s)
        self._max_expiries = int(max_expiries)
        self._min_expiry_days = int(min_expiry_days)
        self._request_pace_s = float(request_pace_s)
        self._max_cycles = max_cycles
        self._k_band = k_band
        self._ticker_factory = ticker_factory
        self._clock = clock
        self._sleep = sleep
        self._cycle_id = 0
        self._consecutive_failures = 0
        # Per-cycle quality log: {cycle_id, ts_start, underlyings:{sym: {...}},
        # n_rows, n_valid, error}
        self.cycle_stats: List[Dict] = []

    # ------------------------------------------------------------ internals
    @staticmethod
    def _years_to_expiry(expiry_str: str, today: _dt.date) -> Optional[float]:
        try:
            d = _dt.date.fromisoformat(expiry_str[:10])
        except ValueError:
            return None
        days = (d - today).days
        if days <= 0:
            return None
        return days / 365.0

    def _spot(self, ticker) -> Optional[float]:
        """Best-effort spot: fast_info then daily close. None on failure."""
        try:
            px = ticker.fast_info.get("lastPrice", None)
            f = _as_float_or_none(px)
            if f is not None and f > 0:
                return f
        except Exception:
            pass
        try:
            hist = ticker.history(period="1d")
            if hist is not None and len(hist):
                f = _as_float_or_none(hist["Close"].iloc[-1])
                if f is not None and f > 0:
                    return f
        except Exception:
            pass
        return None

    def _chain_frames(self, ticker, expiry: str):
        """Return (calls_df, puts_df); empty frames on any failure."""
        import pandas as pd  # local: only needed on this path

        try:
            chain = ticker.option_chain(expiry)
        except Exception:
            return pd.DataFrame(), pd.DataFrame()
        # yfinance versions differ: (calls, puts) or (calls, puts, underlying)
        try:
            calls, puts = chain[0], chain[1]
        except (TypeError, IndexError, KeyError):
            return pd.DataFrame(), pd.DataFrame()
        return calls, puts

    def _quotes_from_frame(
        self,
        df,
        *,
        is_call: bool,
        underlying_id: str,
        expiry_yrs: float,
        spot: float,
        ts: float,
    ) -> List[OptionQuote]:
        out: List[OptionQuote] = []
        if df is None or len(df) == 0:
            return out
        cols = set(df.columns)
        if "strike" not in cols:
            return out
        for _, row in df.iterrows():
            strike = _as_float_or_none(row.get("strike"))
            if strike is None or strike <= 0:
                continue
            if self._k_band is not None and spot > 0:
                k = math.log(strike / spot)
                if not (self._k_band[0] <= k <= self._k_band[1]):
                    continue
            bid = _as_float_or_none(row.get("bid")) if "bid" in cols else None
            ask = _as_float_or_none(row.get("ask")) if "ask" in cols else None
            out.append(
                OptionQuote(
                    ts=ts,
                    underlying_id=underlying_id,
                    expiry=expiry_yrs,
                    strike=strike,
                    is_call=is_call,
                    bid=bid,  # type: ignore[arg-type]  # None = missing, never fabricated
                    ask=ask,  # type: ignore[arg-type]
                    underlying_price=spot,
                    rate=self._rate,
                    div_yield=self._div_yield,
                    is_delayed=IS_DELAYED,
                )
            )
        return out

    # ------------------------------------------------------------------ API
    def poll_cycle(self) -> Tuple[List[OptionQuote], Dict]:
        """Run one full sweep over underlyings x expiries.

        Returns (quotes, stats). A failed underlying is skipped inside the
        cycle (recorded in stats); a totally failed cycle returns ([], stats)
        with stats["error"] set — the caller decides pacing/backoff.
        """
        self._cycle_id += 1
        cid = self._cycle_id
        t_start = self._clock()
        today = _dt.datetime.fromtimestamp(t_start, tz=_dt.timezone.utc).date()
        quotes: List[OptionQuote] = []
        per_sym: Dict[str, Dict] = {}
        cycle_error: Optional[str] = None

        for sym in self._underlyings:
            sym_stat = {"n_rows": 0, "n_valid": 0, "expiries": 0, "error": None}
            per_sym[sym] = sym_stat
            try:
                ticker = self._ticker_factory(sym)
                expiries = list(ticker.options or [])
                if self._min_expiry_days > 0:
                    cutoff = today + _dt.timedelta(days=self._min_expiry_days)

                    def _far_enough(e) -> bool:
                        try:
                            return (_dt.date.fromisoformat(str(e)[:10])
                                    >= cutoff)
                        except ValueError:
                            return False

                    expiries = [e for e in expiries if _far_enough(e)]
                expiries = expiries[: self._max_expiries]
                if not expiries:
                    sym_stat["error"] = "no expiries listed"
                    continue
                spot = self._spot(ticker)
                if spot is None:
                    sym_stat["error"] = "no spot"
                    continue
                for exp in expiries:
                    T = self._years_to_expiry(str(exp), today)
                    if T is None:
                        continue
                    calls, puts = self._chain_frames(ticker, str(exp))
                    ts = self._clock()  # local receive time; Yahoo gives none
                    sym_stat["expiries"] += 1
                    for df, is_call in ((calls, True), (puts, False)):
                        qs = self._quotes_from_frame(
                            df, is_call=is_call, underlying_id=sym,
                            expiry_yrs=T, spot=spot, ts=ts,
                        )
                        sym_stat["n_rows"] += len(qs)
                        sym_stat["n_valid"] += sum(1 for q in qs if q.is_valid())
                        quotes.extend(qs)
                    self._sleep(self._request_pace_s)
            except Exception as exc:  # never let one symbol kill the cycle
                sym_stat["error"] = f"{type(exc).__name__}: {exc}"
                logger.warning("yfinance cycle %d: %s failed: %s",
                               cid, sym, sym_stat["error"])

        n_valid = sum(1 for q in quotes if q.is_valid())
        stats = {
            "cycle_id": cid,
            "ts_start": t_start,
            "n_rows": len(quotes),
            "n_valid": n_valid,
            "completeness": (n_valid / len(quotes)) if quotes else 0.0,
            "underlyings": per_sym,
            "error": cycle_error,
        }
        self.cycle_stats.append(stats)
        return quotes, stats

    def subscribe(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Iterator[OptionQuote]:
        """Yield quotes indefinitely (or ``max_cycles`` cycles).

        Any unexpected exception in a cycle is logged, backed off, and the
        loop continues — this generator never raises on data errors.
        """
        if symbols is not None:
            logger.info("yfinance feed ignores subscribe(symbols); "
                        "universe fixed at construction: %s", self._underlyings)
        cycles = 0
        while True:
            t0 = self._clock()
            try:
                quotes, stats = self.poll_cycle()
                if stats["n_rows"] == 0:
                    self._consecutive_failures += 1
                else:
                    self._consecutive_failures = 0
                for q in quotes:
                    yield q
            except Exception as exc:  # pragma: no cover - defensive
                self._consecutive_failures += 1
                logger.warning("yfinance poll cycle %d raised: %r",
                               self._cycle_id + 1, exc)
            cycles += 1
            if self._max_cycles is not None and cycles >= self._max_cycles:
                return
            # Backoff on repeated empty/failed cycles; otherwise hold cadence.
            if self._consecutive_failures:
                wait = min(_BACKOFF_MAX_S,
                           _BACKOFF_BASE_S * 2 ** (self._consecutive_failures - 1))
            else:
                elapsed = self._clock() - t0
                wait = max(0.0, self._poll_interval_s - elapsed)
            logger.debug("yfinance: cycle done, sleeping %.1fs", wait)
            self._sleep(wait)


__all__ = ["YFinancePollFeed", "YFinanceError", "IS_DELAYED"]
