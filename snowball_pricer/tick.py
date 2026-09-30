"""Option quote tick schema and feed interfaces (P1).

Contracts (see SPEC.md section 2):
- ``OptionQuote`` is the single normalized tick schema. Every adapter
  (synthetic, broker WS, CSV replay) must emit it.
- ``Feed.subscribe()`` yields an unbounded iterator of quotes; the caller
  drives pacing (no hidden threads inside feeds).

The production broker adapter lives in ``snowball_pricer.feeds``
(``QuestradeFeed``); this module keeps only the schema and the interface.
"""
from __future__ import annotations

import abc
import itertools
import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .iv import black_price
from .surface import ssvi_total_var


@dataclass(frozen=True)
class OptionQuote:
    """One normalized option quote tick."""

    ts: float               # exchange timestamp, epoch seconds
    underlying_id: str
    expiry: float           # years to expiry, T > 0
    strike: float           # K > 0
    is_call: bool
    bid: float
    ask: float
    underlying_price: float  # spot S observed at ts
    rate: float              # continuously-compounded risk-free rate r
    div_yield: float         # continuously-compounded dividend yield q
    is_delayed: bool = False  # True when the venue flagged delayed data
                              # (Questrade: no real-time package subscribed)

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def key(self) -> Tuple[str, float, float, bool]:
        return (self.underlying_id, self.expiry, self.strike, self.is_call)

    def is_valid(self) -> bool:
        """Structural validity. Crossed (bid > ask) or non-positive quotes
        are rejected here; near-degenerate but valid quotes are left for
        the IV inverter to accept or refuse.

        Missing quotes (bid/ask None or NaN — e.g. yfinance rows with no
        published quote) are invalid, never fabricated. ``mid``/``spread``
        must only be called on valid quotes.
        """
        for v in (self.bid, self.ask):
            if v is None or v != v:  # None or NaN
                return False
        return (
            self.bid > 0.0
            and self.ask >= self.bid
            and self.strike > 0.0
            and self.expiry > 0.0
            and self.underlying_price > 0.0
        )

    def discount(self) -> float:
        return math.exp(-self.rate * self.expiry)

    def forward_spot(self) -> float:
        """Spot-implied forward. Fallback when no put-call parity pair is
        available for the expiry (see surface._invert_expiry)."""
        return self.underlying_price * math.exp(
            (self.rate - self.div_yield) * self.expiry
        )


class Feed(abc.ABC):
    """Market-data feed contract: unbounded iterator of normalized ticks."""

    @abc.abstractmethod
    def subscribe(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Iterator[OptionQuote]:
        """Yield quotes indefinitely (or until the feed ends)."""
        raise NotImplementedError


class SyntheticTickFeed(Feed):
    """Validation backbone: quotes generated from a known ground-truth SSVI
    surface plus microstructure noise. No broker is connected yet; this feed
    is what calibration accuracy is measured against.

    Microstructure model:
      mid  = Black(F, K, T, iv_ssvi)                      (true model price)
      spread = max(tick_size, spread_bps * mid)
      mid_noisy = mid + N(0, (noise_frac * spread)^2)
      bid/ask  = mid_noisy +/- spread / 2  (bid floored at tick_size)

    Deterministic given ``seed`` (rng is re-seeded on every subscribe()).
    """

    def __init__(
        self,
        *,
        thetas: Sequence[float],
        expiries: Sequence[float],
        rho: float,
        eta: float,
        gamma: float,
        spot: float,
        rate: float,
        div_yield: float,
        underlying_id: str = "SYN",
        k_grid: Optional[Sequence[float]] = None,
        spread_bps: float = 150.0,
        noise_frac: float = 0.1,
        tick_size: float = 0.01,
        tick_dt: float = 0.05,
        seed: int = 0,
        t0: float = 1_700_000_000.0,
    ) -> None:
        assert len(thetas) == len(expiries) and len(expiries) > 0
        assert all(t2 >= t1 for t1, t2 in zip(thetas, thetas[1:])), \
            "ground-truth thetas must be non-decreasing (no calendar arb)"
        self._thetas = tuple(float(t) for t in thetas)
        self._expiries = tuple(float(e) for e in expiries)
        self._rho = float(rho)
        self._eta = float(eta)
        self._gamma = float(gamma)
        self._spot = float(spot)
        self._rate = float(rate)
        self._div_yield = float(div_yield)
        self._underlying_id = underlying_id
        self._k_grid = tuple(k_grid) if k_grid is not None else tuple(
            np.linspace(-0.45, 0.25, 15)
        )
        self._spread_bps = float(spread_bps)
        self._noise_frac = float(noise_frac)
        self._tick_size = float(tick_size)
        self._tick_dt = float(tick_dt)
        self._seed = int(seed)
        self._t0 = float(t0)

        # Pre-compute the static plan: (T, K, is_call) for one full sweep.
        plan: List[Tuple[float, float, bool]] = []
        for T, theta in zip(self._expiries, self._thetas):
            fwd = self._spot * math.exp((self._rate - self._div_yield) * T)
            for k in self._k_grid:
                K = fwd * math.exp(k)
                plan.append((T, K, True))
                plan.append((T, K, False))
        self._plan = plan

    @property
    def sweep_size(self) -> int:
        """Number of quotes in one full surface sweep."""
        return len(self._plan)

    def _true_iv(self, T: float, K: float) -> float:
        theta = self._thetas[self._expiries.index(T)]
        fwd = self._spot * math.exp((self._rate - self._div_yield) * T)
        k = math.log(K / fwd)
        w = float(ssvi_total_var(k, theta, self._rho, self._eta, self._gamma))
        return math.sqrt(w / T)

    def subscribe(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Iterator[OptionQuote]:
        rng = np.random.default_rng(self._seed)
        ts = self._t0
        for T, K, is_call in itertools.cycle(self._plan):
            df = math.exp(-self._rate * T)
            fwd = self._spot * math.exp((self._rate - self._div_yield) * T)
            iv = self._true_iv(T, K)
            mid = black_price(fwd, K, T, iv, df, is_call)
            spread = max(self._tick_size, self._spread_bps * 1e-4 * mid)
            noisy_mid = mid + rng.normal(0.0, self._noise_frac * spread)
            bid = max(self._tick_size, noisy_mid - 0.5 * spread)
            ask = max(bid + self._tick_size, noisy_mid + 0.5 * spread)
            ts += self._tick_dt
            yield OptionQuote(
                ts=ts,
                underlying_id=self._underlying_id,
                expiry=T,
                strike=K,
                is_call=is_call,
                bid=bid,
                ask=ask,
                underlying_price=self._spot,
                rate=self._rate,
                div_yield=self._div_yield,
            )


# NOTE: the P1 ``BrokerWsFeed`` stub was removed on 2026-09-30 and replaced
# by the real Questrade adapter in ``snowball_pricer.feeds`` (``QuestradeFeed``).
# It raised NotImplementedError by design; nothing in the test suite or the
# pipeline referenced it beyond the import, so removal is safe.
