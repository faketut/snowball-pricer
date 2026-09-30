# Questrade setup: live option-quote feed

This is the step-by-step to connect `snowball-pricer` to real market data
via the `QuestradeFeed` adapter (`snowball_pricer/feeds/questrade.py`).

## 1. Questrade account + API access

1. Open a Questrade self-directed account (any type: margin, TFSA, RRSP…).
2. Go to **My Account → Security → Connected Apps & Devices** (the API Hub)
   and register a **personal app**.
3. Grant **Read scope only**. This adapter never places orders; the trade
   scope is partner-restricted anyway and must not be granted.
4. Copy the **refresh token** (shown once). Treat it like a password.

## 2. Market-data package (the part that costs money)

| Package | Cost | What you get |
|---|---|---|
| Standard (included) | $0 | Snap quotes (polled), some streaming data |
| **Real-time streaming** | **$9.95 CAD/mo** | Streaming **Level 1 for US options** + more exchanges |
| Advanced streaming | $44.95 USD/mo | L1+L2, more exchanges |

For this project you want **Real-time streaming ($9.95 CAD/mo)**. Without it,
the stream still works but every message carries `"delay": true` and the
adapter flags all ticks `is_delayed=True` — usable for plumbing tests, not
for measuring the latency thesis.

The API itself is free; there is no per-call charge beyond the package.

## 3. Environment

```bash
export QUESTRADE_REFRESH_TOKEN="paste-your-token-here"
pip install -r requirements.txt   # pulls websocket-client for the stream
```

Rules the adapter enforces:
- The token is read **only** from `QUESTRADE_REFRESH_TOKEN`. It is never a
  function default, never written to disk, never logged (there is a test
  asserting this: `test_credential_never_logged`).
- Do not commit it, do not paste it into chat.

## 4. Run

```bash
# synthetic (default, unchanged behavior)
python3 scripts/live_loop.py --sim-seconds 300

# live Questrade feed (bounded by --max-ticks; Ctrl-C stops any time)
python3 scripts/live_loop.py --feed questrade --underlyings SPY,QQQ \
    --max-ticks 2000 --rate 0.03 --div 0.0
```

What happens on startup: OAuth refresh → resolve underlying symbol ids →
fetch option chains → filter (≥7 days to expiry, strikes within ±35% of
spot) → open the WebSocket stream → normalize every L1 quote to
`OptionQuote`. Reconnects use exponential backoff and re-resolve the
universe (new weekly expiries appear automatically).

## 5. Market hours

Options stream **only during market hours, 09:30–16:00 ET, weekdays**.
Outside hours the socket yields nothing — that is normal. The feed exposes
`feed.health.last_msg_ts`; P5's STALENESS monitor is the intended consumer
(see `snowball_pricer/monitoring.py`).

## 6. Known data caveats (affect interpretation, not plumbing)

- **American exercise.** US equity options are American-style; P1's IV
  inverter assumes European exercise. Early-exercise premium on short-dated
  deep-ITM puts is folded into inverted IV. De-Americanization is future
  work — expect the arbitrage gate to quarantine more aggressively on
  short-dated wings until then.
- **No exchange timestamps on the stream.** `OptionQuote.ts` is the local
  receive time (documented in the adapter, not faked).
- **L1 only.** No depth; size fields are informational.
- **Corporate actions / dividends:** `div_yield` is a config input
  (`--div`), not sourced from Questrade. Set it per underlying.

## 7. Live smoke test checklist (before trusting a surface)

- [ ] `QUESTRADE_REFRESH_TOKEN` exported; token is read-scope
- [ ] Real-time streaming package active ($9.95 CAD/mo)
- [ ] Market hours (09:30–16:00 ET, weekday)
- [ ] Run with `--max-ticks 2000`, confirm `is_delayed=False` on ticks
      (check `feed.health.as_dict()["delayed_quotes"] == 0`)
- [ ] Surface rebuilds publish (not all quarantined) — some quarantine on
      short-dated wings is expected (American-exercise caveat above)
