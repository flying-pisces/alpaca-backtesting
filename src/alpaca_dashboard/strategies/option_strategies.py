"""
Multi-leg options strategy builders for SignalPro Pulse.

Strategies:
  - Long Call        : buy OTM call (YOLO/aggressive — lose 100%, gain unlimited)
  - Long Put         : buy OTM put  (YOLO/aggressive — lose 100%, gain unlimited)
  - Long Straddle    : buy ATM call + buy ATM put (volatility play, pre-ER neutral)
  - Short Straddle   : sell ATM call + sell ATM put (unlimited risk, kept for reference)
  - Bull Call Spread : buy K1 call, sell K2 call (directional debit — strong bullish)
  - Bear Put Spread  : buy K1 put, sell K2 put K1>K2 (directional debit — strong bearish)
  - Bull Put Spread  : sell K2 OTM put, buy K1 lower put (credit — mild bullish, 65-80% PoP)
  - Bear Call Spread : sell K3 OTM call, buy K4 higher call (credit — mild bearish, 65-80% PoP)
  - Iron Condor      : bull put spread + bear call spread (credit, neutral, 70-85% PoP, post-ER)

Pricing: Black-Scholes calls; puts via put-call parity P = C - S + K*exp(-r*T).
All prices are per-share (100x to get dollar P&L per contract).

Strategy selection matrix (PGI-driven, by risk tier):

  Pre-ER (er_dte 1–14):
    YOLO:
      PGI >= 0        → Long Call   (5% OTM, max leverage)
      PGI < 0         → Long Put    (5% OTM, max leverage)
    Aggressive:
      PGI > 30        → Bull Call Spread  (strong bullish directional debit)
      PGI < -30       → Bear Put Spread   (strong bearish directional debit)
      |PGI| <= 30     → Long Straddle     (neutral — ER move play)
    Moderate (default):
      |PGI| < 20      → Long Straddle     (neutral — size of ER move unknown)
      PGI +20 to +50  → Bull Put Spread   (mild bullish — income, 70%+ PoP)
      PGI > +50       → Bull Call Spread  (strong bullish — directional max upside)
      PGI -50 to -20  → Bear Call Spread  (mild bearish — income, 70%+ PoP)
      PGI < -50       → Bear Put Spread   (strong bearish — directional max downside)
    Conservative:
      PGI >= 0        → Bull Put Spread   (income-focused, high PoP)
      PGI < 0         → Bear Call Spread  (income-focused, high PoP)

  Post-ER (er_dte <= 0, IV crush active):
    Iron Condor (if viable: net_credit >= $0.08 and reward_risk >= 0.15)
    Fallback → Long Straddle (if iron condor not viable)
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import List, Optional

from scipy.stats import norm

# Risk-free rate (approximate T-bill)
_RISK_FREE = 0.045


# ── Black-Scholes core ────────────────────────────────────────────────────────

def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Black-Scholes European call price + Greeks."""
    if T <= 1e-10 or sigma <= 1e-10 or S <= 0 or K <= 0:
        intrinsic = max(0.0, S - K) if S > 0 and K > 0 else 0.0
        return {
            "price": intrinsic, "delta": 1.0 if S >= K else 0.0,
            "gamma": 0.0, "theta": 0.0, "vega": 0.0,
            "prob_itm": 1.0 if S >= K else 0.0,
        }

    sqrt_T = sigma * math.sqrt(T)
    if sqrt_T < 1e-10:
        intrinsic = max(0.0, S - K)
        return {
            "price": intrinsic, "delta": 1.0 if S >= K else 0.0,
            "gamma": 0.0, "theta": 0.0, "vega": 0.0,
            "prob_itm": 1.0 if S >= K else 0.0,
        }

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sqrt_T
    d2 = d1 - sigma * math.sqrt(T)

    price  = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    delta  = norm.cdf(d1)
    gamma  = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta  = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
              - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    vega   = S * norm.pdf(d1) * math.sqrt(T) / 100

    return {
        "price":    round(price, 4),
        "delta":    round(delta, 4),
        "gamma":    round(gamma, 6),
        "theta":    round(theta, 4),
        "vega":     round(vega, 4),
        "prob_itm": round(norm.cdf(d2), 4),
    }


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Black-Scholes European put price + Greeks (put-call parity)."""
    call = _bs_call(S, K, T, r, sigma)
    # Put price via put-call parity: P = C - S + K*exp(-r*T)
    put_price = call["price"] - S + K * math.exp(-r * T)
    put_delta = call["delta"] - 1.0   # put delta = call delta - 1
    return {
        "price":    round(max(put_price, 0.0), 4),
        "delta":    round(put_delta, 4),
        "gamma":    call["gamma"],
        "theta":    call["theta"],   # approx same magnitude for ATM
        "vega":     call["vega"],
        "prob_itm": round(1.0 - call["prob_itm"], 4),  # N(-d2) = 1 - N(d2)
    }


def _tick(S: float) -> float:
    """Standard option strike increment for a given stock price.
    Most liquid tickers (SPY, QQQ, AAPL, TSLA, NVDA) have $1 strikes
    at all price levels. Use $1 for S >= 50 to match real chains."""
    if S < 25:    return 0.50
    elif S < 50:  return 1.0
    else:         return 1.0


def _atm_strike(S: float) -> float:
    """Round S to nearest standard option strike increment."""
    t = _tick(S)
    return round(round(S / t) * t, 2)


def _otm_strike(S: float, pct: float) -> float:
    """Strike at pct% away from S, rounded to tick. pct > 0 = above, pct < 0 = below."""
    t = _tick(S)
    return round(round(S * (1 + pct) / t) * t, 2)


def _payoff_curve_straddle(K: float, net_premium: float, side: str, S: float) -> List[dict]:
    """
    Payoff at expiry for straddle positions.
    side = 'long' → long straddle P&L
    side = 'short' → short straddle P&L
    """
    lo = S * 0.65
    hi = S * 1.35
    step = (hi - lo) / 40
    curve = []
    p = lo
    while p <= hi:
        intrinsic = abs(p - K)
        if side == "long":
            pnl = intrinsic - net_premium
        else:
            pnl = net_premium - intrinsic
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step
    return curve


def _payoff_curve_spread(
    K1: float, K2: float, debit: float, spread_type: str, S: float
) -> List[dict]:
    """
    Payoff at expiry for vertical spreads.
    spread_type = 'bull_call' or 'bear_put'
    """
    lo = S * 0.65
    hi = S * 1.35
    step = (hi - lo) / 40
    curve = []
    p = lo
    while p <= hi:
        if spread_type == "bull_call":
            # Buy K1 call, sell K2 call: capped gain at K2-K1
            gain = max(0.0, min(p - K1, K2 - K1))
            pnl  = gain - debit
        else:
            # Bear put: buy K1 put, sell K2 put (K1 > K2)
            gain = max(0.0, min(K1 - p, K1 - K2))
            pnl  = gain - debit
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step
    return curve


def _payoff_curve_credit_spread(
    K_sell: float, K_protect: float, net_credit: float, spread_type: str, S: float
) -> List[dict]:
    """
    Payoff at expiry for vertical credit spreads.
    spread_type = 'bull_put'  : sell K_sell put,  buy K_protect put  (K_protect < K_sell)
    spread_type = 'bear_call' : sell K_sell call, buy K_protect call (K_protect > K_sell)
    """
    lo   = S * 0.65
    hi   = S * 1.35
    step = (hi - lo) / 40
    curve = []
    p = lo
    wing = abs(K_protect - K_sell)
    max_loss = round(wing - net_credit, 4)
    while p <= hi:
        if spread_type == "bull_put":
            if p >= K_sell:
                pnl = net_credit                       # both puts expire worthless
            elif p <= K_protect:
                pnl = -max_loss                        # max loss (capped by long put)
            else:
                pnl = net_credit - (K_sell - p)        # linear transition
        else:  # bear_call
            if p <= K_sell:
                pnl = net_credit                       # both calls expire worthless
            elif p >= K_protect:
                pnl = -max_loss                        # max loss (capped by long call)
            else:
                pnl = net_credit - (p - K_sell)        # linear transition
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step
    return curve


def _payoff_curve_iron_condor(
    K1: float, K2: float, K3: float, K4: float,
    net_credit: float, S: float
) -> List[dict]:
    """
    Payoff at expiry for Iron Condor.
    K1 < K2 < S < K3 < K4
    Put wing: sell K2, buy K1.   Call wing: sell K3, buy K4.
    """
    lo   = S * 0.60
    hi   = S * 1.40
    step = (hi - lo) / 50
    curve = []
    p = lo
    put_wing  = K2 - K1
    call_wing = K4 - K3
    max_loss  = round(max(put_wing, call_wing) - net_credit, 4)
    while p <= hi:
        if K2 <= p <= K3:
            pnl = net_credit                           # full profit zone
        elif p < K2:
            if p <= K1:
                pnl = net_credit - put_wing            # put wing fully exercised
            else:
                pnl = net_credit - (K2 - p)            # put wing partially exercised
        else:  # p > K3
            if p >= K4:
                pnl = net_credit - call_wing           # call wing fully exercised
            else:
                pnl = net_credit - (p - K3)            # call wing partially exercised
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step
    return curve


# ── Strategy builders ─────────────────────────────────────────────────────────

def build_long_straddle(S: float, sigma: float, expiry: date) -> dict:
    """
    Long Straddle: buy ATM call + buy ATM put at the same strike.

    Best environment: pre-ER neutral bias, expecting a large move in either
    direction but uncertain which way. Profits when actual move > total premium.

    Returns full strategy dict with legs, break-evens, max_loss, payoff_curve.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days
    K     = _atm_strike(S)

    call = _bs_call(S, K, T, _RISK_FREE, sigma)
    put  = _bs_put(S, K, T, _RISK_FREE, sigma)

    net_debit = round(call["price"] + put["price"], 4)
    be_upper  = round(K + net_debit, 2)
    be_lower  = round(K - net_debit, 2)
    em_1sigma = round(S * sigma * math.sqrt(T), 2)

    # Is the expected move larger than the straddle cost? (sanity check)
    em_vs_premium = round(em_1sigma / net_debit, 2) if net_debit > 0 else 0

    # PoP = P(S_T > be_upper) + P(S_T < be_lower)
    # Using BS risk-neutral: P(S_T > K) = N(d2) where d2 = [ln(S/K)+(r-σ²/2)T]/(σ√T)
    _d2_upper = (math.log(S / be_upper) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T)) if sigma > 0 else 0
    _d2_lower = (math.log(S / be_lower) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T)) if sigma > 0 else 0
    prob_profit = round((norm.cdf(_d2_upper) + (1 - norm.cdf(_d2_lower))) * 100, 1)

    return {
        "strategy_type":    "long_straddle",
        "strategy_label":   "Long Straddle",
        "strategy_emoji":   "⚡",
        "rationale":        (
            "Pre-ER neutral: buy call + put at same strike. "
            f"Profit if stock moves more than ${net_debit:.2f}/share "
            f"(±{round(net_debit/S*100,1)}%) from ${K}."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K,
        "legs": [
            {
                "action": "BUY", "option_type": "CALL", "strike": K,
                "premium": call["price"], "delta": call["delta"],
                "theta": call["theta"], "vega": call["vega"],
            },
            {
                "action": "BUY", "option_type": "PUT", "strike": K,
                "premium": put["price"], "delta": put["delta"],
                "theta": put["theta"], "vega": put["vega"],
            },
        ],
        "net_debit":        net_debit,
        "net_credit":       None,
        "max_profit":       None,       # theoretically unlimited
        "max_loss":         net_debit,  # capped at premium paid if stock pins at K
        "breakeven_upper":  be_upper,
        "breakeven_lower":  be_lower,
        "expected_move_1s": em_1sigma,
        "em_vs_premium":    em_vs_premium,
        "prob_profit":      prob_profit,
        "payoff_curve":     _payoff_curve_straddle(K, net_debit, "long", S),
        "price_source":     "bs_theoretical",
    }


def build_short_straddle(S: float, sigma: float, expiry: date) -> dict:
    """
    Short Straddle: sell ATM call + sell ATM put.

    Best environment: post-ER IV crush, stock expected to range-bound after
    earnings shock clears. Collect elevated pre-ER premium as IV collapses.

    Max profit = net_credit (if stock pins at K at expiry).
    Max loss = theoretically unlimited (upside) / large (downside to 0).
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days
    K     = _atm_strike(S)

    call = _bs_call(S, K, T, _RISK_FREE, sigma)
    put  = _bs_put(S, K, T, _RISK_FREE, sigma)

    net_credit = round(call["price"] + put["price"], 4)
    be_upper   = round(K + net_credit, 2)
    be_lower   = round(K - net_credit, 2)

    return {
        "strategy_type":    "short_straddle",
        "strategy_label":   "Short Straddle",
        "strategy_emoji":   "💎",
        "rationale":        (
            "Post-ER IV crush: sell call + put at same strike. "
            f"Collect ${net_credit:.2f}/share premium. "
            f"Profit if stock stays between ${be_lower} and ${be_upper}."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K,
        "legs": [
            {
                "action": "SELL", "option_type": "CALL", "strike": K,
                "premium": call["price"], "delta": call["delta"],
                "theta": call["theta"], "vega": call["vega"],
            },
            {
                "action": "SELL", "option_type": "PUT", "strike": K,
                "premium": put["price"], "delta": put["delta"],
                "theta": put["theta"], "vega": put["vega"],
            },
        ],
        "net_debit":        None,
        "net_credit":       net_credit,
        "max_profit":       net_credit,
        "max_loss":         None,       # unlimited upside / large downside
        "breakeven_upper":  be_upper,
        "breakeven_lower":  be_lower,
        "payoff_curve":     _payoff_curve_straddle(K, net_credit, "short", S),
        "price_source":     "bs_theoretical",
    }


def build_bull_call_spread(S: float, sigma: float, expiry: date) -> dict:
    """
    Bull Call Spread: buy lower-strike call (K1, ATM or slight OTM),
    sell higher-strike call (K2, ~1σ above S).

    Best environment: pre-ER bullish PGI. Defined max loss = net debit.
    Max gain = (K2 - K1) - debit if stock closes above K2 at expiry.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    # K1: buy call at ATM; K2: sell call at ~5% OTM
    K1 = _atm_strike(S)
    K2 = _otm_strike(S, +0.05)
    if K2 <= K1:
        K2 = round(K1 + _tick(S), 2)

    call_buy  = _bs_call(S, K1, T, _RISK_FREE, sigma)
    call_sell = _bs_call(S, K2, T, _RISK_FREE, sigma)

    debit      = round(call_buy["price"] - call_sell["price"], 4)
    max_gain   = round((K2 - K1) - debit, 4)
    breakeven  = round(K1 + debit, 2)
    reward_risk = round(max_gain / debit, 2) if debit > 0 else 0

    # PoP = P(S_T > breakeven) = N(d2) with K=breakeven
    _d2_be = (math.log(S / breakeven) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T)) if sigma > 0 else 0
    prob_profit = round(norm.cdf(_d2_be) * 100, 1)

    return {
        "strategy_type":    "bull_call_spread",
        "strategy_label":   "Bull Call Spread",
        "strategy_emoji":   "📈",
        "rationale":        (
            f"Bullish PGI: buy ${K1}C, sell ${K2}C. "
            f"Net debit ${debit:.2f}/share. "
            f"Max gain ${max_gain:.2f} if stock above ${K2} at expiry. "
            f"Reward/risk {reward_risk:.1f}×."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K1,
        "strike_sell":      K2,
        "legs": [
            {
                "action": "BUY",  "option_type": "CALL", "strike": K1,
                "premium": call_buy["price"],  "delta": call_buy["delta"],
                "theta": call_buy["theta"],    "vega": call_buy["vega"],
            },
            {
                "action": "SELL", "option_type": "CALL", "strike": K2,
                "premium": call_sell["price"], "delta": call_sell["delta"],
                "theta": call_sell["theta"],   "vega": call_sell["vega"],
            },
        ],
        "net_debit":        debit,
        "net_credit":       None,
        "max_profit":       max_gain,
        "max_loss":         debit,
        "breakeven_upper":  breakeven,
        "breakeven_lower":  None,
        "reward_risk":      reward_risk,
        "prob_profit":      prob_profit,
        "payoff_curve":     _payoff_curve_spread(K1, K2, debit, "bull_call", S),
        "price_source":     "bs_theoretical",
    }


def build_bear_put_spread(S: float, sigma: float, expiry: date) -> dict:
    """
    Bear Put Spread: buy higher-strike put (K1, ATM), sell lower-strike put (K2).

    Best environment: pre-ER bearish PGI. Defined max loss = net debit.
    Max gain = (K1 - K2) - debit if stock closes below K2 at expiry.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    # K1: buy put at ATM; K2: sell put at ~5% OTM below
    K1 = _atm_strike(S)
    K2 = _otm_strike(S, -0.05)
    if K2 >= K1:
        K2 = round(K1 - _tick(S), 2)

    put_buy  = _bs_put(S, K1, T, _RISK_FREE, sigma)
    put_sell = _bs_put(S, K2, T, _RISK_FREE, sigma)

    debit       = round(put_buy["price"] - put_sell["price"], 4)
    max_gain    = round((K1 - K2) - debit, 4)
    breakeven   = round(K1 - debit, 2)
    reward_risk = round(max_gain / debit, 2) if debit > 0 else 0

    # PoP = P(S_T < breakeven) = 1 - N(d2) with K=breakeven
    _d2_be = (math.log(S / breakeven) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T)) if sigma > 0 else 0
    prob_profit = round((1 - norm.cdf(_d2_be)) * 100, 1)

    return {
        "strategy_type":    "bear_put_spread",
        "strategy_label":   "Bear Put Spread",
        "strategy_emoji":   "📉",
        "rationale":        (
            f"Bearish PGI: buy ${K1}P, sell ${K2}P. "
            f"Net debit ${debit:.2f}/share. "
            f"Max gain ${max_gain:.2f} if stock below ${K2} at expiry. "
            f"Reward/risk {reward_risk:.1f}×."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K1,
        "strike_sell":      K2,
        "legs": [
            {
                "action": "BUY",  "option_type": "PUT", "strike": K1,
                "premium": put_buy["price"],  "delta": put_buy["delta"],
                "theta": put_buy["theta"],    "vega": put_buy["vega"],
            },
            {
                "action": "SELL", "option_type": "PUT", "strike": K2,
                "premium": put_sell["price"], "delta": put_sell["delta"],
                "theta": put_sell["theta"],   "vega": put_sell["vega"],
            },
        ],
        "net_debit":        debit,
        "net_credit":       None,
        "max_profit":       max_gain,
        "max_loss":         debit,
        "breakeven_upper":  None,
        "breakeven_lower":  breakeven,
        "reward_risk":      reward_risk,
        "prob_profit":      prob_profit,
        "payoff_curve":     _payoff_curve_spread(K1, K2, debit, "bear_put", S),
        "price_source":     "bs_theoretical",
    }


def build_bull_put_spread(S: float, sigma: float, expiry: date) -> dict:
    """
    Bull Put Spread (vertical credit spread — bullish / neutral-bullish).

    Sell an OTM put (K_sell, ~5% below S) and buy a further OTM put for
    protection (K_protect, ~10% below S). Collect net credit upfront.

    Win condition: stock stays ABOVE K_sell at expiry (65–80% PoP typical).
    Best environment: mild bullish PGI, or pre-ER where stock has support.
    Defined risk: max loss = wing_width − net_credit (if stock crashes below K_protect).
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K_sell    = _otm_strike(S, -0.05)   # sell ~5% OTM put (more premium)
    K_protect = _otm_strike(S, -0.10)   # buy  ~10% OTM put (protection)
    if K_protect >= K_sell:
        K_protect = round(K_sell - _tick(S), 2)

    put_sell = _bs_put(S, K_sell,    T, _RISK_FREE, sigma)
    put_buy  = _bs_put(S, K_protect, T, _RISK_FREE, sigma)

    net_credit = round(put_sell["price"] - put_buy["price"], 4)
    wing       = round(K_sell - K_protect, 2)
    max_loss   = round(wing - net_credit, 4)
    breakeven  = round(K_sell - net_credit, 2)
    reward_risk = round(net_credit / max_loss, 2) if max_loss > 0 else 0
    # Prob profit ≈ prob stock stays above K_sell (N(d2) for K_sell put)
    prob_profit = round((1 - put_sell["prob_itm"]) * 100, 1)

    return {
        "strategy_type":    "bull_put_spread",
        "strategy_label":   "Bull Put Spread",
        "strategy_emoji":   "🐂",
        "rationale":        (
            f"Mild bullish: sell ${K_sell}P, buy ${K_protect}P for protection. "
            f"Collect ${net_credit:.2f}/share credit. "
            f"Max loss ${max_loss:.2f} if stock falls below ${K_protect}. "
            f"{prob_profit:.0f}% probability of full profit."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K_sell,
        "strike_protect":   K_protect,
        "legs": [
            {
                "action": "SELL", "option_type": "PUT", "strike": K_sell,
                "premium": put_sell["price"], "delta": put_sell["delta"],
                "theta": put_sell["theta"],   "vega": put_sell["vega"],
            },
            {
                "action": "BUY",  "option_type": "PUT", "strike": K_protect,
                "premium": put_buy["price"],  "delta": put_buy["delta"],
                "theta": put_buy["theta"],    "vega": put_buy["vega"],
            },
        ],
        "net_debit":        None,
        "net_credit":       net_credit,
        "max_profit":       net_credit,
        "max_loss":         max_loss,
        "breakeven_upper":  None,
        "breakeven_lower":  breakeven,
        "reward_risk":      reward_risk,
        "prob_profit":      prob_profit,
        "wing_width":       wing,
        "payoff_curve":     _payoff_curve_credit_spread(K_sell, K_protect, net_credit, "bull_put", S),
        "price_source":     "bs_theoretical",
    }


def build_bear_call_spread(S: float, sigma: float, expiry: date) -> dict:
    """
    Bear Call Spread (vertical credit spread — bearish / neutral-bearish).

    Sell an OTM call (K_sell, ~5% above S) and buy a further OTM call for
    protection (K_protect, ~10% above S). Collect net credit upfront.

    Win condition: stock stays BELOW K_sell at expiry (65–80% PoP typical).
    Best environment: mild bearish PGI, or post-ER when stock likely to fade.
    Defined risk: max loss = wing_width − net_credit (if stock surges above K_protect).
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K_sell    = _otm_strike(S, +0.05)   # sell ~5% OTM call
    K_protect = _otm_strike(S, +0.10)   # buy  ~10% OTM call (protection)
    if K_protect <= K_sell:
        K_protect = round(K_sell + _tick(S), 2)

    call_sell = _bs_call(S, K_sell,    T, _RISK_FREE, sigma)
    call_buy  = _bs_call(S, K_protect, T, _RISK_FREE, sigma)

    net_credit  = round(call_sell["price"] - call_buy["price"], 4)
    wing        = round(K_protect - K_sell, 2)
    max_loss    = round(wing - net_credit, 4)
    breakeven   = round(K_sell + net_credit, 2)
    reward_risk = round(net_credit / max_loss, 2) if max_loss > 0 else 0
    # PoP = prob stock stays BELOW K_sell at expiry = 1 − prob_itm(K_sell call)
    prob_profit = round((1.0 - call_sell["prob_itm"]) * 100, 1)

    return {
        "strategy_type":    "bear_call_spread",
        "strategy_label":   "Bear Call Spread",
        "strategy_emoji":   "🐻",
        "rationale":        (
            f"Mild bearish: sell ${K_sell}C, buy ${K_protect}C for protection. "
            f"Collect ${net_credit:.2f}/share credit. "
            f"Max loss ${max_loss:.2f} if stock surges above ${K_protect}. "
            f"{prob_profit:.0f}% probability of full profit."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K_sell,
        "strike_protect":   K_protect,
        "legs": [
            {
                "action": "SELL", "option_type": "CALL", "strike": K_sell,
                "premium": call_sell["price"], "delta": call_sell["delta"],
                "theta": call_sell["theta"],   "vega": call_sell["vega"],
            },
            {
                "action": "BUY",  "option_type": "CALL", "strike": K_protect,
                "premium": call_buy["price"],  "delta": call_buy["delta"],
                "theta": call_buy["theta"],    "vega": call_buy["vega"],
            },
        ],
        "net_debit":        None,
        "net_credit":       net_credit,
        "max_profit":       net_credit,
        "max_loss":         max_loss,
        "breakeven_upper":  breakeven,
        "breakeven_lower":  None,
        "reward_risk":      reward_risk,
        "prob_profit":      prob_profit,
        "wing_width":       wing,
        "payoff_curve":     _payoff_curve_credit_spread(K_sell, K_protect, net_credit, "bear_call", S),
        "price_source":     "bs_theoretical",
    }


def build_iron_condor(S: float, sigma: float, expiry: date,
                       short_otm: float = 0.05, wing_width: float = 0.05) -> dict:
    """
    Iron Condor: Bull Put Spread + Bear Call Spread combined.

    Structure: sell K2 put, buy K1 put | sell K3 call, buy K4 call
               K1 < K2 < S < K3 < K4

    Collect credit from both wings. Maximum profit when stock stays between
    K2 and K3 (the profit tent). Fully defined risk on both sides.

    short_otm  : how far OTM the short strikes are (default 5%).
                 Use 0.08–0.10 for conservative/baseline plays (~75–80% PoP).
    wing_width : distance from short to long strike as % of S (default 5%).

    Best environment: post-ER IV crush — stock expected to range-bound.
    70–85% PoP typical. Far superior to short straddle (unlimited risk).
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    # Put wing: sell K2 (short_otm OTM below), buy K1 (short_otm + wing_width OTM below)
    K2 = _otm_strike(S, -short_otm)
    K1 = _otm_strike(S, -(short_otm + wing_width))
    if K1 >= K2:
        K1 = round(K2 - _tick(S), 2)

    # Call wing: sell K3 (short_otm OTM above), buy K4 (short_otm + wing_width OTM above)
    K3 = _otm_strike(S, +short_otm)
    K4 = _otm_strike(S, +(short_otm + wing_width))
    if K4 <= K3:
        K4 = round(K3 + _tick(S), 2)

    put_sell  = _bs_put(S,  K2, T, _RISK_FREE, sigma)
    put_buy   = _bs_put(S,  K1, T, _RISK_FREE, sigma)
    call_sell = _bs_call(S, K3, T, _RISK_FREE, sigma)
    call_buy  = _bs_call(S, K4, T, _RISK_FREE, sigma)

    put_credit  = round(put_sell["price"]  - put_buy["price"],  4)
    call_credit = round(call_sell["price"] - call_buy["price"], 4)
    net_credit  = round(put_credit + call_credit, 4)

    put_wing  = round(K2 - K1, 2)
    call_wing = round(K4 - K3, 2)
    # Max loss occurs on whichever wing gets breached (usually equal width)
    max_loss  = round(max(put_wing, call_wing) - net_credit, 4)

    be_upper = round(K3 + net_credit, 2)
    be_lower = round(K2 - net_credit, 2)
    reward_risk = round(net_credit / max_loss, 2) if max_loss > 0 else 0

    # PoP = P(K2 <= S_T <= K3) = 1 - P(S_T < K2) - P(S_T > K3)
    prob_profit = round(
        max(min((1.0 - put_sell["prob_itm"] - call_sell["prob_itm"]) * 100, 99.0), 10.0), 1
    )

    # ── QUALITY GATE: reject bad iron condors ────────────────────────────
    # Rule: max_profit must be >= 20% of max_loss (i.e., reward_risk >= 0.20)
    # AND PoP must be >= 80% for it to be a sensible risk/reward.
    # A $0.03 credit on $1.97 risk (R/R 0.015) is a terrible trade.
    if max_loss > 0 and net_credit / max_loss < 0.20:
        return None  # caller handles None as "no viable iron condor"
    if prob_profit < 80.0:
        return None  # not enough probability to justify the limited reward

    return {
        "strategy_type":    "iron_condor",
        "strategy_label":   "Iron Condor",
        "strategy_emoji":   "🦅",
        "rationale":        (
            f"Post-ER IV crush: sell ${K2}P/${K3}C, buy ${K1}P/${K4}C wings. "
            f"Collect ${net_credit:.2f}/share total credit. "
            f"Max profit if stock stays between ${K2} and ${K3}. "
            f"Max loss ${max_loss:.2f} — fully defined risk."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K2,         # put sell strike (primary reference)
        "strike_call_sell": K3,
        "strike_put_buy":   K1,
        "strike_call_buy":  K4,
        "profit_zone":      {"lower": K2, "upper": K3},
        "legs": [
            {
                "action": "SELL", "option_type": "PUT",  "strike": K2,
                "premium": put_sell["price"],  "delta": put_sell["delta"],
                "theta": put_sell["theta"],    "vega": put_sell["vega"],
            },
            {
                "action": "BUY",  "option_type": "PUT",  "strike": K1,
                "premium": put_buy["price"],   "delta": put_buy["delta"],
                "theta": put_buy["theta"],     "vega": put_buy["vega"],
            },
            {
                "action": "SELL", "option_type": "CALL", "strike": K3,
                "premium": call_sell["price"], "delta": call_sell["delta"],
                "theta": call_sell["theta"],   "vega": call_sell["vega"],
            },
            {
                "action": "BUY",  "option_type": "CALL", "strike": K4,
                "premium": call_buy["price"],  "delta": call_buy["delta"],
                "theta": call_buy["theta"],    "vega": call_buy["vega"],
            },
        ],
        "net_debit":        None,
        "net_credit":       net_credit,
        "put_credit":       put_credit,
        "call_credit":      call_credit,
        "max_profit":       net_credit,
        "max_loss":         max_loss,
        "breakeven_upper":  be_upper,
        "breakeven_lower":  be_lower,
        "reward_risk":      reward_risk,
        "prob_profit":      prob_profit,
        "put_wing":         put_wing,
        "call_wing":        call_wing,
        "payoff_curve":     _payoff_curve_iron_condor(K1, K2, K3, K4, net_credit, S),
        "price_source":     "bs_theoretical",
    }


def build_long_call_condor(
    S: float, sigma: float, expiry: date, pct_spread: float = 0.03
) -> dict:
    """
    Long Call Condor: buy K1 call, sell K2 call, sell K3 call, buy K4 call.

    Equally spaced strikes: K1 = ATM, K2 = K1+spread, K3 = K2+spread, K4 = K3+spread.

    Best environment: neutral-to-slightly-bullish, low volatility, stock expected
    to remain in the K2–K3 profit zone. Pure debit — max loss = net debit paid.
    Works well for boring ETFs (SPY, QQQ, IWM, VOO) in low-vol environments.

    Max loss   = net_debit (stock < K1 or > K4 at expiry)
    Max profit = wing_width - net_debit  (stock stays between K2 and K3)
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K1 = _atm_strike(S)
    K2 = _otm_strike(S, pct_spread)
    K3 = _otm_strike(S, pct_spread * 2)
    K4 = _otm_strike(S, pct_spread * 3)

    tick = _tick(S)
    if K2 <= K1: K2 = round(K1 + tick, 2)
    if K3 <= K2: K3 = round(K2 + tick, 2)
    if K4 <= K3: K4 = round(K3 + tick, 2)

    c1 = _bs_call(S, K1, T, _RISK_FREE, sigma)   # buy
    c2 = _bs_call(S, K2, T, _RISK_FREE, sigma)   # sell
    c3 = _bs_call(S, K3, T, _RISK_FREE, sigma)   # sell
    c4 = _bs_call(S, K4, T, _RISK_FREE, sigma)   # buy

    net_debit  = round(c1["price"] - c2["price"] - c3["price"] + c4["price"], 4)
    if net_debit < 0:
        net_debit = abs(net_debit)

    wing_width = round(K2 - K1, 2)
    max_profit = round(wing_width - net_debit, 4)
    be_lower   = round(K1 + net_debit, 2)
    be_upper   = round(K4 - net_debit, 2)
    reward_risk = round(max_profit / net_debit, 2) if net_debit > 0 else 0

    prob_profit = round(
        max((c2["prob_itm"] - c3["prob_itm"]) * 100, 5.0), 1
    )

    lo   = K1 * 0.93
    hi   = K4 * 1.07
    step = (hi - lo) / 60
    curve, p = [], lo
    while p <= hi:
        if p <= K1:
            pnl = -net_debit
        elif p <= K2:
            pnl = -net_debit + (p - K1)
        elif p <= K3:
            pnl = max_profit
        elif p <= K4:
            pnl = max_profit - (p - K3)
        else:
            pnl = -net_debit
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type":  "long_call_condor",
        "strategy_label": "Long Call Condor",
        "strategy_emoji": "🦁",
        "rationale": (
            f"Buy ${K1}C, sell ${K2}C/{K3}C, buy ${K4}C. "
            f"Pay ${net_debit:.2f}/share. Max profit ${max_profit:.2f} if stock stays "
            f"${K2}–${K3}. Max loss capped at ${net_debit:.2f} — fully defined risk."
        ),
        "expiration_iso":  expiry.isoformat(),
        "expiration":      expiry.strftime("%b %d '%y"),
        "dte":             dte,
        "strike":          K1,
        "strike_sell_1":   K2,
        "strike_sell_2":   K3,
        "strike_buy_2":    K4,
        "profit_zone":     {"lower": K2, "upper": K3},
        "legs": [
            {"action": "BUY",  "option_type": "CALL", "strike": K1,
             "premium": c1["price"], "delta": c1["delta"],
             "theta": c1["theta"],   "vega": c1["vega"]},
            {"action": "SELL", "option_type": "CALL", "strike": K2,
             "premium": c2["price"], "delta": c2["delta"],
             "theta": c2["theta"],   "vega": c2["vega"]},
            {"action": "SELL", "option_type": "CALL", "strike": K3,
             "premium": c3["price"], "delta": c3["delta"],
             "theta": c3["theta"],   "vega": c3["vega"]},
            {"action": "BUY",  "option_type": "CALL", "strike": K4,
             "premium": c4["price"], "delta": c4["delta"],
             "theta": c4["theta"],   "vega": c4["vega"]},
        ],
        "net_debit":       net_debit,
        "net_credit":      None,
        "max_profit":      max_profit,
        "max_loss":        net_debit,
        "breakeven_lower": be_lower,
        "breakeven_upper": be_upper,
        "reward_risk":     reward_risk,
        "prob_profit":     prob_profit,
        "payoff_curve":    curve,
        "price_source":    "bs_theoretical",
    }


def build_covered_call(
    S: float, sigma: float, expiry: date, pct_otm: float = 0.07
) -> dict:
    """
    Covered Call: own shares + sell OTM call.

    Best environment: neutral-to-slightly-bullish, harvest premium on existing or
    new position. Collect call premium; sacrifice upside above the strike.

    Max gain  = (K - S) + premium  (stock rises to/above K at expiry)
    Breakeven = S - premium
    Max loss  = S - premium (theoretical, stock goes to 0)
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K       = _otm_strike(S, pct_otm)
    call    = _bs_call(S, K, T, _RISK_FREE, sigma)
    premium = call["price"]

    breakeven        = round(S - premium, 2)
    max_gain         = round((K - S) + premium, 4)
    annualized_yield = round(premium / S * (365 / dte) * 100, 1) if dte > 0 else 0
    prob_profit      = round((1 - call["prob_itm"]) * 100, 1)

    lo, hi = S * 0.80, K * 1.15
    step = (hi - lo) / 50
    payoff, p = [], lo
    while p <= hi:
        pnl = max_gain if p >= K else round(p - breakeven, 4)
        payoff.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type":     "covered_call",
        "strategy_label":    "Covered Call",
        "strategy_emoji":    "📞",
        "rationale": (
            f"Sell ${K}C ({pct_otm*100:.0f}% OTM), expiring {expiry.strftime('%b %d')}. "
            f"Collect ${premium:.2f}/share ({annualized_yield:.0f}% ann.). "
            f"Max gain ${max_gain:.2f} if stock ≥ ${K}. "
            f"Breakeven ${breakeven:.2f}. Upside capped at ${K}."
        ),
        "expiration_iso":    expiry.isoformat(),
        "expiration":        expiry.strftime("%b %d '%y"),
        "dte":               dte,
        "strike":            K,
        "premium_collected": round(premium, 4),
        "annualized_yield":  annualized_yield,
        "legs": [
            {"action": "OWN",  "option_type": "SHARES", "strike": round(S, 2),
             "premium": round(S, 2), "delta": 1.0, "theta": 0.0, "vega": 0.0},
            {"action": "SELL", "option_type": "CALL",   "strike": K,
             "premium": premium, "delta": call["delta"],
             "theta": call["theta"], "vega": call["vega"]},
        ],
        "net_debit":         None,
        "net_credit":        round(premium, 4),
        "max_profit":        max_gain,
        "max_loss":          round(breakeven, 4),
        "breakeven_upper":   None,
        "breakeven_lower":   breakeven,
        "reward_risk":       round(max_gain / S, 2) if S > 0 else 0,
        "prob_profit":       prob_profit,
        "payoff_curve":      payoff,
        "price_source":      "bs_theoretical",
    }


def build_dividend_capture(
    S: float, sigma: float, expiry: date, dividend: float,
    ex_date: date, mode: str = "atm",
) -> dict:
    """
    Dividend Capture: buy stock + sell ATM/ITM covered call through ex-div.

    The idea: buy stock before ex-dividend date, sell an ATM or slightly ITM call.
    Collect dividend + call premium. The call is sold with expiry AFTER ex-div so
    we hold shares through the ex-date and receive the dividend.

    Modes:
      "atm" — sell call at-the-money (K ≈ S). Max premium, likely assignment.
      "itm" — sell call 2-3% in-the-money (K = S*0.97). Higher premium, almost certain assignment.

    Total return = dividend + call_premium - max(0, S - K)
    Breakeven = S - dividend - call_premium + max(0, S - K)
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days
    days_to_ex = (ex_date - today).days

    # Strike: ATM or slightly ITM
    if mode == "itm":
        K = round(S * 0.97 / 0.5) * 0.5   # round to nearest $0.50, ~3% ITM
    else:
        K = round(S / 0.5) * 0.5           # ATM rounded to $0.50

    call    = _bs_call(S, K, T, _RISK_FREE, sigma)
    premium = call["price"]

    # Intrinsic value of ITM call (how much we "lose" on stock being called at K)
    intrinsic = max(0, S - K)

    # Total income per share
    total_income = dividend + premium - intrinsic
    breakeven    = round(S - total_income, 2)
    max_gain     = round(total_income, 4)

    # Annualized yield of the total capture
    ann_yield = round(total_income / S * (365 / max(dte, 1)) * 100, 1)
    div_yield = round(dividend / S * 100, 2)

    # Prob profit: stock must stay above breakeven
    # Use put probability: P(S_T > breakeven) ≈ 1 - P(put ITM at breakeven strike)
    put_at_be = _bs_put(S, breakeven, T, _RISK_FREE, sigma) if breakeven > 0 else {"prob_itm": 0}
    prob_profit = round((1 - put_at_be.get("prob_itm", 0.3)) * 100, 1)

    lo, hi = S * 0.85, S * 1.10
    step = (hi - lo) / 50
    payoff, p = [], lo
    while p <= hi:
        # At expiry: if p >= K, stock called away at K, keep premium + dividend
        # If p < K, keep stock (worth p), keep premium + dividend, lose (S - p) on stock
        if p >= K:
            pnl = max_gain
        else:
            pnl = round(p - S + premium + dividend, 4)
        payoff.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type":     "dividend_capture",
        "strategy_label":    "Dividend Capture",
        "strategy_emoji":    "💰",
        "rationale": (
            f"Buy shares @ ${S:.2f}, sell ${K}C {'(ATM)' if mode == 'atm' else '(ITM)'} "
            f"expiring {expiry.strftime('%b %d')} for ${premium:.2f}/share. "
            f"Collect ${dividend:.2f} dividend (ex-date {ex_date.strftime('%b %d')}, {days_to_ex}d away). "
            f"Total income ${total_income:.2f}/share ({ann_yield:.0f}% ann). "
            f"Likely assigned at ${K} — keep all income."
        ),
        "expiration_iso":    expiry.isoformat(),
        "expiration":        expiry.strftime("%b %d '%y"),
        "dte":               dte,
        "ex_date_iso":       ex_date.isoformat(),
        "days_to_ex":        days_to_ex,
        "strike":            K,
        "dividend_amount":   dividend,
        "dividend_yield":    div_yield,
        "premium_collected": round(premium, 4),
        "annualized_yield":  ann_yield,
        "total_income":      round(total_income, 4),
        "mode":              mode,
        "legs": [
            {"action": "BUY",  "option_type": "SHARES", "strike": round(S, 2),
             "premium": round(S, 2), "delta": 1.0, "theta": 0.0, "vega": 0.0},
            {"action": "SELL", "option_type": "CALL",   "strike": K,
             "premium": premium, "delta": call["delta"],
             "theta": call["theta"], "vega": call["vega"]},
        ],
        "net_debit":         None,
        "net_credit":        round(premium + dividend, 4),
        "max_profit":        max_gain,
        "max_loss":          round(breakeven, 4),
        "breakeven_upper":   None,
        "breakeven_lower":   breakeven,
        "reward_risk":       round(max_gain / S, 2) if S > 0 else 0,
        "prob_profit":       prob_profit,
        "payoff_curve":      payoff,
        "price_source":      "bs_theoretical",
    }


def build_long_call(S: float, sigma: float, expiry: date, pct_otm: float = 0.05) -> dict:
    """
    Long Call: buy a single OTM call.

    YOLO/Aggressive play. Max loss = premium paid (100% of position).
    Unlimited upside above breakeven. High leverage, high risk.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K       = _otm_strike(S, pct_otm)
    call    = _bs_call(S, K, T, _RISK_FREE, sigma)
    premium = call["price"]
    breakeven = round(K + premium, 2)

    # PoP = P(S_T > breakeven)
    if sigma > 0 and breakeven > 0:
        _d2_be = (math.log(S / breakeven) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        prob_profit = round(norm.cdf(_d2_be) * 100, 1)
    else:
        prob_profit = 0.0

    # Payoff curve
    lo, hi = S * 0.70, S * 1.60
    step = (hi - lo) / 50
    payoff, p = [], lo
    while p <= hi:
        payoff.append({"price": round(p, 2), "pnl": round(max(0, p - K) - premium, 4)})
        p += step

    return {
        "strategy_type":    "long_call",
        "strategy_label":   "Long Call",
        "strategy_emoji":   "🚀",
        "rationale": (
            f"YOLO directional: buy ${K}C expiring {expiry.strftime('%b %d')}. "
            f"Pay ${premium:.2f}/share. Breaks even at ${breakeven}. "
            f"Lose 100% below ${K}, unlimited gain above ${breakeven}."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K,
        "legs": [{
            "action": "BUY", "option_type": "CALL", "strike": K,
            "premium": premium, "delta": call["delta"],
            "theta": call["theta"], "vega": call["vega"],
        }],
        "net_debit":        round(premium, 4),
        "net_credit":       None,
        "max_profit":       None,   # unlimited
        "max_loss":         round(premium, 4),
        "breakeven_upper":  breakeven,
        "breakeven_lower":  None,
        "reward_risk":      None,   # unlimited
        "prob_profit":      prob_profit,
        "payoff_curve":     payoff,
        "price_source":     "bs_theoretical",
    }


def build_long_put(S: float, sigma: float, expiry: date, pct_otm: float = 0.05) -> dict:
    """
    Long Put: buy a single OTM put.

    YOLO/Aggressive bearish play. Max loss = premium paid (100% of position).
    Max gain = strike - premium (stock goes to 0). High leverage, high risk.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K     = _otm_strike(S, -pct_otm)
    put   = _bs_put(S, K, T, _RISK_FREE, sigma)
    premium   = put["price"]
    breakeven = round(K - premium, 2)

    # PoP = P(S_T < breakeven)
    if sigma > 0 and breakeven > 0:
        _d2_be = (math.log(S / breakeven) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        prob_profit = round((1 - norm.cdf(_d2_be)) * 100, 1)
    else:
        prob_profit = 0.0

    # Payoff curve
    lo, hi = S * 0.50, S * 1.30
    step = (hi - lo) / 50
    payoff, p = [], lo
    while p <= hi:
        payoff.append({"price": round(p, 2), "pnl": round(max(0, K - p) - premium, 4)})
        p += step

    return {
        "strategy_type":    "long_put",
        "strategy_label":   "Long Put",
        "strategy_emoji":   "🔻",
        "rationale": (
            f"YOLO directional: buy ${K}P expiring {expiry.strftime('%b %d')}. "
            f"Pay ${premium:.2f}/share. Breaks even at ${breakeven}. "
            f"Lose 100% above ${K}, max gain if stock crashes."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K,
        "legs": [{
            "action": "BUY", "option_type": "PUT", "strike": K,
            "premium": premium, "delta": put["delta"],
            "theta": put["theta"], "vega": put["vega"],
        }],
        "net_debit":        round(premium, 4),
        "net_credit":       None,
        "max_profit":       round(K - premium, 4),   # stock goes to 0
        "max_loss":         round(premium, 4),
        "breakeven_upper":  None,
        "breakeven_lower":  breakeven,
        "reward_risk":      round((K - premium) / premium, 2) if premium > 0 else None,
        "prob_profit":      prob_profit,
        "payoff_curve":     payoff,
        "price_source":     "bs_theoretical",
    }


def build_cash_secured_put(S: float, sigma: float, expiry: date, pct_otm: float = 0.07) -> dict:
    """
    Cash-Secured Put: sell an OTM put, back it with full cash collateral.

    Best for: neutral-to-bullish outlook; get paid to wait or buy the stock at a
    discount. Conservative tiers use deeper OTM (higher PoP); aggressive tiers
    sell closer to ATM (more premium, more assignment risk).

    pct_otm=0.15  → conservative  (15% OTM, ~75–80% PoP)
    pct_otm=0.07  → moderate      (7% OTM,  ~65–70% PoP)
    pct_otm=0.03  → aggressive    (3% OTM,  ~55–60% PoP)

    Max profit = net credit (stock stays above strike at expiry).
    Max loss   = strike − premium (stock goes to zero; mitigated by selling at a discount).
    Break-even = strike − premium.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K       = _otm_strike(S, -pct_otm)
    put     = _bs_put(S, K, T, _RISK_FREE, sigma)
    premium = put["price"]
    breakeven      = round(K - premium, 2)
    collateral_share = K                         # cash needed per share
    max_loss         = round(breakeven, 4)       # stock goes to 0 (extreme)
    reward_risk      = round(premium / (K - premium), 2) if (K - premium) > 0 else 0

    # PoP = P(S_T > K) = 1 - prob_itm(put)
    prob_profit = round((1 - put["prob_itm"]) * 100, 1)

    # Payoff at expiry
    lo, hi = S * 0.60, S * 1.20
    step = (hi - lo) / 50
    payoff, p = [], lo
    while p <= hi:
        if p >= K:
            pnl = premium               # put expires worthless, keep premium
        else:
            pnl = premium - (K - p)     # assignment loss
        payoff.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    otm_pct_label = f"{pct_otm*100:.0f}%"
    return {
        "strategy_type":    "cash_secured_put",
        "strategy_label":   "Cash-Secured Put",
        "strategy_emoji":   "💵",
        "rationale": (
            f"Sell ${K}P ({otm_pct_label} OTM), expiring {expiry.strftime('%b %d')}. "
            f"Collect ${premium:.2f}/share. Keep premium if stock stays above ${K}. "
            f"Effective buy price if assigned: ${breakeven:.2f} "
            f"({round((breakeven/S - 1)*100, 1)}% vs today). "
            f"{prob_profit:.0f}% PoP. Need ${collateral_share*100:,.0f} cash per contract."
        ),
        "expiration_iso":    expiry.isoformat(),
        "expiration":        expiry.strftime("%b %d '%y"),
        "dte":               dte,
        "strike":            K,
        "collateral_per_share": collateral_share,
        "effective_buy_price":  breakeven,
        "otm_pct":           round(pct_otm * 100, 1),
        "legs": [{
            "action": "SELL", "option_type": "PUT", "strike": K,
            "premium": premium, "delta": put["delta"],
            "theta": put["theta"], "vega": put["vega"],
        }],
        "net_debit":         None,
        "net_credit":        round(premium, 4),
        "max_profit":        round(premium, 4),
        "max_loss":          max_loss,
        "breakeven_upper":   None,
        "breakeven_lower":   breakeven,
        "reward_risk":       reward_risk,
        "prob_profit":       prob_profit,
        "payoff_curve":      payoff,
        "price_source":      "bs_theoretical",
    }


def build_stock_buy(
    S: float,
    target_pct: float = 0.08,
    stop_pct: float   = 0.05,
    label: str        = "Moderate",
) -> dict:
    """
    Directional stock buy with fixed target, fixed stop, AND trailing stop.

    The trailing stop follows the stock up and protects gains if the trend
    reverses after a rally. It never moves down — only ratchets up.

    Tier parameters:
      conservative : stop 3%,  target  5%, trail 3%  — tight risk
      moderate     : stop 5%,  target  8%, trail 4%  — balanced
      aggressive   : stop 7%,  target 15%, trail 5%  — wide stops
      yolo         : stop 10%, target 25%, trail 7%  — high conviction
    """
    target     = round(S * (1 + target_pct), 2)
    stop       = round(S * (1 - stop_pct), 2)
    max_profit = round(target - S, 2)
    max_loss   = round(S - stop, 2)
    rr         = round(target_pct / stop_pct, 2)

    # Trailing stop: trails at ~80% of the fixed stop distance
    trailing_stop_pct = round(stop_pct * 0.8, 3)
    trailing_stop_price = round(S * (1 - trailing_stop_pct), 2)

    # Limit order: scale out at 50% of target, let rest ride to full target
    scale_out_price = round(S * (1 + target_pct * 0.5), 2)

    hold_days = 21
    expiry = date.today() + timedelta(days=hold_days)

    return {
        "strategy_type":    "stock_buy",
        "strategy_label":   f"Stock Buy ({label})",
        "strategy_emoji":   "🏦",
        "rationale": (
            f"Buy shares at ${S:.2f}. Target ${target} (+{target_pct*100:.0f}%). "
            f"Stop ${stop} (-{stop_pct*100:.0f}%). "
            f"Trailing stop {trailing_stop_pct*100:.1f}% from high. "
            f"Scale out 50% at ${scale_out_price}. "
            f"Reward/risk {rr:.1f}×. Hold up to {hold_days} days."
        ),
        "entry_price":         round(S, 2),
        "target_price":        target,
        "stop_price":          stop,
        "trailing_stop_pct":   trailing_stop_pct,
        "trailing_stop_price": trailing_stop_price,
        "scale_out_price":     scale_out_price,
        "expiration_iso":      expiry.isoformat(),
        "expiration":          expiry.strftime("%b %d '%y"),
        "dte":                 hold_days,
        "legs": [{
            "action": "BUY", "option_type": "SHARES", "strike": round(S, 2),
            "premium": round(S, 2), "delta": 1.0, "theta": 0.0, "vega": 0.0,
        }],
        "net_debit":        round(S, 2),
        "net_credit":       None,
        "max_profit":       max_profit,
        "max_loss":         max_loss,
        "breakeven_upper":  None,
        "breakeven_lower":  round(S, 2),
        "reward_risk":      rr,
        "prob_profit":      None,
        "payoff_curve":     [
            {"price": round(stop,   2), "pnl": round(-max_loss,   4)},
            {"price": round(S,      2), "pnl": 0.0},
            {"price": round(scale_out_price, 2), "pnl": round(max_profit * 0.5, 4)},
            {"price": round(target, 2), "pnl": round(max_profit,  4)},
        ],
        "price_source":     "market",
    }


def build_stock_short(
    S: float,
    target_pct: float = 0.07,
    stop_pct: float   = 0.04,
    label: str        = "Moderate",
    alt_ticker: str   = "",
) -> dict:
    """
    Short-sell directional signal with defined target and stop.
    Target is BELOW entry; stop is ABOVE entry.
    alt_ticker: suggested inverse ETF (e.g. SQQQ, QID, SH) when applicable.
    """
    target     = round(S * (1 - target_pct), 2)
    stop       = round(S * (1 + stop_pct),   2)
    max_profit = round(S - target,  2)
    max_loss   = round(stop - S,    2)
    rr         = round(target_pct / stop_pct, 2)

    alt_note = (
        f" Alternatively, consider inverse ETF {alt_ticker} for leveraged downside exposure."
        if alt_ticker else ""
    )

    return {
        "strategy_type":    "stock_short",
        "strategy_label":   f"Short Sell ({label})",
        "strategy_emoji":   "🔻",
        "rationale": (
            f"Short at ${S:.2f}. Target ${target} (-{target_pct*100:.0f}%). "
            f"Cover stop ${stop} (+{stop_pct*100:.0f}%). "
            f"Reward/risk {rr:.1f}×.{alt_note}"
        ),
        "entry_price":      round(S, 2),
        "target_price":     target,
        "stop_price":       stop,
        "alt_ticker":       alt_ticker,
        "legs": [{
            "action": "SELL", "option_type": "SHARES", "strike": round(S, 2),
            "premium": round(S, 2), "delta": -1.0, "theta": 0.0, "vega": 0.0,
        }],
        "net_debit":        None,
        "net_credit":       round(S, 2),
        "max_profit":       max_profit,
        "max_loss":         max_loss,
        "breakeven_upper":  None,
        "breakeven_lower":  None,
        "reward_risk":      rr,
        "prob_profit":      None,
        "payoff_curve":     [
            {"price": round(target, 2), "pnl": round(max_profit,  4)},
            {"price": round(S,      2), "pnl": 0.0},
            {"price": round(stop,   2), "pnl": round(-max_loss,   4)},
        ],
        "price_source":     "market",
    }


def build_call_calendar(
    S: float,
    sigma: float,
    near_expiry: date,
    far_expiry: Optional[date] = None,
) -> dict:
    """
    Call Calendar Spread: sell near-term ATM call, buy far-term ATM call (same strike).

    Profit from time-decay differential — the near-term option decays faster.
    Best pre-ER when front-month IV is elevated: sell expensive near IV, buy cheaper far IV.
    Max profit ≈ when stock pins ATM at near expiry.
    Max loss = net debit (if stock moves far from ATM before near expiry).

    near_expiry : short leg (sell) — typically the Friday before ER
    far_expiry  : long  leg (buy)  — defaults to near_expiry + 28 days
    """
    today = date.today()

    # Default far expiry: next standard Friday ~28 days after near
    if far_expiry is None:
        cand = near_expiry + timedelta(days=21)
        while cand.weekday() != 4:
            cand += timedelta(days=1)
        far_expiry = cand

    T_near = max((near_expiry - today).days / 365.0, 1 / 365)
    T_far  = max((far_expiry  - today).days / 365.0, 1 / 365)
    if T_far <= T_near:
        T_far = T_near + 21 / 365

    dte_near = (near_expiry - today).days
    dte_far  = (far_expiry  - today).days

    K = _atm_strike(S)

    # Pre-ER: near IV slightly higher (IV term-structure kink at event date)
    sigma_near = min(sigma * 1.15, 1.20)   # near leg: elevated IV
    sigma_far  = sigma                      # far leg: normal IV

    call_sell = _bs_call(S, K, T_near, _RISK_FREE, sigma_near)
    call_buy  = _bs_call(S, K, T_far,  _RISK_FREE, sigma_far)

    net_debit = round(call_buy["price"] - call_sell["price"], 4)
    if net_debit <= 0:
        # Degenerate: sell is more expensive than buy (shouldn't happen normally)
        net_debit = round(abs(net_debit), 4)

    # Approximate max profit: far leg value at near expiry if stock stays at K
    # Use vega of far leg × IV differential as proxy
    approx_max_profit = round(call_buy["vega"] * abs(sigma_near - sigma_far) * 100 + call_buy["theta"] * (-dte_near), 4)
    approx_max_profit = max(approx_max_profit, net_debit * 0.5)   # at least 0.5× debit as rough estimate

    reward_risk = round(approx_max_profit / net_debit, 2) if net_debit > 0 else 0

    # Approximate breakeven range: within ±net_debit of strike at near expiry
    be_upper = round(K + net_debit * 1.5, 2)
    be_lower = round(K - net_debit * 1.5, 2)

    # Simple payoff approximation (near expiry slice — far leg retains residual value)
    lo, hi = S * 0.80, S * 1.20
    step = (hi - lo) / 40
    payoff, p = [], lo
    while p <= hi:
        # At near expiry: short call intrinsic loss + far leg residual
        short_pnl = -max(0.0, p - K) + call_sell["price"]
        # Far leg residual ≈ intrinsic + 50% of remaining time value
        far_residual = max(0.0, p - K) + call_buy["price"] * 0.35
        pnl = short_pnl + far_residual - net_debit
        payoff.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type":    "call_calendar",
        "strategy_label":   "Call Calendar Spread",
        "strategy_emoji":   "📅",
        "rationale": (
            f"Sell ${K}C expiring {near_expiry.strftime('%b %d')} ({dte_near}d), "
            f"buy ${K}C expiring {far_expiry.strftime('%b %d')} ({dte_far}d). "
            f"Net debit ${net_debit:.2f}/share. "
            f"Profit from front-month IV crush post-ER while long term position stays open. "
            f"Max loss = debit if stock moves far from ${K}."
        ),
        "expiration_iso":    near_expiry.isoformat(),
        "expiration":        near_expiry.strftime("%b %d '%y"),
        "expiration_far":    far_expiry.isoformat(),
        "expiration_far_fmt": far_expiry.strftime("%b %d '%y"),
        "dte":               dte_near,
        "dte_far":           dte_far,
        "strike":            K,
        "legs": [
            {
                "action": "SELL", "option_type": "CALL", "strike": K,
                "premium": call_sell["price"], "delta": call_sell["delta"],
                "theta": call_sell["theta"],   "vega": call_sell["vega"],
                "expiry": near_expiry.isoformat(),
            },
            {
                "action": "BUY",  "option_type": "CALL", "strike": K,
                "premium": call_buy["price"],  "delta": call_buy["delta"],
                "theta": call_buy["theta"],    "vega": call_buy["vega"],
                "expiry": far_expiry.isoformat(),
            },
        ],
        "net_debit":         net_debit,
        "net_credit":        None,
        "max_profit":        round(approx_max_profit, 4),
        "max_loss":          net_debit,
        "breakeven_upper":   be_upper,
        "breakeven_lower":   be_lower,
        "reward_risk":       reward_risk,
        "prob_profit":       None,   # complex to compute analytically; shown as N/A
        "payoff_curve":      payoff,
        "price_source":      "bs_theoretical",
    }


def build_long_strangle(S: float, sigma: float, expiry: date, pct_otm: float = 0.04) -> dict:
    """
    Long Strangle: buy OTM call + OTM put at different strikes.

    Cheaper than long straddle — both legs OTM → less premium, larger move needed.
    Best environment: neutral pre-ER, low-to-normal IV, expecting a large move.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K_call = _otm_strike(S, +pct_otm)
    K_put  = _otm_strike(S, -pct_otm)

    call = _bs_call(S, K_call, T, _RISK_FREE, sigma)
    put  = _bs_put(S,  K_put,  T, _RISK_FREE, sigma)

    net_debit    = round(call["price"] + put["price"], 4)
    be_upper     = round(K_call + net_debit, 2)
    be_lower     = round(K_put  - net_debit, 2)
    em_1sigma    = round(S * sigma * math.sqrt(T), 2)
    em_vs_premium = round(em_1sigma / net_debit, 2) if net_debit > 0 else 0

    if sigma > 0 and T > 0:
        _d2u = (math.log(S / be_upper) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        _d2l = (math.log(S / be_lower) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        prob_profit = round((norm.cdf(_d2u) + (1 - norm.cdf(_d2l))) * 100, 1)
    else:
        prob_profit = 0.0

    lo, hi = S * 0.65, S * 1.35
    step = (hi - lo) / 40
    curve, p = [], lo
    while p <= hi:
        intrinsic = max(0.0, p - K_call) + max(0.0, K_put - p)
        curve.append({"price": round(p, 2), "pnl": round(intrinsic - net_debit, 4)})
        p += step

    return {
        "strategy_type":    "long_strangle",
        "strategy_label":   "Long Strangle",
        "strategy_emoji":   "↔️",
        "rationale": (
            f"Neutral vol play: buy ${K_call}C + ${K_put}P ({pct_otm*100:.0f}% OTM each). "
            f"Pay ${net_debit:.2f}/share. Profit above ${be_upper} or below ${be_lower}. "
            f"Expected 1σ move ${em_1sigma} vs cost ${net_debit:.2f} (ratio {em_vs_premium:.1f}×). "
            f"Cheaper than straddle — needs bigger move."
        ),
        "expiration_iso":    expiry.isoformat(),
        "expiration":        expiry.strftime("%b %d '%y"),
        "dte":               dte,
        "strike":            K_put,
        "strike_call":       K_call,
        "legs": [
            {"action": "BUY", "option_type": "CALL", "strike": K_call,
             "premium": call["price"], "delta": call["delta"],
             "theta": call["theta"], "vega": call["vega"]},
            {"action": "BUY", "option_type": "PUT",  "strike": K_put,
             "premium": put["price"],  "delta": put["delta"],
             "theta": put["theta"],  "vega": put["vega"]},
        ],
        "net_debit":         net_debit,
        "net_credit":        None,
        "max_profit":        None,
        "max_loss":          net_debit,
        "breakeven_upper":   be_upper,
        "breakeven_lower":   be_lower,
        "expected_move_1s":  em_1sigma,
        "em_vs_premium":     em_vs_premium,
        "prob_profit":       prob_profit,
        "payoff_curve":      curve,
        "price_source":      "bs_theoretical",
    }


def build_short_strangle(S: float, sigma: float, expiry: date, pct_otm: float = 0.05) -> dict:
    """
    Short Strangle: sell OTM call + sell OTM put at different strikes.

    Wider profit zone than short straddle, less premium collected.
    Best environment: high IV, neutral, post-ER range-bound. ~70-80% PoP.
    WARNING: unlimited risk outside breakevens — prefer Iron Condor when possible.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K_call = _otm_strike(S, +pct_otm)
    K_put  = _otm_strike(S, -pct_otm)

    call = _bs_call(S, K_call, T, _RISK_FREE, sigma)
    put  = _bs_put(S,  K_put,  T, _RISK_FREE, sigma)

    net_credit = round(call["price"] + put["price"], 4)
    be_upper   = round(K_call + net_credit, 2)
    be_lower   = round(K_put  - net_credit, 2)

    prob_profit = round(
        max(min((1 - call["prob_itm"] - put["prob_itm"]) * 100, 92.0), 40.0), 1
    )

    lo, hi = S * 0.70, S * 1.30
    step = (hi - lo) / 40
    curve, p = [], lo
    while p <= hi:
        pnl = net_credit - max(0.0, p - K_call) - max(0.0, K_put - p)
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type":    "short_strangle",
        "strategy_label":   "Short Strangle",
        "strategy_emoji":   "📐",
        "rationale": (
            f"High IV neutral: sell ${K_call}C + ${K_put}P ({pct_otm*100:.0f}% OTM each). "
            f"Collect ${net_credit:.2f}/share. Profit if stock stays between ${K_put}–${K_call}. "
            f"Unlimited risk outside ${be_lower}–${be_upper}. "
            f"⚠️ Use Iron Condor for defined-risk version."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K_put,
        "strike_call":      K_call,
        "legs": [
            {"action": "SELL", "option_type": "CALL", "strike": K_call,
             "premium": call["price"], "delta": call["delta"],
             "theta": call["theta"], "vega": call["vega"]},
            {"action": "SELL", "option_type": "PUT",  "strike": K_put,
             "premium": put["price"],  "delta": put["delta"],
             "theta": put["theta"],  "vega": put["vega"]},
        ],
        "net_debit":        None,
        "net_credit":       net_credit,
        "max_profit":       net_credit,
        "max_loss":         None,
        "breakeven_upper":  be_upper,
        "breakeven_lower":  be_lower,
        "prob_profit":      prob_profit,
        "payoff_curve":     curve,
        "price_source":     "bs_theoretical",
    }


def build_iron_butterfly(S: float, sigma: float, expiry: date,
                          wing_pct: float = 0.07) -> dict:
    """
    Iron Butterfly: sell ATM call + sell ATM put (body), buy OTM call + OTM put (wings).

    Like Iron Condor but short strikes are ATM (same strike).
    More premium collected, narrower profit zone, lower PoP (~50-65% vs 70-85% for IC).
    Best for: post-ER IV crush, high IV, expecting tight range around current price.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K_body   = _atm_strike(S)
    K_put_w  = _otm_strike(S, -wing_pct)
    K_call_w = _otm_strike(S, +wing_pct)

    tick = _tick(S)
    if K_put_w  >= K_body:  K_put_w  = round(K_body - tick, 2)
    if K_call_w <= K_body:  K_call_w = round(K_body + tick, 2)

    call_sell = _bs_call(S, K_body,   T, _RISK_FREE, sigma)
    put_sell  = _bs_put(S,  K_body,   T, _RISK_FREE, sigma)
    call_buy  = _bs_call(S, K_call_w, T, _RISK_FREE, sigma)
    put_buy   = _bs_put(S,  K_put_w,  T, _RISK_FREE, sigma)

    net_credit  = round(call_sell["price"] + put_sell["price"]
                       - call_buy["price"] - put_buy["price"], 4)
    call_wing   = round(K_call_w - K_body, 2)
    put_wing    = round(K_body - K_put_w,  2)
    max_loss    = round(max(call_wing, put_wing) - net_credit, 4)
    be_upper    = round(K_body + net_credit, 2)
    be_lower    = round(K_body - net_credit, 2)
    reward_risk = round(net_credit / max_loss, 2) if max_loss > 0 else 0

    if sigma > 0 and T > 0:
        _d2u = (math.log(S / be_upper) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        _d2l = (math.log(S / be_lower) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        prob_profit = round(max(min((norm.cdf(_d2u) - (1 - norm.cdf(_d2l))) * 100, 90.0), 20.0), 1)
    else:
        prob_profit = 0.0

    lo, hi = S * 0.80, S * 1.20
    step = (hi - lo) / 50
    curve, p = [], lo
    while p <= hi:
        if K_put_w <= p <= K_call_w:
            pnl = net_credit - abs(p - K_body)
        elif p < K_put_w:
            pnl = net_credit - put_wing
        else:
            pnl = net_credit - call_wing
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type":     "iron_butterfly",
        "strategy_label":    "Iron Butterfly",
        "strategy_emoji":    "🦋",
        "rationale": (
            f"IV crush: sell ${K_body}C + ${K_body}P (ATM), buy ${K_put_w}P + ${K_call_w}C wings. "
            f"Collect ${net_credit:.2f}/share. Max profit if stock pins at ${K_body}. "
            f"Max loss ${max_loss:.2f} — more premium than Iron Condor, narrower zone."
        ),
        "expiration_iso":    expiry.isoformat(),
        "expiration":        expiry.strftime("%b %d '%y"),
        "dte":               dte,
        "strike":            K_body,
        "strike_put_wing":   K_put_w,
        "strike_call_wing":  K_call_w,
        "profit_zone":       {"lower": be_lower, "upper": be_upper},
        "legs": [
            {"action": "SELL", "option_type": "CALL", "strike": K_body,
             "premium": call_sell["price"], "delta": call_sell["delta"],
             "theta": call_sell["theta"],   "vega": call_sell["vega"]},
            {"action": "SELL", "option_type": "PUT",  "strike": K_body,
             "premium": put_sell["price"],  "delta": put_sell["delta"],
             "theta": put_sell["theta"],    "vega": put_sell["vega"]},
            {"action": "BUY",  "option_type": "CALL", "strike": K_call_w,
             "premium": call_buy["price"],  "delta": call_buy["delta"],
             "theta": call_buy["theta"],    "vega": call_buy["vega"]},
            {"action": "BUY",  "option_type": "PUT",  "strike": K_put_w,
             "premium": put_buy["price"],   "delta": put_buy["delta"],
             "theta": put_buy["theta"],     "vega": put_buy["vega"]},
        ],
        "net_debit":         None,
        "net_credit":        net_credit,
        "max_profit":        net_credit,
        "max_loss":          max_loss,
        "breakeven_upper":   be_upper,
        "breakeven_lower":   be_lower,
        "reward_risk":       reward_risk,
        "prob_profit":       prob_profit,
        "call_wing":         call_wing,
        "put_wing":          put_wing,
        "payoff_curve":      curve,
        "price_source":      "bs_theoretical",
    }


def build_protective_put(S: float, sigma: float, expiry: date,
                          pct_otm: float = 0.07) -> dict:
    """
    Protective Put (Married Put): own shares + buy OTM put as insurance.

    Caps downside below the put strike while retaining unlimited upside.
    Best environment: bullish long-term but concerned about short-term downside
    (pre-ER event, macro uncertainty). Cost = insurance premium.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K       = _otm_strike(S, -pct_otm)
    put     = _bs_put(S, K, T, _RISK_FREE, sigma)
    premium = put["price"]

    breakeven        = round(S + premium, 2)
    max_loss         = round((S - K) + premium, 4)
    prot_pct         = round((1 - K / S) * 100, 1)
    annualized_cost  = round(premium / S * (365 / dte) * 100, 1) if dte > 0 else 0

    if sigma > 0 and T > 0:
        _d2_be = (math.log(S / breakeven) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        prob_profit = round(norm.cdf(_d2_be) * 100, 1)
    else:
        prob_profit = 0.0

    lo, hi = S * 0.65, S * 1.35
    step = (hi - lo) / 40
    curve, p = [], lo
    while p <= hi:
        stock_pnl = p - S
        put_pnl   = max(0.0, K - p) - premium
        curve.append({"price": round(p, 2), "pnl": round(stock_pnl + put_pnl, 4)})
        p += step

    return {
        "strategy_type":     "protective_put",
        "strategy_label":    "Protective Put",
        "strategy_emoji":    "🛡️",
        "rationale": (
            f"Hedged bullish: buy ${K}P ({pct_otm*100:.0f}% OTM) on shares at ${S:.2f}. "
            f"Pay ${premium:.2f}/share ({annualized_cost:.0f}% ann. cost). "
            f"Downside capped at -${max_loss:.2f}/sh (protected below ${K}, "
            f"{prot_pct:.0f}% OTM floor). Breakeven ${breakeven:.2f}."
        ),
        "expiration_iso":    expiry.isoformat(),
        "expiration":        expiry.strftime("%b %d '%y"),
        "dte":               dte,
        "strike":            K,
        "protection_pct":    prot_pct,
        "annualized_cost":   annualized_cost,
        "legs": [
            {"action": "OWN",  "option_type": "SHARES", "strike": round(S, 2),
             "premium": round(S, 2), "delta": 1.0, "theta": 0.0, "vega": 0.0},
            {"action": "BUY",  "option_type": "PUT", "strike": K,
             "premium": premium, "delta": put["delta"],
             "theta": put["theta"], "vega": put["vega"]},
        ],
        "net_debit":         round(premium, 4),
        "net_credit":        None,
        "max_profit":        None,
        "max_loss":          max_loss,
        "breakeven_upper":   breakeven,
        "breakeven_lower":   None,
        "reward_risk":       None,
        "prob_profit":       prob_profit,
        "payoff_curve":      curve,
        "price_source":      "bs_theoretical",
    }


# ── Oversold criteria standard ────────────────────────────────────────────────
#
# A ticker is "oversold" when multiple technical indicators converge on
# capitulation. The married put is the ideal entry: buy the dip for upside
# but cap downside with a put in case the sell-off continues.
#
# OVERSOLD CRITERIA (must meet ≥3 of 5):
#   1. RSI-14 < 30 (deeply oversold on momentum)
#   2. Price > 20% below 200-day SMA (extended decline)
#   3. 5-day return < -8% (acute selling pressure)
#   4. 30-day return < -20% (sustained crash, e.g., TTD -50%)
#   5. HV > 40% (panic volatility, put premiums are rich but worth it)
#
# Examples of tickers that trigger this: TTD (2026 crash), PYPL (2022),
# META (2022), NFLX (2022 subscriber miss).

def is_oversold(rsi: float, price: float, sma200: float, ret_5d: float,
                ret_30d: float, hv: float) -> bool:
    """Check if a ticker meets the oversold standard (≥3 of 5 criteria)."""
    score = 0
    if rsi < 30:             score += 1  # deeply oversold momentum
    if sma200 > 0 and price < sma200 * 0.80:  score += 1  # 20%+ below 200 SMA
    if ret_5d < -0.08:       score += 1  # acute 5-day selloff
    if ret_30d < -0.20:      score += 1  # sustained crash
    if hv > 0.40:            score += 1  # panic volatility
    return score >= 3


def build_married_put(S: float, sigma: float, expiry: date,
                      pct_otm: float = 0.03) -> dict:
    """
    Married Put: BUY stock + BUY put simultaneously as a new position.

    Unlike a protective put (added to an existing position), a married put
    is entered as a single trade idea for oversold bounce plays:
    - Buy shares at the depressed price (upside if recovery)
    - Buy a near-ATM put (cap downside if sell-off continues)
    - Net cost = stock price + put premium
    - Max loss = stock price - put strike + put premium (DEFINED)
    - Max profit = unlimited (stock recovery)

    Best for: deeply oversold stocks (RSI < 30, 20%+ drawdown from highs)
    where you believe the long-term thesis is intact but the near-term
    is uncertain. Example: TTD after a 50% crash.

    pct_otm: how far OTM the put is. Default 3% (near-ATM for max protection).
    For oversold plays, tighter protection is better than cheaper premium.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    # Near-ATM put (3% OTM) for maximum protection on oversold bounces
    K       = _otm_strike(S, -pct_otm)
    put     = _bs_put(S, K, T, _RISK_FREE, sigma)
    premium = put["price"]

    total_cost = round(S + premium, 2)
    breakeven  = total_cost
    max_loss   = round((S - K) + premium, 4)
    prot_pct   = round((1 - K / S) * 100, 1)

    # PoP: probability stock ends above breakeven
    if sigma > 0 and T > 0:
        _d2_be = (math.log(S / breakeven) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        prob_profit = round(norm.cdf(_d2_be) * 100, 1)
    else:
        prob_profit = 0.0

    # Payoff curve
    lo, hi = S * 0.60, S * 1.50  # wider range for oversold bounce
    step = (hi - lo) / 40
    curve, p = [], lo
    while p <= hi:
        stock_pnl = p - S
        put_pnl   = max(0.0, K - p) - premium
        curve.append({"price": round(p, 2), "pnl": round(stock_pnl + put_pnl, 4)})
        p += step

    return {
        "strategy_type":     "married_put",
        "strategy_label":    "Married Put (Buy Dip + Protect)",
        "strategy_emoji":    "💍",
        "rationale": (
            f"OVERSOLD BOUNCE: buy {int(S)}-level shares + buy ${K}P ({pct_otm*100:.0f}% OTM). "
            f"Put costs ${premium:.2f}/sh — insurance against further drop. "
            f"Max loss capped at ${max_loss:.2f}/sh (floor at ${K}). "
            f"If stock recovers, unlimited upside minus ${premium:.2f} insurance cost. "
            f"Breakeven ${breakeven:.2f}."
        ),
        "expiration_iso":    expiry.isoformat(),
        "expiration":        expiry.strftime("%b %d '%y"),
        "dte":               dte,
        "strike":            K,
        "protection_pct":    prot_pct,
        "legs": [
            {"action": "BUY",  "option_type": "SHARES", "strike": round(S, 2),
             "premium": round(S, 2), "delta": 1.0, "theta": 0.0, "vega": 0.0},
            {"action": "BUY",  "option_type": "PUT", "strike": K,
             "premium": premium, "delta": put["delta"],
             "theta": put["theta"], "vega": put["vega"]},
        ],
        "net_debit":         round(S + premium, 4),
        "net_credit":        None,
        "max_profit":        None,  # unlimited
        "max_loss":          max_loss,
        "breakeven_upper":   breakeven,
        "breakeven_lower":   None,
        "reward_risk":       None,
        "prob_profit":       prob_profit,
        "payoff_curve":      curve,
        "price_source":      "bs_theoretical",
    }


def build_synthetic_long(S: float, sigma: float, expiry: date) -> dict:
    """
    Synthetic Long Stock: BUY ATM call + SELL ATM put.

    Replicates owning 100 shares for a fraction of the capital.
    Net debit is typically small (near zero for ATM). Delta ≈ 1.0.

    Used by institutions as a capital-efficient stock substitute.
    Risk: same as owning stock (unlimited downside to zero from short put).
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K    = _atm_strike(S)
    call = _bs_call(S, K, T, _RISK_FREE, sigma)
    put  = _bs_put(S, K, T, _RISK_FREE, sigma)

    net_debit = round(call["price"] - put["price"], 4)
    breakeven = round(K + net_debit, 2)

    lo, hi = S * 0.70, S * 1.30
    step = (hi - lo) / 40
    curve, p = [], lo
    while p <= hi:
        pnl = (p - K) - net_debit  # call payoff - put obligation - net cost
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type":     "synthetic_long",
        "strategy_label":    "Synthetic Long Stock",
        "strategy_emoji":    "🔄",
        "rationale": (
            f"Capital-efficient stock substitute: buy ${K}C + sell ${K}P. "
            f"Net debit ${net_debit:.2f}/sh. Same upside as owning shares "
            f"for ~{abs(net_debit)/S*100:.0f}% of the capital. "
            f"Breakeven ${breakeven}. Delta ≈ 1.0."
        ),
        "expiration_iso":    expiry.isoformat(),
        "expiration":        expiry.strftime("%b %d '%y"),
        "dte":               dte,
        "strike":            K,
        "legs": [
            {"action": "BUY",  "option_type": "CALL", "strike": K,
             "premium": call["price"], "delta": call["delta"],
             "theta": call["theta"], "vega": call["vega"]},
            {"action": "SELL", "option_type": "PUT",  "strike": K,
             "premium": put["price"],  "delta": put["delta"],
             "theta": put["theta"],  "vega": put["vega"]},
        ],
        "net_debit":         max(0, net_debit),
        "net_credit":        max(0, -net_debit) if net_debit < 0 else None,
        "max_profit":        None,    # unlimited upside
        "max_loss":          round(K + net_debit, 2),  # stock goes to 0
        "breakeven_upper":   breakeven,
        "breakeven_lower":   None,
        "reward_risk":       None,
        "prob_profit":       round(call["prob_itm"] * 100, 1),
        "payoff_curve":      curve,
        "price_source":      "bs_theoretical",
    }


def build_synthetic_short(S: float, sigma: float, expiry: date) -> dict:
    """
    Synthetic Short Stock: SELL ATM call + BUY ATM put.

    Replicates short-selling 100 shares without borrowing.
    Net credit is typically small. Delta ≈ -1.0.

    Institutional hedge or bearish conviction play.
    Risk: unlimited upside risk (same as short stock).
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K    = _atm_strike(S)
    call = _bs_call(S, K, T, _RISK_FREE, sigma)
    put  = _bs_put(S, K, T, _RISK_FREE, sigma)

    net_credit = round(put["price"] - call["price"], 4)
    # If call > put (typical), it's a net debit
    net_cost = call["price"] - put["price"]
    breakeven = round(K - net_cost, 2) if net_cost > 0 else round(K + abs(net_credit), 2)

    lo, hi = S * 0.70, S * 1.30
    step = (hi - lo) / 40
    curve, p = [], lo
    while p <= hi:
        pnl = (K - p) + net_cost  # put payoff - call obligation
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type":     "synthetic_short",
        "strategy_label":    "Synthetic Short Stock",
        "strategy_emoji":    "🔃",
        "rationale": (
            f"Bearish stock substitute: sell ${K}C + buy ${K}P. "
            f"No shares to borrow. Profits when stock falls below ${breakeven}. "
            f"Delta ≈ -1.0. Same risk profile as short selling."
        ),
        "expiration_iso":    expiry.isoformat(),
        "expiration":        expiry.strftime("%b %d '%y"),
        "dte":               dte,
        "strike":            K,
        "legs": [
            {"action": "SELL", "option_type": "CALL", "strike": K,
             "premium": call["price"], "delta": call["delta"],
             "theta": call["theta"], "vega": call["vega"]},
            {"action": "BUY",  "option_type": "PUT",  "strike": K,
             "premium": put["price"],  "delta": put["delta"],
             "theta": put["theta"],  "vega": put["vega"]},
        ],
        "net_debit":         max(0, net_cost),
        "net_credit":        max(0, -net_cost) if net_cost < 0 else None,
        "max_profit":        round(K, 2),  # stock goes to 0
        "max_loss":          None,         # unlimited
        "breakeven_upper":   None,
        "breakeven_lower":   breakeven,
        "reward_risk":       None,
        "prob_profit":       round(put["prob_itm"] * 100, 1),
        "payoff_curve":      curve,
        "price_source":      "bs_theoretical",
    }


def build_collar(S: float, sigma: float, expiry: date,
                 put_otm: float = 0.05, call_otm: float = 0.07) -> dict:
    """
    Collar: own shares + buy OTM put (floor) + sell OTM call (cap).

    Net premium often near zero — a "zero-cost" hedge that sacrifices upside.
    Max gain = (K_call − S) + net_premium (capped).
    Max loss = (S − K_put) − net_premium (floored).
    Best for: protect long stock position, lock in gains, reduce protective put cost.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K_put  = _otm_strike(S, -put_otm)
    K_call = _otm_strike(S, +call_otm)

    tick = _tick(S)
    if K_put  >= S:     K_put  = round(S - tick, 2)
    if K_call <= S:     K_call = round(S + tick, 2)

    put_buy   = _bs_put(S,  K_put,  T, _RISK_FREE, sigma)
    call_sell = _bs_call(S, K_call, T, _RISK_FREE, sigma)

    net_premium  = round(call_sell["price"] - put_buy["price"], 4)
    max_gain     = round((K_call - S) + net_premium, 4)
    max_loss_net = round((S - K_put) - net_premium, 4)
    breakeven    = round(S - net_premium, 2)
    cost_desc    = f"${abs(net_premium):.2f}/sh {'credit' if net_premium > 0 else 'debit'}"

    if sigma > 0 and T > 0:
        _d2_be = (math.log(S / breakeven) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        prob_profit = round(norm.cdf(_d2_be) * 100, 1)
    else:
        prob_profit = 50.0

    lo, hi = S * 0.82, K_call * 1.10
    step = (hi - lo) / 40
    curve, p_pt = [], lo
    while p_pt <= hi:
        s_pnl    = p_pt - S
        put_pnl  = max(0.0, K_put - p_pt) - put_buy["price"]
        call_pnl = call_sell["price"] - max(0.0, p_pt - K_call)
        pnl      = min(s_pnl + put_pnl + call_pnl, max_gain)
        pnl      = max(pnl, -max_loss_net)
        curve.append({"price": round(p_pt, 2), "pnl": round(pnl, 4)})
        p_pt += step

    return {
        "strategy_type":    "collar",
        "strategy_label":   "Collar",
        "strategy_emoji":   "🔒",
        "rationale": (
            f"Position hedge: buy ${K_put}P floor, sell ${K_call}C cap ({cost_desc}). "
            f"Max gain ${max_gain:.2f}/sh at ${K_call}. "
            f"Max loss ${max_loss_net:.2f}/sh (protected below ${K_put}). "
            f"Locks in ${K_put}–${K_call} range."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K_put,
        "strike_call":      K_call,
        "legs": [
            {"action": "OWN",  "option_type": "SHARES", "strike": round(S, 2),
             "premium": round(S, 2), "delta": 1.0, "theta": 0.0, "vega": 0.0},
            {"action": "BUY",  "option_type": "PUT",  "strike": K_put,
             "premium": put_buy["price"],   "delta": put_buy["delta"],
             "theta": put_buy["theta"],     "vega": put_buy["vega"]},
            {"action": "SELL", "option_type": "CALL", "strike": K_call,
             "premium": call_sell["price"], "delta": call_sell["delta"],
             "theta": call_sell["theta"],   "vega": call_sell["vega"]},
        ],
        "net_debit":        None if net_premium > 0 else abs(net_premium),
        "net_credit":       net_premium if net_premium > 0 else None,
        "max_profit":       max_gain,
        "max_loss":         max_loss_net,
        "breakeven_upper":  breakeven,
        "breakeven_lower":  None,
        "reward_risk":      round(max_gain / max_loss_net, 2) if max_loss_net > 0 else 0,
        "prob_profit":      prob_profit,
        "payoff_curve":     curve,
        "price_source":     "bs_theoretical",
    }


def build_call_butterfly(S: float, sigma: float, expiry: date,
                          pct_spread: float = 0.05) -> dict:
    """
    Long Call Butterfly: buy K1 call, sell 2× K2 call, buy K3 call.
    K2 = ATM (body), K1 = K2−spread, K3 = K2+spread (equidistant wings).

    Max profit at K2 (stock pins at middle strike). Defined risk = net debit.
    Better than condor when you have a specific price target.
    PoP: 30–50%. Cost: low.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K2   = _atm_strike(S)
    K1   = _otm_strike(S, -pct_spread)
    K3   = _otm_strike(S, +pct_spread)
    tick = _tick(S)
    if K1 >= K2: K1 = round(K2 - tick, 2)
    if K3 <= K2: K3 = round(K2 + tick, 2)
    wing = min(round(K2 - K1, 2), round(K3 - K2, 2))

    c1 = _bs_call(S, K1, T, _RISK_FREE, sigma)
    c2 = _bs_call(S, K2, T, _RISK_FREE, sigma)
    c3 = _bs_call(S, K3, T, _RISK_FREE, sigma)

    net_debit   = round(c1["price"] - 2 * c2["price"] + c3["price"], 4)
    if net_debit < 0: net_debit = abs(net_debit)
    max_profit  = round(wing - net_debit, 4)
    be_lower    = round(K1 + net_debit, 2)
    be_upper    = round(K3 - net_debit, 2)
    reward_risk = round(max_profit / net_debit, 2) if net_debit > 0 else 0

    if sigma > 0 and T > 0:
        _d2l = (math.log(S / be_lower) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        _d2u = (math.log(S / be_upper) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        prob_profit = round(max(min((norm.cdf(_d2u) - norm.cdf(_d2l)) * 100, 92.0), 5.0), 1)
    else:
        prob_profit = 0.0

    lo, hi = K1 * 0.94, K3 * 1.06
    step = (hi - lo) / 50
    curve, p = [], lo
    while p <= hi:
        if p <= K1:
            pnl = -net_debit
        elif p <= K2:
            pnl = (p - K1) - net_debit
        elif p <= K3:
            pnl = (K3 - p) - net_debit
        else:
            pnl = -net_debit
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type":    "call_butterfly",
        "strategy_label":   "Call Butterfly",
        "strategy_emoji":   "🦋",
        "rationale": (
            f"Neutral pinpoint: buy ${K1}C, sell 2× ${K2}C, buy ${K3}C. "
            f"Pay ${net_debit:.2f}/share. Max profit ${max_profit:.2f} if stock pins at ${K2}. "
            f"Reward/risk {reward_risk:.1f}×. Break-evens ${be_lower}–${be_upper}."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K1,
        "strike_body":      K2,
        "strike_upper":     K3,
        "profit_zone":      {"lower": be_lower, "upper": be_upper},
        "legs": [
            {"action": "BUY",  "option_type": "CALL", "strike": K1,
             "premium": c1["price"], "delta": c1["delta"],
             "theta": c1["theta"], "vega": c1["vega"]},
            {"action": "SELL", "option_type": "CALL", "strike": K2,
             "premium": c2["price"], "delta": c2["delta"],
             "theta": c2["theta"], "vega": c2["vega"], "qty": 2},
            {"action": "BUY",  "option_type": "CALL", "strike": K3,
             "premium": c3["price"], "delta": c3["delta"],
             "theta": c3["theta"], "vega": c3["vega"]},
        ],
        "net_debit":        net_debit,
        "net_credit":       None,
        "max_profit":       max_profit,
        "max_loss":         net_debit,
        "breakeven_lower":  be_lower,
        "breakeven_upper":  be_upper,
        "reward_risk":      reward_risk,
        "prob_profit":      prob_profit,
        "wing_width":       wing,
        "payoff_curve":     curve,
        "price_source":     "bs_theoretical",
    }


def build_put_butterfly(S: float, sigma: float, expiry: date,
                         pct_spread: float = 0.05) -> dict:
    """
    Long Put Butterfly: buy K3 put, sell 2× K2 put, buy K1 put.
    K2 = ATM (body), K3 = K2+spread (higher), K1 = K2−spread (lower).

    Same P&L profile as call butterfly, using puts instead.
    Slightly bearish framing — max profit when stock stays near or drifts to K2.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K2   = _atm_strike(S)
    K3   = _otm_strike(S, +pct_spread)
    K1   = _otm_strike(S, -pct_spread)
    tick = _tick(S)
    if K1 >= K2: K1 = round(K2 - tick, 2)
    if K3 <= K2: K3 = round(K2 + tick, 2)
    wing = min(round(K2 - K1, 2), round(K3 - K2, 2))

    p1 = _bs_put(S, K1, T, _RISK_FREE, sigma)
    p2 = _bs_put(S, K2, T, _RISK_FREE, sigma)
    p3 = _bs_put(S, K3, T, _RISK_FREE, sigma)

    net_debit   = round(p3["price"] - 2 * p2["price"] + p1["price"], 4)
    if net_debit < 0: net_debit = abs(net_debit)
    max_profit  = round(wing - net_debit, 4)
    be_upper    = round(K3 - net_debit, 2)
    be_lower    = round(K1 + net_debit, 2)
    reward_risk = round(max_profit / net_debit, 2) if net_debit > 0 else 0

    if sigma > 0 and T > 0:
        _d2l = (math.log(S / be_lower) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        _d2u = (math.log(S / be_upper) + (_RISK_FREE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        prob_profit = round(max(min((norm.cdf(_d2u) - norm.cdf(_d2l)) * 100, 92.0), 5.0), 1)
    else:
        prob_profit = 0.0

    lo, hi = K1 * 0.94, K3 * 1.06
    step = (hi - lo) / 50
    curve, p = [], lo
    while p <= hi:
        if p >= K3:
            pnl = -net_debit
        elif p >= K2:
            pnl = (K3 - p) - net_debit
        elif p >= K1:
            pnl = (p - K1) - net_debit
        else:
            pnl = -net_debit
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type":    "put_butterfly",
        "strategy_label":   "Put Butterfly",
        "strategy_emoji":   "🦋",
        "rationale": (
            f"Neutral pinpoint (puts): buy ${K3}P, sell 2× ${K2}P, buy ${K1}P. "
            f"Pay ${net_debit:.2f}/share. Max profit ${max_profit:.2f} if stock pins at ${K2}. "
            f"Reward/risk {reward_risk:.1f}×."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K1,
        "strike_body":      K2,
        "strike_upper":     K3,
        "profit_zone":      {"lower": be_lower, "upper": be_upper},
        "legs": [
            {"action": "BUY",  "option_type": "PUT", "strike": K3,
             "premium": p3["price"], "delta": p3["delta"],
             "theta": p3["theta"], "vega": p3["vega"]},
            {"action": "SELL", "option_type": "PUT", "strike": K2,
             "premium": p2["price"], "delta": p2["delta"],
             "theta": p2["theta"], "vega": p2["vega"], "qty": 2},
            {"action": "BUY",  "option_type": "PUT", "strike": K1,
             "premium": p1["price"], "delta": p1["delta"],
             "theta": p1["theta"], "vega": p1["vega"]},
        ],
        "net_debit":        net_debit,
        "net_credit":       None,
        "max_profit":       max_profit,
        "max_loss":         net_debit,
        "breakeven_lower":  be_lower,
        "breakeven_upper":  be_upper,
        "reward_risk":      reward_risk,
        "prob_profit":      prob_profit,
        "wing_width":       wing,
        "payoff_curve":     curve,
        "price_source":     "bs_theoretical",
    }


def build_put_backspread(S: float, sigma: float, expiry: date,
                          pct_short: float = 0.03, pct_long: float = 0.08) -> dict:
    """
    Put Backspread (Reverse Put Ratio Spread): sell 1 near-ATM put, buy 2 OTM puts.

    Aggressive bearish acceleration play — profits from a large downside crash.
    Often entered for a small credit or near-zero cost.
    Max loss at K_long (both long puts expire worthless, short put fully ITM).
    Profit accelerates below the lower breakeven — unlimited to downside.
    Best for: PGI < −60, strongly bearish catalyst, expecting a crash not just a drift.
    """
    today = date.today()
    T     = max((expiry - today).days / 365.0, 1 / 365)
    dte   = (expiry - today).days

    K_short = _otm_strike(S, -pct_short)
    K_long  = _otm_strike(S, -pct_long)
    tick    = _tick(S)
    if K_long >= K_short:
        K_long = round(K_short - tick * 2, 2)

    p_short = _bs_put(S, K_short, T, _RISK_FREE, sigma)
    p_long  = _bs_put(S, K_long,  T, _RISK_FREE, sigma)

    net_premium = round(p_short["price"] - 2 * p_long["price"], 4)
    # Max loss at K_long: both long puts worthless, short put = K_short − K_long deep ITM
    max_loss    = round((K_short - K_long) - net_premium, 4) if net_premium >= 0 \
                  else round((K_short - K_long) + abs(net_premium), 4)
    # Upper BE: above K_short (keep net credit, or add debit to entry)
    be_upper    = round(K_short - net_premium, 2) if net_premium >= 0 \
                  else round(K_short + abs(net_premium), 2)
    # Lower BE: 2×K_long − K_short + net_premium
    be_lower    = round(2 * K_long - K_short + net_premium, 2)
    cost_desc   = f"${abs(net_premium):.2f}/sh {'credit' if net_premium >= 0 else 'debit'}"

    lo, hi = S * 0.55, S * 1.10
    step = (hi - lo) / 60
    curve, p = [], lo
    while p <= hi:
        short_pnl = p_short["price"] - max(0.0, K_short - p)
        long_pnl  = 2 * (max(0.0, K_long - p) - p_long["price"])
        curve.append({"price": round(p, 2), "pnl": round(short_pnl + long_pnl, 4)})
        p += step

    return {
        "strategy_type":    "put_backspread",
        "strategy_label":   "Put Backspread",
        "strategy_emoji":   "💥",
        "rationale": (
            f"Bearish crash play: sell 1× ${K_short}P, buy 2× ${K_long}P ({cost_desc}). "
            f"Max loss ${max_loss:.2f}/sh near ${K_long}. "
            f"Profit accelerates below ${be_lower} — unlimited downside gain. "
            f"Upper breakeven ${be_upper}. Best for strong bearish catalyst."
        ),
        "expiration_iso":   expiry.isoformat(),
        "expiration":       expiry.strftime("%b %d '%y"),
        "dte":              dte,
        "strike":           K_short,
        "strike_long":      K_long,
        "legs": [
            {"action": "SELL", "option_type": "PUT", "strike": K_short,
             "premium": p_short["price"], "delta": p_short["delta"],
             "theta": p_short["theta"],   "vega": p_short["vega"], "qty": 1},
            {"action": "BUY",  "option_type": "PUT", "strike": K_long,
             "premium": p_long["price"],  "delta": p_long["delta"],
             "theta": p_long["theta"],    "vega": p_long["vega"],  "qty": 2},
        ],
        "net_debit":        abs(net_premium) if net_premium < 0 else None,
        "net_credit":       net_premium if net_premium >= 0 else None,
        "max_profit":       None,
        "max_loss":         max_loss,
        "breakeven_upper":  be_upper,
        "breakeven_lower":  be_lower,
        "reward_risk":      None,
        "prob_profit":      None,
        "payoff_curve":     curve,
        "price_source":     "bs_theoretical",
    }


def build_diagonal_call_spread(
    S: float, sigma: float, near_expiry: date, far_expiry: Optional[date] = None,
    pct_otm_sell: float = 0.05,
) -> dict:
    """
    Diagonal Call Spread (Poor Man's Covered Call / PMCC):
    Buy deep ITM long-dated call, sell short-dated OTM call.

    Behaves like a covered call but with ~80% less capital.
    Max profit = (sell_strike - buy_strike) + net credit from sell - net debit
    Max loss = net debit paid for long leg (if stock drops below buy strike)
    """
    today = date.today()
    if far_expiry is None:
        far_expiry = near_expiry + timedelta(days=30)
    T_near = max((near_expiry - today).days / 365.0, 1 / 365)
    T_far  = max((far_expiry  - today).days / 365.0, 1 / 365)

    K_sell = _otm_strike(S, pct_otm_sell)           # OTM short call
    K_buy  = _otm_strike(S, -0.10)                   # ~10% ITM long call (deep ITM)

    call_sell = _bs_call(S, K_sell, T_near, _RISK_FREE, sigma)
    call_buy  = _bs_call(S, K_buy,  T_far,  _RISK_FREE, sigma)

    net_debit = round(call_buy["price"] - call_sell["price"], 4)
    max_profit = round((K_sell - K_buy) - net_debit, 4) if K_sell > K_buy else round(call_sell["price"], 4)
    prob_profit = round((1 - call_sell["prob_itm"]) * 100, 1)

    lo, hi = S * 0.85, K_sell * 1.15
    step = (hi - lo) / 50
    curve, p = [], lo
    while p <= hi:
        # At near expiry: long call still has time value
        long_val = max(0, p - K_buy) + call_buy["vega"] * sigma * 0.5  # rough residual
        short_val = max(0, p - K_sell)
        pnl = long_val - short_val - net_debit
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    return {
        "strategy_type": "diagonal_call_spread",
        "strategy_label": "PMCC (Diagonal Call)",
        "strategy_emoji": "📐",
        "rationale": (
            f"Poor Man's Covered Call: buy ${K_buy}C far ({far_expiry.strftime('%b %d')}), "
            f"sell ${K_sell}C near ({near_expiry.strftime('%b %d')}). "
            f"Net debit ${net_debit:.2f}. Like a covered call with ~80% less capital. "
            f"Max profit ${max_profit:.2f} if stock at ${K_sell} at near expiry."
        ),
        "expiration_iso": near_expiry.isoformat(),
        "expiration": near_expiry.strftime("%b %d '%y"),
        "dte": (near_expiry - today).days,
        "strike": K_sell,
        "strike_long": K_buy,
        "legs": [
            {"action": "BUY",  "option_type": "CALL", "strike": K_buy,
             "premium": call_buy["price"], "delta": call_buy["delta"],
             "theta": call_buy["theta"], "vega": call_buy["vega"]},
            {"action": "SELL", "option_type": "CALL", "strike": K_sell,
             "premium": call_sell["price"], "delta": call_sell["delta"],
             "theta": call_sell["theta"], "vega": call_sell["vega"]},
        ],
        "net_debit": net_debit,
        "net_credit": None,
        "max_profit": max_profit,
        "max_loss": round(net_debit, 4),
        "breakeven_upper": None,
        "breakeven_lower": round(K_buy + net_debit, 2),
        "reward_risk": round(max_profit / net_debit, 2) if net_debit > 0 else 0,
        "prob_profit": prob_profit,
        "payoff_curve": curve,
        "price_source": "bs_theoretical",
    }


def build_put_calendar(
    S: float, sigma: float, near_expiry: date, far_expiry: Optional[date] = None,
) -> dict:
    """
    Put Calendar Spread: sell near-term ATM put, buy far-term ATM put (same strike).
    Mirror of call_calendar but using puts. Profit from near-term theta decay.
    """
    today = date.today()
    if far_expiry is None:
        cand = near_expiry + timedelta(days=21)
        while cand.weekday() != 4:
            cand += timedelta(days=1)
        far_expiry = cand

    T_near = max((near_expiry - today).days / 365.0, 1 / 365)
    T_far  = max((far_expiry  - today).days / 365.0, 1 / 365)
    if T_far <= T_near:
        T_far = T_near + 21 / 365

    K = _atm_strike(S)
    sigma_near = min(sigma * 1.15, 1.20)
    sigma_far  = sigma

    put_sell = _bs_put(S, K, T_near, _RISK_FREE, sigma_near)
    put_buy  = _bs_put(S, K, T_far,  _RISK_FREE, sigma_far)

    net_debit = round(put_buy["price"] - put_sell["price"], 4)
    if net_debit <= 0:
        net_debit = round(abs(net_debit), 4)

    approx_max = max(round(put_buy["vega"] * abs(sigma_near - sigma_far) * 100, 4), net_debit * 0.5)
    rr = round(approx_max / net_debit, 2) if net_debit > 0 else 0

    lo, hi = S * 0.80, S * 1.20
    step = (hi - lo) / 50
    curve, p = [], lo
    while p <= hi:
        near_val = max(0, K - p)
        far_val  = max(0, K - p) + put_buy["vega"] * sigma * 0.3
        pnl = far_val - near_val - net_debit
        curve.append({"price": round(p, 2), "pnl": round(pnl, 4)})
        p += step

    dte_near = (near_expiry - today).days

    return {
        "strategy_type": "put_calendar",
        "strategy_label": "Put Calendar Spread",
        "strategy_emoji": "📅",
        "rationale": (
            f"Sell ${K}P near ({near_expiry.strftime('%b %d')}), buy ${K}P far ({far_expiry.strftime('%b %d')}). "
            f"Net debit ${net_debit:.2f}. Profit from near-term put theta decay. "
            f"Best when stock stays near ${K}."
        ),
        "expiration_iso": near_expiry.isoformat(),
        "expiration": near_expiry.strftime("%b %d '%y"),
        "dte": dte_near,
        "strike": K,
        "legs": [
            {"action": "SELL", "option_type": "PUT", "strike": K,
             "premium": put_sell["price"], "delta": put_sell["delta"],
             "theta": put_sell["theta"], "vega": put_sell["vega"]},
            {"action": "BUY",  "option_type": "PUT", "strike": K,
             "premium": put_buy["price"],  "delta": put_buy["delta"],
             "theta": put_buy["theta"],    "vega": put_buy["vega"]},
        ],
        "net_debit": net_debit,
        "net_credit": None,
        "max_profit": approx_max,
        "max_loss": round(net_debit, 4),
        "breakeven_upper": round(K + net_debit * 1.5, 2),
        "breakeven_lower": round(K - net_debit * 1.5, 2),
        "reward_risk": rr,
        "prob_profit": None,
        "payoff_curve": curve,
        "price_source": "bs_theoretical",
    }


def build_long_put_condor(
    S: float, sigma: float, expiry: date,
    body_width_pct: float = 0.04, wing_width_pct: float = 0.03,
) -> dict:
    """
    Long Put Condor: buy 1 OTM put, sell 1 closer put, sell 1 closer put, buy 1 ATM put.
    Mirror of long_call_condor but using puts. Profits in a narrow range.
    """
    today = date.today()
    T = max((expiry - today).days / 365.0, 1 / 365)
    dte = (expiry - today).days

    K4 = _otm_strike(S, body_width_pct)           # highest strike (buy)
    K3 = _otm_strike(S, body_width_pct / 2)       # upper body (sell)
    K2 = _otm_strike(S, -body_width_pct / 2)      # lower body (sell)
    K1 = _otm_strike(S, -(body_width_pct + wing_width_pct))  # lowest (buy)

    p4 = _bs_put(S, K4, T, _RISK_FREE, sigma)
    p3 = _bs_put(S, K3, T, _RISK_FREE, sigma)
    p2 = _bs_put(S, K2, T, _RISK_FREE, sigma)
    p1 = _bs_put(S, K1, T, _RISK_FREE, sigma)

    net_debit = round((p4["price"] + p1["price"]) - (p3["price"] + p2["price"]), 4)
    if net_debit < 0:
        net_debit = abs(net_debit)
    body_width = K3 - K2 if K3 > K2 else K4 - K1
    max_profit = round(body_width - net_debit, 4) if body_width > net_debit else round(net_debit * 0.5, 4)

    lo, hi = K1 * 0.95, K4 * 1.05
    step = (hi - lo) / 50
    curve, p = [], lo
    while p <= hi:
        pnl_p4 = max(0, K4 - p) - p4["price"]
        pnl_p3 = -(max(0, K3 - p) - p3["price"])
        pnl_p2 = -(max(0, K2 - p) - p2["price"])
        pnl_p1 = max(0, K1 - p) - p1["price"]
        curve.append({"price": round(p, 2), "pnl": round(pnl_p4 + pnl_p3 + pnl_p2 + pnl_p1, 4)})
        p += step

    return {
        "strategy_type": "long_put_condor",
        "strategy_label": "Long Put Condor",
        "strategy_emoji": "🦅",
        "rationale": (
            f"Buy ${K4}P/${K1}P wings, sell ${K3}P/${K2}P body. Net debit ${net_debit:.2f}. "
            f"Max profit ${max_profit:.2f} if stock stays between ${K2}–${K3} at expiry. "
            f"Defined risk: lose net debit if outside range."
        ),
        "expiration_iso": expiry.isoformat(),
        "expiration": expiry.strftime("%b %d '%y"),
        "dte": dte,
        "strike": K3,
        "profit_zone": {"lower": K2, "upper": K3},
        "legs": [
            {"action": "BUY",  "option_type": "PUT", "strike": K4,
             "premium": p4["price"], "delta": p4["delta"], "theta": p4["theta"], "vega": p4["vega"]},
            {"action": "SELL", "option_type": "PUT", "strike": K3,
             "premium": p3["price"], "delta": p3["delta"], "theta": p3["theta"], "vega": p3["vega"]},
            {"action": "SELL", "option_type": "PUT", "strike": K2,
             "premium": p2["price"], "delta": p2["delta"], "theta": p2["theta"], "vega": p2["vega"]},
            {"action": "BUY",  "option_type": "PUT", "strike": K1,
             "premium": p1["price"], "delta": p1["delta"], "theta": p1["theta"], "vega": p1["vega"]},
        ],
        "net_debit": net_debit,
        "net_credit": None,
        "max_profit": max_profit,
        "max_loss": round(net_debit, 4),
        "breakeven_upper": round(K4 - net_debit, 2),
        "breakeven_lower": round(K1 + net_debit, 2),
        "reward_risk": round(max_profit / net_debit, 2) if net_debit > 0 else 0,
        "prob_profit": None,
        "payoff_curve": curve,
        "price_source": "bs_theoretical",
    }


def _is_viable(strategy: Optional[dict]) -> bool:
    """
    Quality gate: reject strategies where the P&L math doesn't make sense.

    Credit strategies (CSP, credit spreads): net_credit >= $0.08/share ($8/contract).
    Debit spreads: reward_risk >= 0.8 (don't pay $2 to win $1.50).
    Single-leg buys (long call/put/straddle/calendar): always viable — loss is bounded.
    Stock buy: always viable (natural entry, defined stop).
    """
    if not strategy:
        return False
    stype = strategy.get("strategy_type", "")
    if stype in (
        "long_call", "long_put", "long_straddle", "long_strangle",
        "call_calendar", "stock_buy", "protective_put", "collar",
        "put_backspread", "call_butterfly", "put_butterfly",
        "married_put", "synthetic_long", "synthetic_short",
        "pmcc", "zero_dte_iron_condor", "dividend_capture",
    ):
        return True   # buyer/defined-risk positions; always valid to show
    credit = strategy.get("net_credit")
    if credit is not None:
        return credit >= 0.08
    debit = strategy.get("net_debit")
    rr    = strategy.get("reward_risk", 0) or 0
    if debit is not None:
        return rr >= 0.8
    return True


# ── Strategy selector ─────────────────────────────────────────────────────────

def select_strategy(
    S: float,
    sigma: float,
    expiry: date,
    er_dte: int,
    pgi: float,
    er_label: str = "Pre-ER",
) -> Optional[dict]:
    """
    Select the best moderate-tier strategy for the given context.

    Post-ER (er_dte <= 0): Iron Condor for IV crush — with quality gate.
    Pre-ER (er_dte >= 1, any distance): PGI-driven spread or straddle.

    Returns None if S <= 0, sigma <= 0, or strategy fails quality gate.
    """
    return select_strategy_for_tier(S, sigma, expiry, er_dte, pgi, er_label, "moderate")


# ── LEAPS configuration per algo ──────────────────────────────────────────────
# Each algo has a probability of generating a LEAPS strategy instead of short-term.
# LEAPS = Long-Term Equity Anticipation Strategies (45d / 6mo / 12mo)

_LEAPS_CONFIG = {
    "fortress": {
        "probability": 0.75,  # 75% chance of LEAPS
        "dte_options": [365, 180, 45],  # 12mo, 6mo, 45d
        "universe": "large_cap_etf",  # SPY, QQQ, IWM, DIA, VOO, VTI
    },
    "sentinel": {
        "probability": 0.50,  # 50% chance of LEAPS
        "dte_options": [365, 180, 45],
        "universe": "sp500",  # S&P 500 stocks only
    },
    "moderate": {
        "probability": 0.25,  # 25% chance of LEAPS
        "dte_options": [365, 180, 45],
        "universe": "weekly_options",  # stocks with weekly options (good liquidity)
    },
}

_LARGE_CAP_ETF = {"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "XLF", "XLE", "XLK", "GLD"}
_SP500_TOP = {"AAPL", "MSFT", "NVDA", "GOOG", "META", "AMZN", "TSLA", "BRK.B", "JPM", "V",
              "UNH", "HD", "PG", "MA", "JNJ", "LLY", "AVGO", "XOM", "COST", "MRK",
              "SPY", "QQQ", "IWM", "VOO", "VTI"}


def _should_use_leaps(risk_tier: str, ticker: str = "") -> tuple:
    """Returns (use_leaps: bool, dte: int) based on algo config."""
    import random
    cfg = _LEAPS_CONFIG.get(risk_tier)
    if not cfg:
        return (False, 0)

    # Universe check
    universe = cfg["universe"]
    t = ticker.upper()
    if universe == "large_cap_etf" and t and t not in _LARGE_CAP_ETF:
        return (False, 0)
    if universe == "sp500" and t and t not in _SP500_TOP:
        return (False, 0)
    # weekly_options: any ticker is fine (assumed liquid)

    if random.random() < cfg["probability"]:
        dte = random.choice(cfg["dte_options"])
        return (True, dte)
    return (False, 0)


def select_strategy_for_tier(
    S: float,
    sigma: float,
    expiry: date,
    er_dte: int,
    pgi: float,
    er_label: str = "Pre-ER",
    risk_tier: str = "moderate",
    far_expiry: Optional[date] = None,
    ticker: str = "",
) -> Optional[dict]:
    """
    Tier-aware strategy selection.

    conservative : Cash-Secured Put (15% OTM) → income, defined cost if assigned.
                   Falls back to Bull/Bear Put/Call Spread if CSP low-premium.
    moderate     : Stock Buy (pre-ER momentum) + CSP overlay, or PGI-driven spreads.
                   Calendar spread if neutral and far_expiry available.
    aggressive   : Directional debit spreads (Bull/Bear Call/Put); straddle if neutral.
    yolo         : Long call or long put (OTM, lose 100%, unlimited upside).

    Post-ER (er_dte <= 0): Iron Condor with quality gate for all tiers.
    Quality gate: credit strategies need net_credit >= $0.08; debit needs R/R >= 0.8.

    CSP pct_otm by tier (same strategy, different risk):
      conservative 15% OTM → ~75–80% PoP, low premium, minimal assignment risk
      moderate      7% OTM → ~65–70% PoP, solid premium, balanced
      aggressive    3% OTM → ~55–60% PoP, high premium, real assignment risk

    Stock Buy target/stop by tier:
      conservative  stop 3%,  target  5%
      moderate      stop 5%,  target  8%
      aggressive    stop 7%,  target 15%
      yolo          stop 10%, target 25%
    """
    if S <= 0 or sigma <= 0:
        return None

    # ── LEAPS check: long-term strategies for Fortress/Sentinel/Moderate ──
    use_leaps, leaps_dte = _should_use_leaps(risk_tier, ticker)
    if use_leaps and leaps_dte > 0:
        leaps_expiry = date.today() + timedelta(days=leaps_dte)
        # LEAPS sigma: reduce annualized vol for longer-dated options (mean reversion)
        leaps_sigma = sigma * 0.85 if leaps_dte >= 180 else sigma * 0.95
        leaps_label = f"LEAPS {leaps_dte}d"

        if risk_tier in ("fortress", "iron_vault"):
            # Fortress LEAPS: deep OTM cash-secured puts on ETFs (income)
            csp = build_cash_secured_put(S, leaps_sigma, leaps_expiry, pct_otm=0.15)
            if csp and _is_viable(csp):
                csp["selection_reason"] = (
                    f"Fortress {leaps_label}: sell ${csp['strike']}P ({leaps_dte}d) on {ticker or 'ETF'}. "
                    f"Collect ${csp.get('net_credit',0):.2f}/sh, {csp.get('prob_profit',0):.0f}% PoP. "
                    f"Long-term income — high PoP at deep OTM."
                )
                csp["algo_id"] = "fortress"
                csp["leaps"] = True
                csp["leaps_dte"] = leaps_dte
                return csp

        elif risk_tier == "sentinel":
            # Sentinel LEAPS: collar or covered call (hedged long-term)
            strat = build_collar(S, leaps_sigma, leaps_expiry)
            if strat and _is_viable(strat):
                strat["selection_reason"] = (
                    f"Sentinel {leaps_label}: collar on {ticker or 'S&P500'} ({leaps_dte}d). "
                    f"Hedged long-term position — limited downside, capped upside."
                )
                strat["algo_id"] = "sentinel"
                strat["leaps"] = True
                strat["leaps_dte"] = leaps_dte
                return strat
            # Fallback: covered call
            cc = build_covered_call(S, leaps_sigma, leaps_expiry, pct_otm=0.08)
            if cc and _is_viable(cc):
                cc["selection_reason"] = (
                    f"Sentinel {leaps_label}: covered call ${cc['strike']}C ({leaps_dte}d). "
                    f"Income + upside participation to ${cc['strike']}."
                )
                cc["algo_id"] = "sentinel"
                cc["leaps"] = True
                cc["leaps_dte"] = leaps_dte
                return cc

        elif risk_tier == "moderate":
            # Moderate LEAPS: bull call spread or calendar (defined risk, long-term)
            if pgi >= 0:
                strat = build_bull_call_spread(S, leaps_sigma, leaps_expiry)
            else:
                strat = build_bear_put_spread(S, leaps_sigma, leaps_expiry)
            if strat and _is_viable(strat):
                direction = "bullish" if pgi >= 0 else "bearish"
                strat["selection_reason"] = (
                    f"Moderate {leaps_label}: {direction} spread on {ticker} ({leaps_dte}d). "
                    f"Defined risk, extended timeframe for thesis to play out."
                )
                strat["algo_id"] = "moderate"
                strat["leaps"] = True
                strat["leaps_dte"] = leaps_dte
                return strat

        # If LEAPS didn't produce viable strategy, fall through to short-term

    is_post_er = er_dte <= 0   # ER has already happened — IV crush regime
    strategy: Optional[dict] = None

    if is_post_er:
        # Post-ER: Iron Condor for IV crush — quality gate, then straddle fallback
        cond = build_iron_condor(S, sigma, expiry)
        if _is_viable(cond):
            cond["selection_reason"] = (
                f"Post-ER IV crush → condor collects ${cond.get('net_credit',0):.2f}/share "
                f"({cond.get('prob_profit',0):.0f}% PoP)"
            )
            return cond
        strad = build_long_straddle(S, sigma, expiry)
        strad["selection_reason"] = "Post-ER condor not viable (low IV) → straddle fallback"
        return strad

    # ── Pre-ER strategies by tier ─────────────────────────────────────────────

    # ── Sentinel: Triple-confirm consensus gate ─────────────────────────────
    # Requires 3 of 4 PGI components to agree on direction before entering.
    # Only fires when |PGI| > 50 (strong signal). Uses fully-hedged strategies.
    if risk_tier == "sentinel":
        # Sentinel: fully-hedged, high-conviction only.
        # Strategy priority: Collar (80% WR) → Protective Put → Covered Call
        # Uses 30-45 DTE for longer hedge horizon.
        if abs(pgi) < 35:
            return None  # Need |PGI| >= 35 for strong consensus

        if pgi > 0:
            # Bullish: collar first (buy put floor + sell call cap)
            strat = build_collar(S, sigma, expiry, put_otm=0.05, call_otm=0.08)
            if _is_viable(strat):
                strat["selection_reason"] = (
                    f"Sentinel bullish (PGI={pgi:+.0f}): collar — "
                    f"buy ${strat.get('legs',[{}])[1].get('strike','?')}P floor + "
                    f"sell ${strat.get('legs',[{},{},{}])[2].get('strike','?')}C cap. "
                    f"Defined risk on both sides."
                )
                strat["algo_id"] = "sentinel"
                return strat
            # Fallback: protective put (stock + buy put, no cap on upside)
            pp = build_protective_put(S, sigma, expiry, pct_otm=0.07)
            if pp and _is_viable(pp):
                pp["selection_reason"] = (
                    f"Sentinel bullish (PGI={pgi:+.0f}): protective put — "
                    f"own shares + buy ${pp['strike']}P ({pp.get('protection_pct',7):.0f}% OTM floor). "
                    f"Unlimited upside, insurance cost ${pp.get('net_debit',0):.2f}/sh."
                )
                pp["algo_id"] = "sentinel"
                return pp
            # Last: covered call
            strat = build_covered_call(S, sigma, expiry, pct_otm=0.08)
            if _is_viable(strat):
                strat["selection_reason"] = (
                    f"Sentinel bullish (PGI={pgi:+.0f}): covered call ${strat['strike']}C (8% OTM)."
                )
                strat["algo_id"] = "sentinel"
                return strat
        else:
            # Bearish: collar (hedged both sides), then protective put (hedge downside)
            strat = build_collar(S, sigma, expiry, put_otm=0.03, call_otm=0.05)
            if _is_viable(strat):
                strat["selection_reason"] = (
                    f"Sentinel bearish (PGI={pgi:+.0f}): tight collar. "
                    f"Protective put near ATM + covered call for income."
                )
                strat["algo_id"] = "sentinel"
                return strat
            # Fallback: protective put only (bearish hedge without call cap)
            pp = build_protective_put(S, sigma, expiry, pct_otm=0.03)
            if pp and _is_viable(pp):
                pp["selection_reason"] = (
                    f"Sentinel bearish (PGI={pgi:+.0f}): protective put — "
                    f"tight 3% OTM floor at ${pp['strike']}. Defensive positioning."
                )
                pp["algo_id"] = "sentinel"
                return pp
            strat = build_bear_call_spread(S, sigma, expiry)
            if strat and _is_viable(strat):
                strat["selection_reason"] = f"Sentinel bearish (PGI={pgi:+.0f}): bear call credit spread"
                strat["algo_id"] = "sentinel"
                return strat
        return None

    # ── Fortress: Neutral-only theta harvesting ────────────────────────────
    # Only trades when |PGI| < 15 (market is range-bound).
    # Iron condors only — sell both sides, collect premium from time decay.
    # Steps aside when market is trending or VIX is too low/high.
    if risk_tier in ("fortress", "iron_vault"):
        if abs(pgi) >= 20:
            return None  # Market is trending — Fortress sits out (need |PGI| < 20)

        # IV quality check: need sufficient premium to justify the risk
        # sigma (our HV proxy for IV) should be 15-50% annualized
        if sigma < 0.12:
            return None  # Premiums too thin — not worth the risk
        if sigma > 0.55:
            return None  # Vol too high — wing breach risk too large

        # Build iron condor with 1σ wings, tighter than Moderate
        # Use 6% OTM short strikes (tighter than Moderate's 8%) for more premium
        # Wing width 4% (tighter = less max loss but less room)
        short_otm = 0.06
        wing_width = 0.04

        # Prefer 14 DTE (accelerated theta) over 21 DTE
        vault_expiry = date.today() + timedelta(days=14)
        if (expiry - date.today()).days <= 14:
            vault_expiry = expiry  # Use provided if already short-dated

        ic = build_iron_condor(S, sigma, vault_expiry,
                               short_otm=short_otm, wing_width=wing_width)
        if ic and _is_viable(ic):
            pop = ic.get("prob_profit", 0)
            # Iron Vault quality: PoP must be >= 55% (relaxed from 70% for real-world vol)
            # At 55%+ PoP, the expected value is still positive due to premium collected
            if pop < 55:
                return None  # Not enough edge — sit out

            ic["selection_reason"] = (
                f"Fortress neutral (PGI={pgi:+.0f}, σ={sigma*100:.0f}%): "
                f"IC ${ic.get('profit_zone',{}).get('lower','?')}/"
                f"${ic.get('profit_zone',{}).get('upper','?')} range. "
                f"Collect ${ic.get('net_credit',0):.2f}/sh, {pop:.0f}% PoP. "
                f"Pure theta decay — no directional bias."
            )
            ic["algo_id"] = "fortress"
            return ic

        # IC not viable → try collar as defensive fallback (45-60 DTE)
        # Collar: stock + buy put + sell call = near-zero-cost hedge
        collar_expiry = date.today() + timedelta(days=45)
        if (expiry - date.today()).days >= 30:
            collar_expiry = expiry
        collar = build_collar(S, sigma, collar_expiry, put_otm=0.05, call_otm=0.07)
        if collar and _is_viable(collar):
            collar["selection_reason"] = (
                f"Fortress neutral (PGI={pgi:+.0f}, σ={sigma*100:.0f}%): "
                f"collar — buy put floor + sell call cap. Near-zero-cost hedge. "
                f"IC not viable at current vol, collar provides income + protection."
            )
            collar["algo_id"] = "fortress"
            return collar

        return None  # No viable neutral strategy — sit out

    # ── Surge: Directional debit spreads with conviction gating ────────────
    # Raised threshold from |30| → |40| to filter weak momentum signals.
    # Historical: 56% win rate at |30| threshold. Target: 70%+ by requiring
    # stronger conviction and wider OTM for higher PoP.
    if risk_tier in ("surge", "momentum_hunter"):
        if abs(pgi) < 40:
            return None  # Not enough directional signal (raised from 30)

        # Conviction levels based on PGI magnitude (wider OTM = higher PoP)
        # |40-55|: moderate conviction → 6% OTM spreads (was 5%)
        # |55-70|: high conviction → 4% OTM spreads (was 3%)
        # |70+|:   extreme conviction → 2% ATM + 2× sizing
        if abs(pgi) >= 70:
            spread_otm = 0.02
            conviction = "extreme"
        elif abs(pgi) >= 55:
            spread_otm = 0.04
            conviction = "high"
        else:
            spread_otm = 0.06
            conviction = "moderate"

        # Bullish: Bull Call Spread (buy lower call, sell higher call)
        if pgi > 0:
            strat = build_bull_call_spread(S, sigma, expiry)
            if strat and _is_viable(strat):
                strat["selection_reason"] = (
                    f"Surge BULLISH ({conviction}, PGI={pgi:+.0f}): "
                    f"bull call spread. Pay ${strat.get('net_debit',0):.2f}/sh for "
                    f"${strat.get('max_profit',0):.2f} max gain. "
                    f"{'2× conviction sizing. ' if conviction == 'extreme' else ''}"
                    f"Ride the trend with defined risk."
                )
                strat["algo_id"] = "surge"
                strat["conviction"] = conviction
                strat["sizing_multiplier"] = 2.0 if conviction == "extreme" else 1.0
                return strat
            # Fallback: protective put (stock + buy put) — defined risk momentum play
            # Surge uses 5% OTM put (tighter floor) with 7-14 DTE
            pp = build_protective_put(S, sigma, expiry, pct_otm=0.05)
            if pp and _is_viable(pp):
                pp["selection_reason"] = (
                    f"Surge BULLISH ({conviction}, PGI={pgi:+.0f}): "
                    f"protective put — own shares + buy ${pp['strike']}P floor. "
                    f"Ride momentum with {pp.get('protection_pct',5):.0f}% downside cap. "
                    f"Cost ${pp.get('net_debit',0):.2f}/sh insurance."
                )
                pp["algo_id"] = "surge"
                pp["conviction"] = conviction
                return pp
            # Last fallback: stock buy with tight stop
            stock = build_stock_buy(S, target_pct=0.10, stop_pct=0.04, label="Surge")
            stock["selection_reason"] = (
                f"Surge BULLISH ({conviction}, PGI={pgi:+.0f}): "
                f"stock buy, target +10%, stop -4%. Protective put not viable."
            )
            stock["algo_id"] = "surge"
            stock["conviction"] = conviction
            return stock

        # Bearish: Bear Put Spread (buy higher put, sell lower put)
        else:
            strat = build_bear_put_spread(S, sigma, expiry)
            if strat and _is_viable(strat):
                strat["selection_reason"] = (
                    f"Surge BEARISH ({conviction}, PGI={pgi:+.0f}): "
                    f"bear put spread. Pay ${strat.get('net_debit',0):.2f}/sh for "
                    f"${strat.get('max_profit',0):.2f} max gain. "
                    f"{'2× conviction sizing. ' if conviction == 'extreme' else ''}"
                    f"Defined-risk short play."
                )
                strat["algo_id"] = "surge"
                strat["conviction"] = conviction
                strat["sizing_multiplier"] = 2.0 if conviction == "extreme" else 1.0
                return strat
            # Fallback: long put for strong bearish conviction
            if abs(pgi) >= 50:
                lp = build_long_put(S, sigma, expiry, pct_otm=spread_otm)
                lp["selection_reason"] = (
                    f"Surge BEARISH ({conviction}, PGI={pgi:+.0f}): "
                    f"long put ${lp['strike']}. Spread not viable, high conviction."
                )
                lp["algo_id"] = "surge"
                lp["conviction"] = conviction
                return lp
        return None  # No viable directional play

    if risk_tier == "conservative":
        # Cash-Secured Put: get paid to wait; if assigned, own stock at discount
        # 15% OTM = very safe. Falls back to credit spread if premium too thin.
        csp = build_cash_secured_put(S, sigma, expiry, pct_otm=0.15)
        if _is_viable(csp):
            csp["selection_reason"] = (
                f"Conservative: sell ${csp['strike']}P (15% OTM, {csp.get('prob_profit',0):.0f}% PoP). "
                f"Collect ${csp.get('net_credit',0):.2f}/sh. Effective buy price ${csp.get('effective_buy_price',0):.2f}."
            )
            return csp
        # Fallback: credit spread
        fallback = build_bull_put_spread(S, sigma, expiry) if pgi >= 0 else build_bear_call_spread(S, sigma, expiry)
        fallback["selection_reason"] = f"Conservative: CSP thin → {'bull put' if pgi >= 0 else 'bear call'} spread (PGI={pgi:+.0f})"
        return fallback if _is_viable(fallback) else None

    elif risk_tier == "moderate":
        # Moderate: tuned for 70%+ win rate.
        # Historical: CSP=89% (excellent), stock_buy=45% (target too wide),
        # covered_call=15% (removed as primary). Focus on high-PoP credit strategies.
        #
        # Changes:
        # - CSP widened to 10% OTM (was 7%) → higher PoP, ~80-85%
        # - Stock buy tightened: +5% target / -3% stop (was +8%/-5%)
        # - Removed covered_call as primary (15% win rate)
        # - Raised bearish threshold to |25| (was 20)
        # - Default to CSP (highest win rate) instead of straddle fallback

        if abs(pgi) < 15 and far_expiry is not None:
            # Neutral + two expirations → calendar spread
            cal = build_call_calendar(S, sigma, expiry, far_expiry)
            cal["selection_reason"] = (
                f"Moderate neutral (PGI={pgi:+.0f}): calendar spread. "
                f"Sell near IV (${cal['strike']} {expiry.strftime('%b %d')}), "
                f"hold far leg ({far_expiry.strftime('%b %d')}). Net ${cal.get('net_debit',0):.2f}/sh."
            )
            return cal
        if pgi >= 25:
            # Bullish: protective put (stock + buy put) for hedged upside
            # Use 21-30 DTE put as insurance, 7% OTM floor
            pp = build_protective_put(S, sigma, expiry, pct_otm=0.07)
            if pp and _is_viable(pp):
                pp["selection_reason"] = (
                    f"Moderate bullish (PGI={pgi:+.0f}): protective put — "
                    f"own shares + buy ${pp['strike']}P ({pp.get('protection_pct',7):.0f}% OTM floor). "
                    f"Unlimited upside, capped downside at -${pp.get('max_loss',0):.2f}/sh. "
                    f"Insurance cost ${pp.get('net_debit',0):.2f}/sh."
                )
                pp["algo_id"] = "moderate"
                return pp
            # Fallback: stock buy with tight stop
            stock = build_stock_buy(S, target_pct=0.05, stop_pct=0.03, label="Moderate")
            stock["selection_reason"] = (
                f"Moderate bullish (PGI={pgi:+.0f}): buy shares, target +5%, stop -3%. "
                f"Protective put not viable — using tight stop instead."
            )
            return stock
        if pgi <= -25:
            # Bearish: CSP first (high PoP income), then bear call spread
            csp = build_cash_secured_put(S, sigma, expiry, pct_otm=0.10)
            if _is_viable(csp):
                csp["selection_reason"] = (
                    f"Moderate bearish (PGI={pgi:+.0f}): sell ${csp['strike']}P (10% OTM, "
                    f"{csp.get('prob_profit',0):.0f}% PoP). High-probability income play."
                )
                return csp
            spread = build_bear_call_spread(S, sigma, expiry)
            if _is_viable(spread):
                spread["selection_reason"] = f"Moderate bearish (PGI={pgi:+.0f}) → bear call spread"
                return spread
        # Default: CSP 10% OTM — highest win-rate strategy (89% historically)
        csp = build_cash_secured_put(S, sigma, expiry, pct_otm=0.10)
        if _is_viable(csp):
            csp["selection_reason"] = (
                f"Moderate neutral (PGI={pgi:+.0f}): sell ${csp['strike']}P (10% OTM, "
                f"{csp.get('prob_profit',0):.0f}% PoP). Collect ${csp.get('net_credit',0):.2f}/sh."
            )
            return csp
        # Fallback: bull put spread (defined risk credit strategy)
        bps = build_bull_put_spread(S, sigma, expiry)
        if bps and _is_viable(bps):
            bps["selection_reason"] = f"Moderate neutral (PGI={pgi:+.0f}) → bull put spread (CSP thin)"
            return bps
        return None  # Moderate doesn't force low-quality trades

    elif risk_tier == "aggressive":
        # Aggressive: directional stock buy or spread; CSP 3% OTM for premium hunters
        if pgi > 30:
            stock = build_stock_buy(S, target_pct=0.15, stop_pct=0.07, label="Aggressive")
            stock["selection_reason"] = (
                f"Aggressive bullish (PGI={pgi:+.0f}): buy shares, target +15%, stop -7%. "
                f"High conviction pre-ER run."
            )
            return stock
        if pgi < -30:
            spread = build_bear_put_spread(S, sigma, expiry)
            spread["selection_reason"] = f"Aggressive bearish (PGI={pgi:+.0f}) → bear put spread"
            if _is_viable(spread):
                return spread
        # Neutral or spread not viable: CSP 3% OTM (high premium, assignment-tolerant)
        csp = build_cash_secured_put(S, sigma, expiry, pct_otm=0.03)
        if _is_viable(csp):
            csp["selection_reason"] = (
                f"Aggressive: sell ${csp['strike']}P (3% OTM, "
                f"{csp.get('prob_profit',0):.0f}% PoP). "
                f"High premium ${csp.get('net_credit',0):.2f}/sh. Assignment = buy at discount."
            )
            return csp
        strad = build_long_straddle(S, sigma, expiry)
        strad["selection_reason"] = f"Aggressive neutral (PGI={pgi:+.0f}) → straddle for ER move"
        return strad

    # ── Degen: 0DTE + short-dated binary event plays ────────────────────────
    # 50% chance of 0DTE (today expiry) on high-liquidity ETFs
    # Remaining: 1-7 DTE for max gamma leverage
    # Targets 3-10× returns. Accepts 25-45% PoP.
    _DEGEN_0DTE_TICKERS = {"SPY", "QQQ", "IWM", "GLD", "SLV", "UVXY", "TLT",
                           "XLF", "XLE", "XLK", "AAPL", "TSLA", "NVDA", "META"}

    if risk_tier == "degen":
        import random as _rnd

        t = ticker.upper() if ticker else ""
        is_0dte_eligible = t in _DEGEN_0DTE_TICKERS

        # 50% chance of 0DTE for eligible tickers, otherwise 1-7 DTE
        use_0dte = is_0dte_eligible and _rnd.random() < 0.50
        if use_0dte:
            degen_dte = 0
            degen_expiry = date.today()  # expires today
            otm = _rnd.choice([0.005, 0.01, 0.015, 0.02])  # tight strikes for 0DTE
            label = "0DTE"
        else:
            degen_dte = _rnd.choice([1, 2, 3, 5, 7])
            degen_expiry = date.today() + timedelta(days=degen_dte)
            otm = _rnd.choice([0.03, 0.05, 0.08])
            label = f"{degen_dte}DTE"

        # Use market-open sigma boost for 0DTE (gamma is much higher)
        degen_sigma = sigma * 1.5 if use_0dte else sigma

        if abs(pgi) < 10 or _rnd.random() < 0.2:
            # Neutral / uncertain → straddle (profits from any big move)
            strat = build_long_straddle(S, degen_sigma, degen_expiry)
            strat["selection_reason"] = (
                f"Degen {label} STRADDLE on {t or 'ticker'}: "
                f"ATM straddle. Profits from any big move either direction. "
                f"Max loss = premium. 🎰"
            )
        elif pgi >= 0:
            strat = build_long_call(S, degen_sigma, degen_expiry, pct_otm=otm)
            strike = strat.get('strike', '?')
            strat["selection_reason"] = (
                f"Degen {label} CALL on {t or 'ticker'} (PGI={pgi:+.0f}): "
                f"buy ${strike}C {otm*100:.1f}% OTM. "
                f"{'Intraday gamma play.' if use_0dte else f'{degen_dte}d swing.'} "
                f"Risk 100% of premium. 🚀"
            )
        else:
            strat = build_long_put(S, degen_sigma, degen_expiry, pct_otm=otm)
            strike = strat.get('strike', '?')
            strat["selection_reason"] = (
                f"Degen {label} PUT on {t or 'ticker'} (PGI={pgi:+.0f}): "
                f"buy ${strike}P {otm*100:.1f}% OTM. "
                f"{'Intraday gamma play.' if use_0dte else f'{degen_dte}d swing.'} "
                f"Max gain if stock tanks. 🚀"
            )

        strat["algo_id"] = "degen"
        strat["is_0dte"] = use_0dte
        strat["max_position_pct"] = 0.01
        return strat

    else:  # fallback to yolo (legacy)
        if pgi >= 0:
            strategy = build_long_call(S, sigma, expiry, pct_otm=0.08)
            strategy["selection_reason"] = (
                f"YOLO bullish (PGI={pgi:+.0f}) → buy 8% OTM call."
            )
        else:
            strategy = build_long_put(S, sigma, expiry, pct_otm=0.08)
            strategy["selection_reason"] = (
                f"YOLO bearish (PGI={pgi:+.0f}) → buy 8% OTM put."
            )
        return strategy


# ── Multi-strategy selector (used by scanner for credit-per-swipe revenue) ───

def select_multi_strategies(
    S: float,
    sigma: float,
    expiry: date,
    er_dte: int,
    pgi: float,
    far_expiry: Optional[date] = None,
    max_count: int = 4,
) -> List[dict]:
    """
    Returns 2–4 ranked strategy suggestions for a ticker, ordered primary → alternative → hedge/income.

    Each strategy gains a 'selection_rank' field (1 = primary, 2 = alt, 3 = hedge/income, 4 = aggressive).
    The scanner creates one Pulse per strategy entry → enabling 1-credit-per-strategy revenue.

    PGI-driven selection matrix:
      Post-ER:          Iron Condor → Iron Butterfly → Long Call Condor → Short Strangle
      PGI > +60:        Bull Call Spread → Long Call → Bull Put Spread → Cash-Secured Put
      PGI +20 to +60:   Bull Put Spread → Cash-Secured Put → Covered Call → Bull Call Spread
      PGI -20 to +20:   Iron Condor → Iron Butterfly or Long Strangle → Call Butterfly → Calendar
      PGI -20 to -60:   Bear Call Spread → Bear Put Spread → Collar → Long Put
      PGI < -60:        Bear Put Spread → Put Backspread → Bear Call Spread → Long Put
    """
    if S <= 0 or sigma <= 0:
        return []

    is_post_er = er_dte <= 0
    high_vol   = sigma > 0.30
    results: List[dict] = []

    def _add(s: Optional[dict], rank: int, reason: str = "") -> None:
        if s and _is_viable(s):
            s = dict(s)
            s["selection_rank"]   = rank
            s["selection_reason"] = reason or s.get("selection_reason", "")
            results.append(s)

    # ── Post-ER: IV crush plays (sell premium on both sides) ──────────
    if is_post_er:
        ic = build_iron_condor(S, sigma, expiry)
        _add(ic, 1, f"Post-ER IV crush: iron condor ({ic.get('prob_profit',0):.0f}% PoP)" if ic else "")
        ib = build_iron_butterfly(S, sigma, expiry)
        _add(ib, 2, "Post-ER: iron butterfly — more premium, narrower zone")
        if high_vol:
            ss = build_short_strangle(S, sigma, expiry)
            _add(ss, 3, "Post-ER high IV: short strangle — wider profit zone")
            sd = build_short_straddle(S, sigma, expiry)
            _add(sd, 4, "Post-ER extreme IV: short straddle — max premium at ATM")
        else:
            lcc = build_long_call_condor(S, sigma, expiry)
            _add(lcc, 3, "Post-ER: long call condor — defined-risk neutral")
            pfly = build_put_butterfly(S, sigma, expiry)
            _add(pfly, 4, "Post-ER: put butterfly — pinpoint bearish recovery bet")

    # ── Strong bull (PGI > +60): aggressive directional + income ────
    elif pgi > 60:
        bcs = build_bull_call_spread(S, sigma, expiry)
        _add(bcs, 1, f"Strong bull (PGI={pgi:+.0f}): bull call spread — directional debit")
        lc = build_long_call(S, sigma, expiry, pct_otm=0.08)
        _add(lc, 2, "Strong bull: long call — max gamma leverage")
        syn = build_synthetic_long(S, sigma, expiry)
        _add(syn, 3, "Strong bull: synthetic long — capital-efficient stock substitute, delta ≈ 1.0")
        bps = build_bull_put_spread(S, sigma, expiry)
        _add(bps, 4, "Strong bull income: bull put spread — credit with room")

    # ── Mild bull (PGI +20 to +60): income-focused with upside ──────
    elif pgi > 20:
        bps = build_bull_put_spread(S, sigma, expiry)
        _add(bps, 1, f"Mild bull (PGI={pgi:+.0f}): bull put spread — {bps.get('prob_profit',0):.0f}% PoP")
        csp = build_cash_secured_put(S, sigma, expiry, pct_otm=0.07)
        _add(csp, 2, "Mild bull: cash-secured put — income while waiting to buy")
        cc = build_covered_call(S, sigma, expiry)
        _add(cc, 3, "Mild bull: covered call — harvest premium on shares held")
        cfly = build_call_butterfly(S, sigma, expiry)
        _add(cfly, 4, "Mild bull pinpoint: call butterfly — cheap defined risk")

    # ── Strong bear (PGI < -60): aggressive directional + hedges ────
    elif pgi < -60:
        bput = build_bear_put_spread(S, sigma, expiry)
        _add(bput, 1, f"Strong bear (PGI={pgi:+.0f}): bear put spread — directional debit")
        syn_s = build_synthetic_short(S, sigma, expiry)
        _add(syn_s, 2, "Strong bear: synthetic short — no borrowing needed, delta ≈ -1.0")
        lp = build_long_put(S, sigma, expiry, pct_otm=0.08)
        _add(lp, 3, "Strong bear: long put — max downside leverage")
        pb = build_put_backspread(S, sigma, expiry)
        _add(pb, 4, "Strong bear: put backspread (1×2) — profits accelerate on crash")

    # ── Mild bear (PGI -20 to -60): income + protection ────────────
    elif pgi < -20:
        bcs = build_bear_call_spread(S, sigma, expiry)
        _add(bcs, 1, f"Mild bear (PGI={pgi:+.0f}): bear call spread — {bcs.get('prob_profit',0):.0f}% PoP")
        col = build_collar(S, sigma, expiry)
        _add(col, 2, "Mild bear: collar — zero-cost protection on long shares")
        bput = build_bear_put_spread(S, sigma, expiry)
        _add(bput, 3, "Mild bear: bear put spread — defined debit directional")
        pfly = build_put_butterfly(S, sigma, expiry)
        _add(pfly, 4, "Mild bear pinpoint: put butterfly — cheap defined risk")

    # ── Neutral (|PGI| < 20): theta + vol plays ────────────────────
    else:
        ic = build_iron_condor(S, sigma, expiry)
        _add(ic, 1, f"Neutral (PGI={pgi:+.0f}): iron condor — sell both sides")
        if high_vol:
            ib = build_iron_butterfly(S, sigma, expiry)
            _add(ib, 2, "Neutral high IV: iron butterfly — max ATM premium")
            ss = build_short_strangle(S, sigma, expiry)
            _add(ss, 3, "Neutral high IV: short strangle — wider profit zone")
        else:
            ls = build_long_strangle(S, sigma, expiry)
            _add(ls, 2, "Neutral low IV: long strangle — cheap bet on big move")
            cfly = build_call_butterfly(S, sigma, expiry)
            _add(cfly, 3, "Neutral pinpoint: call butterfly — high reward/risk")
        if far_expiry is not None:
            cal = build_call_calendar(S, sigma, expiry, far_expiry)
            _add(cal, 4, "Neutral: call calendar — sell near IV, hold far leg")
            pcal = build_put_calendar(S, sigma, expiry, far_expiry)
            _add(pcal, 5, "Neutral: put calendar — theta play on put side")
        else:
            lpc = build_long_put_condor(S, sigma, expiry)
            _add(lpc, 4, "Neutral: long put condor — bearish-lean neutral")

    # Strip any None results (from quality gates like iron condor R/R check)
    return [r for r in results[:max_count] if r is not None]
