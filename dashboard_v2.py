#!/usr/bin/env python3
"""
Morning Pre-Session Bias Dashboard v2
======================================
Tashkent-optimized (UTC+5) scalping dashboard.

Data Sources (ALL FREE, NO ACCOUNT NEEDED):
  - Deribit Public API  → BTC/ETH options chain → GEX calculation, funding, OI
  - Yahoo Finance       → SPY/GLD/USO options chain → full Greeks (Δ Γ Θ V ρ) + 5d history
  - Black-Scholes       → Greeks calculated locally from IV (no vendor dependency)

Install:
  pip install requests yfinance scipy numpy

Run:
  python dashboard_v2.py              # full dashboard (one-shot)
  python dashboard_v2.py --serve      # live server, auto-refresh every 10 min
  python dashboard_v2.py --verify     # print raw API responses for audit
  python dashboard_v2.py --crypto     # crypto only (skip SPY/GLD)

Output: morning_dashboard.html (open in browser)

IMPORTANT: Every number in the output is tagged with:
  [LIVE]  = fetched from API in this run
  [CALC]  = calculated from fetched data using Black-Scholes
  [FAIL]  = fetch failed, no data shown (never synthesized)
"""

import requests, json, math, sys, os, time, threading, csv, io
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    import numpy as np
    from scipy.stats import norm
except ImportError:
    print("ERROR: pip install requests yfinance scipy numpy"); sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    yf = None
    print("WARNING: yfinance not installed — SPY/GLD skipped. pip install yfinance")

UZT = timezone(timedelta(hours=5))
DERIBIT = "https://www.deribit.com/api/v2/public"
DERIBIT_PERPS = {
    "BTC": "BTC-PERPETUAL",
    "ETH": "ETH-PERPETUAL",
    "SOL": "SOL_USDC-PERPETUAL",
}
VERIFY = "--verify" in sys.argv
CRYPTO_ONLY = "--crypto" in sys.argv
SERVE = "--serve" in sys.argv
SERVE_PORT = 8050
REFRESH_INTERVAL = 300  # seconds (5 min)
YF_MAX_EXPIRIES = 12
DXY_TICKERS = ["DX-Y.NYB", "DX=F", "UUP"]
FRED_BROAD_USD_SERIES = "DTWEXBGS"
fetch_log = []
_cached_html = ""
_last_fetch_ts = 0
_is_fetching = False
_fetch_lock = threading.Lock()

def log(src, ep, status, detail=""):
    fetch_log.append({"source": src, "endpoint": ep, "status": status, "detail": detail[:200]})
    if VERIFY: print(f"  [VERIFY] {src} | {ep} | {status} | {detail[:120]}")

# ── Black-Scholes ────────────────────────────────────────────────────────────
def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    except: return 0.0

def bs_greeks(S, K, T, r, sigma, opt_type):
    """Full Greeks: (delta, gamma, theta, vega, rho) or zeros."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return (0.0,) * 5
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        nd1 = norm.pdf(d1)
        if opt_type == "call":
            delta = norm.cdf(d1)
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        gamma = nd1 / (S * sigma * math.sqrt(T))
        theta = (-(S * nd1 * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * (norm.cdf(d2) if opt_type == "call" else norm.cdf(-d2))) / 365
        vega = S * nd1 * math.sqrt(T) / 100
        return delta, gamma, theta, vega, rho
    except:
        return (0.0,) * 5

def _safe_oi(raw):
    return int(raw) if raw is not None and not (isinstance(raw, float) and math.isnan(raw)) else 0

def _safe_float(raw):
    return float(raw) if raw is not None and not (isinstance(raw, float) and math.isnan(raw)) else 0.0

def _perp_funding_rate(perp):
    if not perp:
        return None
    if perp.get("funding_8h") is not None:
        return perp.get("funding_8h")
    return perp.get("current_funding")

def _yf_expiry_years(exp_date):
    # Treat expiry as end-of-day UTC so same-day chains are not dropped at midnight.
    expiry_dt = datetime.strptime(exp_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    secs = (expiry_dt - datetime.now(timezone.utc)).total_seconds()
    return max(secs / (365.25 * 24 * 3600), 0.0)

def _yf_weight_value(row, field):
    if field == "volume":
        return _safe_oi(row.get("volume", 0))
    return _safe_oi(row.get("openInterest", 0))

def _pct_change(series, lookback):
    if not series or len(series) <= lookback:
        return None
    base = series[-(lookback + 1)]
    if not base:
        return None
    return (series[-1] / base - 1) * 100

def _color_pct(pct):
    if pct is None:
        return "#888"
    return "#E24B4A" if pct > 0 else "#1D9E75" if pct < 0 else "#888"

def _fmt_pct(pct, digits=2):
    if pct is None:
        return "N/A"
    return f"{pct:+.{digits}f}%"

def _macro_leg_bias(pct_5d, pct_20d):
    score = 0
    if pct_5d is not None:
        if pct_5d >= 0.30:
            score += 1
        elif pct_5d <= -0.30:
            score -= 1
    if pct_20d is not None:
        if pct_20d >= 0.75:
            score += 1
        elif pct_20d <= -0.75:
            score -= 1
    if score >= 2:
        return "bull"
    if score <= -2:
        return "bear"
    return "neutral"

def yf_macro_proxy(symbols, days=25):
    if not yf:
        return None
    for symbol in symbols:
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period=f"{days + 10}d")
            closes = [float(v) for v in hist.get("Close", []).dropna().tail(days + 1)]
            if len(closes) < 6:
                continue
            latest = closes[-1]
            pct_5d = _pct_change(closes, 5)
            pct_20d = _pct_change(closes, 20)
            log("yfinance", f"{symbol}/macro", "OK", f"last={latest:.2f} 5d={pct_5d}")
            return {"symbol": symbol, "latest": latest, "pct_5d": pct_5d, "pct_20d": pct_20d, "_src": "Yahoo"}
        except Exception as e:
            log("yfinance", f"{symbol}/macro", "FAIL", str(e))
    return None

def fred_series(series_id, days=25):
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": series_id}, timeout=10)
        r.raise_for_status()
        rows = []
        for row in csv.DictReader(io.StringIO(r.text)):
            val = row.get(series_id)
            if not val or val == ".":
                continue
            rows.append((row.get("DATE"), float(val)))
        if len(rows) < 6:
            log("FRED", series_id, "FAIL", "insufficient rows")
            return None
        rows = rows[-(days + 1):]
        values = [v for _, v in rows]
        pct_5d = _pct_change(values, 5)
        pct_20d = _pct_change(values, 20)
        log("FRED", series_id, "OK", f"last={values[-1]:.2f} 5d={pct_5d}")
        return {"series": series_id, "date": rows[-1][0], "latest": values[-1],
                "pct_5d": pct_5d, "pct_20d": pct_20d, "_src": "FRED"}
    except Exception as e:
        log("FRED", series_id, "FAIL", str(e))
        return None

def usd_macro():
    dxy = yf_macro_proxy(DXY_TICKERS)
    broad = fred_series(FRED_BROAD_USD_SERIES)
    if not dxy and not broad:
        return None

    dxy_bias = _macro_leg_bias(dxy.get("pct_5d") if dxy else None, dxy.get("pct_20d") if dxy else None)
    broad_bias = _macro_leg_bias(broad.get("pct_5d") if broad else None, broad.get("pct_20d") if broad else None)

    if dxy and broad and dxy_bias == broad_bias and dxy_bias in ("bull", "bear"):
        bias = dxy_bias
    elif dxy and not broad and dxy_bias in ("bull", "bear"):
        bias = dxy_bias
    elif broad and not dxy and broad_bias in ("bull", "bear"):
        bias = broad_bias
    else:
        bias = "neutral"

    if bias == "bull":
        label, color = "USD strong", "#E24B4A"
        reason = "Dollar strength is a macro headwind for BTC, gold, crude, and broad risk."
    elif bias == "bear":
        label, color = "USD weak", "#1D9E75"
        reason = "Dollar weakness is a macro tailwind for BTC, gold, crude, and broad risk."
    else:
        label, color = "USD mixed", "#EF9F27"
        reason = "DXY and the broad dollar index are not aligned strongly enough for a directional macro call."

    parts = []
    if dxy:
        parts.append(f"DXY proxy {_fmt_pct(dxy.get('pct_5d'))} (5d)")
    if broad:
        parts.append(f"broad USD {_fmt_pct(broad.get('pct_5d'))} (5d)")

    return {"bias": bias, "label": label, "color": color, "reason": reason,
            "summary": " · ".join(parts), "dxy": dxy, "broad_usd": broad}

# ── Deribit ───────────────────────────────────────────────────────────────────
def binance_funding(ccy):
    """Fetch Binance perpetual funding rate — largest market by OI/volume."""
    sym = f"{ccy}USDT"
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                         params={"symbol": sym}, timeout=10)
        r.raise_for_status()
        d = r.json()
        rate = float(d.get("lastFundingRate", 0))
        log("Binance", f"funding/{sym}", "OK", f"rate={rate:.6f}")
        return {"funding_rate": rate, "mark_price": float(d.get("markPrice", 0)),
                "next_funding_time": d.get("nextFundingTime", 0)}
    except Exception as e:
        log("Binance", f"funding/{sym}", "FAIL", str(e)); return None

def dget(method, params=None):
    try:
        r = requests.get(f"{DERIBIT}/{method}", params=params or {}, timeout=15)
        r.raise_for_status()
        res = r.json().get("result")
        log("Deribit", method, "OK", f"{params}")
        return res
    except Exception as e:
        log("Deribit", method, "FAIL", str(e)); return None

def deribit_perp(ccy):
    instrument_name = DERIBIT_PERPS.get(ccy, f"{ccy}-PERPETUAL")
    t = dget("ticker", {"instrument_name": instrument_name})
    if not t: return None
    return {"current_funding": t.get("current_funding"), "funding_8h": t.get("funding_8h"),
            "open_interest": t.get("open_interest"), "index_price": t.get("index_price"),
            "instrument_name": instrument_name,
            "volume_24h": t.get("stats",{}).get("volume"), "_src": "Deribit"}

def deribit_gex(ccy):
    idx = dget("get_index_price", {"index_name": f"{ccy.lower()}_usd"})
    if not idx: return None
    spot = idx["index_price"]
    instruments = dget("get_instruments", {"currency": ccy, "kind": "option", "expired": "false"})
    if not instruments: return None
    book = dget("get_book_summary_by_currency", {"currency": ccy, "kind": "option"})
    if not book: return None

    oi_map = {b["instrument_name"]: {"oi": b.get("open_interest",0), "iv": b.get("mark_iv",0)} for b in book}
    cands = [{"name": i["instrument_name"], "strike": i["strike"], "type": i["option_type"],
              "oi": oi_map.get(i["instrument_name"],{}).get("oi",0),
              "exp_ts": i.get("expiration_timestamp",0)}
             for i in instruments if oi_map.get(i["instrument_name"],{}).get("oi",0) > 0]
    cands.sort(key=lambda x: x["oi"], reverse=True)
    top = cands[:80]

    gex_by_strike = defaultdict(float)
    call_oi, put_oi = 0, 0
    call_oi_s, put_oi_s = defaultdict(float), defaultdict(float)
    n_ok = 0
    print(f"    Fetching Greeks for {len(top)} contracts...")
    for i, c in enumerate(top):
        tk = dget("ticker", {"instrument_name": c["name"]})
        if not tk: continue
        g = tk.get("greeks",{}).get("gamma", 0)
        if g == 0:
            iv = oi_map.get(c["name"],{}).get("iv",0)
            if iv > 0:
                T = max((c["exp_ts"] - datetime.now(timezone.utc).timestamp()*1000) / (365.25*24*3600*1000), 0.001)
                g = bs_gamma(spot, c["strike"], T, 0.05, iv/100)
                if g > 0: log("BS", c["name"], "CALC", f"gamma={g:.8f}")
        if g == 0: continue
        n_ok += 1
        s, oi = c["strike"], c["oi"]
        if c["type"] == "call":
            gex_by_strike[s] += g * oi * spot**2 * 0.01; call_oi += oi; call_oi_s[s] += oi
        else:
            gex_by_strike[s] -= g * oi * spot**2 * 0.01; put_oi += oi; put_oi_s[s] += oi
        if (i+1) % 20 == 0: print(f"    ...{i+1}/{len(top)}")

    if n_ok == 0: return None
    levels = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)
    flip = None; cum = 0; prev = None
    for s in sorted(gex_by_strike):
        cum += gex_by_strike[s]; cs = 1 if cum >= 0 else -1
        if prev is not None and cs != prev: flip = s
        prev = cs
    mp = None; ml = float('inf')
    for tp in set(call_oi_s) | set(put_oi_s):
        loss = sum(max(0, tp-s)*oi for s,oi in call_oi_s.items()) + sum(max(0,s-tp)*oi for s,oi in put_oi_s.items())
        if loss < ml: ml = loss; mp = tp

    # Detect crypto OpEx — find expirations within 24h
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    expiring_soon = defaultdict(lambda: {"call_oi": 0, "put_oi": 0})
    for c in cands:
        hrs_to_exp = (c["exp_ts"] - now_ms) / 3600000
        if 0 < hrs_to_exp <= 24:
            expiring_soon["24h"]["call_oi" if c["type"] == "call" else "put_oi"] += c["oi"]
        elif 0 < hrs_to_exp <= 48:
            expiring_soon["48h"]["call_oi" if c["type"] == "call" else "put_oi"] += c["oi"]
    crypto_opex = None
    total_live_oi = sum(c["oi"] for c in cands)
    if expiring_soon.get("24h") and (expiring_soon["24h"]["call_oi"] + expiring_soon["24h"]["put_oi"]) > 0:
        d = expiring_soon["24h"]
        total_oi = d["call_oi"] + d["put_oi"]
        crypto_opex = {"call_oi": d["call_oi"], "put_oi": d["put_oi"],
                       "total_oi": total_oi, "window": "24h",
                       "share_pct": (total_oi / total_live_oi * 100) if total_live_oi else None}

    return {"spot": spot, "gex_levels": levels[:8], "gamma_flip": flip, "max_pain": mp,
            "pc_ratio": put_oi/call_oi if call_oi else None, "call_oi": call_oi, "put_oi": put_oi,
            "net_gex": sum(gex_by_strike.values()), "n": n_ok,
            "opex": crypto_opex, "_src": "Deribit+BS"}

# ── Yahoo Finance — full Greeks + 5d history ─────────────────────────────────
def yf_history(symbol, days=5):
    """5-day price history, realized vol, trend."""
    if not yf: return None
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period=f"{days+5}d")
        if hist.empty or len(hist) < 2: return None
        hist = hist.tail(days + 1)
        closes = hist["Close"].values
        spot = float(closes[-1])
        returns = np.diff(np.log(closes))
        hv_ann = float(np.std(returns) * math.sqrt(252))
        pct_chg = (closes[-1] / closes[0] - 1) * 100
        return {"spot": spot, "hv_ann": hv_ann, "pct_chg": pct_chg,
                "hi_5d": float(np.max(closes)), "lo_5d": float(np.min(closes)),
                "closes": closes.tolist(), "returns": returns.tolist(),
                "dates": [str(d.date()) for d in hist.index]}
    except Exception as e:
        log("yfinance", f"{symbol}/history", "FAIL", str(e)); return None

def _opex_type(exp_date):
    """Classify expiration: daily, weekly, monthly, quarterly, yearly."""
    d = datetime.strptime(exp_date, "%Y-%m-%d")
    # 3rd Friday of month = monthly OpEx
    import calendar
    cal = calendar.monthcalendar(d.year, d.month)
    fridays = [w[calendar.FRIDAY] for w in cal if w[calendar.FRIDAY] != 0]
    third_fri = fridays[2] if len(fridays) >= 3 else fridays[-1]
    is_monthly = d.day == third_fri and d.weekday() == 4
    is_quarterly = is_monthly and d.month in (3, 6, 9, 12)
    if is_quarterly: return "QUARTERLY"
    if is_monthly: return "MONTHLY"
    if d.weekday() == 4: return "weekly"
    return "daily"

def _detect_opex(symbol, valid_exps, tk, spot):
    """Detect upcoming expirations and their OI impact."""
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    results = []
    for exp in valid_exps[:6]:  # check next 6 expirations
        if exp > tomorrow: break  # only care about today/tomorrow
        otype = _opex_type(exp)
        T_days = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
        try:
            chain = tk.option_chain(exp)
            total_call_oi = sum(_safe_oi(r.get("openInterest", 0)) for _, r in chain.calls.iterrows())
            total_put_oi = sum(_safe_oi(r.get("openInterest", 0)) for _, r in chain.puts.iterrows())
            total_oi = total_call_oi + total_put_oi
            if total_oi <= 0:
                continue
            atm_call_oi = sum(_safe_oi(r.get("openInterest", 0)) for _, r in chain.calls.iterrows()
                              if _safe_oi(r.get("openInterest", 0)) > 0 and
                              spot * 0.90 < r.get("strike", 0) < spot * 1.10)
            atm_put_oi = sum(_safe_oi(r.get("openInterest", 0)) for _, r in chain.puts.iterrows()
                             if _safe_oi(r.get("openInterest", 0)) > 0 and
                             spot * 0.90 < r.get("strike", 0) < spot * 1.10)
            atm_total_oi = atm_call_oi + atm_put_oi
            atm_share_pct = (atm_total_oi / total_oi * 100) if total_oi else 0
            moderate_threshold = max(100, int(total_oi * 0.02))
            high_threshold = max(300, int(total_oi * 0.05))
            if atm_total_oi >= high_threshold:
                impact = "high"
            elif atm_total_oi >= moderate_threshold:
                impact = "moderate"
            else:
                impact = "low"
            results.append({"exp": exp, "type": otype, "days": T_days,
                            "call_oi": total_call_oi, "put_oi": total_put_oi, "total_oi": total_oi,
                            "atm_call_oi": atm_call_oi, "atm_put_oi": atm_put_oi, "atm_total_oi": atm_total_oi,
                            "atm_share_pct": atm_share_pct, "impact": impact,
                            "show_alert": atm_total_oi > 0 and impact != "low",
                            "is_today": exp == today, "is_tomorrow": exp == tomorrow})
        except:
            pass
    return results if results else None

def yf_greeks(symbol, hist_data):
    """Full Greeks computation for SPY/GLD: returns GEX + all Greeks aggregated."""
    if not yf: log("yfinance", symbol, "FAIL", "not installed"); return None
    try:
        tk = yf.Ticker(symbol)
        spot = hist_data["spot"]
        hv = hist_data["hv_ann"]
        log("yfinance", f"{symbol}/price", "OK", f"spot={spot}")
        exps = tk.options
        if not exps: log("yfinance", f"{symbol}/options", "FAIL", "none"); return None
        today_str = datetime.now().strftime("%Y-%m-%d")
        valid_exps = [e for e in exps if e >= today_str]
        if not valid_exps: log("yfinance", f"{symbol}/options", "FAIL", "all expired"); return None
        sampled_exps = valid_exps[:YF_MAX_EXPIRIES]
        log("yfinance", f"{symbol}/options", "OK", f"{len(valid_exps)} valid exps, sampling {sampled_exps}")

        gex_by_strike = defaultdict(float)
        agg = defaultdict(lambda: {"delta":0,"gamma":0,"theta":0,"vega":0,"call_oi":0,"put_oi":0})
        c_oi = 0; p_oi = 0; n = 0; iv_sum = 0
        chains = []
        oi_rows = 0
        vol_rows = 0

        for exp in sampled_exps:
            try:
                chain = tk.option_chain(exp)
            except Exception as e:
                log("yfinance", f"{symbol}/{exp}", "FAIL", str(e))
                continue
            log("yfinance", f"{symbol}/{exp}", "OK", f"c={len(chain.calls)} p={len(chain.puts)}")
            T = _yf_expiry_years(exp)
            if T <= 0:
                continue
            exp_oi_rows = 0
            exp_vol_rows = 0
            for df in (chain.calls, chain.puts):
                for _, r in df.iterrows():
                    s = _safe_float(r.get("strike", 0))
                    iv = _safe_float(r.get("impliedVolatility", 0))
                    if iv <= 0 or s < spot * 0.85 or s > spot * 1.15:
                        continue
                    if _safe_oi(r.get("openInterest", 0)) > 0:
                        exp_oi_rows += 1
                    if _safe_oi(r.get("volume", 0)) > 0:
                        exp_vol_rows += 1
            oi_rows += exp_oi_rows
            vol_rows += exp_vol_rows
            chains.append({"exp": exp, "chain": chain, "T": T, "oi_rows": exp_oi_rows, "vol_rows": exp_vol_rows})

        if not chains:
            return None

        weight_source = "openInterest" if oi_rows > 0 else "volume"
        weight_label = "OI" if weight_source == "openInterest" else "Vol"
        pc_label = "P/C" if weight_source == "openInterest" else "P/C vol"
        sampled_rows = oi_rows if weight_source == "openInterest" else vol_rows
        if weight_source == "volume":
            log("yfinance", f"{symbol}/weights", "CALC", "open interest unavailable, using live volume fallback")
        else:
            log("yfinance", f"{symbol}/weights", "OK", f"using open interest ({oi_rows} usable rows)")

        for item in chains:
            exp = item["exp"]
            chain = item["chain"]
            T = item["T"]
            if _safe_oi(item["oi_rows"] if weight_source == "openInterest" else item["vol_rows"]) == 0:
                continue
            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                for _, r in df.iterrows():
                    s = r.get("strike", 0)
                    oi = _yf_weight_value(r, weight_source)
                    iv = _safe_float(r.get("impliedVolatility", 0))
                    if oi <= 0 or iv <= 0: continue
                    if s < spot * 0.85 or s > spot * 1.15: continue

                    d, g, t, v, rho = bs_greeks(spot, s, T, 0.05, iv, opt_type)
                    if g <= 0: continue
                    sign = 1 if opt_type == "call" else -1

                    gex_by_strike[s] += g * oi * 100 * spot**2 * 0.01 * sign
                    agg[s]["delta"] += d * oi * 100 * sign
                    agg[s]["gamma"] += g * oi * 100 * spot**2 * 0.01 * sign
                    agg[s]["theta"] += t * oi * 100
                    agg[s]["vega"]  += v * oi * 100
                    if opt_type == "call": agg[s]["call_oi"] += oi; c_oi += oi
                    else:                  agg[s]["put_oi"]  += oi; p_oi += oi
                    iv_sum += iv; n += 1

        if n == 0:
            return None

        # Totals
        totals = {k: sum(agg[s][k] for s in agg) for k in ["delta","gamma","theta","vega","call_oi","put_oi"]}
        net_gex = totals["gamma"]

        # Gamma flip
        flip = None; cum = 0; prev = None
        for s in sorted(agg):
            cum += agg[s]["gamma"]; cs = 1 if cum >= 0 else -1
            if prev is not None and cs != prev: flip = s
            prev = cs

        # Max pain
        call_oi_by_s = {s: agg[s]["call_oi"] for s in agg}
        put_oi_by_s  = {s: agg[s]["put_oi"]  for s in agg}
        mp = None; ml = float("inf")
        for tp in set(call_oi_by_s) | set(put_oi_by_s):
            loss = (sum(max(0, tp - s) * oi for s, oi in call_oi_by_s.items()) +
                    sum(max(0, s - tp) * oi for s, oi in put_oi_by_s.items()))
            if loss < ml: ml = loss; mp = tp

        # Top strikes by |GEX|
        top_strikes = sorted(agg.items(), key=lambda x: abs(x[1]["gamma"]), reverse=True)[:8]
        levels = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)[:6]

        avg_iv = iv_sum / n if n else 0
        pc = p_oi / c_oi if c_oi > 0 else None

        # OpEx detection — OI expiring today/tomorrow
        opex_info = _detect_opex(symbol, valid_exps, tk, spot)

        confidence_label = "OI-backed" if weight_source == "openInterest" else "Volume proxy"
        confidence_color = "#1D9E75" if weight_source == "openInterest" else "#EF9F27"

        return {"symbol": symbol, "spot": spot, "gex_levels": levels, "gamma_flip": flip,
                "max_pain": mp, "net_gex": net_gex, "pc_ratio": pc,
                "call_oi": c_oi, "put_oi": p_oi, "n": n, "exps": [c["exp"] for c in chains],
                "regime": "Positive gamma (pinning)" if net_gex > 0 else "Negative gamma (trending)",
                "totals": totals, "top_strikes": top_strikes,
                "avg_iv": avg_iv, "avg_iv_vs_hv": avg_iv - hv, "hv": hv,
                "opex": opex_info, "weight_source": weight_source,
                "weight_label": weight_label, "pc_label": pc_label,
                "confidence_label": confidence_label, "confidence_color": confidence_color,
                "directional_quality": "full" if weight_source == "openInterest" else "proxy",
                "weight_note": "" if weight_source == "openInterest" else "Volume-weighted fallback — Yahoo open interest is zero across sampled expiries.",
                "sampled_rows": sampled_rows, "_src": "Yahoo+BS"}
    except Exception as e:
        log("yfinance", symbol, "FAIL", str(e)); return None

# ── HTML helpers ──────────────────────────────────────────────────────────────
def fk(n):
    if n is None: return "N/A"
    if abs(n)>=1e9: return f"${n/1e9:,.1f}B"
    if abs(n)>=1e6: return f"${n/1e6:,.1f}M"
    if abs(n)>=1e3: return f"${n/1e3:,.1f}K"
    return f"${n:,.0f}"

def fb(rate):
    if rate is None: return "N/A","#888","No data"
    p=rate*100
    if p>0.03: return f"{p:.4f}%","#E24B4A","LONGS CROWDED"
    elif p>0.01: return f"{p:.4f}%","#EF9F27","Slight long crowding"
    elif p<-0.03: return f"{p:.4f}%","#1D9E75","SHORTS CROWDED"
    elif p<-0.01: return f"{p:.4f}%","#1D9E75","Slight short crowding"
    else: return f"{p:.4f}%","#378ADD","Neutral"

def gex_bars(levels):
    if not levels: return ""
    mx = max(abs(v) for _,v in levels) or 1
    h = ""
    for s,v in sorted(levels, key=lambda x: x[0]):
        c = "#1D9E75" if v>0 else "#E24B4A"
        w = min(abs(v)/mx*100,100)
        tag = "support" if v>0 else "resistance"
        h += f'<div style="display:flex;align-items:center;gap:6px;margin:1px 0;font-size:11px"><span style="min-width:70px;text-align:right;font-family:monospace">${s:,.0f}</span><div style="height:5px;width:{w:.0f}%;background:{c};border-radius:2px;min-width:2px"></div><span style="color:#aaa;font-size:10px">{tag}</span></div>'
    return h

def sparkline_svg(closes):
    if not closes or len(closes) < 2: return ""
    mn, mx = min(closes), max(closes)
    rng = mx - mn or 1; w, h = 100, 24
    pts = [f"{i/(len(closes)-1)*w:.1f},{h-(c-mn)/rng*h:.1f}" for i, c in enumerate(closes)]
    color = "#1D9E75" if closes[-1] >= closes[0] else "#E24B4A"
    return f'<svg width="{w}" height="{h}" style="vertical-align:middle"><polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'

def bar_html(val, mx, max_w=80):
    if mx == 0: return ""
    w = min(abs(val)/mx*max_w, max_w)
    c = "#1D9E75" if val >= 0 else "#E24B4A"
    return f'<div style="height:5px;width:{w:.0f}px;background:{c};border-radius:2px;min-width:2px;display:inline-block"></div>'

def build_macro_card(macro):
    if not macro:
        return '''<div class="card"><div style="display:flex;justify-content:space-between;margin-bottom:6px">
          <div><b style="font-size:16px">DXY</b> <span style="color:#E24B4A;font-size:12px">[FAIL]</span></div>
          <span class="tag">Yahoo+FRED</span></div>
          <div style="font-size:10px;color:#888">USD macro data unavailable.</div></div>'''
    dxy = macro.get("dxy")
    broad = macro.get("broad_usd")
    tone = macro["label"]
    tone_color = macro["color"]

    dxy_symbol = dxy.get("symbol") if dxy else "N/A"
    broad_symbol = broad.get("series") if broad else "N/A"
    dxy_latest = f"{dxy['latest']:.2f}" if dxy else "N/A"
    broad_latest = f"{broad['latest']:.2f}" if broad else "N/A"
    dxy_color = _color_pct(dxy.get("pct_5d") if dxy else None)
    broad_color = _color_pct(broad.get("pct_5d") if broad else None)

    return f'''<div class="card">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <div><b style="font-size:16px">DXY</b> <span style="color:#888;font-size:13px">{dxy_latest} <span class="tag">[LIVE]</span></span></div>
        <span class="tag">Yahoo+FRED</span></div>
      <div class="metric-row" style="margin-bottom:5px">
        <div class="metric">
          <div class="metric-label">DXY proxy {dxy_symbol}</div>
          <div class="metric-value" style="color:{dxy_color}">{_fmt_pct(dxy.get("pct_5d") if dxy else None)} 5d</div>
          <div style="font-size:10px;color:#888">{_fmt_pct(dxy.get("pct_20d") if dxy else None)} 20d</div>
        </div>
        <div class="metric">
          <div class="metric-label">Broad USD {broad_symbol}</div>
          <div class="metric-value" style="color:{broad_color}">{_fmt_pct(broad.get("pct_5d") if broad else None)} 5d</div>
          <div style="font-size:10px;color:#888">{_fmt_pct(broad.get("pct_20d") if broad else None)} 20d</div>
        </div>
        <div class="metric">
          <div class="metric-label">Macro read</div>
          <div class="metric-value" style="color:{tone_color}">{tone}</div>
          <div style="font-size:10px;color:#888">{macro.get("reason","")}</div>
        </div>
      </div>
      <div style="padding:3px 8px;border-radius:4px;font-size:10px;color:{tone_color};background:{tone_color}08">{macro.get("summary","")}</div>
    </div>'''

# ── Checklist ─────────────────────────────────────────────────────────────────
def _market_bias(label, gex_data, hist_data=None, crypto_data=None, macro=None):
    """Compute directional bias for a specific market open. Returns (direction, reasons, color)."""
    signals = []
    if gex_data:
        asset = gex_data.get("symbol", label)
        options_confident = gex_data.get("weight_source", "openInterest") == "openInterest"
        if options_confident:
            if gex_data["net_gex"] > 0:
                signals.append(("bull", "positive γ — pinning/mean-revert"))
            else:
                signals.append(("bear", "negative γ — trend/breakout"))
            if gex_data.get("pc_ratio") is not None:
                pc = gex_data["pc_ratio"]
                if pc > 0.7: signals.append(("bear", f"P/C {pc:.2f} — put heavy"))
                elif pc < 0.3: signals.append(("bull", f"P/C {pc:.2f} — call heavy"))
        elif gex_data.get("weight_source") == "volume":
            signals.append(("neutral", f"{asset} options proxy only — volume fallback"))
        if gex_data.get("opex"):
            opex = gex_data["opex"]
            if isinstance(opex, list):
                for o in opex:
                    if o.get("type") in ("MONTHLY", "QUARTERLY") and o.get("show_alert"):
                        signals.append(("bear" if o["atm_put_oi"] > o["atm_call_oi"] else "bull",
                                        f"{o['type']} OpEx {o['exp']} — {o['atm_total_oi']:,} ATM OI expiring"))
            elif isinstance(opex, dict):
                share = f" ({opex['share_pct']:.1f}% OI)" if opex.get("share_pct") is not None else ""
                signals.append(("neutral", f"Crypto OpEx {opex['window']} — {opex['total_oi']:,.0f} contracts{share}"))
    if hist_data:
        if hist_data["pct_chg"] > 0.5: signals.append(("bull", f"5d momentum +{hist_data['pct_chg']:.1f}%"))
        elif hist_data["pct_chg"] < -0.5: signals.append(("bear", f"5d momentum {hist_data['pct_chg']:.1f}%"))
    if gex_data and gex_data.get("avg_iv_vs_hv") is not None:
        iv_hv = gex_data["avg_iv_vs_hv"]
        if gex_data.get("weight_source", "openInterest") == "openInterest":
            if iv_hv > 0.10: signals.append(("bear", f"IV−HV +{iv_hv*100:.0f}% — fear elevated"))
            elif iv_hv < -0.05: signals.append(("bull", f"IV−HV {iv_hv*100:.0f}% — vol cheap"))
    if crypto_data:
        for sym, thresh in [("BTC", 0.02), ("ETH", 0.02)]:
            bn = crypto_data.get(sym, {}).get("binance")
            p = crypto_data.get(sym, {}).get("perp")
            # prefer Binance (more representative), fall back to Deribit
            if bn:
                fr = bn["funding_rate"] * 100
                src = "Binance"
            else:
                fallback_rate = _perp_funding_rate(p)
                if fallback_rate is None:
                    continue
                fr = fallback_rate * 100
                src = "Deribit"
            if fr > thresh: signals.append(("bear", f"{sym} funding +{fr:.4f}% ({src}) — longs crowded"))
            elif fr < -thresh: signals.append(("bull", f"{sym} funding {fr:.4f}% ({src}) — shorts crowded"))
    if macro:
        if macro.get("bias") == "bull":
            signals.append(("bear", "USD strong — macro headwind"))
        elif macro.get("bias") == "bear":
            signals.append(("bull", "USD weak — macro tailwind"))

    bulls = sum(1 for d, _ in signals if d == "bull")
    bears = sum(1 for d, _ in signals if d == "bear")
    if bulls > bears:
        return "LONG", signals, "#1D9E75"
    elif bears > bulls:
        return "SHORT", signals, "#E24B4A"
    else:
        return "NEUTRAL", signals, "#EF9F27"

def build_narrative(crypto, us_data):
    """Generate professional desk-note style market regime analysis from real data."""
    spy = next((g for s,_,g in us_data if s=="SPY" and g), None)
    spy_h = next((h for s,h,_ in us_data if s=="SPY" and h), None)
    gld = next((g for s,_,g in us_data if s=="GLD" and g), None)
    gld_h = next((h for s,h,_ in us_data if s=="GLD" and h), None)
    uso = next((g for s,_,g in us_data if s=="USO" and g), None)
    uso_h = next((h for s,h,_ in us_data if s=="USO" and h), None)
    btc_g = crypto.get("BTC",{}).get("gex")
    btc_p = crypto.get("BTC",{}).get("perp")
    eth_g = crypto.get("ETH",{}).get("gex")

    paras = []

    # 1. SPY regime narrative
    if spy and spy_h:
        pc = spy.get("pc_ratio")
        iv = spy["avg_iv"]*100
        hv = spy["hv"]*100
        iv_hv = spy["avg_iv_vs_hv"]*100
        net_gex = spy["net_gex"]
        regime = spy["regime"]
        pct = spy_h["pct_chg"]
        flip = spy.get("gamma_flip")
        pain = spy.get("max_pain")

        # P/C interpretation
        pc_str = ""
        if pc is not None:
            if pc > 1.2:
                pc_str = f"The SPY P/C at {pc:.2f} is an extreme reading that historically precedes either a sharp capitulation flush (puts pay off, real washout) or a contrarian reversal where excessive fear marks the bottom."
            elif pc > 0.8:
                pc_str = f"SPY P/C at {pc:.2f} shows elevated put demand — hedging activity is above average, consistent with cautious positioning ahead of potential moves."
            elif pc < 0.4:
                pc_str = f"SPY P/C at {pc:.2f} is unusually low — call-heavy positioning suggests complacency. Historically, extreme low P/C readings precede mean-reversion pullbacks."
            else:
                pc_str = f"SPY P/C at {pc:.2f} is balanced — no extreme positioning signal."

        # IV vs HV interpretation
        vol_str = ""
        if iv_hv > 15:
            vol_str = f"IV at {iv:.1f}% vs HV at {hv:.1f}% (spread +{iv_hv:.1f}%) signals the market is pricing significantly more risk than realized moves justify. This fear premium often marks tradeable lows — but can also precede the event that justifies the premium."
        elif iv_hv > 5:
            vol_str = f"IV at {iv:.1f}% vs HV at {hv:.1f}% (spread +{iv_hv:.1f}%) shows mild fear premium. Options are somewhat expensive relative to actual moves."
        elif iv_hv < -5:
            vol_str = f"IV at {iv:.1f}% below HV at {hv:.1f}% (spread {iv_hv:.1f}%) is unusual — vol is cheap. This complacency typically resolves with a sharp move that reprices options."
        else:
            vol_str = f"IV at {iv:.1f}% roughly matches HV at {hv:.1f}% — vol is fairly priced with no extreme signal."

        # Gamma regime
        if net_gex > 0:
            gex_str = f"Positive gamma regime means dealer hedging dampens moves — expect pinning near ${flip:,.0f}" if flip else "Positive gamma regime means dealer hedging dampens moves — expect range-bound action near key strikes"
            gex_str += f". Max pain at ${pain:,.0f} acts as a gravitational center." if pain else "."
        else:
            gex_str = "Negative gamma regime means dealer hedging amplifies moves — trend-following setups are favored."
            if flip: gex_str += f" Gamma flip at ${flip:,.0f} is the key level: above it dealers stabilize, below they accelerate selling."
            if pain: gex_str += f" Max pain at ${pain:,.0f} could act as a magnet on expiration."

        # 5d momentum
        if abs(pct) > 2:
            mom_str = f"SPY {pct:+.1f}% over 5 days is a significant move. "
            if pct < -2: mom_str += "Sharp drawdowns of this magnitude in negative gamma often see continuation before exhaustion."
            else: mom_str += "Strong rallies in this regime tend to face resistance at gamma flip levels."
        elif abs(pct) > 0.5:
            mom_str = f"SPY at {pct:+.1f}% (5d) shows moderate directional pressure."
        else:
            mom_str = f"SPY flat at {pct:+.1f}% (5d) — range-bound, waiting for catalyst."

        # Bull/bear counter-argument
        bulls_case, bears_case = [], []
        if net_gex > 0: bulls_case.append("positive gamma supports dip-buying")
        if pc and pc > 0.8: bulls_case.append(f"high P/C ({pc:.2f}) is contrarian bullish")
        if iv_hv > 10: bulls_case.append(f"fear premium (+{iv_hv:.0f}%) often marks lows")
        if pct > 0.5: bulls_case.append(f"5d momentum positive (+{pct:.1f}%)")
        if net_gex < 0: bears_case.append("negative gamma amplifies selloffs")
        if pc and pc > 1.0: bears_case.append(f"extreme P/C ({pc:.2f}) may reflect real hedging need")
        if pct < -0.5: bears_case.append(f"5d slide ({pct:.1f}%) shows selling pressure")
        if iv_hv > 5: bears_case.append(f"IV still elevated — market expects more downside")

        counter = ""
        if bulls_case and bears_case:
            counter = f"<br><span style='color:#1D9E75'>Bulls:</span> {'; '.join(bulls_case)}. <span style='color:#E24B4A'>Bears:</span> {'; '.join(bears_case)}."

        paras.append(f"<b>SPY — {regime}</b><br>{pc_str} {vol_str}<br>{gex_str}<br>{mom_str}{counter}")

    # 2. GLD narrative
    if gld and gld_h:
        pct_g = gld_h["pct_chg"]
        pc_g = gld.get("pc_ratio")
        iv_g = gld["avg_iv"]*100
        if abs(pct_g) > 3:
            gld_str = f"GLD {pct_g:+.1f}% (5d) is a major move for gold. "
            if pct_g < -3: gld_str += "Sharp gold selloffs often coincide with margin calls / risk-off liquidation cascades. Watch for stabilization near "
            else: gld_str += "Strong gold rallies signal flight-to-safety. Watch resistance at "
            if gld.get("gamma_flip"): gld_str += f"gamma flip ${gld['gamma_flip']:,.0f}."
            elif gld.get("max_pain"): gld_str += f"max pain ${gld['max_pain']:,.0f}."
        elif abs(pct_g) > 1:
            gld_str = f"GLD {pct_g:+.1f}% (5d) shows moderate pressure."
        else:
            gld_str = f"GLD flat at {pct_g:+.1f}% (5d)."
        if pc_g and pc_g > 1.0: gld_str += f" P/C at {pc_g:.2f} shows heavy put positioning."
        if iv_g > 30: gld_str += f" IV at {iv_g:.1f}% is elevated — expect outsized moves."
        paras.append(f"<b>GLD</b><br>{gld_str}")

    # 2b. USO (Oil) narrative
    if uso and uso_h:
        pct_o = uso_h["pct_chg"]
        pc_o = uso.get("pc_ratio")
        iv_o = uso["avg_iv"]*100
        hv_o = uso["hv"]*100
        net_o = uso["net_gex"]
        flip_o = uso.get("gamma_flip")
        pain_o = uso.get("max_pain")

        if abs(pct_o) > 5:
            oil_str = f"USO {pct_o:+.1f}% (5d) is a major oil move. "
            if pct_o < -5: oil_str += "Sharp crude selloffs signal demand destruction fears or OPEC supply surprises. "
            else: oil_str += "Strong crude rally signals supply tightening or geopolitical risk premium. "
        elif abs(pct_o) > 2:
            oil_str = f"USO {pct_o:+.1f}% (5d) shows meaningful directional pressure in crude. "
        else:
            oil_str = f"USO flat at {pct_o:+.1f}% (5d) — crude consolidating. "

        if net_o > 0:
            oil_str += "Positive gamma — dealer hedging pins price near key strikes. "
        else:
            oil_str += "Negative gamma — dealer hedging amplifies moves, trend-following favored. "

        iv_hv_o = uso["avg_iv_vs_hv"]*100
        if iv_hv_o > 10:
            oil_str += f"IV at {iv_o:.1f}% vs HV at {hv_o:.1f}% (spread +{iv_hv_o:.1f}%) — market pricing elevated crude risk. "
        elif iv_hv_o < -5:
            oil_str += f"IV at {iv_o:.1f}% below HV at {hv_o:.1f}% — vol is cheap, complacency in crude options. "

        if flip_o: oil_str += f"Gamma flip at ${flip_o:,.2f}. "
        if pain_o: oil_str += f"Max pain at ${pain_o:,.2f}. "
        if pc_o and pc_o > 1.0: oil_str += f"P/C at {pc_o:.2f} — heavy put hedging in crude."
        elif pc_o and pc_o < 0.4: oil_str += f"P/C at {pc_o:.2f} — call-heavy, bullish crude speculation."
        paras.append(f"<b>USO (Crude Oil)</b><br>{oil_str}")

    # 3. Crypto narrative
    if btc_g and btc_p:
        fr = (btc_p.get("current_funding",0) or 0)*100
        spot = btc_g["spot"]
        pc_b = btc_g.get("pc_ratio")
        flip_b = btc_g.get("gamma_flip")
        pain_b = btc_g.get("max_pain")
        net_b = btc_g["net_gex"]

        btc_str = f"BTC at ${spot:,.0f} with "
        if net_b > 0:
            btc_str += f"positive gamma — dealers are short calls, hedging stabilizes price. "
        else:
            btc_str += f"negative gamma — dealers amplify moves, trend-following favored. "
        if abs(fr) < 0.01:
            btc_str += "Funding neutral — no crowding signal. "
        elif fr > 0.02:
            btc_str += f"Funding at +{fr:.4f}% shows longs are paying — crowded long positioning increases flush risk. "
        elif fr < -0.02:
            btc_str += f"Funding at {fr:.4f}% shows shorts paying — short squeeze risk elevated. "
        if pc_b is not None:
            btc_str += f"P/C at {pc_b:.2f} "
            if pc_b > 0.7: btc_str += "shows elevated put hedging — fear in the options market."
            elif pc_b < 0.3: btc_str += "is very call-heavy — bullish speculation dominates."
            else: btc_str += "is balanced."
        if flip_b and pain_b:
            btc_str += f" Key levels: gamma flip ${flip_b:,.0f} (above = stability), max pain ${pain_b:,.0f} (gravitational center)."
        # OpEx
        if btc_g.get("opex"):
            o = btc_g["opex"]
            share = f" ({o['share_pct']:.1f}% of live OI)" if o.get("share_pct") is not None else ""
            btc_str += f" <b style='color:#E24B4A'>OpEx alert:</b> {o['total_oi']:,.0f} contracts expire in {o['window']}{share} — possible gamma unwind / vol pickup."
        paras.append(f"<b>BTC</b><br>{btc_str}")

    if not paras:
        return ""

    body = "</div><div style='margin-top:8px;line-height:1.6;font-size:11px'>".join(paras)
    return f'''<div class="card" style="margin-bottom:10px">
      <div class="card-title">Market regime analysis</div>
      <div style="line-height:1.6;font-size:11px">{body}</div></div>'''

def build_checklist(crypto, us_data, conviction, now, macro=None):
    steps = []
    h = now.hour
    weekday = now.weekday()
    is_market_day = weekday < 5
    power_hours = [(5,0,"Tokyo/Seoul open"),(6,30,"Shanghai/HK open"),(13,0,"London open"),(19,30,"US open"),(0,0,"US power hour"),(1,0,"US close")]

    # Step 1: Session timing — rendered by JS (live countdown)
    steps.append(("1","Session timing", '<span id="cl-timing">calculating...</span>', "#378ADD"))

    # Step 2: OpEx warning (if any)
    opex_warnings = []
    for sym in ["BTC", "ETH"]:
        g = crypto.get(sym, {}).get("gex")
        if g and g.get("opex"):
            o = g["opex"]
            share = f" ({o['share_pct']:.1f}% of live OI)" if o.get("share_pct") is not None else ""
            opex_warnings.append(f"<b>{sym}</b> {o['total_oi']:,.0f} contracts expire in {o['window']}{share} — possible gamma unwind / vol pickup")
    for sym, hist, greeks in us_data:
        if greeks and greeks.get("opex"):
            for o in greeks["opex"]:
                if not o.get("show_alert"):
                    continue
                tag = f"<b style='color:#E24B4A'>{o['type']}</b>" if o["type"] in ("MONTHLY","QUARTERLY") else o["type"]
                day = "TODAY" if o["is_today"] else "TOMORROW"
                opex_warnings.append(f"<b>{sym}</b> {tag} OpEx {day} ({o['exp']}) — {o['total_oi']:,} total OI | {o['atm_total_oi']:,} near ATM ({o['atm_share_pct']:.1f}%) | C:{o['atm_call_oi']:,} P:{o['atm_put_oi']:,}")
    if opex_warnings:
        steps.append(("2","OpEx alert", "<br>".join(opex_warnings), "#E24B4A"))
    else:
        steps.append(("2","OpEx", "No major expirations today/tomorrow", "#888"))

    # Step 3: Pre-open bias per market — directional call BEFORE each open
    spy_greeks = next((g for s,_,g in us_data if s=="SPY" and g), None)
    spy_hist = next((h for s,h,_ in us_data if s=="SPY" and h), None)
    gld_greeks = next((g for s,_,g in us_data if s=="GLD" and g), None)
    gld_hist = next((h for s,h,_ in us_data if s=="GLD" and h), None)
    uso_greeks = next((g for s,_,g in us_data if s=="USO" and g), None)
    uso_hist = next((h for s,h,_ in us_data if s=="USO" and h), None)
    btc_gex = crypto.get("BTC",{}).get("gex")
    eth_gex = crypto.get("ETH",{}).get("gex")

    market_biases = []

    # Tokyo/Seoul (05:00 UZT) — crypto-driven, use BTC/ETH data
    d, sigs, c = _market_bias("Tokyo/Seoul", btc_gex, crypto_data=crypto, macro=macro)
    key_lvls = []
    if btc_gex:
        if btc_gex.get("gamma_flip"): key_lvls.append(f"BTC flip ${btc_gex['gamma_flip']:,.0f}")
        if btc_gex.get("max_pain"): key_lvls.append(f"pain ${btc_gex['max_pain']:,.0f}")
    reasons = " · ".join(r for _,r in sigs[:3])
    lvl_str = f" → {' | '.join(key_lvls)}" if key_lvls else ""
    market_biases.append(f"<div id='bias-tokyo' style='margin-bottom:6px;transition:all .3s'><b>Tokyo/Seoul 05:00</b> — <span style='color:{c};font-weight:700'>{d}</span>{lvl_str}<br><span style='color:#888;font-size:10px'>{reasons}</span></div>")

    # London (13:00 UZT) — crypto + equity overlap
    d, sigs, c = _market_bias("London", spy_greeks, spy_hist, crypto, macro)
    key_lvls = []
    if spy_greeks and spy_greeks.get("gamma_flip"): key_lvls.append(f"SPY flip ${spy_greeks['gamma_flip']:,.0f}")
    if spy_greeks and spy_greeks.get("max_pain"): key_lvls.append(f"SPY pain ${spy_greeks['max_pain']:,.0f}")
    if btc_gex and btc_gex.get("gamma_flip"): key_lvls.append(f"BTC flip ${btc_gex['gamma_flip']:,.0f}")
    reasons = " · ".join(r for _,r in sigs[:3])
    lvl_str = f" → {' | '.join(key_lvls)}" if key_lvls else ""
    market_biases.append(f"<div id='bias-london' style='margin-bottom:6px;transition:all .3s'><b>London 13:00</b> — <span style='color:{c};font-weight:700'>{d}</span>{lvl_str}<br><span style='color:#888;font-size:10px'>{reasons}</span></div>")

    # US open (19:30 UZT) — full equity focus
    d, sigs, c = _market_bias("US", spy_greeks, spy_hist, crypto, macro)
    key_lvls = []
    if spy_greeks and spy_greeks.get("gamma_flip"): key_lvls.append(f"SPY flip ${spy_greeks['gamma_flip']:,.0f}")
    if spy_greeks and spy_greeks.get("max_pain"): key_lvls.append(f"SPY pain ${spy_greeks['max_pain']:,.0f}")
    if gld_greeks and gld_greeks.get("gamma_flip"): key_lvls.append(f"GLD flip ${gld_greeks['gamma_flip']:,.0f}")
    if uso_greeks and uso_greeks.get("gamma_flip"): key_lvls.append(f"USO flip ${uso_greeks['gamma_flip']:,.2f}")
    if uso_greeks and uso_greeks.get("max_pain"): key_lvls.append(f"USO pain ${uso_greeks['max_pain']:,.2f}")
    reasons = " · ".join(r for _,r in sigs[:4])
    lvl_str = f" → {' | '.join(key_lvls)}" if key_lvls else ""
    market_biases.append(f"<div id='bias-us' style='margin-bottom:6px;transition:all .3s'><b>US open 19:30</b> — <span style='color:{c};font-weight:700'>{d}</span>{lvl_str}<br><span style='color:#888;font-size:10px'>{reasons}</span></div>")

    steps.append(("3","Pre-open bias", "".join(market_biases), "#7F77DD"))

    # Step 4: Overall conviction
    if conviction:
        c = conviction
        bu = c.get("bulls", 0); be = c.get("bears", 0); td = c.get("total_dir", 1) or 1
        pct = bu/td*100 if td else 0
        conf_color = "#1D9E75" if c.get("level",0)>=2 else "#EF9F27" if c.get("level",0)==0 else "#E24B4A"
        steps.append(("4","Conviction", f"<b style='color:{c['color']}'>{c['label']}</b> — {c['sizing']} <span style='color:{conf_color};font-size:10px'>({bu} bull · {be} bear · level {c.get('level',0):+d}/3)</span>", c["color"]))
    else:
        steps.append(("4","Conviction","Insufficient data — sit out","#888"))

    # Step 5: Action plan with real levels
    if conviction:
        parts = [f"BTC flip ${btc_gex['gamma_flip']:,.0f}" if btc_gex and btc_gex.get("gamma_flip") else None,
                 f"BTC pain ${btc_gex['max_pain']:,.0f}" if btc_gex and btc_gex.get("max_pain") else None,
                 f"SPY flip ${spy_greeks['gamma_flip']:,.0f}" if spy_greeks and spy_greeks.get("gamma_flip") else None,
                 f"SPY pain ${spy_greeks['max_pain']:,.0f}" if spy_greeks and spy_greeks.get("max_pain") else None,
                 f"USO flip ${uso_greeks['gamma_flip']:,.2f}" if uso_greeks and uso_greeks.get("gamma_flip") else None,
                 f"USO pain ${uso_greeks['max_pain']:,.2f}" if uso_greeks and uso_greeks.get("max_pain") else None]
        lvl = " | ".join(filter(None, parts)) or "[FAIL]"
        label = conviction["label"]
        if "BULL" in label: action = f"Long on dips → {lvl}"
        elif "BEAR" in label: action = f"Short on bounces → {lvl}"
        else: action = f"Half size or sit out → {lvl}"
        steps.append(("5","Action plan", action, conviction["color"]))

    html = '<div class="card" style="margin-bottom:10px"><div class="card-title">Pre-session bias — clear instructions before each market open</div>'
    for num,title,body,color in steps:
        html += f'''<div style="display:flex;gap:10px;margin-bottom:7px;align-items:flex-start">
          <div style="width:20px;height:20px;border-radius:50%;background:{color}18;border:1.5px solid {color};display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:9px;font-weight:700;color:{color}">{num}</div>
          <div><div style="font-size:9px;font-weight:600;color:#999;text-transform:uppercase;letter-spacing:.5px">{title}</div>
          <div style="font-size:11px;margin-top:1px;line-height:1.5">{body}</div></div></div>'''
    html += '</div>'
    return html

# ── Power hour JS ─────────────────────────────────────────────────────────────
POWER_HOUR_JS = """
<div id="ph-banner" class="card" style="padding:8px 14px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center">
  <span id="ph-label" style="font-weight:600;font-size:12px"></span>
  <span id="ph-countdown" style="font-family:'SF Mono',Monaco,monospace;font-size:14px;font-weight:700"></span>
</div>
<script>
(function(){
  var PH=[[5,0,"Tokyo/Seoul open"],[6,30,"Shanghai/HK open"],[13,0,"London open"],[19,30,"US open (main)"],[0,0,"US power hour"],[1,0,"US close"]];
  var pad=function(n){return n<10?'0'+n:n;};
  function tick(){
    var now=new Date();
    var uztMs=now.getTime()+5*3600000;
    var uzt=new Date(uztMs);
    var dow=uzt.getUTCDay(),hh=uzt.getUTCHours(),mm=uzt.getUTCMinutes(),ss=uzt.getUTCSeconds();
    var cur=hh*3600+mm*60+ss;
    var b=document.getElementById('ph-banner'),l=document.getElementById('ph-label'),c=document.getElementById('ph-countdown');
    if(dow===0||dow===6){
      var d=dow===0?1:2,s=d*86400-cur+5*3600;
      if(s<0) s+=86400;
      var h2=Math.floor(s/3600),m2=Math.floor((s%3600)/60),s2=s%60;
      l.textContent='Weekend — next: Monday Tokyo/Seoul open';
      c.textContent=h2+'h '+pad(m2)+'m '+pad(s2)+'s';
      b.style.background='#f8f8f6';b.style.borderColor='#e5e5e5';c.style.color='#888';return;
    }
    var best=null,bestSec=99999;
    for(var i=0;i<PH.length;i++){var t=PH[i][0]*3600+PH[i][1]*60;var diff=t-cur;if(diff<0)diff+=86400;if(diff<bestSec){bestSec=diff;best=PH[i];}}
    var wraps=(best[0]*3600+best[1]*60)<cur;
    if(wraps){var nd=dow%7+1;if(nd===6){bestSec+=2*86400;best=[5,0,"Tokyo/Seoul open"];}else if(nd===0){bestSec+=86400;best=[5,0,"Tokyo/Seoul open"];}}
    var h2=Math.floor(bestSec/3600),m2=Math.floor((bestSec%3600)/60),s2=bestSec%60;
    var ts=h2>0?h2+'h '+pad(m2)+'m '+pad(s2)+'s':pad(m2)+'m '+pad(s2)+'s';
    l.textContent=bestSec<=1800?'POWER HOUR — '+best[2]:'Next — '+best[2];
    c.textContent=ts;
    if(bestSec<=1800){b.style.background='#E24B4A12';b.style.borderColor='#E24B4A44';c.style.color='#E24B4A';}
    else if(bestSec<=3600){b.style.background='#EF9F2712';b.style.borderColor='#EF9F2744';c.style.color='#EF9F27';}
    else{b.style.background='#f8f8f6';b.style.borderColor='#e5e5e5';c.style.color='#1a1a1a';}
  }
  tick();setInterval(tick,1000);
})();
</script>
"""

# ── Main HTML build ───────────────────────────────────────────────────────────
def build_html(crypto, us_data, conviction, macro=None, serve_mode=False):
    now = datetime.now(UZT)
    fetch_epoch = int(now.timestamp())
    h = now.hour; wd = now.weekday()
    if wd >= 5: sess, sc = "Weekend", "#888"
    elif 0<=h<5:  sess,sc = "Off-hours","#888"
    elif 5<=h<7:  sess,sc = "Tokyo / Seoul","#EF9F27"
    elif 7<=h<13: sess,sc = "Asia (Tokyo/Seoul/Shanghai)","#378ADD"
    elif 13<=h<19:sess,sc = "London / Pre-US","#7F77DD"
    elif 19<=h<24:sess,sc = "US session","#E24B4A"
    else:         sess,sc = "Off-hours","#888"

    checklist = build_checklist(crypto, us_data, conviction, now, macro=macro)

    # Refresh button + data age bar
    rerun_note = "" if serve_mode else ' <span style="font-size:8px;color:#bbb">(re-run script for fresh data)</span>'
    refresh_bar = f'''
    <!-- Updating banner (hidden by default, shown by JS when fetching) -->
    <div id="updating-banner" style="display:none;background:#1D9E7512;border:1px solid #1D9E7544;border-radius:8px;padding:6px 14px;margin-bottom:8px;font-size:10px;font-weight:600;color:#1D9E75;align-items:center;gap:8px">
      <div style="width:10px;height:10px;border:2px solid #1D9E75;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0"></div>
      Fetching fresh data in background... dashboard will reload automatically when ready.
    </div>
    <style>@keyframes spin{{to{{transform:rotate(360deg)}}}}</style>
    <div class="card" style="padding:6px 14px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">
      <div style="display:flex;align-items:center;gap:8px">
        <button id="refresh-btn" onclick="triggerRefresh()" style="background:#333;color:#fff;border:none;border-radius:4px;padding:4px 12px;font-size:10px;font-weight:600;cursor:pointer;font-family:inherit">Refresh data</button>
        <span style="font-size:10px;color:#aaa">Data fetched: <b>{now.strftime("%H:%M:%S")} UZT</b></span>{rerun_note}
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <span style="font-size:10px;color:#aaa">Age:</span>
        <span id="data-age" style="font-family:'SF Mono',Monaco,monospace;font-size:11px;font-weight:600;color:#333">0:00</span>
        {"<span id='next-refresh' style='font-size:9px;color:#bbb;margin-left:8px'></span>" if serve_mode else ""}
      </div>
    </div>
    <script>
    (function(){{
      var fetchTs={fetch_epoch};
      var serve={str(serve_mode).lower()};
      var interval={REFRESH_INTERVAL};
      var polling=false;
      function pad(n){{return n<10?'0'+n:n;}}

      // Age counter (pure display, no reload logic)
      function ageT(){{
        var ago=Math.floor(Date.now()/1000)-fetchTs;
        var m=Math.floor(ago/60),s=ago%60;
        var el=document.getElementById('data-age');
        if(el){{el.textContent=m+':'+pad(s);
          el.style.color=ago>600?'#E24B4A':ago>300?'#EF9F27':'#333';}}
      }}
      setInterval(ageT,1000); ageT();

      // Status polling — only when serve mode
      function pollStatus(){{
        if(!serve) return;
        fetch('/api/status').then(function(r){{return r.json();}}).then(function(d){{
          var banner=document.getElementById('updating-banner');
          var nr=document.getElementById('next-refresh');
          var btn=document.getElementById('refresh-btn');
          if(d.fetching){{
            if(banner){{banner.style.display='flex';}}
            if(btn){{btn.disabled=true;btn.style.opacity='0.5';}}
            if(nr) nr.textContent='updating...';
          }} else {{
            if(banner){{banner.style.display='none';}}
            if(btn){{btn.disabled=false;btn.style.opacity='1';}}
            // New data ready — reload page
            if(d.last_updated > fetchTs){{
              window.location.reload(); return;
            }}
            // Show next refresh countdown
            if(nr){{
              var left=d.next_refresh-Math.floor(Date.now()/1000);
              if(left>0){{var rm=Math.floor(left/60),rs=left%60;nr.textContent='next refresh: '+rm+':'+pad(rs);}}
              else nr.textContent='updating soon...';
            }}
          }}
        }}).catch(function(){{}});
      }}
      if(serve){{pollStatus();setInterval(pollStatus, 5000);}}

      window.triggerRefresh=function(){{
        var banner=document.getElementById('updating-banner');
        var btn=document.getElementById('refresh-btn');
        if(banner){{banner.style.display='flex';}}
        if(btn){{btn.disabled=true;btn.style.opacity='0.5';}}
        if(!serve){{window.location.reload();return;}}
        window.location.href='/refresh';
      }};
    }})();
    </script>'''

    # ── Crypto cards ──
    crypto_cards = ""
    for asset, d in crypto.items():
        perp, gex, bn_data = d.get("perp"), d.get("gex"), d.get("binance")
        spot = gex["spot"] if gex else (perp.get("index_price") if perp else (bn_data.get("mark_price") if bn_data else None))
        if spot is None:
            crypto_cards += f'<div class="card"><b>{asset}</b> <span style="color:#E24B4A;font-size:12px">[FAIL]</span></div>'
            continue
        # Binance funding (primary signal — cross-exchange OI weighted)
        bn_rate = bn_data["funding_rate"] if bn_data else None
        bn_v, bn_c, bn_s = fb(bn_rate)
        # Deribit funding (secondary)
        fv, fc, fs = fb(_perp_funding_rate(perp))
        oi_usd = perp["open_interest"]*spot if perp and perp.get("open_interest") else None
        srcs = []
        if perp: srcs.append("Deribit")
        if bn_data: srcs.append("Binance")
        if gex: srcs.append("Options")
        src_label = "+".join(srcs) if srcs else "N/A"
        gh = ""
        if gex:
            net=gex["net_gex"]; rc="#1D9E75" if net>0 else "#E24B4A"
            rt="POSITIVE γ — mean-revert" if net>0 else "NEGATIVE γ — trend-follow"
            fl=f"${gex['gamma_flip']:,.0f}" if gex.get("gamma_flip") else "N/A"
            mp=f"${gex['max_pain']:,.0f}" if gex.get("max_pain") else "N/A"
            pc=f"{gex['pc_ratio']:.2f}" if gex.get("pc_ratio") is not None else "N/A"
            crypto_opex_badge = ""
            if gex.get("opex"):
                o = gex["opex"]
                share = f' ({o["share_pct"]:.1f}% OI)' if o.get("share_pct") is not None else ""
                crypto_opex_badge = f'<div style="padding:4px 10px;border-radius:4px;font-size:10px;font-weight:600;color:#E24B4A;background:#E24B4A10;margin-bottom:6px">⚡ {o["total_oi"]:,.0f} contracts expire in {o["window"]}{share} — possible gamma unwind / vol pickup</div>'
            gh = f'''<div style="margin-top:10px;padding-top:8px;border-top:1px solid #eee">
              {crypto_opex_badge}
              <div style="font-size:10px;font-weight:600;color:#666;margin-bottom:4px">GEX <span style="color:#bbb;font-weight:400">[CALC {gex["n"]} contracts]</span></div>
              {gex_bars(gex["gex_levels"])}
              <div class="metric-row" style="margin-top:6px">
                <div class="metric"><div class="metric-label">Flip</div><div class="metric-value">{fl}</div></div>
                <div class="metric"><div class="metric-label">Max pain</div><div class="metric-value">{mp}</div></div>
                <div class="metric"><div class="metric-label">P/C</div><div class="metric-value">{pc}</div></div>
              </div>
              <div style="margin-top:5px;padding:3px 8px;border-radius:4px;font-size:10px;font-weight:600;color:{rc};background:{rc}10">{rt}</div></div>'''
        # Choose dominant signal for badge
        dom_c, dom_s = (bn_c, bn_s) if bn_data else (fc, fs)
        crypto_cards += f'''<div class="card">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <div><b style="font-size:16px">{asset}</b> <span style="color:#888;font-size:13px">${spot:,.2f} <span class="tag">[LIVE]</span></span></div>
            <span class="tag">{src_label}</span></div>
          <div class="metric-row" style="margin-bottom:5px">
            <div class="metric">
              <div class="metric-label">BINANCE FUNDING <span class="tag">[LIVE]</span></div>
              <div class="metric-value" style="color:{bn_c}">{bn_v if bn_data else "N/A"}</div>
            </div>
            <div class="metric">
              <div class="metric-label">DERIBIT 8H FUNDING <span class="tag">[LIVE]</span></div>
              <div class="metric-value" style="color:{fc}">{fv}</div>
            </div>
            <div class="metric"><div class="metric-label">OI [LIVE]</div><div class="metric-value">{fk(oi_usd)}</div></div>
          </div>
          <div style="padding:3px 8px;border-radius:4px;font-size:10px;color:{dom_c};background:{dom_c}08">{dom_s}</div>
          {gh}</div>'''

    crypto_cards += build_macro_card(macro)

    # ── US cards with full Greeks ──
    us_cards = ""
    for sym, hist, greeks in us_data:
        if not hist: continue
        sl = sparkline_svg(hist["closes"])
        pchg_col = "#1D9E75" if hist["pct_chg"] > 0 else "#E24B4A"
        daily_rets = " ".join(f'<span style="color:{"#1D9E75" if r>0 else "#E24B4A"}">{r*100:+.2f}%</span>' for r in hist["returns"])
        conf_label = greeks.get("confidence_label", "N/A") if greeks else "N/A"
        conf_color = greeks.get("confidence_color", "#888") if greeks else "#888"

        greeks_section = ""
        if greeks:
            rc = "#1D9E75" if greeks["net_gex"] > 0 else "#E24B4A"
            rt = greeks["regime"]
            fl = f"${greeks['gamma_flip']:,.0f}" if greeks.get("gamma_flip") else "N/A"
            mp_v = f"${greeks['max_pain']:,.0f}" if greeks.get("max_pain") else "N/A"
            pc_v = f"{greeks['pc_ratio']:.2f}" if greeks.get("pc_ratio") is not None else "N/A"
            iv_hv_col = "#E24B4A" if greeks["avg_iv_vs_hv"] > 0.05 else "#1D9E75" if greeks["avg_iv_vs_hv"] < -0.02 else "#888"
            weight_note = f'<div style="margin-bottom:6px;font-size:10px;color:#EF9F27">{greeks["weight_note"]}</div>' if greeks.get("weight_note") else ""

            # Strike-level Greeks table
            strike_rows = ""
            if greeks.get("top_strikes"):
                mx_gex = max((abs(v["gamma"]) for _,v in greeks["top_strikes"]), default=1)
                for s, v in sorted(greeks["top_strikes"], key=lambda x: x[0]):
                    atm = abs(s - hist["spot"]) / hist["spot"] < 0.005
                    strike_rows += f'''<tr style="border-top:1px solid #f0f0ee;{"background:#fffbe6" if atm else ""}">
                      <td style="padding:2px 5px;font-size:10px;font-family:monospace">${s:,.0f}{"*" if atm else ""}</td>
                      <td style="padding:2px 5px;font-size:10px;color:#378ADD">{v["delta"]:+,.0f}</td>
                      <td style="padding:2px 5px;font-size:10px">{bar_html(v["gamma"],mx_gex)} <span style="color:{"#1D9E75" if v["gamma"]>0 else "#E24B4A"}">{v["gamma"]:+,.0f}</span></td>
                      <td style="padding:2px 5px;font-size:10px;color:#888">{v["theta"]:+,.0f}</td>
                      <td style="padding:2px 5px;font-size:10px;color:#7F77DD">{v["vega"]:+,.0f}</td>
                      <td style="padding:2px 5px;font-size:10px;color:#aaa">{v["call_oi"]:,}/{v["put_oi"]:,}</td></tr>'''

            # OpEx badge
            opex_badge = ""
            if greeks.get("opex"):
                for o in greeks["opex"]:
                    if not o.get("show_alert"):
                        continue
                    oc = "#E24B4A" if o["type"] in ("MONTHLY","QUARTERLY") else "#EF9F27"
                    day_label = "TODAY" if o["is_today"] else "TOMORROW"
                    opex_badge += f'<div style="padding:4px 10px;border-radius:4px;font-size:10px;font-weight:600;color:{oc};background:{oc}10;margin-bottom:6px">⚡ {o["type"]} OpEx {day_label} — {o["total_oi"]:,} total OI | {o["atm_total_oi"]:,} near ATM ({o["atm_share_pct"]:.1f}%)</div>'

            greeks_section = f'''
            {opex_badge}
            {weight_note}
            <!-- Vol metrics -->
            <div class="metric-row" style="margin:8px 0">
              <div class="metric"><div class="metric-label">5d HV</div><div class="metric-value">{hist["hv_ann"]*100:.1f}%</div></div>
              <div class="metric"><div class="metric-label">Avg IV</div><div class="metric-value">{greeks["avg_iv"]*100:.1f}%</div></div>
              <div class="metric"><div class="metric-label">IV−HV</div><div class="metric-value" style="color:{iv_hv_col}">{greeks["avg_iv_vs_hv"]*100:+.1f}%</div></div>
              <div class="metric"><div class="metric-label">{greeks.get("pc_label", "P/C")}</div><div class="metric-value">{pc_v}</div></div>
            </div>
            <!-- GEX + key levels -->
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">
              <div>
                <div style="font-size:10px;font-weight:600;color:#666;margin-bottom:3px">GEX levels <span class="tag">[CALC {greeks["n"]} rows · {greeks.get("weight_label", "OI")}]</span></div>
                {gex_bars(greeks["gex_levels"])}
              </div>
              <div>
                <div class="metric-row" style="margin-bottom:4px">
                  <div class="metric"><div class="metric-label">Flip</div><div class="metric-value">{fl}</div></div>
                  <div class="metric"><div class="metric-label">Pain</div><div class="metric-value">{mp_v}</div></div>
                </div>
                <div style="padding:3px 8px;border-radius:4px;font-size:10px;font-weight:600;color:{rc};background:{rc}10;margin-bottom:4px">{rt}</div>
                <div class="metric-row">
                  <div class="metric"><div class="metric-label">Net Delta</div><div class="metric-value" style="color:#378ADD">{greeks["totals"]["delta"]:+,.0f}</div></div>
                  <div class="metric"><div class="metric-label">Net Theta/d</div><div class="metric-value" style="color:#888">{greeks["totals"]["theta"]:+,.0f}</div></div>
                </div>
                <div class="metric-row" style="margin-top:3px">
                  <div class="metric"><div class="metric-label">Net Vega/1%</div><div class="metric-value" style="color:#7F77DD">{greeks["totals"]["vega"]:+,.0f}</div></div>
                </div>
              </div>
            </div>
            <!-- Greeks by strike -->
            <details style="margin-top:4px"><summary style="font-size:10px;color:#aaa;cursor:pointer">Greeks by strike (top {len(greeks.get("top_strikes",[]))} by |GEX|)</summary>
            <table style="width:100%;border-collapse:collapse;margin-top:4px">
              <tr style="background:#f8f8f6;font-size:9px;color:#aaa">
                <th style="padding:2px 5px;text-align:left">Strike</th><th style="text-align:left;padding:2px 5px">Net Δ</th>
                <th style="text-align:left;padding:2px 5px">Net Γ (GEX)</th><th style="text-align:left;padding:2px 5px">Net Θ</th>
                <th style="text-align:left;padding:2px 5px">Net V</th><th style="text-align:left;padding:2px 5px">C/P {greeks.get("weight_label", "OI")}</th></tr>
              {strike_rows}
            </table></details>'''

        us_cards += f'''<div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:6px">
            <div>
              <b style="font-size:16px">{sym}</b>
              <span style="color:#888;font-size:13px">${hist["spot"]:,.2f} <span class="tag">[LIVE]</span></span>
              <span style="color:{pchg_col};font-size:11px;margin-left:4px">{hist["pct_chg"]:+.2f}% (5d)</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px">{sl}<span style="font-size:9px;padding:2px 6px;border-radius:999px;background:{conf_color}12;color:{conf_color};font-weight:700">{conf_label}</span><span class="tag">Yahoo+BS</span></div>
          </div>
          <div style="font-size:9px;color:#bbb;margin-bottom:6px">5d returns: {daily_rets}</div>
          {greeks_section}</div>'''

    # ── Conviction ──
    cv = ""
    if conviction:
        c = conviction
        sigs = "".join(f'<div style="display:flex;align-items:center;gap:5px;margin:1px 0;font-size:11px"><div style="width:5px;height:5px;border-radius:50%;background:{"#1D9E75" if d=="bullish" else "#E24B4A" if d=="bearish" else "#EF9F27"};flex-shrink:0"></div><span>{t}</span></div>' for t,d in c["signals"])
        # 7-level probability scale visual
        scale_labels = [
            ("EXT BEAR", "#6B0000", -3),
            ("BEAR",     "#E24B4A", -2),
            ("SL BEAR",  "#F09595", -1),
            ("NEUTRAL",  "#EF9F27",  0),
            ("SL BULL",  "#5DCAA5",  1),
            ("BULL",     "#1D9E75",  2),
            ("EXT BULL", "#0B6E3D",  3),
        ]
        lvl = c.get("level", 0)
        scale_cells = ""
        for sl_label, sl_color, sl_lvl in scale_labels:
            active = sl_lvl == lvl
            bg = sl_color if active else sl_color + "22"
            border = f"2px solid {sl_color}" if active else f"1px solid {sl_color}44"
            txt_color = "#fff" if active else sl_color
            scale_cells += f'<div style="flex:1;text-align:center;padding:{("6px 2px" if active else "4px 2px")};border-radius:4px;background:{bg};border:{border};font-size:{"9px" if active else "8px"};font-weight:{"800" if active else "500"};color:{txt_color};transition:all .2s">{sl_label}</div>'
        scale_html = f'<div style="display:flex;gap:3px;margin:10px 0 6px">{scale_cells}</div>'
        bu = c.get("bulls", 0); be = c.get("bears", 0); td = c.get("total_dir", 1) or 1
        pct_bull = bu/td*100; pct_bear = be/td*100
        bar_html_conv = f'<div style="display:flex;height:4px;border-radius:2px;overflow:hidden;margin-bottom:6px"><div style="width:{pct_bull:.0f}%;background:#1D9E75"></div><div style="width:{pct_bear:.0f}%;background:#E24B4A"></div></div>'
        cv = f'''<div class="card" style="margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:4px">
            <div><div style="font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:.5px">Session bias</div><div style="font-size:22px;font-weight:800;color:{c["color"]};letter-spacing:-.5px">{c["label"]}</div></div>
            <div style="text-align:right"><div style="font-size:9px;color:#aaa">Sizing</div><div style="font-size:11px;font-weight:600">{c["sizing"]}</div>
            <div style="font-size:9px;color:#aaa;margin-top:2px">{bu} bull · {be} bear signals</div></div></div>
          {scale_html}
          {bar_html_conv}
          {sigs}</div>'''

    links = '<div style="display:flex;flex-wrap:wrap;gap:3px;margin:8px 0">' + "".join(
        f'<a href="{u}" target="_blank" style="font-size:9px;padding:2px 6px;border-radius:3px;background:#f0f0ee;color:#555;text-decoration:none">{l}</a>'
        for l,u in [("CoinGlass Funding","https://www.coinglass.com/FundingRate"),
                    ("Liq Heatmap","https://www.coinglass.com/pro/futures/LiquidationHeatMap"),
                    ("GammaFlip","https://gammaflip.io/"),
                    ("SPX GEX","https://www.barchart.com/stocks/quotes/$SPX/gamma-exposure"),
                    ("GLD GEX","https://www.barchart.com/etfs-funds/quotes/GLD/gamma-exposure"),
                    ("USO GEX","https://www.barchart.com/etfs-funds/quotes/USO/gamma-exposure"),
                    ("Crude CL","https://www.tradingview.com/symbols/NYMEX-CL1!/"),
                    ("CME BTC Gap","https://www.tradingview.com/symbols/CME-BTC1!/"),
                    ("CoinGlass Options","https://www.coinglass.com/options")]) + "</div>"

    lr = "".join(f'<tr><td style="font-size:9px;color:#999;padding:1px 4px">{e["source"]}</td><td style="font-size:9px;padding:1px 4px;font-family:monospace">{e["endpoint"][:35]}</td><td style="font-size:9px;color:{"#1D9E75" if e["status"]=="OK" else "#E24B4A" if e["status"]=="FAIL" else "#378ADD"};font-weight:600;padding:1px 4px">{e["status"]}</td></tr>' for e in fetch_log)

    return f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bias Dashboard — {now.strftime('%H:%M')} UZT</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text','Segoe UI',sans-serif;background:#f5f5f3;color:#1a1a1a;padding:14px;max-width:1100px;margin:0 auto;font-size:13px;-webkit-font-smoothing:antialiased}}
.card{{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:14px 16px}}
.card-title{{font-size:12px;font-weight:700;color:#333;margin-bottom:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}}
.tag{{font-size:8px;color:#bbb;font-weight:400}}
.metric-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(60px,1fr));gap:4px}}
.metric{{background:#f8f8f6;padding:4px 6px;border-radius:4px}}
.metric-label{{font-size:8px;color:#aaa;text-transform:uppercase;letter-spacing:.3px}}
.metric-value{{font-size:11px;font-weight:600}}
details summary{{cursor:pointer;font-size:10px;color:#aaa}}
a:hover{{opacity:.7}}
</style></head><body>
<div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;margin-bottom:8px">
  <div><h1 style="font-size:17px;font-weight:800;letter-spacing:-.3px">Pre-session bias dashboard</h1>
  <div id="live-clock" style="font-size:10px;color:#aaa">{now.strftime('%A %B %d, %Y — %H:%M:%S')} UZT</div></div>
<script>
(function(){{
  var days=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
  var months=['January','February','March','April','May','June','July','August','September','October','November','December'];
  function pad(n){{return n<10?'0'+n:n;}}
  function uztClock(){{
    var u=new Date(Date.now()+5*3600000);
    var utc=new Date(u.getUTCFullYear(),u.getUTCMonth(),u.getUTCDate(),u.getUTCHours(),u.getUTCMinutes(),u.getUTCSeconds());
    var s=days[utc.getDay()]+' '+months[utc.getMonth()]+' '+pad(utc.getDate())+', '+utc.getFullYear()+' \u2014 '+pad(utc.getHours())+':'+pad(utc.getMinutes())+':'+pad(utc.getSeconds())+' UZT';
    var el=document.getElementById('live-clock');
    if(el) el.textContent=s;
  }}
  uztClock();setInterval(uztClock,1000);
}})();
</script>
  <span id="session-badge" style="padding:3px 10px;border-radius:4px;font-size:10px;font-weight:600;color:{sc};background:{sc}12">{sess}</span></div>
{refresh_bar}
    {POWER_HOUR_JS}
    {checklist}
<script>
(function(){{
  var PH2=[[5,0,"Tokyo/Seoul open"],[6,30,"Shanghai/HK open"],[13,0,"London open"],[19,30,"US open"],[0,0,"US power hour"],[1,0,"US close"]];
  var SESS=[[0,5,"Off-hours","#888"],[5,7,"Tokyo / Seoul","#EF9F27"],[7,13,"Asia (Tokyo/Seoul/Shanghai)","#378ADD"],[13,19,"London / Pre-US","#7F77DD"],[19,24,"US session","#E24B4A"]];
  var MKTIDS=["bias-tokyo","bias-london","bias-us"];
  var MKTH=[[5,0],[13,0],[19,30]];
  function pad(n){{return n<10?'0'+n:n;}}
  function dynTick(){{
    var now=new Date();
    var uztMs=now.getTime()+5*3600000;
    var uzt=new Date(uztMs);
    var dow=uzt.getUTCDay(),hh=uzt.getUTCHours(),mm=uzt.getUTCMinutes(),ss=uzt.getUTCSeconds();
    var cur=hh*3600+mm*60+ss;
    var el=document.getElementById('cl-timing');
    var badge=document.getElementById('session-badge');
    // Session badge
    if(badge){{
      if(dow===0||dow===6){{badge.textContent='Weekend';badge.style.color='#888';badge.style.background='#88812';}}
      else{{
        var found=false;
        for(var i=0;i<SESS.length;i++){{
          if(hh>=SESS[i][0]&&hh<SESS[i][1]){{badge.textContent=SESS[i][2];badge.style.color=SESS[i][3];badge.style.background=SESS[i][3]+'12';found=true;break;}}
        }}
        if(!found){{badge.textContent='Off-hours';badge.style.color='#888';badge.style.background='#88812';}}
      }}
    }}
    // Checklist timing
    if(el){{
      if(dow===0||dow===6){{
        var d2=dow===0?1:2,s2=d2*86400-cur+5*3600;if(s2<0)s2+=86400;
        var h2=Math.floor(s2/3600),m2=Math.floor((s2%3600)/60);
        el.innerHTML='Weekend — next: Monday Tokyo/Seoul open in '+h2+'h '+pad(m2)+'m';
      }}else{{
        var best2=null,bestS2=99999;
        for(var i=0;i<PH2.length;i++){{var t=PH2[i][0]*3600+PH2[i][1]*60;var diff=t-cur;if(diff<0)diff+=86400;if(diff<bestS2){{bestS2=diff;best2=PH2[i];}}}}
        var wraps2=(best2[0]*3600+best2[1]*60)<cur;
        if(wraps2){{var nd2=(dow%7)+1;if(nd2===6){{bestS2+=2*86400;best2=[5,0,"Tokyo/Seoul open"];}}else if(nd2===0){{bestS2+=86400;best2=[5,0,"Tokyo/Seoul open"];}}}}
        var h3=Math.floor(bestS2/3600),m3=Math.floor((bestS2%3600)/60),s3=bestS2%60;
        var ts2=h3>0?h3+'h '+pad(m3)+'m '+pad(s3)+'s':pad(m3)+'m '+pad(s3)+'s';
        if(bestS2<=1800){{el.innerHTML="<b style='color:#E24B4A'>"+ts2+' to '+best2[2]+'!</b>';}}
        else if(bestS2<=3600){{el.innerHTML="<b style='color:#EF9F27'>"+ts2+' to '+best2[2]+'</b>';}}
        else{{el.textContent=ts2+' to '+best2[2];}}
      }}
    }}
    // Highlight next market in pre-open bias
    // Map power hour to bias panel: Tokyo=[5,6:30], London=[13], US=[19:30,0,1]
    var nextBias='';
    if(best2){{
      var bh=best2[0];
      if(bh===5||bh===6) nextBias='bias-tokyo';
      else if(bh===13) nextBias='bias-london';
      else if(bh===19||bh===0||bh===1) nextBias='bias-us';
    }}
    for(var j=0;j<MKTIDS.length;j++){{
      var me=document.getElementById(MKTIDS[j]);
      if(!me)continue;
      if(MKTIDS[j]===nextBias){{
        me.style.background='#fffbe6';me.style.borderRadius='6px';me.style.padding='6px 10px';
        me.style.borderLeft='3px solid #EF9F27';
      }}else{{
        me.style.background='';me.style.borderRadius='';me.style.padding='';me.style.borderLeft='';
      }}
    }}
  }}
  dynTick();setInterval(dynTick,1000);
}})();
</script>
{cv}
{links}
    <div style="font-size:10px;font-weight:600;color:#999;margin:10px 0 6px;text-transform:uppercase;letter-spacing:.5px">Crypto & Dollar</div>
<div class="grid" style="margin-bottom:10px">{crypto_cards}</div>
<div style="font-size:10px;font-weight:600;color:#999;margin:10px 0 6px;text-transform:uppercase;letter-spacing:.5px">Equities & Commodities — full Greeks</div>
<div class="grid">{us_cards}</div>
<details style="margin-top:12px"><summary>Data provenance ({len(fetch_log)} calls — {sum(1 for e in fetch_log if e["status"]=="OK")} OK, {sum(1 for e in fetch_log if e["status"]=="FAIL")} FAIL)</summary>
<table style="width:100%;margin-top:4px;border-collapse:collapse"><tr style="background:#f0f0ee"><th style="text-align:left;font-size:9px;padding:2px 4px">Source</th><th style="text-align:left;font-size:9px;padding:2px 4px">Endpoint</th><th style="text-align:left;font-size:9px;padding:2px 4px">Status</th></tr>{lr}</table></details>
<div style="margin-top:8px;text-align:center;font-size:8px;color:#ccc">[LIVE]=API fetched · [CALC]=Black-Scholes from IV · Zero synthesized values</div></body></html>'''

# ── Conviction scorer ─────────────────────────────────────────────────────────
def score(crypto, us_data, macro=None):
    sigs = []
    b = crypto.get("BTC",{})
    bp, bg, bn = b.get("perp"), b.get("gex"), b.get("binance")
    # Funding — prefer Binance (most representative OI), fall back to Deribit
    if bn:
        fr = bn["funding_rate"]
        if fr*100>0.02: sigs.append((f"BTC Binance funding +{fr*100:.4f}%: longs crowded [LIVE]","bearish"))
        elif fr*100<-0.02: sigs.append((f"BTC Binance funding {fr*100:.4f}%: shorts crowded [LIVE]","bullish"))
        else: sigs.append((f"BTC Binance funding {fr*100:.4f}%: neutral [LIVE]","neutral"))
    elif bp and _perp_funding_rate(bp) is not None:
        fr = _perp_funding_rate(bp)
        if fr*100>0.02: sigs.append((f"BTC Deribit funding +{fr*100:.4f}%: longs crowded [LIVE]","bearish"))
        elif fr*100<-0.02: sigs.append((f"BTC Deribit funding {fr*100:.4f}%: shorts crowded [LIVE]","bullish"))
        else: sigs.append((f"BTC Deribit funding {fr*100:.4f}%: neutral [LIVE]","neutral"))
    # ETH Binance funding
    eth_bn = crypto.get("ETH",{}).get("binance")
    if eth_bn:
        fr_e = eth_bn["funding_rate"]
        if fr_e*100>0.02: sigs.append((f"ETH Binance funding +{fr_e*100:.4f}%: longs crowded [LIVE]","bearish"))
        elif fr_e*100<-0.02: sigs.append((f"ETH Binance funding {fr_e*100:.4f}%: shorts crowded [LIVE]","bullish"))
    if bg:
        if bg["net_gex"]>0: sigs.append(("BTC GEX: positive γ [CALC]","neutral"))
        else: sigs.append(("BTC GEX: negative γ [CALC]","neutral"))
        if bg.get("pc_ratio") is not None:
            pc = bg["pc_ratio"]
            if pc>0.7: sigs.append((f"BTC P/C {pc:.2f}: bearish [CALC]","bearish"))
            elif pc<0.3: sigs.append((f"BTC P/C {pc:.2f}: bullish [CALC]","bullish"))
            else: sigs.append((f"BTC P/C {pc:.2f}: balanced [CALC]","neutral"))
    if macro:
        if macro.get("bias") == "bull":
            sigs.append(("USD strong: macro headwind for BTC/commodities/risk [LIVE]","bearish"))
        elif macro.get("bias") == "bear":
            sigs.append(("USD weak: macro tailwind for BTC/commodities/risk [LIVE]","bullish"))
        else:
            sigs.append(("USD mixed: no clear dollar impulse [LIVE]","neutral"))
    for sym, hist, greeks in us_data:
        if not greeks: continue
        oi_backed = greeks.get("weight_source", "openInterest") == "openInterest"
        if sym == "SPY":
            if oi_backed:
                if greeks["net_gex"] > 0: sigs.append(("SPY: positive γ — risk-on [CALC]","bullish"))
                else: sigs.append(("SPY: negative γ — risk-off [CALC]","bearish"))
                iv_hv = greeks["avg_iv_vs_hv"]
                if iv_hv > 0.10: sigs.append((f"SPY IV−HV: +{iv_hv*100:.0f}% — fear elevated [CALC]","bearish"))
                elif iv_hv < -0.05: sigs.append((f"SPY IV−HV: {iv_hv*100:.0f}% — vol cheap [CALC]","bullish"))
                if greeks.get("pc_ratio") is not None:
                    pc_r = greeks["pc_ratio"]
                    if pc_r > 0.7: sigs.append((f"SPY P/C {pc_r:.2f}: put heavy [CALC]","bearish"))
                    elif pc_r < 0.3: sigs.append((f"SPY P/C {pc_r:.2f}: call heavy [CALC]","bullish"))
            else:
                sigs.append(("SPY options: volume proxy only — not scored directionally [CALC]","neutral"))
        if sym == "SPY" and hist:
            if hist["pct_chg"] > 0.5: sigs.append((f"SPY 5d: +{hist['pct_chg']:.1f}% [LIVE]","bullish"))
            elif hist["pct_chg"] < -0.5: sigs.append((f"SPY 5d: {hist['pct_chg']:.1f}% [LIVE]","bearish"))
        if sym == "USO":
            if oi_backed:
                if greeks["net_gex"] > 0: sigs.append(("USO: positive γ — crude pinning [CALC]","neutral"))
                else: sigs.append(("USO: negative γ — crude trending [CALC]","neutral"))
                if greeks.get("pc_ratio") is not None:
                    pc_o = greeks["pc_ratio"]
                    if pc_o > 1.0: sigs.append((f"USO P/C {pc_o:.2f}: put heavy [CALC]","bearish"))
                    elif pc_o < 0.4: sigs.append((f"USO P/C {pc_o:.2f}: call heavy [CALC]","bullish"))
            else:
                sigs.append(("USO options: volume proxy only — informational only [CALC]","neutral"))
            if hist and abs(hist["pct_chg"]) > 3:
                if hist["pct_chg"] > 3: sigs.append((f"USO 5d: +{hist['pct_chg']:.1f}% — crude rally [LIVE]","bullish"))
                else: sigs.append((f"USO 5d: {hist['pct_chg']:.1f}% — crude selloff [LIVE]","bearish"))

    bu = sum(1 for _,s in sigs if s=="bullish")
    be = sum(1 for _,s in sigs if s=="bearish")
    diff = bu - be
    total_dir = bu + be

    # 7-level probability scale based on net directional alignment
    if total_dir == 0: lvl = 0
    else:
        pct = diff / total_dir
        if pct >= 0.70: lvl = 3
        elif pct >= 0.35: lvl = 2
        elif pct > 0.05: lvl = 1
        elif pct >= -0.05: lvl = 0
        elif pct >= -0.35: lvl = -1
        elif pct >= -0.70: lvl = -2
        else: lvl = -3

    LEVELS = {
        3:  ("EXTREME BULL",    "#0B6E3D", "Full size — aggressive longs"),
        2:  ("BULLISH",         "#1D9E75", "Full size — longs"),
        1:  ("SLIGHTLY BULLISH","#5DCAA5", "Normal — scalp longs"),
        0:  ("NEUTRAL",         "#EF9F27", "Half size or sit out"),
        -1: ("SLIGHTLY BEARISH","#F09595", "Normal — scalp shorts"),
        -2: ("BEARISH",         "#E24B4A", "Full size — shorts"),
        -3: ("EXTREME BEAR",    "#6B0000", "Full size — aggressive shorts"),
    }
    label, color, sizing = LEVELS[lvl]
    return {"label": label, "color": color, "sizing": sizing, "signals": sigs, "level": lvl,
            "bulls": bu, "bears": be, "total_dir": total_dir}

# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_all(serve_mode=False):
    """Fetch all data and return HTML string."""
    global fetch_log, _cached_html, _last_fetch_ts, _is_fetching
    with _fetch_lock:
        if _is_fetching:
            print("  [SKIP] Already fetching, skipping duplicate call.")
            return _cached_html
        _is_fetching = True
    try:
        fetch_log = []  # reset log for each fetch cycle
        now = datetime.now(UZT)
        print(f"\n{'='*60}\n  BIAS DASHBOARD v2 — {now.strftime('%A %H:%M UZT')}")
        if VERIFY: print("  ** VERIFY MODE **")
        if serve_mode: print(f"  [SERVE] refresh cycle @ {now.strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

        crypto = {}
        for a in ["BTC","ETH"]:
            print(f"  [{a}] Perpetual...")
            p = deribit_perp(a)
            deribit_rate = _perp_funding_rate(p)
            if p:
                deribit_str = f"{deribit_rate*100:.4f}%" if deribit_rate is not None else "N/A"
                print(f"  [{a}] ${p['index_price']:,.2f}  deribit_8h={deribit_str}")
            bn = binance_funding(a)
            if bn:
                print(f"  [{a}] Binance funding={bn['funding_rate']*100:.4f}%")
            g = None
            if a in ["BTC","ETH"]:
                print(f"  [{a}] Options GEX (~30s)...")
                g = deribit_gex(a)
                if g: print(f"  [{a}] {g['n']} contracts, flip={'${:,.0f}'.format(g['gamma_flip']) if g.get('gamma_flip') else 'N/A'}")
            crypto[a] = {"perp": p, "gex": g, "binance": bn}

        us_data = []
        if not CRYPTO_ONLY and yf:
            for sym in ["SPY","GLD","USO"]:
                print(f"\n  [{sym}] 5-day history...")
                hist = yf_history(sym)
                if hist:
                    print(f"  [{sym}] spot=${hist['spot']:,.2f}  5d={hist['pct_chg']:+.2f}%  HV={hist['hv_ann']*100:.1f}%")
                    print(f"  [{sym}] Computing full Greeks...")
                    greeks = yf_greeks(sym, hist)
                    if greeks:
                        print(f"  [{sym}] {greeks['n']} rows | {greeks['regime']} | weight={greeks.get('weight_label','OI')} | flip={'${:,.0f}'.format(greeks['gamma_flip']) if greeks.get('gamma_flip') else 'N/A'} | IV={greeks['avg_iv']*100:.1f}%")
                        print(f"  [{sym}] Net Δ={greeks['totals']['delta']:+,.0f}  Θ={greeks['totals']['theta']:+,.0f}/d  V={greeks['totals']['vega']:+,.0f}")
                    us_data.append((sym, hist, greeks))
                else:
                    print(f"  [{sym}] FAIL: no history"); us_data.append((sym, None, None))
        elif CRYPTO_ONLY: print("\n  Skipping SPY/GLD (--crypto)")

        macro = usd_macro()
        if macro:
            dxy_desc = _fmt_pct(macro.get("dxy", {}).get("pct_5d")) if macro.get("dxy") else "N/A"
            broad_desc = _fmt_pct(macro.get("broad_usd", {}).get("pct_5d")) if macro.get("broad_usd") else "N/A"
            print(f"\n  [USD] {macro['label']} | DXY 5d={dxy_desc} | Broad USD 5d={broad_desc}")

        conv = score(crypto, us_data, macro=macro)
        html = build_html(crypto, us_data, conv, macro=macro, serve_mode=serve_mode)
        _cached_html = html
        _last_fetch_ts = time.time()

        out = Path("morning_dashboard.html")
        out.write_text(html)
        ok = sum(1 for e in fetch_log if e['status']=='OK')
        fail = sum(1 for e in fetch_log if e['status']=='FAIL')
        print(f"\n  Saved: {out.absolute()}")
        print(f"  Fetches: {ok} OK / {fail} FAIL\n")

        try:
            Path("/mnt/user-data/outputs/morning_dashboard.html").write_text(html)
            Path("/mnt/user-data/outputs/dashboard_v2.py").write_text(Path(__file__).read_text())
        except: pass
        return html
    finally:
        with _fetch_lock:
            _is_fetching = False

# ── HTTP server for --serve mode ──────────────────────────────────────────────
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            payload = json.dumps({
                "last_updated": int(_last_fetch_ts),
                "fetching": _is_fetching,
                "next_refresh": int(_last_fetch_ts + REFRESH_INTERVAL),
            })
            self.wfile.write(payload.encode('utf-8'))
        elif self.path == '/refresh':
            if not _is_fetching:
                threading.Thread(target=fetch_all, args=(True,), daemon=True).start()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            # Serve a waiting page that polls /api/status and reloads when done
            ts_now = int(_last_fetch_ts)
            self.wfile.write(f'''<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Refreshing...</title>
<style>body{{font-family:-apple-system,sans-serif;background:#f5f5f3;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;flex-direction:column;gap:12px}}
.spinner{{width:32px;height:32px;border:3px solid #e5e5e5;border-top-color:#1D9E75;border-radius:50%;animation:spin 0.8s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
</style></head><body>
<div class="spinner"></div>
<div style="font-size:14px;font-weight:600;color:#333">Fetching live data...</div>
<div style="font-size:11px;color:#aaa">Takes ~2 min. Page will update automatically.</div>
<script>
var knownTs = {ts_now};
function poll() {{
    fetch('/api/status').then(r=>r.json()).then(d=>{{
        if (!d.fetching && d.last_updated > knownTs) {{
            window.location.href = '/';
        }} else {{
            setTimeout(poll, 3000);
        }}
    }}).catch(()=>setTimeout(poll, 5000));
}}
setTimeout(poll, 3000);
</script>
</body></html>'''.encode('utf-8'))
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(_cached_html.encode('utf-8'))
    def log_message(self, fmt, *args):
        pass  # suppress HTTP log spam

def refresh_loop():
    """Background thread: re-fetches data every REFRESH_INTERVAL seconds after last fetch completes."""
    while True:
        # Sleep from last known fetch time, not wall clock — avoids drift
        elapsed = time.time() - _last_fetch_ts
        wait = max(0, REFRESH_INTERVAL - elapsed)
        if wait > 0:
            time.sleep(wait)
        print(f"\n  [AUTO-REFRESH] Starting data refresh cycle...")
        try:
            fetch_all(serve_mode=True)
        except Exception as e:
            print(f"  [AUTO-REFRESH] ERROR: {e}")
            _is_fetching = False  # ensure flag is cleared on exception

def serve():
    """Run dashboard as a live-updating local server."""
    print(f"\n  [SERVE] Initial data fetch...")
    fetch_all(serve_mode=True)

    # Start background refresh thread
    t = threading.Thread(target=refresh_loop, daemon=True)
    t.start()
    print(f"  [SERVE] Auto-refresh every {REFRESH_INTERVAL//60} min")

    server = HTTPServer(('localhost', SERVE_PORT), DashboardHandler)
    url = f"http://localhost:{SERVE_PORT}"
    print(f"  [SERVE] Dashboard live at {url}")
    print(f"  [SERVE] Press Ctrl+C to stop\n")

    # Open in browser
    import webbrowser
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  [SERVE] Stopped.")
        server.server_close()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if SERVE:
        serve()
    else:
        fetch_all(serve_mode=False)

if __name__ == "__main__":
    main()
