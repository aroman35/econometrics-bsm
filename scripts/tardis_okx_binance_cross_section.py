import os
import sys
import csv
import gzip
import math
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from glob import glob
from statistics import stdev
import pandas as pd
from tardis_dev import datasets

# ================== USER CONFIG ==================

# Snapshot (UTC)
SNAPSHOT_DATE = "2024-10-01"    # YYYY-MM-DD
SNAPSHOT_TIME_UTC = "12:00:00"  # HH:MM:SS

# OKX options via okex-options + options_chain
EXCHANGE_OPTIONS = "okex-options"
OPTIONS_SYMBOL = "OPTIONS"

# Optional filter by underlying_index inside options_chain (e.g. "BTC-USD" or "BTC-USDT").
# If empty string -> no filter.
TARGET_UNDERLYING_INDEX = ""

# OKX spot BTC-USDT
OKX_SPOT_EXCHANGE = "okex"
OKX_SPOT_SYMBOL = "BTC-USDT"

# Binance spot BTCUSDT
BINANCE_SPOT_EXCHANGE = "binance"
BINANCE_SPOT_SYMBOL = "BTCUSDT"

# OKX perpetual swap BTC-USDT-SWAP
OKX_PERP_EXCHANGE = "okex-swap"
OKX_PERP_SYMBOL = "BTC-USDT-SWAP"

# Binance perp futures btcusdt (USDT-margined)
BINANCE_PERP_EXCHANGE = "binance-futures"
BINANCE_PERP_SYMBOL = "btcusdt"

# Realized vol window (days) for snapshot sigma proxy
REALIZED_VOL_WINDOW_DAYS = 14

# Directories
OUT_DIR = "../data"
DATASETS_DIR = "../datasets"

# Tardis API key
TARDIS_API_KEY = os.getenv("TARDIS_API_KEY", "YOUR_API_KEY_HERE")

# Logging
LOG_LEVEL = logging.INFO

# =================================================

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
for h in list(logger.handlers):
    logger.removeHandler(h)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# ================== HELPERS ==================


def ensure_api_key():
    if not TARDIS_API_KEY or TARDIS_API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError(
            "Tardis API key is not set. "
            "Set TARDIS_API_KEY environment variable or TARDIS_API_KEY in the script."
        )


def snapshot_dt() -> datetime:
    return datetime.fromisoformat(
        f"{SNAPSHOT_DATE}T{SNAPSHOT_TIME_UTC}"
    ).replace(tzinfo=timezone.utc)


def to_microseconds(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def from_microseconds(us: int) -> datetime:
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def tardis_download(exchange: str, data_types, from_date: str, to_date: str, symbols, desc: str):
    """
    Wrapper with logging so heavy downloads are visible.
    """
    logger.info(
        f"Downloading {desc}: exchange={exchange}, data_types={data_types}, "
        f"from={from_date}, to={to_date}, symbols={symbols}"
    )
    datasets.download(
        exchange=exchange,
        data_types=data_types,
        from_date=from_date,
        to_date=to_date,
        symbols=symbols,
        api_key=TARDIS_API_KEY,
        download_dir=DATASETS_DIR,
    )
    logger.info(f"Finished downloading {desc}")


def find_single_file(pattern: str) -> Path:
    files = glob(pattern)
    if not files:
        raise RuntimeError(f"No files found for pattern: {pattern}")
    if len(files) > 1:
        logger.warning(f"Multiple files found for pattern {pattern}, using first.")
    return Path(files[0])


def get_last_trade_price_before(exchange: str, symbol: str, snapshot_us: int) -> float:
    """
    Last trade price at or before snapshot_us from Tardis trades CSVs.
    """
    pattern = f"{DATASETS_DIR}/{exchange}_trades_*_{symbol}.csv.gz"
    files = sorted(glob(pattern))
    if not files:
        raise RuntimeError(f"No trades files found for {exchange} {symbol} with pattern {pattern}")

    last_price = None

    for file in files:
        with gzip.open(file, "rt", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = int(row["timestamp"])
                if ts > snapshot_us:
                    break
                price = safe_float(row.get("price") or row.get("match_price"))
                if price is not None:
                    last_price = price

    if last_price is None:
        raise RuntimeError(
            f"No trades found at or before snapshot for {exchange} {symbol}"
        )

    logger.info(f"Last trade price before snapshot for {exchange} {symbol}: {last_price}")
    return last_price


def compute_realized_vol_from_binance_spot(snapshot: datetime, window_days: int) -> float | None:
    """
    Annualized realized vol from Binance spot BTCUSDT trades.
    """
    logger.info(
        f"Computing realized volatility from Binance spot trades, window={window_days} days..."
    )

    start_date = (snapshot.date() - timedelta(days=window_days + 3)).isoformat()
    end_date = snapshot.date().isoformat()

    tardis_download(
        BINANCE_SPOT_EXCHANGE,
        ["trades"],
        start_date,
        end_date,
        [BINANCE_SPOT_SYMBOL],
        desc="Binance spot trades for RV",
    )

    pattern = f"{DATASETS_DIR}/{BINANCE_SPOT_EXCHANGE}_trades_*_{BINANCE_SPOT_SYMBOL}.csv.gz"
    files = sorted(glob(pattern))
    if not files:
        logger.warning("No Binance spot trades files found for RV computation.")
        return None

    daily_close = {}

    for file in files:
        with gzip.open(file, "rt", newline="") as f:
            reader = csv.DictReader(f)
            last_price = None
            last_ts = None
            for row in reader:
                ts = int(row["timestamp"])
                dt = from_microseconds(ts)
                if dt.date() >= snapshot.date():
                    break
                price = safe_float(row.get("price") or row.get("match_price"))
                if price is not None:
                    last_price = price
                    last_ts = ts

            if last_price is not None and last_ts is not None:
                d = from_microseconds(last_ts).date()
                daily_close[d] = last_price

    dates = sorted(d for d in daily_close.keys() if d < snapshot.date())
    if len(dates) < window_days + 1:
        logger.warning(
            f"Not enough days for RV: have={len(dates)}, need>={window_days+1}"
        )
        return None

    dates = dates[-(window_days + 1):]
    closes = [daily_close[d] for d in dates]

    rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            rets.append(math.log(closes[i] / closes[i - 1]))

    if len(rets) < 2:
        logger.warning("Not enough returns for RV.")
        return None

    daily_vol = stdev(rets)
    ann_vol = daily_vol * math.sqrt(365.0)
    logger.info(f"Realized vol (ann) from Binance spot: {ann_vol:.6f}")
    return ann_vol


def download_okx_options_chain_for_day(date_str: str) -> Path:
    tardis_download(
        EXCHANGE_OPTIONS,
        ["options_chain"],
        date_str,
        (datetime.fromisoformat(date_str) + timedelta(days=1)).date().isoformat(),
        [OPTIONS_SYMBOL],
        desc="OKX options_chain",
    )
    pattern = f"{DATASETS_DIR}/{EXCHANGE_OPTIONS}_options_chain_{date_str}_{OPTIONS_SYMBOL}.csv.gz"
    return find_single_file(pattern)


# ================== CROSS-SECTION BUILDER ==================


def build_options_cross_section(
    options_chain_file: Path,
    snapshot_us: int,
    s_ref: float,
    rv_ann: float,
):
    """
    Build cross-section at snapshot:
    - one row per (type, strike, expiry)
    - last quote <= snapshot
    - outputs only clean, agreed factor set.
    """
    logger.info(f"Building options cross-section from {options_chain_file}...")
    snapshot = from_microseconds(snapshot_us)

    latest = {}

    with gzip.open(options_chain_file, "rt", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["timestamp"])
            if ts > snapshot_us:
                break

            opt_type = (row.get("type") or "").lower()  # 'call' / 'put'
            strike = safe_float(row.get("strike_price"))
            exp_us = safe_float(row.get("expiration"))
            underlying_index = (row.get("underlying_index") or "").upper()

            # Optional filter by underlying_index if specified
            if TARGET_UNDERLYING_INDEX and underlying_index != TARGET_UNDERLYING_INDEX:
                continue

            if opt_type not in ("call", "put") or strike is None or exp_us is None:
                continue

            key = (opt_type, strike, int(exp_us))

            prev = latest.get(key)
            if prev is None or ts > prev["timestamp"]:
                bid_price = safe_float(row.get("bid_price"))
                ask_price = safe_float(row.get("ask_price"))
                mark_price = safe_float(row.get("mark_price"))
                mark_iv = safe_float(row.get("mark_iv"))

                oi = (
                    safe_float(row.get("open_interest"))
                    or safe_float(row.get("open_interest_qty"))
                    or safe_float(row.get("oi"))
                )

                latest[key] = {
                    "timestamp": ts,
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                    "mark_price": mark_price,
                    "mark_iv": mark_iv,
                    "open_interest": oi,
                }

    rows = []
    max_oi = 0.0

    for (opt_type, strike, exp_us), info in latest.items():
        expiry_dt = from_microseconds(exp_us)
        if expiry_dt <= snapshot:
            continue

        ttm_years = (expiry_dt - snapshot).total_seconds() / (365.0 * 24 * 60 * 60)
        if ttm_years <= 0:
            continue

        bid = info["bid_price"]
        ask = info["bid_price"] if info["bid_price"] is not None else info["bid_price"]
        ask = info["ask_price"]
        mark = info["mark_price"]
        mark_iv = info["mark_iv"]
        oi_raw = info.get("open_interest") or 0.0

        # Choose option price: mark, mid, bid/ask
        if mark is not None:
            opt_price = mark
        elif bid is not None and ask is not None:
            opt_price = 0.5 * (bid + ask)
        elif bid is not None:
            opt_price = bid
        elif ask is not None:
            opt_price = ask
        else:
            continue

        if strike <= 0 or s_ref is None or s_ref <= 0:
            continue

        # Core factor: log moneyness
        moneyness = s_ref / strike
        if moneyness <= 0:
            continue

        log_m = math.log(moneyness)
        abs_log_m = abs(log_m)

        is_call = 1 if opt_type == "call" else 0

        # Interaction: T * |log moneyness|
        ttm_x_abs_lm = ttm_years * abs_log_m

        # IV - RV gap
        iv_rv_gap = None
        if mark_iv is not None and rv_ann is not None:
            iv_rv_gap = mark_iv - rv_ann

        # Relative bid-ask spread
        mid_price = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid_price = 0.5 * (bid + ask)
        elif mark is not None and mark > 0:
            mid_price = mark

        rel_spread = None
        if (
            mid_price is not None
            and bid is not None
            and ask is not None
            and ask >= bid
            and mid_price > 0
        ):
            rel_spread = (ask - bid) / mid_price

        if oi_raw and oi_raw > max_oi:
            max_oi = oi_raw

        rows.append(
            {
                "snapshot_time_utc": snapshot.isoformat(),
                "underlying_price_ref": s_ref,
                "realized_vol_14d_ann": rv_ann,
                "option_type": opt_type.upper(),
                "is_call": is_call,
                "strike": strike,
                "expiry_utc": expiry_dt.isoformat(),
                "time_to_maturity_years": ttm_years,
                "option_price": opt_price,
                "log_moneyness": log_m,
                "ttm_x_abs_log_moneyness": ttm_x_abs_lm,
                "iv_rv_gap": iv_rv_gap,
                "rel_bid_ask_spread": rel_spread,
                "oi_raw": oi_raw,  # internal, will convert to oi_rel
            }
        )

    if not rows:
        raise RuntimeError("No valid options rows in cross-section. Check snapshot and options_chain data.")

    # Normalize OI -> oi_rel and drop oi_raw from final schema
    if max_oi > 0:
        for r in rows:
            r["oi_rel"] = (r.get("oi_raw") or 0.0) / max_oi
    else:
        for r in rows:
            r["oi_rel"] = None

    for r in rows:
        r.pop("oi_raw", None)

    logger.info(f"Options cross-section built. Rows: {len(rows)}")
    return rows


def save_cross_section(rows):
    os.makedirs(OUT_DIR, exist_ok=True)

    ts = rows[0]["snapshot_time_utc"]
    ts_clean = (
        ts.replace(":", "")
          .replace("-", "")
          .replace("T", "_")
          .replace("+00:00", "Z")
    )

    filename = os.path.join(
        OUT_DIR,
        f"okx_binance_options_cross_section_{ts_clean}.csv"
    )

    # Final column order: base fields + factor set
    fieldnames = [
        "snapshot_time_utc",
        "underlying_price_ref",
        "realized_vol_14d_ann",
        "option_type",
        "is_call",
        "strike",
        "expiry_utc",
        "time_to_maturity_years",
        "option_price",
        "log_moneyness",
        "ttm_x_abs_log_moneyness",
        "iv_rv_gap",
        "rel_bid_ask_spread",
        "oi_rel",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    logger.info(f"CSV saved: {filename}, rows={len(rows)}")
    return filename


def check_factor_collinearity(csv_path: str):
    """
    Diagnostic: check near-perfect linear dependencies among final factor set.
    """
    logger.info(f"Checking factor correlations in {csv_path} ...")
    df = pd.read_csv(csv_path)

    factor_cols = [
        "log_moneyness",
        "time_to_maturity_years",
        "iv_rv_gap",
        "rel_bid_ask_spread",
        "oi_rel",
        "is_call",
        "ttm_x_abs_log_moneyness",
    ]

    cols = [c for c in factor_cols if c in df.columns]
    sub = df[cols].dropna()
    if sub.empty:
        logger.warning("No valid rows for correlation check.")
        return

    corr = sub.corr()
    logger.info("Correlation matrix of selected factors:")
    print(corr)

    threshold = 0.98
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr.iloc[i, j]
            if abs(rho) > threshold:
                logger.warning(
                    f"Potential multicollinearity: {cols[i]} vs {cols[j]} (corr={rho:.4f})"
                )


def main():
    ensure_api_key()

    snap = snapshot_dt()
    snapshot_us = to_microseconds(snap)
    logger.info(f"Snapshot time (UTC): {snap.isoformat()}")

    os.makedirs(DATASETS_DIR, exist_ok=True)

    # 1) OKX options_chain
    options_chain_file = download_okx_options_chain_for_day(SNAPSHOT_DATE)

    # 2) Underlying trades window
    from_date = (snap.date() - timedelta(days=1)).isoformat()
    to_date = (snap.date() + timedelta(days=1)).isoformat()

    # 3) Download spot & perp trades
    tardis_download(
        OKX_SPOT_EXCHANGE,
        ["trades"],
        from_date,
        to_date,
        [OKX_SPOT_SYMBOL],
        desc="OKEX spot trades",
    )
    tardis_download(
        BINANCE_SPOT_EXCHANGE,
        ["trades"],
        from_date,
        to_date,
        [BINANCE_SPOT_SYMBOL],
        desc="Binance spot trades",
    )
    tardis_download(
        OKX_PERP_EXCHANGE,
        ["trades"],
        from_date,
        to_date,
        [OKX_PERP_SYMBOL],
        desc="OKEX perp trades",
    )
    tardis_download(
        BINANCE_PERP_EXCHANGE,
        ["trades"],
        from_date,
        to_date,
        [BINANCE_PERP_SYMBOL],
        desc="Binance perp trades",
    )

    # 4) Underlying prices at snapshot
    s_okx_spot = get_last_trade_price_before(OKX_SPOT_EXCHANGE, OKX_SPOT_SYMBOL, snapshot_us)
    s_binance_spot = get_last_trade_price_before(BINANCE_SPOT_EXCHANGE, BINANCE_SPOT_SYMBOL, snapshot_us)

    # Reference underlying: average spot
    s_ref = (s_okx_spot + s_binance_spot) / 2.0

    # 5) Realized vol from Binance spot
    rv_ann = compute_realized_vol_from_binance_spot(snap, REALIZED_VOL_WINDOW_DAYS)

    # 6) Build cross-section with final factor set
    rows = build_options_cross_section(
        options_chain_file,
        snapshot_us,
        s_ref,
        rv_ann,
    )

    # 7) Save CSV
    csv_path = save_cross_section(rows)

    # 8) Check factor collinearity
    check_factor_collinearity(csv_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Script failed: {e}")
        raise
