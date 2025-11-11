import os
import sys
import math
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============ CONFIG ============

DATA_DIR = Path("../data")
OUT_DATASET_DIR = Path("./../data")
OUT_REPORT_DIR = Path("./../reports") / "regression"

# Risk-free rate (annualized, simple)
RISK_FREE_RATE = 0.03

# Pattern to detect cross-section files
CROSS_SECTION_PATTERN = "okx_binance_options_cross_section_*.csv"

# Main dependent variable: choose between "rel" and "log"
# We'll compute both; LOG is preferred in analysis.
PRIMARY_TARGET = "bsm_error_log"

# ================================


def find_latest_cross_section() -> Path:
    pattern = str(DATA_DIR / CROSS_SECTION_PATTERN)
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No cross-section files found by pattern: {pattern}. "
            f"Run tardis_okx_binance_cross_section.py first."
        )
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return Path(files[0])


def bsm_price(S, K, T, r, sigma, is_call: bool):
    """
    Black-Scholes-Merton price for European option (no dividends).
    Uses simple r, annualized sigma.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma is None or sigma <= 0:
        return np.nan

    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.nan

    # Standard normal CDF via erf
    from math import erf, sqrt

    def N(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    if is_call:
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    else:
        return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def compute_sigma_from_snapshot(df: pd.DataFrame) -> float | None:
    """
    Use realized_vol_14d_ann as snapshot volatility proxy.
    If constant -> single sigma; if varying -> return None (we'll use row-wise).
    """
    if "realized_vol_14d_ann" not in df.columns:
        return None

    vals = df["realized_vol_14d_ann"].dropna().unique()
    if len(vals) == 0:
        return None
    if len(vals) == 1:
        return float(vals[0])
    # varying sigma per row: let caller use row values
    return None


def add_bsm_errors(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "underlying_price_ref",
        "strike",
        "time_to_maturity_years",
        "option_price",
        "is_call",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in cross-section.")

    sigma_const = compute_sigma_from_snapshot(df)

    bsm_prices = []
    err_raw = []
    err_rel = []
    err_log = []

    for _, row in df.iterrows():
        S = row["underlying_price_ref"]
        K = row["strike"]
        T = row["time_to_maturity_years"]
        is_call = bool(row["is_call"])

        if sigma_const is not None:
            sigma = sigma_const
        else:
            sigma = row.get("realized_vol_14d_ann", np.nan)

        p_mkt = row["option_price"]
        p_bsm = bsm_price(S, K, T, RISK_FREE_RATE, sigma, is_call)

        bsm_prices.append(p_bsm)

        if (
            p_bsm is None
            or not np.isfinite(p_bsm)
            or p_bsm <= 0
            or p_mkt is None
            or not np.isfinite(p_mkt)
            or p_mkt <= 0
        ):
            err_raw.append(np.nan)
            err_rel.append(np.nan)
            err_log.append(np.nan)
        else:
            diff = p_mkt - p_bsm
            err_raw.append(diff)
            err_rel.append(diff / p_bsm)  # relative pricing error (in decimals)
            err_log.append(math.log(p_mkt / p_bsm))  # log mispricing

    df["bsm_price"] = bsm_prices
    df["bsm_error_raw"] = err_raw
    df["bsm_error_rel"] = err_rel
    df["bsm_error_log"] = err_log

    # Drop rows where target is invalid
    before = len(df)
    df = df.dropna(subset=["bsm_price", "bsm_error_log"])
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows with invalid BSM errors. Remaining: {after}")
    return df


def add_log_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed spread to mitigate heavy right tail:
    log_rel_bid_ask_spread = ln(1 + rel_bid_ask_spread).
    """
    if "rel_bid_ask_spread" not in df.columns:
        df["log_rel_bid_ask_spread"] = np.nan
        return df

    x = df["rel_bid_ask_spread"].copy()

    # Negative or NaN spreads treated as NaN here.
    x = x.where(x >= 0)

    df["log_rel_bid_ask_spread"] = np.log1p(x)
    return df


def z_score(df: pd.DataFrame, col: str, out_col: str):
    """
    Add z-scored version of column if std > 0.
    """
    series = df[col].astype(float)
    mean = series.mean(skipna=True)
    std = series.std(skipna=True)
    if std and std > 0:
        df[out_col] = (series - mean) / std
    else:
        df[out_col] = np.nan
        print(f"[WARN] Std of {col} is zero or NaN; z-score not informative.")
    return df


def standardize_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add z-scores for continuous factors to make coefficients comparable.
    """
    # Continuous factors
    cont_factors = [
        "log_moneyness",
        "time_to_maturity_years",
        "iv_rv_gap",
        "rel_bid_ask_spread",
        "log_rel_bid_ask_spread",
        "oi_rel",
        "ttm_x_abs_log_moneyness",
    ]

    for col in cont_factors:
        if col in df.columns:
            df = z_score(df, col, col + "_z")

    return df


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Compute VIF for given columns.
    Assumes df has no NaNs in these columns.
    """
    if not cols:
        raise ValueError("No columns provided for VIF calculation.")

    X = df[cols].copy()
    X = sm.add_constant(X)

    vif_data = []
    for i, name in enumerate(X.columns):
        if name == "const":
            continue
        vif_val = variance_inflation_factor(X.values, i)
        vif_data.append({"variable": name, "VIF": vif_val})

    vif_df = pd.DataFrame(vif_data)
    return vif_df.sort_values("VIF", ascending=False)


def main():
    OUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Locate latest cross-section
    cross_path = find_latest_cross_section()
    print(f"[INFO] Using cross-section file: {cross_path}")

    df = pd.read_csv(cross_path)

    # 2) Add BSM prices and mispricing measures
    df = add_bsm_errors(df)

    # 3) Add log-transformed bid-ask spread
    df = add_log_spread(df)

    # 4) Standardize continuous factors (z-scores)
    df = standardize_factors(df)

    # 5) Prepare VIF analysis for standardized features
    # Use z-scored versions + dummy is_call.
    vif_cols = []
    candidate_z = [
        "log_moneyness_z",
        "time_to_maturity_years_z",
        "iv_rv_gap_z",
        "oi_rel_z",
        "ttm_x_abs_log_moneyness_z",
        "log_rel_bid_ask_spread_z",
    ]
    for c in candidate_z:
        if c in df.columns and df[c].notna().std() > 0:
            vif_cols.append(c)

    # include dummy unscaled
    if "is_call" in df.columns and df["is_call"].notna().std() > 0:
        vif_cols.append("is_call")

    # Filter rows for VIF calc
    df_vif = df.dropna(subset=vif_cols).copy()
    if len(df_vif) == 0:
        print("[WARN] No rows available for VIF after dropping NaNs.")
    else:
        vif_df = compute_vif(df_vif, vif_cols)
        vif_path = OUT_REPORT_DIR / "vif_table.csv"
        vif_df.to_csv(vif_path, index=False)
        print("[INFO] VIF table:")
        print(vif_df.to_string(index=False))
        print(f"[OK] Saved VIF table to {vif_path}")

    # 6) Save final regression dataset
    # Columns to keep
    keep_cols = [
        # identifiers
        "snapshot_time_utc",
        "option_type",
        "is_call",
        "strike",
        "expiry_utc",
        # core economics
        "underlying_price_ref",
        "realized_vol_14d_ann",
        "time_to_maturity_years",
        "option_price",
        "bsm_price",
        # targets
        "bsm_error_raw",
        "bsm_error_rel",
        "bsm_error_log",
        # raw factors
        "log_moneyness",
        "ttm_x_abs_log_moneyness",
        "iv_rv_gap",
        "rel_bid_ask_spread",
        "log_rel_bid_ask_spread",
        "oi_rel",
        # standardized factors
        "log_moneyness_z",
        "time_to_maturity_years_z",
        "iv_rv_gap_z",
        "rel_bid_ask_spread_z",
        "log_rel_bid_ask_spread_z",
        "oi_rel_z",
        "ttm_x_abs_log_moneyness_z",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    # Construct output filename from snapshot
    first_ts = df["snapshot_time_utc"].iloc[0].replace(":", "").replace("-", "").replace("T", "_")
    out_path = OUT_DATASET_DIR / f"bsm_regression_dataset_{first_ts}.csv"
    df[keep_cols].to_csv(out_path, index=False)
    print(f"[OK] Saved regression dataset to {out_path}")
    print("[DONE] Regression dataset and VIF report are ready.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
