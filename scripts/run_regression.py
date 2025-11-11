import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ============ CONFIG ============

DATA_DIR = Path("./../data")
PATTERN = "bsm_regression_dataset_*.csv"

# Regressand
TARGET_COL = "bsm_error_log"  # log(p_mkt / p_bsm)

# Regressors (z-scored continous + dummy)
BASE_REGRESSORS = [
    "log_moneyness_z",
    "time_to_maturity_years_z",
    "iv_rv_gap_z",
    "log_rel_bid_ask_spread_z",
    "oi_rel_z",
    "ttm_x_abs_log_moneyness_z",
    "is_call",  # not standardized (dummy)
]

# Robust covariance estimator
ROBUST_COV_TYPE = "HC3"

# ================================


def find_latest_dataset() -> Path:
    pattern = str(DATA_DIR / PATTERN)
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No regression datasets found by pattern: {pattern}. "
            f"Run build_regression_dataset.py first."
        )
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return Path(files[0])


def load_data(path: Path) -> pd.DataFrame:
    print(f"[INFO] Using regression dataset: {path}")
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing in dataset.")

    missing_regressors = [c for c in BASE_REGRESSORS if c not in df.columns]
    if missing_regressors:
        raise ValueError(
            "Some required regressors are missing in dataset: "
            + ", ".join(missing_regressors)
        )

    return df


def fit_ols(df: pd.DataFrame):
    # Subset to relevant columns
    cols = [TARGET_COL] + BASE_REGRESSORS
    df_model = df[cols].dropna()

    if df_model.empty:
        raise ValueError("No rows left after dropping NaNs for model variables.")

    print(f"[INFO] Observations used in regression: {len(df_model)}")

    y = df_model[TARGET_COL].values
    X = df_model[BASE_REGRESSORS].copy()

    # Add constant
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X)
    results = model.fit(cov_type=ROBUST_COV_TYPE)

    return results


def format_results(results: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    params = results.params
    bse = results.bse
    tvalues = results.tvalues
    pvalues = results.pvalues

    rows = []
    for name in params.index:
        rows.append(
            {
                "variable": name,
                "coef": params[name],
                "std_err": bse[name],
                "t": tvalues[name],
                "p_value": pvalues[name],
            }
        )

    df_res = pd.DataFrame(rows)
    df_res = df_res.sort_values(by="variable", ascending=True)

    # Add significance stars
    def stars(p):
        if p < 0.01:
            return "***"
        elif p < 0.05:
            return "**"
        elif p < 0.1:
            return "*"
        else:
            return ""

    df_res["signif"] = df_res["p_value"].apply(stars)
    return df_res


def interpret(results_df: pd.DataFrame):
    """
    Simple text interpretation in Russian:
    - смотрим на знак и значимость;
    - формируем короткие выводы под учебный формат.
    """
    print("\n[INTERPRETATION] Интерпретация коэффициентов (bsm_error_log):\n")

    for _, row in results_df.iterrows():
        name = row["variable"]
        if name == "const":
            continue

        coef = row["coef"]
        p = row["p_value"]
        signif = row["signif"]

        if p < 0.1:
            signif_text = (
                "статистически значим (p < 0.10)"
                if p >= 0.05
                else "статистически значим (p < 0.05)"
                if p >= 0.01
                else "статистически высоко значим (p < 0.01)"
            )
            direction = "положительно" if coef > 0 else "отрицательно"
            if name == "log_moneyness_z":
                print(
                    f"- log_moneyness_z: {signif_text}, {direction} связан с лог-ошибкой BSM. "
                    f"Это отражает наличие улыбки/смёрка волатильности в ценах опционов."
                )
            elif name == "time_to_maturity_years_z":
                print(
                    f"- time_to_maturity_years_z: {signif_text}, {direction} связан с лог-ошибкой. "
                    f"Срок до экспирации влияет на систематическое отклонение от BSM."
                )
            elif name == "iv_rv_gap_z":
                print(
                    f"- iv_rv_gap_z: {signif_text}, {direction} связан с лог-ошибкой. "
                    f"Разрыв между implied и реализованной волатильностью объясняет часть mispricing."
                )
            elif name == "log_rel_bid_ask_spread_z":
                print(
                    f"- log_rel_bid_ask_spread_z: {signif_text}, {direction} связан с лог-ошибкой. "
                    f"Ликвидность и торговые издержки существенно влияют на отклонения от теоретической цены."
                )
            elif name == "oi_rel_z":
                print(
                    f"- oi_rel_z: {signif_text}, {direction} связан с лог-ошибкой. "
                    f"Концентрация открытого интереса по страйкам/срокам коррелирует с mispricing."
                )
            elif name == "ttm_x_abs_log_moneyness_z":
                print(
                    f"- ttm_x_abs_log_moneyness_z: {signif_text}, {direction} связан с лог-ошибкой. "
                    f"Нелинейное взаимодействие срока и глубины опциона статистически важно."
                )
            elif name == "is_call":
                print(
                    f"- is_call: {signif_text}, {direction} влияет на лог-ошибку. "
                    f"Есть асимметрия между CALL и PUT относительно модели BSM."
                )
            else:
                print(
                    f"- {name}: {signif_text}, {direction} влияет на лог-ошибку BSM."
                )
        else:
            # Not significant
            print(
                f"- {name}: незначим статистически (p = {p:.3f}). "
                f"В рамках данного среза его вклад в объяснение ошибки BSM не подтверждён."
            )

    print(
        "\nЗамечание: знак коэффициента интерпретируется для z-скорированных факторов.\n"
        "Положительный коэффициент означает, что рост фактора связан с ростом логарифмической "
        "ошибки BSM (рыночная цена относительно теоретической). Отрицательный — с её снижением."
    )


def main():
    dataset_path = find_latest_dataset()
    df = load_data(dataset_path)

    results = fit_ols(df)
    res_df = format_results(results)

    print("\n[RESULTS] OLS with robust (HC3) standard errors:\n")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    # Optionally also show compact R^2 etc.
    print("\n[MODEL SUMMARY]")
    print(
        f"R-squared: {results.rsquared:.4f}, "
        f"Adj. R-squared: {results.rsquared_adj:.4f}"
    )

    # Save table to file for report
    OUT_REPORT_DIR = Path("./../reports") / "regression"
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_table_path = OUT_REPORT_DIR / "ols_results_bsm_error_log.csv"
    res_df.to_csv(out_table_path, index=False)
    print(f"\n[OK] Saved regression table to {out_table_path}")

    # Textual interpretation
    interpret(res_df)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
