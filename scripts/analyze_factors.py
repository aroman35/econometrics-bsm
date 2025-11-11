import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===

DATA_DIR = Path("../data")
REPORT_DIR = Path("../reports") / "eda"

# Factor columns produced by tardis_okx_binance_cross_section.py
FACTOR_COLS = [
    "log_moneyness",
    "time_to_maturity_years",
    "iv_rv_gap",
    "rel_bid_ask_spread",
    "oi_rel",
    "is_call",
    "ttm_x_abs_log_moneyness",
]


def find_latest_cross_section() -> Path:
    """
    Find the most recent cross-section CSV in data/.
    File pattern: okx_binance_options_cross_section_*.csv
    """
    pattern = str(DATA_DIR / "okx_binance_options_cross_section_*.csv")
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No cross-section files found by pattern: {pattern}\n"
            f"Run tardis_okx_binance_cross_section.py first."
        )
    # sort by mtime descending, take newest
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return Path(files[0])


def ensure_report_dir():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_correlation_outputs(df_factors: pd.DataFrame):
    """
    Save correlation matrix as CSV and Markdown, and plot heatmap.
    """
    corr = df_factors.corr()

    # 1) CSV
    corr_csv_path = REPORT_DIR / "correlation_matrix.csv"
    corr.to_csv(corr_csv_path, float_format="%.6f")

    # 2) Markdown table (for README / docs)
    corr_md_path = REPORT_DIR / "correlation_matrix.md"
    with corr_md_path.open("w", encoding="utf-8") as f:
        f.write("| | " + " | ".join(corr.columns) + " |\n")
        f.write("|" + " --- |" * (len(corr.columns) + 1) + "\n")
        for row_name, row in corr.iterrows():
            vals = " | ".join(f"{v:.3f}" for v in row.values)
            f.write(f"| {row_name} | {vals} |\n")

    # 3) Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation Heatmap of Factors")
    fig.tight_layout()

    heatmap_path = REPORT_DIR / "corr_heatmap.png"
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved correlation matrix to {corr_csv_path} and {corr_md_path}")
    print(f"[OK] Saved correlation heatmap to {heatmap_path}")


def save_variance_plot(df_factors: pd.DataFrame):
    """
    Save bar chart of factor variances.
    """
    variances = df_factors.var().sort_values(ascending=False)

    var_csv_path = REPORT_DIR / "factor_variances.csv"
    variances.to_csv(var_csv_path, header=["variance"], float_format="%.6f")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(variances.index, variances.values)
    ax.set_title("Factor Variances")
    ax.set_ylabel("Variance")
    ax.set_xticks(range(len(variances.index)))
    ax.set_xticklabels(variances.index, rotation=45, ha="right")
    fig.tight_layout()

    var_png_path = REPORT_DIR / "factor_variances.png"
    fig.savefig(var_png_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved factor variances to {var_csv_path} and {var_png_path}")


def save_boxplot(df_factors: pd.DataFrame):
    """
    Save boxplot of factor distributions to inspect dispersion & outliers.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    df_factors.boxplot(column=df_factors.columns.tolist(), grid=False, ax=ax)
    ax.set_title("Distributions of Factors (Boxplot)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    boxplot_path = REPORT_DIR / "factor_boxplot.png"
    fig.savefig(boxplot_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved factor boxplot to {boxplot_path}")


def main():
    ensure_report_dir()

    # 1) Locate latest cross-section
    csv_path = find_latest_cross_section()
    print(f"[INFO] Using cross-section file: {csv_path}")

    # 2) Load and subset factors
    df = pd.read_csv(csv_path)

    missing = [c for c in FACTOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected factor columns in {csv_path}:\n{missing}\n"
            f"Check that tardis_okx_binance_cross_section.py is the latest version."
        )

    df_factors = df[FACTOR_COLS].dropna()
    if df_factors.empty:
        raise ValueError("No valid rows after dropping NaNs in factor columns.")

    # 3) Save correlation matrix + heatmap
    save_correlation_outputs(df_factors)

    # 4) Save variance chart
    save_variance_plot(df_factors)

    # 5) Save boxplot
    save_boxplot(df_factors)

    print("[DONE] EDA reports generated under:", REPORT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
