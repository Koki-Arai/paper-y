"""
共通ユーティリティ関数

論文 Y(荒井 2026 実証編)で使用する共通関数を提供する。
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# パス設定
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [DATA_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 領域定義
# ============================================================

DOMAINS = ["民事", "商事", "行政", "刑事"]
DOMAIN_EN = {"民事": "civil", "商事": "commercial", "行政": "administrative", "刑事": "criminal"}
DOMAIN_SAMPLE_SIZE = {"民事": 3506, "商事": 2093, "行政": 639, "刑事": 486}


# ============================================================
# 変数定義
# ============================================================

# 選択ベクトル m_t = (μ^F, μ^T, ν, π)
SELECTION_VARS = ["mu_F", "mu_T", "nu", "pi_V", "pi_C", "pi_H", "pi_M_S", "pi_M_R"]

# 政策判断明示度の四成分(命題 3 における π = π^V + π^C + π^H + π^M)
PI_COMPONENTS = ["pi_V", "pi_C", "pi_H", "pi_M_S", "pi_M_R"]

# 媒介能力・領域パラメータ
DOMAIN_PARAMS = ["mu_LS", "mu_TH"]


# ============================================================
# ログ設定
# ============================================================

def get_logger(name: str = __name__) -> logging.Logger:
    """標準的なロガーを取得する。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ============================================================
# 共通関数
# ============================================================

def load_data(filename: str = "hanrei_features_v3_3_2015_2024.csv") -> pd.DataFrame:
    """
    判決テキストから抽出した特徴量データを読み込む。

    Parameters
    ----------
    filename : str
        データファイル名(data/ ディレクトリ内)

    Returns
    -------
    df : pd.DataFrame
        判決サンプル DataFrame(6,724 件)

    Notes
    -----
    本データセットは、裁判所サイトから公開されている判決文に基づき、
    語彙頻度ベースで構築された特徴量データである。サンプルサイズは
    領域別に民事 3,506・商事 2,093・行政 639・刑事 486 の計 6,724 件。
    """
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Please place the dataset CSV file in {DATA_DIR}"
        )
    df = pd.read_csv(filepath, parse_dates=["decision_date"])

    # 領域名の正規化
    if "domain" in df.columns:
        df["domain"] = df["domain"].astype("category")

    return df


def save_table(df: pd.DataFrame, name: str, formats: tuple = ("csv", "tex")) -> None:
    """結果テーブルを CSV と LaTeX 形式で保存する。"""
    base = TABLES_DIR / name
    if "csv" in formats:
        df.to_csv(f"{base}.csv", index=False, encoding="utf-8")
    if "tex" in formats:
        df.to_latex(f"{base}.tex", index=False, escape=False)


def save_figure(fig, name: str, formats: tuple = ("png", "pdf"), dpi: int = 300) -> None:
    """結果図表を PNG と PDF 形式で保存する。"""
    base = FIGURES_DIR / name
    if "png" in formats:
        fig.savefig(f"{base}.png", dpi=dpi, bbox_inches="tight")
    if "pdf" in formats:
        fig.savefig(f"{base}.pdf", bbox_inches="tight")


def descriptive_stats(df: pd.DataFrame, vars_: list, by: str = "domain") -> pd.DataFrame:
    """領域別の記述統計を計算する。"""
    return df.groupby(by, observed=False)[vars_].agg(["mean", "std", "min", "max", "count"]).round(4)


def kruskal_wallis_test(df: pd.DataFrame, var: str, by: str = "domain") -> dict:
    """Kruskal-Wallis 検定を実行する(領域横断的な分布差検定)。"""
    from scipy import stats

    groups = [g[var].dropna().values for _, g in df.groupby(by, observed=False)]
    stat, p_value = stats.kruskal(*groups)
    return {"H": stat, "p_value": p_value, "df": len(groups) - 1}


def spearman_rho(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman 順位相関係数とその p 値を計算する。"""
    from scipy import stats

    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 3:
        return {"rho": np.nan, "p_value": np.nan, "n": valid.sum()}
    rho, p_value = stats.spearmanr(x[valid], y[valid])
    return {"rho": rho, "p_value": p_value, "n": valid.sum()}
