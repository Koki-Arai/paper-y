"""
01. データ読み込みと前処理

判決テキストから抽出した特徴量データ(6,724 件、2015〜2024 年)を
読み込み、領域別の記述統計を計算し、後続の分析で使用する形に整形する。

Usage:
    python src/01_data_loader.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import (
    DOMAINS,
    DOMAIN_SAMPLE_SIZE,
    SELECTION_VARS,
    PI_COMPONENTS,
    get_logger,
    load_data,
    save_table,
    descriptive_stats,
)

logger = get_logger(__name__)


def validate_data(df: pd.DataFrame) -> None:
    """データの整合性を検証する。"""
    n_total = len(df)
    logger.info(f"Total sample size: {n_total}")

    if n_total != 6724:
        logger.warning(
            f"Sample size {n_total} differs from expected 6,724. "
            f"This may indicate a different version of the dataset."
        )

    # 領域別サンプルサイズの確認
    if "domain" in df.columns:
        by_domain = df["domain"].value_counts()
        logger.info("Sample size by domain:")
        for domain in DOMAINS:
            actual = by_domain.get(domain, 0)
            expected = DOMAIN_SAMPLE_SIZE[domain]
            status = "OK" if actual == expected else "DIFF"
            logger.info(f"  {domain}: {actual:5d} (expected {expected}) [{status}]")

    # 必須変数の存在確認
    required_vars = SELECTION_VARS + ["domain", "decision_date"]
    missing = [v for v in required_vars if v not in df.columns]
    if missing:
        logger.error(f"Missing required variables: {missing}")
        raise ValueError(f"Missing variables: {missing}")

    # 欠損値の確認
    for var in SELECTION_VARS:
        n_missing = df[var].isna().sum()
        if n_missing > 0:
            logger.warning(f"  {var}: {n_missing} missing values")


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """記述統計を計算し、CSV として保存する。"""
    logger.info("Computing descriptive statistics by domain...")
    stats = descriptive_stats(df, SELECTION_VARS, by="domain")
    save_table(stats.reset_index(), "01_descriptive_stats")
    return stats


def add_period_indicator(df: pd.DataFrame, cutoff: str = "2020-04-01") -> pd.DataFrame:
    """改正民法債権法施行(2020 年 4 月 1 日)前後の期間ダミーを追加する。"""
    cutoff_date = pd.Timestamp(cutoff)
    df = df.copy()
    df["post_civil_reform"] = (df["decision_date"] >= cutoff_date).astype(int)
    n_pre = (df["post_civil_reform"] == 0).sum()
    n_post = (df["post_civil_reform"] == 1).sum()
    logger.info(f"Pre-reform: {n_pre}, Post-reform: {n_post}")
    return df


def compute_pi_total(df: pd.DataFrame) -> pd.DataFrame:
    """政策判断明示度の合計 π_total を計算する。"""
    df = df.copy()
    df["pi_total"] = df[PI_COMPONENTS].sum(axis=1)
    return df


def main() -> pd.DataFrame:
    """データ読み込みと前処理の主要パイプライン。"""
    logger.info("=" * 60)
    logger.info("STEP 01: Data Loading and Preprocessing")
    logger.info("=" * 60)

    # データ読み込み
    df = load_data()

    # 整合性検証
    validate_data(df)

    # 期間ダミー追加
    df = add_period_indicator(df)

    # 政策判断明示度合計
    df = compute_pi_total(df)

    # 記述統計
    compute_descriptive_stats(df)

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    main()
