"""
05. 改正民法 DiD(T-DiD-4)

改正民法債権法(2020 年 4 月 1 日施行)前後を利用した差分の差分推定。
処置群:民事領域、対照群:行政領域。

検証対象:
    Y_it = α + β_1 · Treat_i + β_2 · Post_t + δ · Treat_i × Post_t + ε_it

主要結果(予想):
    π^V : δ̂ = -0.505, p < 0.001  (REJECTED → D_t 制約による修正経路)
    μ^T : δ̂ = -0.219, p = 0.023  (REJECTED → 立法的明確化の翻訳抑制経路)

並行トレンド検定:
    処置前(2015〜2019)期間における年次トレンドの差異を検定

Usage:
    python src/05_did_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.utils import (
    get_logger,
    load_data,
    save_table,
)

logger = get_logger(__name__)

# 改正民法債権法施行日
REFORM_DATE = pd.Timestamp("2020-04-01")


# ============================================================
# サンプル準備
# ============================================================

def prepare_did_sample(df: pd.DataFrame) -> pd.DataFrame:
    """DiD 推定用のサンプルを準備する。"""
    df = df.copy()

    # 処置群と対照群
    df = df[df["domain"].isin(["民事", "行政"])].copy()
    df["treat"] = (df["domain"] == "民事").astype(int)

    # 処置前後ダミー
    df["post"] = (df["decision_date"] >= REFORM_DATE).astype(int)

    # 処置 × 期間の交互作用
    df["treat_post"] = df["treat"] * df["post"]

    # 年次ダミー(並行トレンド検定用)
    df["year"] = pd.to_datetime(df["decision_date"]).dt.year

    n_civil_pre = ((df["domain"] == "民事") & (df["post"] == 0)).sum()
    n_civil_post = ((df["domain"] == "民事") & (df["post"] == 1)).sum()
    n_admin = (df["domain"] == "行政").sum()

    logger.info(f"DiD sample:")
    logger.info(f"  Civil (treat) pre: {n_civil_pre}, post: {n_civil_post}")
    logger.info(f"  Admin (control): {n_admin}")

    return df


# ============================================================
# 並行トレンド検定
# ============================================================

def parallel_trend_test(df: pd.DataFrame, outcome: str) -> dict:
    """処置前期間における並行トレンド仮定の検定。"""
    df_pre = df[df["post"] == 0].copy()

    # treat × year のインタラクション
    df_pre["year_centered"] = df_pre["year"] - df_pre["year"].mean()
    df_pre["treat_year"] = df_pre["treat"] * df_pre["year_centered"]

    formula = f"{outcome} ~ treat + year_centered + treat_year"
    try:
        model = smf.ols(formula, data=df_pre).fit(cov_type="HC1")
        p_value = model.pvalues.get("treat_year", np.nan)
    except Exception as e:
        logger.warning(f"  Parallel trend test failed for {outcome}: {e}")
        p_value = np.nan

    return {
        "outcome": outcome,
        "p_value_parallel_trend": p_value,
        "supports_parallel_trend": p_value > 0.05 if not np.isnan(p_value) else False,
    }


# ============================================================
# DiD 推定
# ============================================================

def did_estimation(df: pd.DataFrame, outcome: str) -> dict:
    """DiD 推定を実行する。"""
    formula = f"{outcome} ~ treat + post + treat_post"

    try:
        model = smf.ols(formula, data=df).fit(cov_type="HC1")
        delta_hat = model.params.get("treat_post", np.nan)
        se = model.bse.get("treat_post", np.nan)
        p_value = model.pvalues.get("treat_post", np.nan)
        n_obs = int(model.nobs)
    except Exception as e:
        logger.error(f"  DiD estimation failed for {outcome}: {e}")
        return {
            "outcome": outcome,
            "delta_hat": np.nan,
            "se": np.nan,
            "p_value": np.nan,
            "n_obs": 0,
            "verdict": "ERROR",
        }

    # 予想方向(理論的予測)
    expected_sign = {
        "mu_F": "negative",
        "mu_T": "positive",
        "pi_V": "positive",
        "pi_C": "neutral",
        "pi_H": "neutral",
        "pi_M_S": "positive",
        "pi_M_R": "neutral",
    }

    actual_sign = "positive" if delta_hat > 0 else "negative"
    direction_match = expected_sign.get(outcome, "any") == actual_sign

    if p_value < 0.05:
        if direction_match:
            verdict = "VERIFIED"
        else:
            verdict = "REJECTED"
    elif p_value < 0.1:
        verdict = "弱有意"
    else:
        verdict = "非有意"

    return {
        "outcome": outcome,
        "delta_hat": delta_hat,
        "se": se,
        "p_value": p_value,
        "n_obs": n_obs,
        "expected_direction": expected_sign.get(outcome, "any"),
        "actual_direction": actual_sign,
        "verdict": verdict,
    }


# ============================================================
# Main
# ============================================================

def main() -> pd.DataFrame:
    """改正民法 DiD の主要パイプライン。"""
    logger.info("=" * 60)
    logger.info("STEP 05: T-DiD-4 (Civil Code Reform DiD)")
    logger.info("=" * 60)

    df = load_data()
    df_did = prepare_did_sample(df)

    outcomes = ["mu_F", "mu_T", "pi_V", "pi_C", "pi_H", "pi_M_S", "pi_M_R"]

    # 並行トレンド検定
    logger.info("\n--- Parallel Trend Tests (pre-period 2015-2019) ---")
    pt_results = []
    for outcome in outcomes:
        result = parallel_trend_test(df_did, outcome)
        pt_results.append(result)
        logger.info(
            f"  {outcome}: p = {result['p_value_parallel_trend']:.3f}, "
            f"parallel: {result['supports_parallel_trend']}"
        )

    df_pt = pd.DataFrame(pt_results)
    save_table(df_pt, "05_did_parallel_trends")

    # DiD 推定
    logger.info("\n--- DiD Estimation ---")
    did_results = []
    for outcome in outcomes:
        result = did_estimation(df_did, outcome)
        did_results.append(result)
        if not np.isnan(result["delta_hat"]):
            logger.info(
                f"  {outcome}: δ̂ = {result['delta_hat']:+.3f}, "
                f"SE = {result['se']:.3f}, p = {result['p_value']:.3f}, "
                f"verdict: {result['verdict']}"
            )

    df_did_results = pd.DataFrame(did_results)
    save_table(df_did_results, "05_did_results")

    # 修正経路への含意
    rejected_count = (df_did_results["verdict"] == "REJECTED").sum()
    if rejected_count > 0:
        logger.info(
            f"\n{rejected_count} outcomes REJECTED → "
            f"Proposition 1 修正経路:D_t 制約のシャドウ・プライス効果(§4.3.2)として解釈"
        )

    return df_did_results


if __name__ == "__main__":
    main()
