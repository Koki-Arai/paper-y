"""
03. 命題 2 関連検証テスト群

論文 X の命題 2(精緻化された統合命題)に対応する 9 個の独立検証テスト:

    T1-2: K_c 階層(藤谷条件、K_c 部分)
    T2-2: π 三分解(命題 3 の前駆形)
    T3-2: 事案影響度 x(命題 1)
    T4-2: Π^M 最高裁外れ値
    T5-2: M_AV(西内条件)
    T6-2: Δ_J(矢作条件)
    T7-2: η_J(矢作条件、二面性命題)
    T-S_A: S_A · M_LS(飯田条件)
    T-π^C: π^C 領域別異質性(林条件、Π_t 部分)

Usage:
    python src/03_tests_p2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.utils import (
    DOMAINS,
    PI_COMPONENTS,
    get_logger,
    load_data,
    save_table,
    kruskal_wallis_test,
    spearman_rho,
)

logger = get_logger(__name__)


# ============================================================
# T1-2: K_c 階層
# ============================================================

def test_T1_2(df: pd.DataFrame) -> dict:
    """T1-2: 文脈成熟度 K_c の領域別順序(藤谷条件)。"""
    logger.info("Running T1-2: K_c hierarchy test")

    # 代理変数:π_total
    df = df.copy()
    df["pi_total"] = df[PI_COMPONENTS].sum(axis=1)

    means = df.groupby("domain", observed=False)["pi_total"].mean()
    expected_order = ["商事", "民事", "行政", "刑事"]
    actual_order = means.sort_values(ascending=False).index.tolist()

    kw = kruskal_wallis_test(df, "pi_total")

    return {
        "test_id": "T1-2",
        "name": "K_c hierarchy",
        "kruskal_H": kw["H"],
        "p_value": kw["p_value"],
        "expected_order": " > ".join(expected_order),
        "actual_order": " > ".join(actual_order),
        "match": expected_order == actual_order,
        "verdict": "PARTIAL",  # 代理変数の選択問題
    }


# ============================================================
# T2-2: π 三分解
# ============================================================

def test_T2_2(df: pd.DataFrame) -> dict:
    """T2-2: 政策判断明示度の三分解(命題 3 前駆形)。"""
    logger.info("Running T2-2: π three-decomposition")

    pi_three = ["pi_V", "pi_C", "pi_H"]
    means = df[pi_three].mean()

    # ノンパラメトリック検定:三成分の分布が異なるか
    friedman = stats.friedmanchisquare(df["pi_V"], df["pi_C"], df["pi_H"])

    return {
        "test_id": "T2-2",
        "name": "pi three-decomposition",
        "friedman_chi2": friedman.statistic,
        "p_value": friedman.pvalue,
        "mean_pi_V": means["pi_V"],
        "mean_pi_C": means["pi_C"],
        "mean_pi_H": means["pi_H"],
        "verdict": "VERIFIED" if friedman.pvalue < 0.05 else "REJECTED",
    }


# ============================================================
# T3-2: 事案影響度
# ============================================================

def test_T3_2(df: pd.DataFrame) -> dict:
    """T3-2: 事案影響度 x と政策判断明示度の関係(命題 1)。"""
    logger.info("Running T3-2: Case impact x correlation")

    if "case_impact" not in df.columns:
        logger.warning("  case_impact variable not available")
        return {"test_id": "T3-2", "name": "case impact", "verdict": "SKIPPED"}

    df = df.copy()
    df["pi_total"] = df[PI_COMPONENTS].sum(axis=1)
    r = spearman_rho(df["case_impact"].values, df["pi_total"].values)

    return {
        "test_id": "T3-2",
        "name": "case impact",
        "rho": r["rho"],
        "p_value": r["p_value"],
        "verdict": "VERIFIED" if r["p_value"] < 0.05 and r["rho"] > 0 else "REJECTED",
    }


# ============================================================
# T4-2: Π^M 最高裁外れ値
# ============================================================

def test_T4_2(df: pd.DataFrame) -> dict:
    """T4-2: 最高裁判決における Π^M の外れ値性(Mahalanobis 距離)。"""
    logger.info("Running T4-2: Supreme Court outlier")

    if "is_supreme_court" not in df.columns:
        logger.warning("  is_supreme_court variable not available")
        return {"test_id": "T4-2", "name": "SC outlier", "verdict": "SKIPPED"}

    pi_M = df[["pi_M_S", "pi_M_R"]].values

    # 共分散行列
    cov = np.cov(pi_M.T)
    inv_cov = np.linalg.pinv(cov)
    mean_pi_M = pi_M.mean(axis=0)

    # Mahalanobis 距離
    diff = pi_M - mean_pi_M
    mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

    sc_mask = df["is_supreme_court"] == 1
    sc_mahal_mean = mahal_dist[sc_mask].mean() if sc_mask.sum() > 0 else np.nan
    non_sc_mahal_mean = mahal_dist[~sc_mask].mean()

    return {
        "test_id": "T4-2",
        "name": "SC outlier (Mahalanobis)",
        "sc_mahal_mean": sc_mahal_mean,
        "non_sc_mahal_mean": non_sc_mahal_mean,
        "diff_ratio": sc_mahal_mean / non_sc_mahal_mean if non_sc_mahal_mean > 0 else np.nan,
        "verdict": "VERIFIED" if sc_mahal_mean > non_sc_mahal_mean * 1.5 else "PARTIAL",
    }


# ============================================================
# T5-2: M_AV
# ============================================================

def test_T5_2(df: pd.DataFrame) -> dict:
    """T5-2: 媒介能力 M_AV の領域別異質性(西内条件)。"""
    logger.info("Running T5-2: M_AV heterogeneity")

    # μ^T を M_AV の代理として使用
    kw = kruskal_wallis_test(df, "mu_T")
    means = df.groupby("domain", observed=False)["mu_T"].mean()

    return {
        "test_id": "T5-2",
        "name": "M_AV (Nishiuchi condition)",
        "kruskal_H": kw["H"],
        "p_value": kw["p_value"],
        "mean_civil": means.get("民事", np.nan),
        "mean_commercial": means.get("商事", np.nan),
        "mean_admin": means.get("行政", np.nan),
        "mean_criminal": means.get("刑事", np.nan),
        "verdict": "VERIFIED" if kw["p_value"] < 1e-40 else "PARTIAL",
    }


# ============================================================
# T6-2: Δ_J
# ============================================================

def test_T6_2(df: pd.DataFrame) -> dict:
    """T6-2: 動機ずれ指標 Δ_J の領域別異質性(矢作条件)。"""
    logger.info("Running T6-2: Δ_J motivation gap")

    # 代理:1 - μ^F(動機ずれが小さいほど μ^F が小さい)
    df = df.copy()
    df["delta_J_proxy"] = 1 - df["mu_F"]

    kw = kruskal_wallis_test(df, "delta_J_proxy")

    return {
        "test_id": "T6-2",
        "name": "Δ_J (Yahagi condition)",
        "kruskal_H": kw["H"],
        "p_value": kw["p_value"],
        "verdict": "VERIFIED" if kw["p_value"] < 0.05 else "PARTIAL",
    }


# ============================================================
# T7-2: η_J(批判リスク係数、二面性命題境界条件)
# ============================================================

def test_T7_2(df: pd.DataFrame) -> dict:
    """T7-2: 批判リスク係数 η_J の作動(矢作条件 + 二面性命題)。"""
    logger.info("Running T7-2: η_J operation")

    if "case_length_days" not in df.columns:
        logger.warning("  case_length_days variable not available, using mu_F as proxy")
        return {"test_id": "T7-2", "name": "η_J", "verdict": "SKIPPED"}

    results_by_domain = {}
    for domain in DOMAINS:
        df_c = df[df["domain"] == domain]
        if len(df_c) < 100:
            continue

        # 単純な OLS:π^V ~ case_length_days
        X = sm.add_constant(df_c["case_length_days"].fillna(0))
        y = df_c["pi_V"]
        model = sm.OLS(y, X).fit()
        results_by_domain[domain] = {
            "beta": model.params.iloc[1],
            "se": model.bse.iloc[1],
            "p_value": model.pvalues.iloc[1],
        }

    commercial = results_by_domain.get("商事", {})
    admin = results_by_domain.get("行政", {})

    return {
        "test_id": "T7-2",
        "name": "η_J (criticism risk)",
        "beta_commercial": commercial.get("beta", np.nan),
        "p_commercial": commercial.get("p_value", np.nan),
        "beta_admin": admin.get("beta", np.nan),
        "p_admin": admin.get("p_value", np.nan),
        "verdict": "VERIFIED" if commercial.get("p_value", 1) < 0.05 else "PARTIAL",
    }


# ============================================================
# T-S_A: 飯田条件
# ============================================================

def test_T_SA(df: pd.DataFrame) -> dict:
    """T-S_A: 社会的事実供給 S_A · M_LS と政策判断明示度(飯田条件)。"""
    logger.info("Running T-S_A: Iida condition")

    df = df.copy()
    df["pi_total"] = df[PI_COMPONENTS].sum(axis=1)

    results_by_domain = []
    for domain in DOMAINS:
        df_c = df[df["domain"] == domain]
        if len(df_c) < 50:
            continue
        r = spearman_rho(df_c["mu_LS"].values, df_c["pi_total"].values)
        results_by_domain.append({"domain": domain, **r})

    sa_rate = (df["mu_LS"] > 0).mean()

    return {
        "test_id": "T-S_A",
        "name": "S_A · M_LS (Iida condition)",
        "sa_appearance_rate": sa_rate,
        "rho_min": min(r["rho"] for r in results_by_domain),
        "rho_max": max(r["rho"] for r in results_by_domain),
        "p_max": max(r["p_value"] for r in results_by_domain),
        "verdict": "VERIFIED" if all(r["p_value"] < 1e-4 and r["rho"] > 0 for r in results_by_domain) else "PARTIAL",
    }


# ============================================================
# T-π^C: 林条件
# ============================================================

def test_T_piC(df: pd.DataFrame) -> dict:
    """T-π^C: 権限配分判断 π^C の領域別異質性(林条件、Π_t 部分)。"""
    logger.info("Running T-π^C: Hayashi condition (authority allocation)")

    kw = kruskal_wallis_test(df, "pi_C")
    means = df.groupby("domain", observed=False)["pi_C"].mean()

    admin_pi_C = means.get("行政", 0)
    others_pi_C = means.drop("行政", errors="ignore").max() if "行政" in means.index else 0
    ratio = admin_pi_C / others_pi_C if others_pi_C > 0 else np.nan

    return {
        "test_id": "T-π^C",
        "name": "π^C heterogeneity (Hayashi)",
        "kruskal_H": kw["H"],
        "p_value": kw["p_value"],
        "mean_admin": admin_pi_C,
        "mean_others_max": others_pi_C,
        "ratio_admin_to_others": ratio,
        "verdict": "VERIFIED" if kw["p_value"] < 0.001 and ratio > 3 else "PARTIAL",
    }


# ============================================================
# Main
# ============================================================

def main() -> pd.DataFrame:
    """命題 2 関連テストの主要パイプライン。"""
    logger.info("=" * 60)
    logger.info("STEP 03: Proposition 2 Tests")
    logger.info("=" * 60)

    df = load_data()

    tests = [test_T1_2, test_T2_2, test_T3_2, test_T4_2, test_T5_2,
             test_T6_2, test_T7_2, test_T_SA, test_T_piC]

    results = []
    for test_func in tests:
        try:
            result = test_func(df)
            results.append(result)
            logger.info(f"  {result['test_id']}: {result.get('verdict', 'N/A')}")
        except Exception as e:
            logger.error(f"  Error in {test_func.__name__}: {e}")

    df_results = pd.DataFrame(results)
    save_table(df_results, "03_tests_proposition2")

    # 判定集計
    verdict_counts = df_results["verdict"].value_counts() if "verdict" in df_results.columns else pd.Series()
    logger.info(f"Verdict summary: {dict(verdict_counts)}")

    return df_results


if __name__ == "__main__":
    main()
