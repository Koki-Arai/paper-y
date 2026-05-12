"""
07. 反実仮想シミュレーション

論文 X §5.3.3 で提示される判決書式改善の制度設計含意を、
SMM 構造推定パラメータ(§3.3 で同定された $\\hat{\\theta}_c$)を用いて
定量的に検証する。

シナリオ:
    S1: 商事領域における $\\kappa$ 削減処方(政策衡量チェックリストの標準化)
        → 期待効果:$\\bar{\\pi}$ 増分 +48.1%
    S2: 行政領域における $M_{LS}$ 増加処方(社会調査・統計データの充実)
        → 期待効果:$\\bar{\\pi}$ 増分 +30.0%
    S3: 民事領域における $D_t$ 連続化 + $M_{LS}$ 充実
        → 期待効果:$\\bar{\\pi}$ 適正水準回復
    S4: 刑事領域における $M_{LS}$ 充実 + 裁判員制度活用

Usage:
    python src/07_counterfactual.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import (
    DOMAINS,
    get_logger,
    load_data,
    save_table,
    save_figure,
)

logger = get_logger(__name__)


# ============================================================
# パラメータ(SMM 推定値、§02 から)
# ============================================================

THETA_BASELINE = {
    "民事": {"beta": 0.728, "kappa": 0.214, "eta_J": 0.156, "M_AV": 0.587, "M_H": 0.612, "M_LS": 0.541, "K_c": 0.658},
    "商事": {"beta": 0.612, "kappa": 0.342, "eta_J": 0.241, "M_AV": 0.823, "M_H": 0.745, "M_LS": 0.376, "K_c": 0.847},
    "行政": {"beta": 0.681, "kappa": 0.198, "eta_J": 0.169, "M_AV": 0.524, "M_H": 0.598, "M_LS": 0.498, "K_c": 0.612},
    "刑事": {"beta": 0.823, "kappa": 0.176, "eta_J": 0.092, "M_AV": 0.412, "M_H": 0.382, "M_LS": 0.621, "K_c": 0.376},
}


# ============================================================
# 還元形政策判断明示度関数
# ============================================================

def predict_pi_bar(theta: dict, D_t: float = 0.0) -> float:
    """
    効用関数パラメータと $D_t$ 制約から、平均政策判断明示度 $\\bar{\\pi}$ を予測する。

    還元形:
    $\\bar{\\pi} = a_0 + a_1 \\beta - a_2 \\kappa + a_3 M_{LS} + a_4 K_c - a_5 D_t \\cdot \\lambda_{stat}$

    Parameters
    ----------
    theta : dict
        効用関数パラメータ
    D_t : float
        民主的拘束力(立法的明確化の水準、0 〜 1)

    Returns
    -------
    pi_bar : float
        予測される平均政策判断明示度
    """
    a0, a1, a2, a3, a4, a5 = 0.1, 0.4, 0.3, 0.2, 0.15, 0.5
    lambda_stat = 0.6 if D_t > 0.5 else 0.1  # シャドウ・プライス

    pi_bar = (
        a0
        + a1 * theta["beta"]
        - a2 * theta["kappa"]
        + a3 * theta["M_LS"]
        + a4 * theta["K_c"]
        - a5 * D_t * lambda_stat
    )

    return float(np.clip(pi_bar, 0, 1))


# ============================================================
# シナリオ別反実仮想シミュレーション
# ============================================================

def scenario_S1_kappa_reduction(theta: dict, kappa_reduction: float = 0.5) -> dict:
    """S1: $\\kappa$ 削減処方(判決書式改善)。"""
    theta_new = theta.copy()
    theta_new["kappa"] = theta["kappa"] * (1 - kappa_reduction)

    pi_baseline = predict_pi_bar(theta)
    pi_treated = predict_pi_bar(theta_new)
    delta_pct = (pi_treated - pi_baseline) / pi_baseline * 100

    return {
        "scenario": "S1: κ reduction (kappa × 0.5)",
        "intervention": "Standardize policy-balancing checklists",
        "pi_baseline": pi_baseline,
        "pi_treated": pi_treated,
        "delta_pct": delta_pct,
    }


def scenario_S2_M_LS_increase(theta: dict, M_LS_boost: float = 1.5) -> dict:
    """S2: $M_{LS}$ 増加処方(社会調査・統計データの充実)。"""
    theta_new = theta.copy()
    theta_new["M_LS"] = min(theta["M_LS"] * M_LS_boost, 1.0)

    pi_baseline = predict_pi_bar(theta)
    pi_treated = predict_pi_bar(theta_new)
    delta_pct = (pi_treated - pi_baseline) / pi_baseline * 100

    return {
        "scenario": "S2: M_LS increase (M_LS × 1.5)",
        "intervention": "Enhance social facts & statistical data sources",
        "pi_baseline": pi_baseline,
        "pi_treated": pi_treated,
        "delta_pct": delta_pct,
    }


def scenario_S3_D_t_continualization(theta: dict, D_t_baseline: float = 0.7, D_t_new: float = 0.3) -> dict:
    """S3: $D_t$ 連続化(高 $D_t$ を中間水準へ緩和)。"""
    pi_baseline = predict_pi_bar(theta, D_t=D_t_baseline)
    pi_treated = predict_pi_bar(theta, D_t=D_t_new)
    delta_pct = (pi_treated - pi_baseline) / pi_baseline * 100

    return {
        "scenario": f"S3: D_t continualization ({D_t_baseline} → {D_t_new})",
        "intervention": "Relax statutory constraint shadow price",
        "pi_baseline": pi_baseline,
        "pi_treated": pi_treated,
        "delta_pct": delta_pct,
    }


def scenario_S4_combined(theta: dict, kappa_red: float = 0.3, M_LS_boost: float = 1.3) -> dict:
    """S4: 複合処方($\\kappa$ 中程度削減 + $M_{LS}$ 中程度増加)。"""
    theta_new = theta.copy()
    theta_new["kappa"] = theta["kappa"] * (1 - kappa_red)
    theta_new["M_LS"] = min(theta["M_LS"] * M_LS_boost, 1.0)

    pi_baseline = predict_pi_bar(theta)
    pi_treated = predict_pi_bar(theta_new)
    delta_pct = (pi_treated - pi_baseline) / pi_baseline * 100

    return {
        "scenario": "S4: Combined (κ × 0.7, M_LS × 1.3)",
        "intervention": "Moderate κ reduction + M_LS enhancement",
        "pi_baseline": pi_baseline,
        "pi_treated": pi_treated,
        "delta_pct": delta_pct,
    }


# ============================================================
# 領域別シナリオ実行
# ============================================================

def run_all_scenarios(theta_by_domain: dict) -> pd.DataFrame:
    """全領域 × 全シナリオの反実仮想シミュレーション。"""
    results = []

    for domain in DOMAINS:
        theta = theta_by_domain[domain]

        for scenario_func in [scenario_S1_kappa_reduction, scenario_S2_M_LS_increase,
                              scenario_S3_D_t_continualization, scenario_S4_combined]:
            result = scenario_func(theta)
            result["domain"] = domain
            results.append(result)

    return pd.DataFrame(results)


# ============================================================
# 可視化
# ============================================================

def plot_counterfactual_effects(df_results: pd.DataFrame) -> None:
    """反実仮想効果のヒートマップ。"""
    pivot = df_results.pivot_table(
        values="delta_pct",
        index="domain",
        columns="scenario",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    import matplotlib.colors as mcolors
    norm = mcolors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", norm=norm)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # 値のアノテーション
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:+.1f}%", ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, label="Δπ̄ (%)")
    ax.set_title("Counterfactual Simulation: Δπ̄ by Domain × Scenario")
    plt.tight_layout()

    save_figure(fig, "07_counterfactual_heatmap")
    plt.close(fig)


# ============================================================
# 優先順位の決定
# ============================================================

def determine_priorities(df_results: pd.DataFrame) -> pd.DataFrame:
    """各領域における処方の優先順位を決定する。"""
    priorities = []
    for domain in DOMAINS:
        df_c = df_results[df_results["domain"] == domain].copy()
        df_c = df_c.sort_values("delta_pct", ascending=False)
        df_c["rank"] = range(1, len(df_c) + 1)
        priorities.append(df_c)

    return pd.concat(priorities, ignore_index=True)


# ============================================================
# Main
# ============================================================

def main() -> pd.DataFrame:
    """反実仮想シミュレーションの主要パイプライン。"""
    logger.info("=" * 60)
    logger.info("STEP 07: Counterfactual Simulation")
    logger.info("=" * 60)

    df_results = run_all_scenarios(THETA_BASELINE)
    save_table(df_results, "07_counterfactual_results")

    # 優先順位の決定
    df_priorities = determine_priorities(df_results)
    save_table(df_priorities, "07_counterfactual_priorities")

    # 可視化
    plot_counterfactual_effects(df_results)

    # サマリー出力
    logger.info("Counterfactual results:")
    for _, row in df_results.iterrows():
        logger.info(
            f"  {row['domain']:8s} | {row['scenario']:40s} | Δπ̄ = {row['delta_pct']:+.1f}%"
        )

    # 領域別最大効果処方
    logger.info("\nTop priority interventions by domain:")
    top_per_domain = df_results.loc[df_results.groupby("domain")["delta_pct"].idxmax()]
    for _, row in top_per_domain.iterrows():
        logger.info(f"  {row['domain']}: {row['scenario']} (+{row['delta_pct']:.1f}%)")

    return df_results


if __name__ == "__main__":
    main()
