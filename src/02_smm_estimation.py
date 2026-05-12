"""
02. SMM(Simulated Method of Moments)構造推定

論文 X の効用関数パラメータを領域別に同定する。

推定対象パラメータ(領域 c ごと):
    β_c   制度的厚生重み
    κ_c   説明費用
    η_J^c 批判リスク
    M_AV^c 媒介能力(西内条件)
    M_H^c 歴史媒介(石川条件)
    M_LS^c 法社会学媒介(飯田条件)
    K_c^c 文脈成熟度(藤谷条件)

モーメント:
    E[μ^F | c], E[μ^T | c], E[π^V | c], E[π^C | c],
    E[π^H | c], E[π^M | c], Var(Λ | c) など

Usage:
    python src/02_smm_estimation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils import (
    DOMAINS,
    PI_COMPONENTS,
    get_logger,
    load_data,
    save_table,
)
from src.__init__ import __version__  # noqa

logger = get_logger(__name__)


# ============================================================
# シミュレーション関数
# ============================================================

def simulate_choices(theta: np.ndarray, n_sim: int, rng: np.random.Generator) -> dict:
    """
    効用関数パラメータ θ から、シミュレートされたモーメントを計算する。

    Parameters
    ----------
    theta : np.ndarray
        パラメータベクトル [β, κ, η_J, M_AV, M_H, M_LS, K_c]
    n_sim : int
        シミュレーション回数
    rng : np.random.Generator
        乱数生成器

    Returns
    -------
    moments : dict
        シミュレートされたモーメント
    """
    beta, kappa, eta_J, M_AV, M_H, M_LS, K_c = theta

    # 文脈成熟度 K_c に応じて受け手分布をシミュレート
    lambda_E = K_c
    lambda_M = 1 - K_c
    lambda_P = 0.0  # 簡略化:E と M のみ

    # ベクトル化されたシミュレーション
    # μ^F の最適化:批判リスク η_J が高いほど縮減
    mu_F_sim = np.clip(0.1 + 0.05 / (1 + eta_J) - 0.02 * lambda_E + 0.01 * rng.standard_normal(n_sim), 0, 1)

    # μ^T の最適化:媒介能力 M_AV が閾値を超える場合に活性化
    M_bar = 0.5  # 媒介能力臨界閾値
    mu_T_sim = np.where(
        M_AV > M_bar,
        np.clip(0.1 + 0.3 * (M_AV - M_bar) + 0.02 * rng.standard_normal(n_sim), 0, 1),
        np.clip(0.02 + 0.01 * rng.standard_normal(n_sim), 0, 0.05),
    )

    # 政策判断明示度
    pi_V_sim = np.clip(0.2 + 0.4 * beta - 0.3 * kappa + 0.02 * rng.standard_normal(n_sim), 0, 1)
    pi_C_sim = np.clip(0.1 + 0.05 * (1 - K_c) + 0.02 * rng.standard_normal(n_sim), 0, 1)
    pi_H_sim = np.clip(0.05 + 0.15 * M_H + 0.02 * rng.standard_normal(n_sim), 0, 1)
    pi_M_S_sim = np.clip(0.1 + 0.1 * K_c + 0.02 * rng.standard_normal(n_sim), 0, 1)
    pi_M_R_sim = np.clip(0.05 + 0.05 * kappa + 0.02 * rng.standard_normal(n_sim), 0, 1)

    # 法社会学媒介
    mu_LS_sim = np.clip(0.1 + 0.4 * M_LS + 0.02 * rng.standard_normal(n_sim), 0, 1)

    return {
        "E_mu_F": np.mean(mu_F_sim),
        "E_mu_T": np.mean(mu_T_sim),
        "E_pi_V": np.mean(pi_V_sim),
        "E_pi_C": np.mean(pi_C_sim),
        "E_pi_H": np.mean(pi_H_sim),
        "E_pi_M_S": np.mean(pi_M_S_sim),
        "E_pi_M_R": np.mean(pi_M_R_sim),
        "E_mu_LS": np.mean(mu_LS_sim),
        "Var_pi_total": np.var(pi_V_sim + pi_C_sim + pi_H_sim + pi_M_S_sim + pi_M_R_sim),
    }


def empirical_moments(df_c: pd.DataFrame) -> dict:
    """データから経験的モーメントを計算する。"""
    return {
        "E_mu_F": df_c["mu_F"].mean(),
        "E_mu_T": df_c["mu_T"].mean(),
        "E_pi_V": df_c["pi_V"].mean(),
        "E_pi_C": df_c["pi_C"].mean(),
        "E_pi_H": df_c["pi_H"].mean(),
        "E_pi_M_S": df_c["pi_M_S"].mean(),
        "E_pi_M_R": df_c["pi_M_R"].mean(),
        "E_mu_LS": df_c["mu_LS"].mean(),
        "Var_pi_total": df_c[PI_COMPONENTS].sum(axis=1).var(),
    }


def smm_objective(theta: np.ndarray, m_data: dict, n_sim: int, seed: int) -> float:
    """SMM 目的関数(モーメント間の二乗距離)。"""
    rng = np.random.default_rng(seed)
    m_sim = simulate_choices(theta, n_sim, rng)

    diff = np.array([m_data[k] - m_sim[k] for k in m_data.keys()])
    # 単位重み行列(対角)
    return float(diff @ diff)


def estimate_domain(df_c: pd.DataFrame, domain_name: str, n_sim: int = 5000, seed: int = 42) -> dict:
    """ある領域における SMM 推定を実行する。"""
    logger.info(f"  Estimating parameters for domain: {domain_name}")

    m_data = empirical_moments(df_c)

    # 初期値とパラメータ範囲
    theta_init = np.array([0.7, 0.2, 0.15, 0.6, 0.6, 0.5, 0.65])
    bounds = [(0.1, 1.0)] * 7

    result = minimize(
        smm_objective,
        theta_init,
        args=(m_data, n_sim, seed),
        method="L-BFGS-B",
        bounds=bounds,
    )

    return {
        "domain": domain_name,
        "beta": result.x[0],
        "kappa": result.x[1],
        "eta_J": result.x[2],
        "M_AV": result.x[3],
        "M_H": result.x[4],
        "M_LS": result.x[5],
        "K_c": result.x[6],
        "objective": result.fun,
        "converged": result.success,
    }


# ============================================================
# 標準誤差(デルタ法)
# ============================================================

def delta_method_se(theta_hat: np.ndarray, df_c: pd.DataFrame, n_sim: int, seed: int) -> np.ndarray:
    """
    デルタ法による標準誤差を計算する。
    ブートストラップが計算量で実行困難な場合の代替手法。
    """
    n = len(df_c)
    eps = 1e-4
    grad = np.zeros((9, 7))  # 9 モーメント × 7 パラメータ

    for j in range(7):
        theta_plus = theta_hat.copy()
        theta_minus = theta_hat.copy()
        theta_plus[j] += eps
        theta_minus[j] -= eps

        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        m_plus = simulate_choices(theta_plus, n_sim, rng1)
        m_minus = simulate_choices(theta_minus, n_sim, rng2)

        for i, key in enumerate(m_plus.keys()):
            grad[i, j] = (m_plus[key] - m_minus[key]) / (2 * eps)

    # データの共分散行列(対角近似)
    sigma_data = np.eye(9) / n

    # デルタ法による分散
    try:
        var_theta = np.linalg.pinv(grad.T @ grad) @ grad.T @ sigma_data @ grad @ np.linalg.pinv(grad.T @ grad)
        se = np.sqrt(np.diag(var_theta))
    except np.linalg.LinAlgError:
        se = np.full(7, np.nan)

    return se


def main() -> pd.DataFrame:
    """SMM 推定の主要パイプライン。"""
    logger.info("=" * 60)
    logger.info("STEP 02: SMM Structural Estimation")
    logger.info("=" * 60)

    df = load_data()

    results = []
    for domain in DOMAINS:
        df_c = df[df["domain"] == domain]
        if len(df_c) < 50:
            logger.warning(f"  Skipping {domain} (only {len(df_c)} observations)")
            continue

        est = estimate_domain(df_c, domain)
        results.append(est)

    df_est = pd.DataFrame(results)
    save_table(df_est, "02_smm_estimates")

    logger.info("SMM estimation complete:")
    logger.info(f"\n{df_est.round(3).to_string(index=False)}")

    return df_est


if __name__ == "__main__":
    main()
