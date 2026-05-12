"""
全体実行スクリプト

論文 Y(荒井 2026 実証編)の全分析を順次実行する。

実行順序:
    01_data_loader.py
    02_smm_estimation.py
    03_tests_p2.py
    04_tests_p3.py
    05_did_analysis.py
    06_pca_analysis.py
    07_counterfactual.py

Usage:
    python src/run_all.py
"""

import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)


def import_module_from_path(module_name: str, file_path: str):
    """ファイルパスから Python モジュールをロードする。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """全体パイプラインを実行する。"""
    logger.info("=" * 60)
    logger.info("Paper Y Replication: Full Pipeline")
    logger.info("=" * 60)

    src_dir = Path(__file__).parent
    scripts = [
        ("01_data_loader", "01_data_loader.py"),
        ("02_smm_estimation", "02_smm_estimation.py"),
        ("03_tests_p2", "03_tests_p2.py"),
        ("04_tests_p3", "04_tests_p3.py"),
        ("05_did_analysis", "05_did_analysis.py"),
        ("06_pca_analysis", "06_pca_analysis.py"),
        ("07_counterfactual", "07_counterfactual.py"),
    ]

    for name, filename in scripts:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# Running {filename}")
        logger.info(f"{'#' * 60}")
        try:
            module = import_module_from_path(name, src_dir / filename)
            if hasattr(module, "main"):
                module.main()
            logger.info(f"  → {filename} completed successfully")
        except Exception as e:
            logger.error(f"  → Error in {filename}: {e}")
            raise

    logger.info("\n" + "=" * 60)
    logger.info("Full pipeline completed successfully")
    logger.info("Results saved to: results/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
