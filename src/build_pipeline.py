"""End-to-end pipeline runner for Quant RAG Copilot.

Orchestrates:
- Stage 1: Market Data Ingestion (market_loader.py)
- Stage 2: Technical Feature Engineering (feature_eng.py)
- Stage 3: News Sentiment Ingestion & FinBERT Scoring (news_sentiment.py)
- Stage 4: Baseline Forecasting & Walk-Forward Validation (baseline_forecasting.py)
"""

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_pipeline")


def run_stage_market() -> None:
    """Execute Stage 1: Market data download."""
    logger.info("Executing Stage 1: Market Data Ingestion...")
    from src.data_ingestion.market_loader import main as market_main
    market_main()


def run_stage_features() -> None:
    """Execute Stage 2: Technical feature engineering."""
    logger.info("Executing Stage 2: Technical Feature Engineering...")
    from src.features.feature_eng import main as features_main
    features_main()


def run_stage_sentiment() -> None:
    """Execute Stage 3: News sentiment pipeline."""
    logger.info("Executing Stage 3: News Sentiment Ingestion & FinBERT Scoring...")
    from src.data_ingestion.news_sentiment import run_production_pipeline
    run_production_pipeline()


def run_stage_forecasting() -> None:
    """Execute Stage 4: Baseline forecasting and walk-forward validation."""
    logger.info("Executing Stage 4: Baseline Forecasting & Walk-Forward Validation...")
    from src.models.baseline_forecasting import main as forecasting_main
    forecasting_main()


def run_all_tests() -> bool:
    """Run test suites across all project modules."""
    logger.info("Executing automated test suites across all pipeline stages...")
    # 1. Sentiment pipeline tests
    from src.data_ingestion.news_sentiment import run_test_mode
    logger.info("--- Testing Stage 3: News Sentiment ---")
    try:
        run_test_mode()
        logger.info("Stage 3: News Sentiment tests PASSED.")
    except Exception as exc:
        logger.error(f"Sentiment test suite failed: {exc}")
        return False

    # 2. Forecasting & walk-forward tests
    from src.models.baseline_forecasting import run_test_suite
    logger.info("--- Testing Stage 4: Baseline Forecasting & Validation ---")
    forecasting_ok = run_test_suite()
    if not forecasting_ok:
        logger.error("Forecasting test suite failed!")
        return False

    logger.info("ALL PIPELINE TEST SUITES PASSED SUCCESSFULLY!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant RAG Copilot Pipeline Runner")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["market", "features", "sentiment", "forecasting", "all"],
        default="forecasting",
        help="Pipeline stage to execute (default: forecasting).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run comprehensive test suites for validation.",
    )

    args = parser.parse_args()

    if args.test:
        success = run_all_tests()
        sys.exit(0 if success else 1)

    if args.stage == "market":
        run_stage_market()
    elif args.stage == "features":
        run_stage_features()
    elif args.stage == "sentiment":
        run_stage_sentiment()
    elif args.stage == "forecasting":
        run_stage_forecasting()
    elif args.stage == "all":
        run_stage_market()
        run_stage_features()
        run_stage_sentiment()
        run_stage_forecasting()


if __name__ == "__main__":
    main()
