"""
run_pipeline.py
===============
Master pipeline script: raw data → feature matrix.

Usage:
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --stage ingest     # ingestion only
    python run_pipeline.py --stage eda        # diagnostic EDA only
    python run_pipeline.py --stage timeseries # time series features only
    python run_pipeline.py --stage validate --validation-level basic
    python run_pipeline.py --stage validate --validation-level strict
    python run_pipeline.py --dev              # dev subset (10 tickers)
"""

import argparse
import logging
import time
from typing import Literal

import yaml

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("pipeline")


def load_config(path: str = "pipeline_config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_ingest(config: dict, dev: bool = False) -> None:
    log.info("── STAGE: INGESTION ──────────────────────────────")
    from src.ingestion.price_loader import load_prices
    from src.ingestion.news_loader import load_news
    from src.ingestion.fred_fetcher import fetch_macro

    tickers = config["tickers"]["dev_subset"] if dev else None
    load_prices(config, tickers=tickers)
    load_news(config)
    fetch_macro(config)


def stage_timeseries(config: dict) -> None:
    log.info("── STAGE: TIME SERIES FEATURES ──────────────────")
    from src.timeseries.features import build_price_features

    build_price_features(config)


def stage_eda(config: dict) -> None:
    log.info("── STAGE: DIAGNOSTIC EDA ──────────────────────")
    from src.eda import run_diagnostic_eda

    run_diagnostic_eda(config)


def stage_nlp(config: dict) -> None:
    log.info("── STAGE: NLP PREPROCESSING ─────────────────────")
    from src.nlp.cleaner import clean_news
    from src.nlp.sentiment import run_finbert

    clean_news(config)
    run_finbert(config)


def stage_integrate(config: dict) -> None:
    log.info("── STAGE: INTEGRATION ───────────────────────────")
    from src.integration.joiner import join_all_sources
    from src.integration.feature_store import save_feature_matrix

    feature_matrix = join_all_sources(config)
    save_feature_matrix(feature_matrix, config)


def stage_validate(config: dict, validation_level: Literal["basic", "strict"] | None = None) -> None:
    log.info("── STAGE: VALIDATION ────────────────────────────")
    from src.utils.validators import run_all_validations

    run_all_validations(config, level=validation_level)


def main() -> None:
    parser = argparse.ArgumentParser(description="Market Pulse Lab Pipeline")
    parser.add_argument(
        "--stage",
        choices=["ingest", "eda", "timeseries", "nlp", "integrate", "validate", "all"],
        default="all",
    )
    parser.add_argument("--dev", action="store_true", help="Use dev subset (10 tickers)")
    parser.add_argument(
        "--validation-level",
        choices=["basic", "strict"],
        default=None,
        help="Validation strictness for validate stage (default: from config).",
    )
    args = parser.parse_args()

    config = load_config()
    t0 = time.time()

    stages = {
        "ingest":     lambda: stage_ingest(config, dev=args.dev),
        "eda":        lambda: stage_eda(config),
        "timeseries": lambda: stage_timeseries(config),
        "nlp":        lambda: stage_nlp(config),
        "integrate":  lambda: stage_integrate(config),
        "validate":   lambda: stage_validate(config, validation_level=args.validation_level),
    }

    if args.stage == "all":
        for fn in stages.values():
            fn()
    else:
        stages[args.stage]()

    elapsed = time.time() - t0
    log.info("Pipeline completed in %.1fs", elapsed)


if __name__ == "__main__":
    main()
