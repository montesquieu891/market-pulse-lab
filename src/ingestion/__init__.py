"""Data ingestion modules."""

from .fred_fetcher import fetch_macro
from .news_loader import load_news
from .price_loader import load_prices

__all__ = ["load_prices", "load_news", "fetch_macro"]
