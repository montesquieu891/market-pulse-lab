from __future__ import annotations

import importlib
import importlib.util
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve required NLP stage paths.

    Args:
        config: Parsed pipeline configuration dictionary.

    Returns:
        Dictionary of input/output/report paths.
    """
    paths_cfg = config.get("paths", {})
    return {
        "interim_news": Path(paths_cfg.get("interim_news", "datasets/interim/news_cleaned.parquet")),
        "news_features": Path(paths_cfg.get("news_features", "datasets/features/news_features.parquet")),
        "tfidf_report": Path("reports/nlp_tfidf_top_terms.md"),
        "phase4_report": Path("reports/nlp_phase4_diagnostics.md"),
        "embedding_plot": Path("reports/plots/nlp_embedding_umap_sample.png"),
    }


def _load_stopwords(extra_stopwords: list[str]) -> list[str]:
    """Build stopword list for vectorization.

    Args:
        extra_stopwords: Domain-specific stopwords from config.

    Returns:
        Combined stopword list.
    """
    words: set[str] = set(map(str.lower, extra_stopwords))

    if importlib.util.find_spec("nltk") is None:
        return sorted(words)

    nltk = importlib.import_module("nltk")
    stopwords_module = importlib.import_module("nltk.corpus")
    stopwords = getattr(stopwords_module, "stopwords")

    with suppress(LookupError):
        words.update(stopwords.words("english"))

    if not words and hasattr(nltk, "download"):
        with suppress(Exception):
            getattr(nltk, "download")("stopwords", quiet=True)
            words.update(stopwords.words("english"))

    return sorted(words)

def _build_text(news_df: pd.DataFrame) -> pd.Series:
    """Return modeling text, preferring cleaned_text when available.

    Args:
        news_df: Input news dataframe.

    Returns:
        Text series aligned to dataframe rows.
    """
    if "cleaned_text" in news_df.columns:
        return news_df["cleaned_text"].fillna("").astype("string")

    article = news_df["article"].fillna("").astype("string").str.strip()
    title = news_df["title"].fillna("").astype("string").str.strip()
    return article.where(article != "", title)


def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D with a NumPy PCA fallback.

    Args:
        embeddings: Dense embedding matrix.

    Returns:
        Two-dimensional projection.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


class _SimpleTfidfVectorizer:
    """Small TF-IDF fallback when scikit-learn is unavailable."""

    def __init__(self, max_features: int, ngram_range: tuple[int, int], stop_words: set[str]) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def _tokenize(self, text: str) -> list[str]:
        tokens = [tok.strip(".,;:!?()[]{}\"'").lower() for tok in text.split() if tok.strip()]
        tokens = [tok for tok in tokens if tok and tok not in self.stop_words]

        all_terms: list[str] = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            if n <= 0 or len(tokens) < n:
                continue
            if n == 1:
                all_terms.extend(tokens)
            else:
                all_terms.extend(" ".join(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1))
        return all_terms

    def fit(self, texts: list[str]) -> "_SimpleTfidfVectorizer":
        doc_freq: dict[str, int] = {}
        n_docs = len(texts)

        for text in texts:
            terms = set(self._tokenize(text))
            for term in terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        sorted_terms = sorted(doc_freq.items(), key=lambda kv: kv[1], reverse=True)
        if self.max_features > 0:
            sorted_terms = sorted_terms[: self.max_features]

        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(sorted_terms)}

        idf = np.zeros(len(self.vocabulary_), dtype=np.float32)
        for term, idx in self.vocabulary_.items():
            df = doc_freq.get(term, 1)
            idf[idx] = np.log((1 + n_docs) / (1 + df)) + 1.0
        self.idf_ = idf
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf_ is None:
            raise ValueError("Vectorizer must be fit before transform")

        matrix = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float32)
        for row, text in enumerate(texts):
            terms = self._tokenize(text)
            if not terms:
                continue

            term_counts: dict[int, int] = {}
            for term in terms:
                idx = self.vocabulary_.get(term)
                if idx is None:
                    continue
                term_counts[idx] = term_counts.get(idx, 0) + 1

            total = sum(term_counts.values())
            if total == 0:
                continue

            for idx, count in term_counts.items():
                tf = count / total
                matrix[row, idx] = tf * self.idf_[idx]

        return matrix

    def get_feature_names_out(self) -> np.ndarray:
        features = [term for term, _ in sorted(self.vocabulary_.items(), key=lambda kv: kv[1])]
        return np.asarray(features, dtype=object)


def _fit_tfidf(
    train_text: pd.Series,
    max_features: int,
    ngram_range: tuple[int, int],
    stop_words: list[str],
) -> Any:
    """Fit TF-IDF vectorizer on training text only.

    Args:
        train_text: Training split text series.
        max_features: Vocabulary cap.
        ngram_range: N-gram range.
        stop_words: Stopword list.

    Returns:
        Fitted vectorizer.
    """
    text_values = train_text.fillna("").astype(str).tolist()

    if importlib.util.find_spec("sklearn") is not None:
        text_module = importlib.import_module("sklearn.feature_extraction.text")
        vectorizer_cls = getattr(text_module, "TfidfVectorizer")
        vectorizer = vectorizer_cls(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words if stop_words else None,
            lowercase=True,
        )
        vectorizer.fit(text_values)
        return vectorizer

    log.warning("scikit-learn not available; using simplified TF-IDF fallback implementation.")
    vectorizer = _SimpleTfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=set(stop_words),
    )
    vectorizer.fit(text_values)
    return vectorizer

def _write_tfidf_report(
    news_df: pd.DataFrame,
    vectorizer: Any,
    train_mask: pd.Series,
    out_path: Path,
) -> None:
    """Write top-20 TF-IDF terms report for AAPL (or fallback ticker).

    Args:
        news_df: News dataframe with text and ticker.
        vectorizer: Fitted TF-IDF vectorizer.
        train_mask: Boolean mask selecting train rows.
        out_path: Report path.
    """
    train_df = news_df.loc[train_mask].copy()
    if train_df.empty:
        train_df = news_df.copy()

    ticker = "AAPL" if (train_df["ticker"] == "AAPL").any() else str(train_df["ticker"].mode().iat[0])
    ticker_df = train_df[train_df["ticker"] == ticker]

    terms: list[tuple[str, float]] = []
    if not ticker_df.empty:
        matrix = vectorizer.transform(ticker_df["model_text"].fillna("").astype(str).tolist())
        mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
        vocab = vectorizer.get_feature_names_out()
        top_idx = np.argsort(mean_scores)[::-1][:20]
        terms = [(str(vocab[i]), float(mean_scores[i])) for i in top_idx if float(mean_scores[i]) > 0]

    lines = [
        "# TF-IDF Top Terms",
        "",
        f"Ticker analyzed: `{ticker}`",
        "",
        "Top terms from train-only TF-IDF vocabulary:",
        "",
    ]

    if terms:
        lines.extend([f"- `{term}`: {score:.6f}" for term, score in terms])
    else:
        lines.append("- No non-zero TF-IDF terms found for selected ticker.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_finbert_batches(
    texts: pd.Series,
    model_name: str,
    cache_dir: str,
    max_tokens: int,
    batch_size: int,
) -> pd.DataFrame:
    """Run FinBERT inference in batches.

    Args:
        texts: Text series for inference.
        model_name: HuggingFace model id.
        cache_dir: Local model cache directory.
        max_tokens: Token truncation length.
        batch_size: Batch size for inference.

    Returns:
        Dataframe with score, label and probability columns.
    """
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    tokenizer_cls = getattr(transformers, "AutoTokenizer")
    model_cls = getattr(transformers, "AutoModelForSequenceClassification")

    tokenizer = tokenizer_cls.from_pretrained(model_name, cache_dir=cache_dir)
    model = model_cls.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    id2label_raw = getattr(model.config, "id2label", {})
    id2label: dict[int, str] = {}
    for key, value in id2label_raw.items():
        if isinstance(key, int):
            idx = key
        elif isinstance(key, str) and key.isdigit():
            idx = int(key)
        else:
            continue
        id2label[idx] = str(value).lower()

    if not id2label:
        id2label = {0: "positive", 1: "negative", 2: "neutral"}

    pos_idx = next((k for k, v in id2label.items() if "pos" in v), 0)
    neg_idx = next((k for k, v in id2label.items() if "neg" in v), 1)
    neu_idx = next((k for k, v in id2label.items() if "neu" in v), 2)

    n_rows = len(texts)
    sentiment_score = np.zeros(n_rows, dtype=np.float32)
    sentiment_label: list[str] = ["neutral"] * n_rows
    prob_pos = np.zeros(n_rows, dtype=np.float32)
    prob_neg = np.zeros(n_rows, dtype=np.float32)
    prob_neu = np.zeros(n_rows, dtype=np.float32)

    text_values = texts.fillna("").astype(str).tolist()

    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        batch = text_values[start:end]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        prob_pos[start:end] = probs[:, pos_idx]
        prob_neg[start:end] = probs[:, neg_idx]
        prob_neu[start:end] = probs[:, neu_idx]
        sentiment_score[start:end] = probs[:, pos_idx] - probs[:, neg_idx]

        argmax_idx = probs.argmax(axis=1)
        for i, label_idx in enumerate(argmax_idx, start=start):
            label = id2label.get(int(label_idx), "neutral")
            if "pos" in label:
                sentiment_label[i] = "positive"
            elif "neg" in label:
                sentiment_label[i] = "negative"
            else:
                sentiment_label[i] = "neutral"

    return pd.DataFrame(
        {
            "finbert_sentiment_score": sentiment_score,
            "finbert_label": sentiment_label,
            "finbert_prob_positive": prob_pos,
            "finbert_prob_negative": prob_neg,
            "finbert_prob_neutral": prob_neu,
            "sentiment_model": "finbert",
        }
    )


def _run_lexicon_fallback(texts: pd.Series) -> pd.DataFrame:
    """Compute lightweight lexicon sentiment when FinBERT is unavailable.

    Args:
        texts: Text series.

    Returns:
        Dataframe with sentiment columns compatible with FinBERT output.
    """
    positive_words = {
        "beat",
        "beats",
        "growth",
        "surge",
        "gain",
        "gains",
        "upgrade",
        "upgraded",
        "profit",
        "strong",
        "bullish",
    }
    negative_words = {
        "miss",
        "missed",
        "loss",
        "losses",
        "drop",
        "drops",
        "downgrade",
        "downgraded",
        "weak",
        "bearish",
        "lawsuit",
    }

    def score_text(text: str) -> tuple[float, float, float, str]:
        tokens = [tok.strip(".,;:!?()[]{}\"'").lower() for tok in text.split() if tok.strip()]
        pos = sum(tok in positive_words for tok in tokens)
        neg = sum(tok in negative_words for tok in tokens)

        raw = pos - neg
        score = float(np.tanh(raw / 3.0))

        pos_prob = float(np.clip(0.34 + max(score, 0.0) * 0.6, 0.0, 1.0))
        neg_prob = float(np.clip(0.34 + max(-score, 0.0) * 0.6, 0.0, 1.0))
        neu_prob = float(np.clip(1.0 - pos_prob - neg_prob, 0.0, 1.0))

        if pos_prob >= neg_prob and pos_prob >= neu_prob:
            label = "positive"
        elif neg_prob >= pos_prob and neg_prob >= neu_prob:
            label = "negative"
        else:
            label = "neutral"

        return score, pos_prob, neg_prob, neu_prob, label

    rows = [score_text(text) for text in texts.fillna("").astype(str).tolist()]
    score_arr = np.array([r[0] for r in rows], dtype=np.float32)
    pos_arr = np.array([r[1] for r in rows], dtype=np.float32)
    neg_arr = np.array([r[2] for r in rows], dtype=np.float32)
    neu_arr = np.array([r[3] for r in rows], dtype=np.float32)
    label_arr = [str(r[4]) for r in rows]

    return pd.DataFrame(
        {
            "finbert_sentiment_score": score_arr,
            "finbert_label": label_arr,
            "finbert_prob_positive": pos_arr,
            "finbert_prob_negative": neg_arr,
            "finbert_prob_neutral": neu_arr,
            "sentiment_model": "lexicon_fallback",
        }
    )


def _aggregate_daily(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level sentiment to ticker x date features.

    Args:
        news_df: Article-level dataframe with sentiment columns.

    Returns:
        Daily aggregated feature dataframe.
    """
    grouped = news_df.groupby(["ticker", "date"], observed=True)

    agg = grouped.agg(
        finbert_sentiment_mean=("finbert_sentiment_score", "mean"),
        finbert_pct_positive=("finbert_label", lambda s: float((s == "positive").mean())),
        finbert_pct_negative=("finbert_label", lambda s: float((s == "negative").mean())),
        finbert_pct_neutral=("finbert_label", lambda s: float((s == "neutral").mean())),
        news_count=("finbert_label", "size"),
        char_count_mean=("char_count", "mean"),
        word_count_mean=("word_count", "mean"),
        avg_word_length_mean=("avg_word_length", "mean"),
    ).reset_index()

    return agg.sort_values(["ticker", "date"]).reset_index(drop=True)


def _lag_news_features(daily_df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """Apply per-ticker lag to all feature columns.

    Args:
        daily_df: Ticker x date aggregated dataframe.
        lag_days: Number of trading days to lag.

    Returns:
        Lagged feature dataframe.
    """
    if lag_days <= 0:
        return daily_df

    out = daily_df.copy()
    feature_cols = [col for col in out.columns if col not in {"ticker", "date"}]

    grouped = out.groupby("ticker", observed=True)
    for col in feature_cols:
        out[col] = grouped[col].shift(lag_days)

    return out


def _build_embedding_plot(news_df: pd.DataFrame, config: dict[str, Any], out_path: Path) -> str:
    """Generate 2D embedding visualization on a sample.

    Args:
        news_df: Article-level dataframe with model_text and sentiment_score.
        config: Parsed pipeline configuration dictionary.
        out_path: Plot output path.

    Returns:
        Method used for dimensionality reduction.
    """
    sample_size = int(config.get("nlp", {}).get("sbert_sample_size", 1500))
    sample_df = news_df.dropna(subset=["model_text"]).copy()
    if sample_df.empty:
        return "none"

    sample_df = sample_df.sample(n=min(sample_size, len(sample_df)), random_state=42)

    coords = np.zeros((len(sample_df), 2), dtype=np.float32)
    method = "none"

    if importlib.util.find_spec("sentence_transformers") is not None:
        with suppress(Exception):
            sentence_transformers = importlib.import_module("sentence_transformers")
            sentence_transformer_cls = getattr(sentence_transformers, "SentenceTransformer")

            model_name = str(config.get("nlp", {}).get("sbert_model", "sentence-transformers/all-MiniLM-L6-v2"))
            cache_dir = str(config.get("nlp", {}).get("sbert_cache_dir", "models/sbert_cache/"))
            model = sentence_transformer_cls(model_name, cache_folder=cache_dir)
            embeddings = model.encode(
                sample_df["model_text"].astype(str).tolist(),
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            if importlib.util.find_spec("umap") is not None:
                umap_module = importlib.import_module("umap")
                reducer_cls = getattr(umap_module, "UMAP")
                reducer = reducer_cls(n_components=2, random_state=42)
                coords = reducer.fit_transform(embeddings)
                method = "sbert_umap"
            else:
                coords = _pca_2d(np.asarray(embeddings, dtype=np.float32))
                method = "sbert_pca"

    if method == "none":
        fallback_vec = _SimpleTfidfVectorizer(max_features=256, ngram_range=(1, 2), stop_words=set())
        fallback_vec.fit(sample_df["model_text"].astype(str).tolist())
        tfidf_matrix = fallback_vec.transform(sample_df["model_text"].astype(str).tolist())
        coords = _pca_2d(np.asarray(tfidf_matrix, dtype=np.float32))
        method = "tfidf_pca_fallback"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=sample_df["finbert_sentiment_score"].to_numpy(),
        cmap="coolwarm",
        s=14,
        alpha=0.7,
    )
    ax.set_title(f"NLP Sample Embeddings ({method.upper()})")
    ax.set_xlabel("component_1")
    ax.set_ylabel("component_2")
    fig.colorbar(scatter, ax=ax, label="sentiment score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return method

def _write_phase4_report(
    news_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    phase4_report: Path,
    tfidf_report: Path,
    embedding_plot: Path,
    embedding_method: str,
    model_name: str,
) -> None:
    """Write compact diagnostics report for Phase 4 artifacts.

    Args:
        news_df: Article-level dataframe after inference.
        daily_df: Daily aggregated dataframe.
        phase4_report: Report output path.
        tfidf_report: TF-IDF report path.
        embedding_plot: Embedding plot path.
        embedding_method: Method used for 2D embedding projection.
        model_name: Sentiment model used for inference.
    """
    article_ratio = float((news_df["text_source"] == "article").mean()) if len(news_df) else 0.0
    title_ratio = float((news_df["text_source"] == "title").mean()) if len(news_df) else 0.0

    lines = [
        "# Phase 4 NLP Diagnostics",
        "",
        f"- Rows processed: {len(news_df)}",
        f"- Daily feature rows: {len(daily_df)}",
        f"- text_source article ratio: {article_ratio:.3f}",
        f"- text_source title ratio: {title_ratio:.3f}",
        f"- Sentiment model used: `{model_name}`",
        f"- TF-IDF report: `{tfidf_report.as_posix()}`",
        f"- Embedding plot: `{embedding_plot.as_posix()}`",
        f"- Embedding reducer: `{embedding_method}`",
        "",
        "Sentiment label distribution:",
        "",
    ]

    label_counts = news_df["finbert_label"].value_counts(dropna=False)
    if label_counts.empty:
        lines.append("- No labels produced.")
    else:
        for label, count in label_counts.items():
            lines.append(f"- {label}: {int(count)}")

    phase4_report.parent.mkdir(parents=True, exist_ok=True)
    phase4_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_finbert(config: dict[str, Any]) -> None:
    """Run sentiment inference and aggregate lagged daily news features.

    The stage enforces train-only fitting for TF-IDF vocabulary and applies a
    one-day lag to aggregated daily sentiment features to avoid leakage.

    Args:
        config: Parsed pipeline configuration dictionary.
    """
    paths = _resolve_paths(config)
    interim_news_path = paths["interim_news"]
    out_path = paths["news_features"]

    if not interim_news_path.exists():
        raise FileNotFoundError(f"Missing cleaned interim news parquet: {interim_news_path}")

    news_df = pd.read_parquet(interim_news_path)

    required_cols = {"ticker", "date", "title", "article", "text_source", "char_count", "word_count", "avg_word_length"}
    missing = sorted(required_cols - set(news_df.columns))
    if missing:
        raise ValueError(f"run_finbert requires cleaned columns missing from interim news: {missing}")

    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce", utc=True)
    news_df = news_df.dropna(subset=["date", "ticker"]).copy()
    news_df["ticker"] = news_df["ticker"].astype("string").str.upper()

    start_date = pd.to_datetime(config["dates"]["start"], utc=True)
    end_date = pd.to_datetime(config["dates"]["end"], utc=True)
    news_df = news_df[(news_df["date"] >= start_date) & (news_df["date"] <= end_date)].copy()
    news_df["date"] = news_df["date"].dt.normalize()

    news_df["model_text"] = _build_text(news_df).fillna("").astype("string")

    train_end = pd.to_datetime(config["dates"]["train_end"], utc=True)
    train_mask = news_df["date"] <= train_end

    nlp_cfg = config.get("nlp", {})
    max_features = int(nlp_cfg.get("tfidf_max_features", 5000))
    ngram_cfg = nlp_cfg.get("tfidf_ngram_range", [1, 2])
    ngram_range = (int(ngram_cfg[0]), int(ngram_cfg[1]))
    extra_stopwords = [str(x) for x in nlp_cfg.get("extra_stopwords", [])]
    stop_words = _load_stopwords(extra_stopwords)

    tfidf_train = news_df.loc[train_mask, "model_text"]
    if tfidf_train.empty:
        tfidf_train = news_df["model_text"]

    vectorizer = _fit_tfidf(
        tfidf_train,
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
    _write_tfidf_report(news_df, vectorizer, train_mask, paths["tfidf_report"])

    finbert_model = str(nlp_cfg.get("finbert_model", "ProsusAI/finbert"))
    finbert_cache_dir = str(nlp_cfg.get("finbert_cache_dir", "models/finbert_cache/"))
    max_tokens = int(nlp_cfg.get("max_tokens", 512))
    batch_size = int(nlp_cfg.get("batch_size", 32))

    sentiment_df: pd.DataFrame | None = None
    model_used = "lexicon_fallback"

    finbert_available = (
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    )
    if finbert_available:
        with suppress(Exception):
            sentiment_df = _run_finbert_batches(
                texts=news_df["model_text"],
                model_name=finbert_model,
                cache_dir=finbert_cache_dir,
                max_tokens=max_tokens,
                batch_size=batch_size,
            )
            model_used = "finbert"

    if sentiment_df is None:
        sentiment_df = _run_lexicon_fallback(news_df["model_text"])
        if finbert_available:
            log.warning("FinBERT failed at runtime; using lexicon fallback for this run.")
        else:
            log.warning("FinBERT dependencies unavailable; using lexicon fallback for this run.")

    news_df = pd.concat([news_df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)

    test_sample = news_df.head(min(1000, len(news_df)))
    if not test_sample.empty:
        sample_counts = test_sample["finbert_label"].value_counts().to_dict()
        log.info("Sentiment test-run sample (up to 1000 rows): %s", sample_counts)

    daily_df = _aggregate_daily(news_df)
    lag_days = int(config.get("features", {}).get("min_lag_days", 1))
    daily_df = _lag_news_features(daily_df, lag_days=lag_days)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_parquet(out_path, index=False)

    embedding_method = _build_embedding_plot(news_df, config, paths["embedding_plot"])
    _write_phase4_report(
        news_df=news_df,
        daily_df=daily_df,
        phase4_report=paths["phase4_report"],
        tfidf_report=paths["tfidf_report"],
        embedding_plot=paths["embedding_plot"],
        embedding_method=embedding_method,
        model_name=model_used,
    )

    log.info(
        "run_finbert completed | input_rows=%s | feature_rows=%s | output=%s | model=%s",
        len(news_df),
        len(daily_df),
        out_path,
        model_used,
    )
