# Data Quality Issues

1. Material missingness in `news.article`.
   Evidence: 100.00% of rows have null article text.
   Proposed treatment: use title fallback and preserve `text_source`.

2. Zero-volume trading rows are present in prices.
   Evidence: 0.36% of rows have `volume_zero_flag == 1`.
   Proposed treatment: keep rows and use `volume_zero_flag` as a quality feature.

3. Duplicate news stories exist for ticker-date-text keys.
   Evidence: 709 exact duplicates found for (`ticker`, `date`, `title`, `article`).
   Proposed treatment: deduplicate only during aggregation, not at raw-interim stage.

4. Some tickers have sparse trading-day coverage.
   Evidence: 1928 tickers exceed 20% missing trading days.
   Proposed treatment: track these symbols and consider minimum-history filters before modeling.
