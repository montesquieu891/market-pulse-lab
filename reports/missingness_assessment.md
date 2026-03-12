# Missingness Assessment

Classification by column as MCAR/MAR/MNAR with rationale and treatment.

| Dataset | Column | Missing % | Mechanism | Rationale | Proposed treatment |
|---|---|---:|---|---|---|
| news | article | 100.00% | MAR | Missing full article text depends on source feed availability; headline usually remains available. | Fallback to title text; keep row. |
| prices | log_return | 0.17% | MNAR | First observation per ticker has no prior close by construction. | Keep NaN for first row and ignore in return-based aggregations. |
