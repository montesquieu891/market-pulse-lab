# Market Pulse Lab (README en Espanol)

## Resumen
Market Pulse Lab es un proyecto de ingenieria de preprocesamiento para datos financieros.
El foco no es ganar una competencia de modelos, sino construir una base de datos confiable
para pronosticar volatilidad a corto plazo, combinando precios, noticias y macroeconomia.

## Contexto
En entornos de riesgo y research cuantitativo, la calidad del preprocesamiento define la calidad
final del sistema. Este proyecto integra tres fuentes heterogeneas:

- Precios historicos de acciones (tabular + series temporales)
- Noticias financieras (NLP)
- Indicadores macro (FRED)

Objetivo operativo:
- Construir una matriz de features unificada (ticker x fecha)
- Mantener integridad temporal estricta (sin leakage)
- Serializar artefactos para uso reproducible

## Como se trabajo
Se siguio un flujo por etapas en pipeline:

1. Ingestion
- Normalizacion de tipos y fechas en UTC
- Exportacion de intermedios a Parquet
- Regla de inmutabilidad para datos crudos

2. Diagnostico EDA
- Perfilado de calidad de datos
- Clasificacion de faltantes y deteccion de cobertura
- Documentacion del diccionario de datos

3. Features de series temporales
- log_return, realized volatility, lags, rolling stats
- RSI(14), Bollinger Bands
- ADF en close y log_return para chequeo de estacionariedad

4. NLP
- Limpieza de texto y TF-IDF (fit solo en train)
- Agregacion diaria de sentimiento por ticker
- Lag temporal de features de noticias

5. Integracion y validacion
- Join de precios + noticias + macro
- Validaciones tipo Great Expectations
- Generacion de feature matrix final y pipeline serializado

6. Baselines
- Split temporal train/validation/test
- Escalado ajustado solo en train
- Comparativa de baselines lineales y no lineales

## Logros
- Pipeline end-to-end ejecuta en modo estricto con estado PASS
- Matriz final integrada:
  - rows: 4,178,400
  - columns: 38
  - feature_count: 33
  - train_rows: 2,811,067
- Validaciones de calidad clave exitosas:
  - target no nulo en train
  - realized_vol_5d positivo
  - ventana objetivo posterior a fecha de features
  - monotonia temporal por ticker
- Baseline con mejor desempeno:
  - XGBoost (price + sentiment):
    - validation_mae: 0.1602995641
    - test_mae: 0.1597641816

## Insights obtenidos
- Integridad temporal:
  - ADF p-value close: 0.981798 (no estacionaria)
  - ADF p-value log_return: 0.000000 (estacionaria)
  - Confirmacion de transformacion correcta para modelado temporal

- Cobertura de noticias:
  - rows_without_news_pct: 0.964701 en feature matrix
  - Este dato explica parte del techo de desempeno en modelos con sentimiento

- Calidad NLP en este entorno:
  - rows processed: 311,619
  - text_source article ratio: 0.000
  - text_source title ratio: 1.000
  - sentiment model usado en corrida final: lexicon_fallback

## Propuestas
1. Re-ejecutar NLP con stack completo FinBERT/SBERT en ambiente con dependencias full.
2. Ampliar cobertura de texto completo (article) para elevar senal semantica.
3. Agregar evaluacion por sector y por ventanas de tiempo para robustez.
4. Incluir backtesting de drift de features y alertas de data quality en produccion.
5. Publicar dashboard ejecutivo de metricas de pipeline y validaciones.

## Impacto de valor
- Riesgo y cumplimiento:
  - Menor probabilidad de leakage temporal y decisiones sesgadas.

- Productividad de equipos de datos:
  - Pipeline reproducible, tipado y validado que acelera iteracion.

- Base para evolucion futura:
  - La arquitectura permite conectar modelos mas avanzados sin rehacer ingestion,
    limpieza e integracion.

## Artefactos clave
- Resultados de baseline: reports/modeling_baseline_results.csv
- Diagnostico NLP: reports/nlp_phase4_diagnostics.md
- Diagnostico temporal: reports/timeseries_diagnostics.md
- Validacion integrada: reports/data_validation_report.html
