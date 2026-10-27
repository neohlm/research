# Model training comparison

Metrics for the successive model-development rounds. The first two notebook runs use same-year data; their scores are not valid estimates of future prediction. The notebook’s direct leakage run includes variables used to construct its target, and the later “fixed” run still contains same-year trend features derived from those variables.

The temporal holdout uses year *t−1* financial data to predict year *t* financial risk. The rolling backtests evaluate successive years 2019–2024 with each test year predicted by models selected using only earlier years. Rolling backtest predictions are repeated annual forecasts, not a new independent final test.

| Round | Main evaluation | Random Forest accuracy | Balanced accuracy | Poor recall | Good specificity | False negatives / positives |
|---|---|---:|---:|---:|---:|---:|
| Notebook initial (leaky) | Random row split, same-year | 0.994 | — | 1.000 | — | Not recorded |
| Notebook second (residual leakage) | Random row split, same-year trends | 0.872 | — | 0.912 | — | Not recorded |
| Temporal holdout, 0.5 threshold | 2021–2024, prior-year features | 0.863 | 0.600 | 0.981 | 0.220 | 14 / 103 |
| Rolling backtest, balanced threshold | 2019–2024, tuned on earlier years | 0.829 | 0.718 | 0.875 | 0.561 | 136 / 83 |
| Rolling backtest, high-recall threshold | 2019–2024, tuned on earlier years | 0.868 | 0.634 | 0.966 | 0.302 | 37 / 132 |

The “high-recall” version catches more poor-risk cases and produces more false alarms. The balanced-threshold version has better equal-class performance. Overall accuracy alone is misleading because poor-risk labels make up about 85% of the later test observations. See `model_training_comparison.csv` for all models, baselines, and metrics.

## Testing-set use

A test set is not learned by the model merely because it has been scored. It becomes part of the development process when its results are repeatedly used to choose models, features, or thresholds. Use cross-validation on training data for those choices, then reserve an untouched final test for the end. Since these 2019–2024 results have now been reviewed, report them as rolling backtest/development evidence; a later labelled year would provide a genuinely new temporal test.
