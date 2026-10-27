# Figure captions and notes

The PNG files are 300 dpi for insertion in Word. PDF versions are vector graphics and are suitable for LaTeX or other publication workflows. The figures were generated from the CSVs in the project folder by `visualize_model_comparisons.py`.

## Figure 1 — Model scores across development rounds

Heatmap of accuracy, balanced accuracy, poor-risk recall, good-risk specificity, poor-class F1, and ROC-AUC for each classifier and modelling round. Validation designs differ, so scores across rounds are descriptive rather than directly comparable. The first two notebook runs used same-year information and are marked as leakage-affected.

## Figure 2 — Temporal holdout performance

Scores for the majority baseline, prior-year persistence, Logistic Regression, Decision Tree, and Random Forest on the 2021–2024 holdout, trained on earlier years and using prior-year financial predictors. Poor-risk cases make up 84.5% of this holdout; therefore, accuracy and F1 should be interpreted alongside balanced accuracy, poor-risk recall, and good-risk specificity.

## Figure 3 — Threshold trade-off in the rolling backtest

Each point shows poor-risk recall and good-risk specificity for one classifier under either a balanced-accuracy threshold or a high-recall threshold. The high-recall rule targets at least 95% poor-risk recall using training-only validation predictions. The line marks that target; observed held-out-year recall can fall below it.

## Figure 4 — Random Forest performance by rolling test year

Poor-risk recall and good-risk specificity for the Random Forest across held-out outcome years 2019–2024. Each year is predicted by a model trained on earlier years. The plot shows that performance, especially specificity under the high-recall threshold, varies across years.

## Figure 5 — Random Forest confusion matrices

Pooled counts of true negatives, false positives, false negatives, and true positives across rolling test years 2019–2024. The high-recall threshold reduces false negatives while increasing false positives relative to the balanced-accuracy threshold.

## Figure 6 — Earlier notebook scores

Accuracy, F1, and ROC-AUC from the notebook’s initial run and its later run after removing the direct target columns. Both runs used same-year random splits; the later run retained same-year trend features that encode target-year expenditure. These figures document the development history and should not be presented as valid future-forecast estimates.

## Figure 7 — Temporal test confusion matrices

Confusion matrices for Logistic Regression, Decision Tree, and Random Forest on the 852 held-out municipality-year cases from 2021–2024, using a fixed 0.50 probability cutoff. These show the false-positive and false-negative counts for each classifier.

## Figure 8 — Temporal test ROC and precision–recall curves

ROC and precision–recall curves for predicted poor-risk probabilities on the 2021–2024 holdout. The dashed precision–recall reference is the test-set poor-risk prevalence. Curves show performance across cutoffs; they do not choose a final operating cutoff.

## Figure 9 — Training validation versus test F1

Comparison of each classifier’s best expanding-window cross-validation F1 within the 2013–2020 training period and its F1 on the 2021–2024 temporal holdout. Validation F1 informed hyperparameter selection, while later years were held out for evaluation in this run.

## Figure 10 — Observed and predicted poor-risk rates by test year

Observed prevalence and the share predicted poor by each classifier for each held-out year. Differences between observed and predicted rates can reveal systematic over- or under-prediction across years.

## Figure 11 — Blue Drop scores by province

Mean 2023 Blue Drop scores for the water-services authorities with a score entered in the notebook’s manually assembled dataset. The number of scored authorities differs by province; these means are descriptive and should not be interpreted as complete provincial estimates.

## Figure 12 — Blue Drop 2023 score distribution

Distribution of the 90 Blue Drop scores entered in the notebook and their performance categories. The source notebook assigns a representative 96% to listed certified authorities, whose report scores are at least 95%; these are not exact scores.

## Figure 13 — Paired Blue Drop scores, 2014 and 2023

Paired scores for the 14 Eastern Cape water-services authorities with both values entered in the notebook. This is a descriptive historical comparison, not a validation of model predictions.

## Interpretation note

All current models predict the project’s threshold-based financial-governance-risk label. They do not predict water-service outcomes. The original notebook’s Blue Drop merge compares a stored financial-risk target (not model predictions) and contains questionable municipality-code mappings; do not present it as external validation until the temporal alignment and crosswalk are corrected. Rolling backtest years have already been examined during development and should not be described as a new untouched final test set.
