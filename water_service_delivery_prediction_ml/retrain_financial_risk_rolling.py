"""Nested rolling-year backtest, threshold selection, and final model refit."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
LABEL_COLS = ["irregular_exp", "unauthorised_exp", "fruitless_wasteful_exp"]
FEATURE_COLS = [
    "fruitless_wasteful_exp", "irregular_exp", "unauthorised_exp",
    "total_capital_exp", "revenue_water", "revenue_waste_water",
    "revenue_electricity", "employee_costs", "contracted_services",
    "debt_impairment", "transfers_operational", "transfers_capital",
    "repairs_maintenance_exp", "irreg_exp_change", "irreg_exp_3yr_avg",
    "repairs_to_capital_ratio", "water_revenue_efficiency",
]


def make_data():
    raw = pd.read_csv(HERE / "municipal_data_clean.csv")
    raw["year"] = pd.to_numeric(raw.year, errors="coerce")
    complete = raw[LABEL_COLS].notna().all(axis=1)
    target = raw.loc[complete, ["muni_code", "year"] + LABEL_COLS].copy()
    target["y"] = (
        (target.irregular_exp > 10_000_000)
        | (target.unauthorised_exp > 10_000_000)
        | (target.fruitless_wasteful_exp > 1_000_000)
    ).astype(int)
    outcomes = target[["muni_code", "year", "y"]]
    prior = raw[["muni_code", "year"] + FEATURE_COLS].copy()
    prior["year"] += 1
    prior = prior.rename(columns={c: f"prior_{c}" for c in FEATURE_COLS})
    data = outcomes.merge(prior, on=["muni_code", "year"], how="inner", validate="one_to_one")
    data = data[data.year.between(2013, 2024)].sort_values(["year", "muni_code"]).reset_index(drop=True)
    features = [f"prior_{c}" for c in FEATURE_COLS]
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    return data, features, int((~complete).sum())


def model_specs():
    prep = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]
    specs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=42),
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"model__max_depth": [3, 6, None], "model__min_samples_leaf": [5, 15],
             "model__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1),
            {"model__max_depth": [None, 8], "model__min_samples_leaf": [2, 8],
             "model__max_features": ["sqrt", 0.8], "model__class_weight": [None, "balanced"]},
        ),
    }
    return {name: (Pipeline(prep + [("model", est)]), grid) for name, (est, grid) in specs.items()}


def year_folds(frame, first_validation_year):
    years = sorted(frame.year.unique())
    folds = []
    for val_year in [y for y in years if y >= first_validation_year]:
        fit_idx = np.flatnonzero(frame.year.to_numpy() < val_year)
        val_idx = np.flatnonzero(frame.year.to_numpy() == val_year)
        if len(fit_idx) and len(val_idx) and frame.iloc[fit_idx].y.nunique() == 2:
            folds.append((fit_idx, val_idx))
    return folds


def choose_threshold(truth, prob, objective="high_recall"):
    # Select a cutoff using training-period out-of-fold probabilities only.
    candidates = np.linspace(0.0, 1.0, 201)
    if objective == "high_recall":
        feasible = []
        for threshold in candidates:
            pred = (prob >= threshold).astype(int)
            recall = recall_score(truth, pred, zero_division=0)
            specificity = evaluate(truth, pred)["specificity_good"]
            if recall >= 0.95:
                feasible.append((specificity, threshold, recall))
        if feasible:
            # Highest specificity while meeting the prespecified 95% poor-case recall target.
            specificity, threshold, recall = max(feasible)
            return float(threshold), float(recall), float(specificity)
        # Fallback: maximize recall, then specificity, if 95% is unattainable.
        fallback = []
        for threshold in candidates:
            pred = (prob >= threshold).astype(int)
            recall = recall_score(truth, pred, zero_division=0)
            specificity = evaluate(truth, pred)["specificity_good"]
            fallback.append((recall, specificity, threshold))
        recall, specificity, threshold = max(fallback)
        return float(threshold), float(recall), float(specificity)
    choices = []
    for threshold in candidates:
        pred = (prob >= threshold).astype(int)
        choices.append((balanced_accuracy_score(truth, pred), -abs(threshold - 0.5), threshold))
    threshold = max(choices)[2]
    pred = (prob >= threshold).astype(int)
    return float(threshold), float(recall_score(truth, pred, zero_division=0)), float(evaluate(truth, pred)["specificity_good"])


def evaluate(y, pred, prob=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y), "poor_rate": np.mean(y), "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_poor": precision_score(y, pred, zero_division=0),
        "recall_poor": recall_score(y, pred, zero_division=0),
        "specificity_good": tn / (tn + fp) if tn + fp else np.nan,
        "F1_poor": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, prob) if prob is not None else np.nan,
        "PR_AUC": average_precision_score(y, prob) if prob is not None else np.nan,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def select_and_fit(train, features, first_validation_year, objective="high_recall"):
    folds = year_folds(train, first_validation_year)
    if len(folds) < 2:
        raise ValueError(f"Need at least two year-ordered validation folds; got {len(folds)}")
    fitted, thresholds, cv_scores = {}, {}, {}
    for name, (pipe, grid) in model_specs().items():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=folds, refit=True,
                              n_jobs=1, error_score="raise")
        search.fit(train[features], train.y)
        # Generate inner out-of-fold probabilities for threshold selection.
        truths, probs = [], []
        for fit_idx, val_idx in folds:
            fold_model = clone(search.best_estimator_)
            fold_model.fit(train.iloc[fit_idx][features], train.iloc[fit_idx].y)
            truths.extend(train.iloc[val_idx].y.tolist())
            probs.extend(fold_model.predict_proba(train.iloc[val_idx][features])[:, 1].tolist())
        threshold, cv_recall, cv_specificity = choose_threshold(
            np.asarray(truths), np.asarray(probs), objective=objective)
        fitted[name] = search.best_estimator_
        thresholds[name] = threshold
        cv_scores[name] = {"mean_roc_auc": float(search.best_score_),
                           "threshold_cv_recall": cv_recall,
                           "threshold_cv_specificity": cv_specificity,
                           "best_params": search.best_params_}
    return fitted, thresholds, cv_scores


def main():
    data, features, unknown_count = make_data()
    print(f"Known-label, one-year-ahead rows: {len(data):,}; unknown-label rows excluded: {unknown_count:,}")
    print(f"Outcome years: {int(data.year.min())}–{int(data.year.max())}; poor prevalence: {data.y.mean():.1%}")
    objective = "high_recall"
    print("Rolling one-year-ahead backtest years: 2019–2024")
    print("Threshold objective: maximize good-case specificity subject to at least 95% poor-case recall (training CV only)")

    # Each outer year is evaluated only with earlier years; inner folds tune models/cutoffs.
    predictions = []
    for test_year in range(2019, 2025):
        train = data[data.year < test_year].copy()
        test = data[data.year == test_year].copy()
        fitted, thresholds, cv_info = select_and_fit(
            train, features, first_validation_year=2016, objective=objective)
        for name, model in fitted.items():
            prob = model.predict_proba(test[features])[:, 1]
            pred = (prob >= thresholds[name]).astype(int)
            for i, (_, row) in enumerate(test.iterrows()):
                predictions.append({"year": test_year, "muni_code": row.muni_code,
                                    "model": name, "truth": int(row.y),
                                    "probability": float(prob[i]), "prediction": int(pred[i]),
                                    "threshold": float(thresholds[name])})
        # Baselines for the same outer year.
        majority_value = int(train.y.mean() >= 0.5)
        prior_targets = data[["muni_code", "year", "y"]].copy()
        prior_targets["year"] += 1
        prior_map = prior_targets.set_index(["muni_code", "year"]).y
        for _, row in test.iterrows():
            predictions.append({"year": test_year, "muni_code": row.muni_code,
                                "model": "Majority baseline", "truth": int(row.y),
                                "probability": np.nan, "prediction": majority_value,
                                "threshold": np.nan})
            prev = prior_map.get((row.muni_code, test_year), np.nan)
            if pd.notna(prev):
                predictions.append({"year": test_year, "muni_code": row.muni_code,
                                    "model": "Prior-year persistence", "truth": int(row.y),
                                    "probability": np.nan, "prediction": int(prev),
                                    "threshold": np.nan})
        print(f"Completed outer test year {test_year} (train through {test_year-1})")

    pred_df = pd.DataFrame(predictions)
    metrics = []
    for name, part in pred_df.groupby("model", sort=False):
        metrics.append({"Model": name, **evaluate(part.truth.to_numpy(), part.prediction.to_numpy(),
                                                   part.probability.to_numpy() if part.probability.notna().all() else None)})
    metrics_df = pd.DataFrame(metrics).set_index("Model")
    print("\nNested rolling-year out-of-time results (each year predicted by an earlier-only model):")
    print(metrics_df.round(3).to_string())
    yearly_rows = []
    for (year, name), part in pred_df.groupby(["year", "model"], sort=True):
        yearly_rows.append({"year": year, "Model": name,
                            **evaluate(part.truth.to_numpy(), part.prediction.to_numpy(),
                                       part.probability.to_numpy() if part.probability.notna().all() else None)})
    yearly_df = pd.DataFrame(yearly_rows)
    metrics_df.to_csv(HERE / "rolling_year_model_results_high_recall.csv")
    yearly_df.to_csv(HERE / "rolling_year_model_results_by_year_high_recall.csv", index=False)
    pred_df.to_csv(HERE / "rolling_year_model_predictions_high_recall.csv", index=False)

    # Finally, select/refit on every known label for future use. This is not another test.
    final_models, final_thresholds, final_cv = select_and_fit(
        data, features, first_validation_year=2017, objective=objective)
    bundle = {"models": final_models, "thresholds": final_thresholds,
              "features": features, "target_definition": {
                  "irregular_exp_gt": 10_000_000,
                  "unauthorised_exp_gt": 10_000_000,
                  "fruitless_wasteful_exp_gt": 1_000_000,
                  "prediction_horizon": "one financial year",
              }, "threshold_objective": objective, "final_expanding_cv": final_cv}
    joblib.dump(bundle, HERE / "financial_risk_models_high_recall.joblib")
    (HERE / "financial_risk_final_cv_high_recall.json").write_text(json.dumps(final_cv, indent=2, default=str))
    print(f"\nSaved rolling backtest results and final model bundle in {HERE}")
    print("Final models are fitted on all known labelled years (2013–2024); their performance is estimated by the rolling backtest above.")


if __name__ == "__main__":
    main()
