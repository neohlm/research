"""Leakage-controlled, next-year municipal financial-risk model comparison."""

from pathlib import Path
import numpy as np
import pandas as pd
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
DATA_PATH = HERE / "municipal_data_clean.csv"
LABEL_COLS = ["irregular_exp", "unauthorised_exp", "fruitless_wasteful_exp"]
FEATURE_COLS = [
    "fruitless_wasteful_exp", "irregular_exp", "unauthorised_exp",
    "total_capital_exp", "revenue_water", "revenue_waste_water",
    "revenue_electricity", "employee_costs", "contracted_services",
    "debt_impairment", "transfers_operational", "transfers_capital",
    "repairs_maintenance_exp", "irreg_exp_change", "irreg_exp_3yr_avg",
    "repairs_to_capital_ratio", "water_revenue_efficiency",
]


def score(name, truth, prediction, probability=None):
    return {
        "Model": name,
        "Accuracy": accuracy_score(truth, prediction),
        "Balanced accuracy": balanced_accuracy_score(truth, prediction),
        "Precision": precision_score(truth, prediction, zero_division=0),
        "Recall": recall_score(truth, prediction, zero_division=0),
        "F1": f1_score(truth, prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(truth, probability) if probability is not None else np.nan,
        "PR-AUC": average_precision_score(truth, probability) if probability is not None else np.nan,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Do not call unreported outcomes "good": all label components must be observed.
    complete = df[LABEL_COLS].notna().all(axis=1)
    outcomes = df.loc[complete, ["muni_code", "year"] + LABEL_COLS].copy()
    outcomes["risk_target"] = (
        (outcomes.irregular_exp > 10_000_000)
        | (outcomes.unauthorised_exp > 10_000_000)
        | (outcomes.fruitless_wasteful_exp > 1_000_000)
    ).astype(int)
    outcomes = outcomes[["muni_code", "year", "risk_target"]]

    # Shift each feature forward so year t's prediction uses only year t-1 data.
    previous = df[["muni_code", "year"] + FEATURE_COLS].copy()
    previous["year"] += 1
    previous = previous.rename(columns={c: f"prior_{c}" for c in FEATURE_COLS})
    data = outcomes.merge(previous, on=["muni_code", "year"], how="inner", validate="one_to_one")
    data = data[data.year.between(2013, 2024)].copy()
    data["risk_target"] = data["risk_target"].astype(int)
    features = [f"prior_{c}" for c in FEATURE_COLS]
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)

    # Final untouched test period is the later four observed outcome years.
    train = data[data.year <= 2020].copy()
    test = data[data.year >= 2021].copy()
    X_train, y_train = train[features], train.risk_target
    X_test, y_test = test[features], test.risk_target

    # Expanding-window folds: validation year is always later than its training years.
    folds = []
    for validation_year in (2017, 2018, 2019, 2020):
        fit = np.flatnonzero(train.year.to_numpy() < validation_year)
        valid = np.flatnonzero(train.year.to_numpy() == validation_year)
        if len(fit) and len(valid) and train.iloc[fit].risk_target.nunique() == 2:
            folds.append((fit, valid))
    if not folds or y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Insufficient classes for chronological training and test evaluation.")

    prep = [
        # Keep columns that have no values in early folds so feature shape stays stable.
        ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                 keep_empty_features=True)),
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
            RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            {"model__max_depth": [None, 8], "model__min_samples_leaf": [2, 8],
             "model__max_features": ["sqrt", 0.8], "model__class_weight": [None, "balanced"]},
        ),
    }
    fitted = {}
    cv_f1 = {}
    test_prediction_rows = []
    for name, (estimator, grid) in specs.items():
        pipe = Pipeline(prep + [("model", estimator)])
        search = GridSearchCV(pipe, grid, scoring="f1", cv=folds, refit=True,
                              n_jobs=-1, error_score="raise")
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        cv_f1[name] = search.best_score_

    records = []
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    records.append(score("Majority baseline", y_test, dummy.predict(X_test)))

    # Persistence baseline: use last year's known risk label for that municipality.
    prev_target = outcomes.copy()
    prev_target["year"] += 1
    persistence = test[["muni_code", "year", "risk_target"]].merge(
        prev_target.rename(columns={"risk_target": "prior_target"}),
        on=["muni_code", "year"], how="left", validate="one_to_one"
    )
    keep = persistence.prior_target.notna()
    records.append(score("Prior-year risk persistence", persistence.loc[keep, "risk_target"],
                         persistence.loc[keep, "prior_target"].astype(int)))

    test_score_by_model = {}
    for name, model in fitted.items():
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        test_score_by_model[name] = score(name, y_test, pred, prob)
        records.append(test_score_by_model[name])
        for (_, row), prediction, probability in zip(test.iterrows(), pred, prob):
            test_prediction_rows.append({
                "year": int(row.year), "muni_code": row.muni_code,
                "actual_risk": int(row.risk_target), "predicted_risk": int(prediction),
                "predicted_poor_probability": float(probability), "model": name,
                "classification_threshold": 0.5,
            })
    results = pd.DataFrame(records).set_index("Model")

    print(f"Dataset: {DATA_PATH.name} ({len(df):,} municipality-year records)")
    print(f"Known labels: {int(complete.sum()):,}; unknown labels excluded: {int((~complete).sum()):,}")
    print(f"Model rows with exact prior-year features: {len(data):,}")
    print(f"Training: {len(train):,} rows, outcomes {int(train.year.min())}–{int(train.year.max())}")
    print(f"Testing: {len(test):,} rows, outcomes {int(test.year.min())}–{int(test.year.max())}")
    print("Training and test outcome years overlap:", bool(set(train.year) & set(test.year)))
    print("Training and test municipality-year rows overlap:",
          bool(set(zip(train.muni_code, train.year)) & set(zip(test.muni_code, test.year))))
    print("Municipalities appear in both periods:",
          bool(set(train.muni_code) & set(test.muni_code)))
    print("\nBest expanding-window CV F1 and parameters:")
    for name, model in fitted.items():
        print(f"{name}: CV F1={cv_f1[name]:.3f}; {model.named_steps['model'].get_params()}")
    print("\nUntouched later-year test results:")
    print(results.round(3).to_string())
    print("\nConfusion matrices [[TN, FP], [FN, TP]]:")
    for name, model in fitted.items():
        print(f"{name}:\n{confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])}")

    rf = fitted["Random Forest"]
    yearly = []
    for year, part in test.groupby("year"):
        pred = rf.predict(part[features])
        yearly.append({"year": year, "n": len(part), "poor_rate": part.risk_target.mean(),
                       "accuracy": accuracy_score(part.risk_target, pred),
                       "recall": recall_score(part.risk_target, pred, zero_division=0),
                       "F1": f1_score(part.risk_target, pred, zero_division=0)})
    print("\nRandom Forest test performance by year:")
    print(pd.DataFrame(yearly).set_index("year").round(3).to_string())
    results.to_csv(HERE / "temporal_model_results.csv")
    predictions = pd.DataFrame(test_prediction_rows)
    predictions.to_csv(HERE / "temporal_test_predictions.csv", index=False)
    comparison = []
    for name in fitted:
        comparison.append({"Model": name, "Training expanding-CV F1": cv_f1[name],
                           "Test F1": test_score_by_model[name]["F1"],
                           "Test balanced accuracy": test_score_by_model[name]["Balanced accuracy"],
                           "Test ROC-AUC": test_score_by_model[name]["ROC-AUC"]})
    pd.DataFrame(comparison).to_csv(HERE / "temporal_cv_vs_test.csv", index=False)
    print(f"\nSaved {HERE / 'temporal_model_results.csv'}, temporal_test_predictions.csv, and temporal_cv_vs_test.csv")


if __name__ == "__main__":
    main()
