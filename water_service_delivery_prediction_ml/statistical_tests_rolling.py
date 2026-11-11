"""Statistical comparison of rolling backtest predictions (2019 to 2024).

Bootstrap 95% confidence intervals for differences in balanced accuracy and exact
McNemar tests on paired correct/incorrect predictions. Writes statistical_tests_rolling.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
N_BOOT = 5000


def balanced_accuracy(truth, pred):
    pos, neg = truth == 1, truth == 0
    return 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())


def main():
    d = pd.read_csv(HERE / "rolling_year_model_predictions.csv")
    wide = d.pivot_table(index=["year", "muni_code"], columns="model", values="prediction")
    truth = d.pivot_table(index=["year", "muni_code"], columns="model", values="truth").iloc[:, 0].to_numpy()
    rng = np.random.default_rng(42)
    samples = [rng.integers(0, len(truth), len(truth)) for _ in range(N_BOOT)]
    rows = []
    for other in ("Prior-year persistence", "Majority baseline", "Decision Tree", "Logistic Regression"):
        rf, ot = wide["Random Forest"].to_numpy(), wide[other].to_numpy()
        diff = balanced_accuracy(truth, rf) - balanced_accuracy(truth, ot)
        boots = [balanced_accuracy(truth[i], rf[i]) - balanced_accuracy(truth[i], ot[i]) for i in samples]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rf_ok, ot_ok = rf == truth, ot == truth
        only_rf, only_other = int((rf_ok & ~ot_ok).sum()), int((~rf_ok & ot_ok).sum())
        p = binomtest(min(only_rf, only_other), only_rf + only_other, 0.5).pvalue
        rows.append({"comparison": f"Random Forest vs {other}", "balanced_accuracy_difference": diff,
                     "ci_low": lo, "ci_high": hi, "only_random_forest_correct": only_rf,
                     "only_other_correct": only_other, "mcnemar_exact_p": p})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "statistical_tests_rolling.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
