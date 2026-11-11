"""SHAP analysis of the final temporal Random Forest (prior-year features, median imputation)."""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ML = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
sys.path.insert(0, str(ML))
from retrain_financial_risk_rolling import make_data  # noqa: E402

LABELS = {
    "fruitless_wasteful_exp": "Fruitless and wasteful expenditure", "irregular_exp": "Irregular expenditure",
    "unauthorised_exp": "Unauthorised expenditure", "total_capital_exp": "Total capital expenditure",
    "revenue_water": "Water revenue", "revenue_waste_water": "Waste water revenue",
    "revenue_electricity": "Electricity revenue", "employee_costs": "Employee costs",
    "contracted_services": "Contracted services", "debt_impairment": "Debt impairment",
    "transfers_operational": "Operational transfers", "transfers_capital": "Capital transfers",
    "repairs_maintenance_exp": "Repairs and maintenance", "irreg_exp_change": "Change in irregular expenditure",
    "irreg_exp_3yr_avg": "Irregular expenditure, 3 year average",
    "repairs_to_capital_ratio": "Repairs to capital ratio", "water_revenue_efficiency": "Water revenue efficiency",
}


def nice(name):
    name = name.replace("missingindicator_", "MISSING:")
    missing = name.startswith("MISSING:")
    base = name.replace("MISSING:", "").replace("prior_", "")
    label = LABELS.get(base, base)
    return f"{label} (not reported)" if missing else f"{label} (previous year)"


bundle = joblib.load(ML / "financial_risk_models.joblib")
pipe = bundle["models"]["Random Forest"]
data, features, _ = make_data()
X = data[features]
prep = pipe[:-1]
Xt = prep.transform(X)
names = [nice(n) for n in prep.get_feature_names_out()]
rf = pipe[-1]
explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(Xt)
sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv[1]
imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
imp.to_csv(ML / "shap_random_forest_temporal.csv", header=["mean_abs_shap"])
print(imp.head(12).round(4).to_string())

top = imp.head(12)[::-1]
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.barh(top.index, top.values, color="#2E75B6")
ax.set_xlabel("Mean absolute SHAP value (impact on predicted risk)")
ax.set_title("Random Forest feature importance (SHAP)\nFinal temporal model, previous year features")
fig.tight_layout()
fig.savefig(ML / "thesis_figures" / "fig14_shap_random_forest_bar.png", dpi=300)

order = [names.index(n) for n in imp.head(12).index]
plt.figure()
shap.summary_plot(sv[:, order], Xt[:, order], feature_names=[names[i] for i in order], show=False, max_display=12)
plt.title("SHAP summary: final temporal Random Forest")
plt.tight_layout()
plt.savefig(ML / "thesis_figures" / "fig15_shap_random_forest_summary.png", dpi=300, bbox_inches="tight")
print("rows", len(X))
