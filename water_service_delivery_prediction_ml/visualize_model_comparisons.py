"""Create thesis-ready figures from the saved model-comparison results."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score
from scipy import stats
import json
import warnings
import io
from contextlib import redirect_stdout

HERE = Path(__file__).resolve().parent
OUT = HERE / "thesis_figures"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans", "axes.titleweight": "bold"})


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def all_rounds_heatmap():
    data = pd.read_csv(HERE / "model_training_comparison.csv")
    model_order = ["Decision Tree", "Logistic Regression", "Random Forest"]
    run_order = [
        "Notebook initial run (leaky)",
        "Notebook second run (residual leakage)",
        "Temporal holdout (0.5 threshold)",
        "Rolling backtest (balanced threshold)",
        "Rolling backtest (high-recall threshold)",
    ]
    data = data[data.Model.isin(model_order) & data["Training round"].isin(run_order)].copy()
    data["Training round"] = pd.Categorical(data["Training round"], run_order, ordered=True)
    data["Model"] = pd.Categorical(data.Model, model_order, ordered=True)
    data = data.sort_values(["Training round", "Model"])
    row_labels=[]
    for _,r in data.iterrows():
        prefix = "! " if ("leakage" in r["Training round"] or "leaky" in r["Training round"]) else ""
        row_labels.append(f"{prefix}{r['Training round']}  |  {r.Model}")
    cols = ["Accuracy", "Balanced accuracy", "Recall (poor)",
            "Specificity (good)", "F1 (poor)", "ROC-AUC"]
    matrix = data[cols].copy()
    matrix.columns = ["Accuracy", "Balanced\naccuracy", "Poor\nrecall",
                      "Good\nspecificity", "Poor-class\nF1", "ROC-AUC"]
    fig,ax=plt.subplots(figsize=(13,8.5))
    sns.heatmap(matrix,annot=True,fmt=".2f",cmap="YlGnBu",vmin=0,vmax=1,
                linewidths=.5,linecolor="white",mask=matrix.isna(),
                cbar_kws={"label":"Score"},ax=ax,annot_kws={"fontsize":8})
    ax.set_yticklabels(row_labels,rotation=0,fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=0,fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Model scores across development rounds\nValidation designs differ; shaded/missing cells were not reported",
                 pad=14)
    ax.text(0,-0.10,"⚠ Same-year notebook runs have target leakage and are shown only to document the development history.",
            transform=ax.transAxes,fontsize=9,va="top",color="#8c2d04")
    fig.tight_layout()
    save(fig,"fig01_all_rounds_score_heatmap")


def temporal_holdout():
    data=pd.read_csv(HERE/"model_training_comparison.csv")
    run="Temporal holdout (0.5 threshold)"
    data=data[data["Training round"]==run].copy()
    # Prior-year persistence specificity can be recovered from balanced accuracy and recall.
    i=data["Model"]=="Prior-year risk persistence"
    data.loc[i,"Specificity (good)"]=2*data.loc[i,"Balanced accuracy"]-data.loc[i,"Recall (poor)"]
    names=["Majority baseline","Prior-year risk persistence","Logistic Regression","Decision Tree","Random Forest"]
    data["Model"]=pd.Categorical(data.Model,names,ordered=True)
    data=data.sort_values("Model")
    metrics=[("Accuracy","Accuracy"),("Balanced accuracy","Balanced accuracy"),
             ("Recall (poor)","Recall: poor-risk cases"),
             ("Specificity (good)","Specificity: good-risk cases"),
             ("F1 (poor)","F1: poor-risk cases"),("ROC-AUC","ROC-AUC")]
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    palette=sns.color_palette("colorblind",len(names))
    for ax,(col,title) in zip(axes.flat,metrics):
        vals=data[col].to_numpy(dtype=float)
        ax.barh(range(len(names)),vals,color=palette,edgecolor="white")
        ax.set_yticks(range(len(names)),names if ax in axes[:,0] else [])
        ax.invert_yaxis();ax.set_xlim(0,1);ax.set_title(title,fontsize=11)
        ax.set_xlabel("Score")
        for j,v in enumerate(vals):
            if pd.notna(v): ax.text(v+.015,j,f"{v:.2f}",va="center",fontsize=8)
            else: ax.text(.02,j,"n/a",va="center",fontsize=8,color="gray")
    fig.suptitle("Temporal holdout performance (test outcome years 2021–2024)",y=1.02,fontsize=14,fontweight="bold")
    fig.text(.5,-.015,"Features use the prior financial year. The test set contains 84.5% poor-risk labels; accuracy and F1 should be read with class-specific recall and specificity.",ha="center",fontsize=9)
    fig.tight_layout()
    save(fig,"fig02_temporal_holdout_metrics")


def threshold_tradeoff():
    a=pd.read_csv(HERE/"rolling_year_model_results.csv",index_col=0).reset_index(names="Model")
    b=pd.read_csv(HERE/"rolling_year_model_results_high_recall.csv",index_col=0).reset_index(names="Model")
    a["Threshold policy"]="Balanced-accuracy threshold"
    b["Threshold policy"]="High-recall threshold"
    data=pd.concat([a,b],ignore_index=True)
    data=data[data.Model.isin(["Logistic Regression","Decision Tree","Random Forest"])]
    fig,ax=plt.subplots(figsize=(9.5,7))
    colors={"Logistic Regression":"#0072B2","Decision Tree":"#D55E00","Random Forest":"#009E73"}
    markers={"Balanced-accuracy threshold":"o","High-recall threshold":"^"}
    for _,r in data.iterrows():
        ax.scatter(r.specificity_good,r.recall_poor,s=130,color=colors[r.Model],
                   marker=markers[r["Threshold policy"]],edgecolor="white",linewidth=.8,zorder=3)
    ax.axhline(.95,color="gray",linestyle="--",linewidth=1,label="95% recall target")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Specificity for good-risk cases",ylabel="Recall for poor-risk cases",
           title="Threshold trade-off in the rolling backtest")
    model_handles=[Line2D([0],[0],marker="o",color="white",markerfacecolor=color,
                          markeredgecolor="white",label=model,markersize=9)
                   for model,color in colors.items()]
    policy_handles=[Line2D([0],[0],marker=marker,color="#555555",linestyle="None",
                           label=policy,markersize=8)
                    for policy,marker in markers.items()]
    first=ax.legend(handles=model_handles,loc="lower right",title="Model",frameon=True)
    ax.add_artist(first)
    ax.legend(handles=policy_handles+[Line2D([0],[0],color="gray",linestyle="--",
                                             label="95% recall target")],
              loc="lower left",title="Threshold policy",frameon=True)
    ax.set_title("Threshold trade-off in the rolling backtest\n(2019–2024)",pad=12)
    fig.tight_layout()
    save(fig,"fig03_threshold_tradeoff")


def rf_by_year():
    balanced=pd.read_csv(HERE/"rolling_year_model_results_by_year.csv")
    high=pd.read_csv(HERE/"rolling_year_model_results_by_year_high_recall.csv")
    data=pd.concat([balanced.assign(policy="Balanced threshold"),high.assign(policy="High-recall threshold")])
    data=data[data.Model=="Random Forest"]
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True,sharey=True)
    for ax,metric,title in zip(axes,["recall_poor","specificity_good"],["Poor-risk recall","Good-risk specificity"]):
        for policy,part in data.groupby("policy",sort=False):
            ax.plot(part.year,part[metric],marker="o",linewidth=2,label=policy)
        ax.set_xticks(sorted(data.year.unique()));ax.set_ylim(0,1)
        ax.set_xlabel("Held-out outcome year");ax.set_title(title)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="best",frameon=True)
    fig.suptitle("Random Forest performance by rolling test year",y=1.03,fontsize=14,fontweight="bold")
    fig.text(.5,-.025,"Each year's threshold was selected using earlier training years only. Performance varies across years.",ha="center",fontsize=9)
    fig.tight_layout()
    save(fig,"fig04_random_forest_by_year")


def rf_confusion_matrices():
    mats={"Balanced threshold":[[106,83],[136,953]],
          "High-recall threshold":[[57,132],[37,1052]]}
    fig,axes=plt.subplots(1,2,figsize=(10,4.7))
    for ax,(title,m) in zip(axes,mats.items()):
        matrix=pd.DataFrame(m,index=["Good actual","Poor actual"],columns=["Predicted good","Predicted poor"])
        sns.heatmap(matrix,annot=True,fmt="d",cmap="Blues",cbar=False,linewidths=1,
                    linecolor="white",ax=ax,annot_kws={"fontsize":14})
        ax.set_title(title);ax.set_xlabel("Model prediction");ax.set_ylabel("Observed label")
    fig.suptitle("Random Forest confusion matrices\nPooled rolling backtest, 2019–2024",y=1.04,fontsize=14,fontweight="bold")
    fig.text(.5,-.025,"Rows are observed labels; off-diagonal cells are errors. High-recall threshold reduces missed poor cases (false negatives) while increasing false alarms.",ha="center",fontsize=9)
    fig.tight_layout()
    save(fig,"fig05_random_forest_confusion_matrices")


def legacy_notebook_runs():
    data=pd.read_csv(HERE/"model_training_comparison.csv")
    runs=["Notebook initial run (leaky)","Notebook second run (residual leakage)"]
    data=data[data["Training round"].isin(runs)]
    data["Training round"]=pd.Categorical(data["Training round"],runs,ordered=True)
    metrics=[("Accuracy","Accuracy"),("F1 (poor)","F1-score"),("ROC-AUC","ROC-AUC")]
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharey=True)
    names=["Decision Tree","Logistic Regression","Random Forest"]
    colors=sns.color_palette("colorblind",len(names))
    for ax,(col,title) in zip(axes,metrics):
        pivot=data.pivot(index="Training round",columns="Model",values=col).reindex(runs)
        pivot=pivot[names]
        pivot.plot(kind="bar",ax=ax,color=colors,width=.78)
        ax.set_title(title);ax.set_ylim(0,1.05);ax.set_xlabel("");ax.set_ylabel("Score" if ax is axes[0] else "")
        ax.tick_params(axis="x",rotation=20,labelsize=8)
        if ax is not axes[0]: ax.get_legend().remove()
        else: ax.legend(fontsize=8,loc="lower left")
    fig.suptitle("Notebook scores before and after removing direct target columns",y=1.04,fontsize=14,fontweight="bold")
    fig.text(.5,-.07,"Both runs used same-year random splits; trend features in the second run still contain target-year expenditure. Neither is a valid future-forecast estimate.",ha="center",fontsize=9,color="#8c2d04")
    fig.tight_layout()
    save(fig,"fig06_legacy_notebook_scores")


def temporal_test_confusion():
    d=pd.read_csv(HERE/"temporal_test_predictions.csv")
    models=["Logistic Regression","Decision Tree","Random Forest"]
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    for ax,name in zip(axes,models):
        part=d[d.model==name]
        m=confusion_matrix(part.actual_risk,part.predicted_risk,labels=[0,1])
        sns.heatmap(pd.DataFrame(m,index=["Good actual","Poor actual"],
                                 columns=["Predicted good","Predicted poor"]),
                    annot=True,fmt="d",cmap="Blues",cbar=False,linewidths=1,
                    linecolor="white",ax=ax,annot_kws={"fontsize":12})
        ax.set_title(name);ax.set_xlabel("Prediction");ax.set_ylabel("Observed label")
    fig.suptitle("Confusion matrices on the 2021–2024 temporal holdout\nFixed probability cutoff = 0.50",y=1.04,fontsize=14,fontweight="bold")
    fig.text(.5,-.025,"Each panel uses the same 852 held-out municipality-year cases. Rows are actual labels; columns are model predictions.",ha="center",fontsize=9)
    fig.tight_layout()
    save(fig,"fig07_temporal_test_confusion_matrices")


def temporal_test_curves():
    d=pd.read_csv(HERE/"temporal_test_predictions.csv")
    prevalence=d[d.model=="Random Forest"].actual_risk.mean()
    colors={"Logistic Regression":"#0072B2","Decision Tree":"#D55E00","Random Forest":"#009E73"}
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for name,color in colors.items():
        part=d[d.model==name]
        fpr,tpr,_=roc_curve(part.actual_risk,part.predicted_poor_probability)
        roc_auc=auc(fpr,tpr)
        axes[0].plot(fpr,tpr,label=f"{name} (AUC={roc_auc:.2f})",color=color,linewidth=2)
        prec,rec,_=precision_recall_curve(part.actual_risk,part.predicted_poor_probability)
        ap=average_precision_score(part.actual_risk,part.predicted_poor_probability)
        axes[1].plot(rec,prec,label=f"{name} (AP={ap:.2f})",color=color,linewidth=2)
    axes[0].plot([0,1],[0,1],"--",color="gray",linewidth=1,label="No-discrimination reference")
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel="False-positive rate",ylabel="True-positive rate",title="ROC curves")
    axes[1].axhline(prevalence,linestyle="--",color="gray",linewidth=1,label=f"Test prevalence={prevalence:.2f}")
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel="Recall",ylabel="Precision",title="Precision–recall curves")
    for ax in axes: ax.legend(loc="lower left",fontsize=8,frameon=True)
    fig.suptitle("Probability ranking on the 2021–2024 temporal holdout",y=1.03,fontsize=14,fontweight="bold")
    fig.text(.5,-.025,"Curves assess ranking across cutoffs; they do not select a final operational threshold. The positive class is poor financial risk.",ha="center",fontsize=9)
    fig.tight_layout()
    save(fig,"fig08_temporal_test_roc_pr_curves")


def cv_vs_test():
    d=pd.read_csv(HERE/"temporal_cv_vs_test.csv")
    order=["Logistic Regression","Decision Tree","Random Forest"]
    d["Model"]=pd.Categorical(d.Model,order,ordered=True);d=d.sort_values("Model")
    x=range(len(d));w=.34
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    ax.bar([i-w/2 for i in x],d["Training expanding-CV F1"],width=w,label="Training expanding-CV F1",color="#4C78A8")
    ax.bar([i+w/2 for i in x],d["Test F1"],width=w,label="Temporal test F1",color="#F58518")
    ax.set_xticks(list(x),d.Model);ax.set_ylim(0,1);ax.set_ylabel("F1 for poor-risk class")
    ax.set_title("Training-period validation versus later-year test F1")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),ncol=2,fontsize=9)
    for i,v in enumerate(d["Training expanding-CV F1"]):ax.text(i-w/2,v+.015,f"{v:.2f}",ha="center",fontsize=8)
    for i,v in enumerate(d["Test F1"]):ax.text(i+w/2,v+.015,f"{v:.2f}",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.12,1,1])
    save(fig,"fig09_training_validation_vs_test")


def test_rates_by_year():
    d=pd.read_csv(HERE/"temporal_test_predictions.csv")
    actual=d[d.model=="Random Forest"].groupby("year").actual_risk.mean()
    predicted=d.groupby(["model","year"]).predicted_risk.mean().unstack(0)
    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.plot(actual.index,actual.values,marker="o",linewidth=2.5,color="#222222",label="Observed poor-risk rate")
    for model,color in [("Logistic Regression","#0072B2"),("Decision Tree","#D55E00"),("Random Forest","#009E73")]:
        ax.plot(predicted.index,predicted[model],marker="s",linewidth=1.8,color=color,label=f"Predicted poor rate: {model}")
    ax.set(xticks=sorted(actual.index),ylim=(0,1),xlabel="Held-out outcome year",ylabel="Share classified as poor risk",
           title="Observed and predicted poor-risk rates in the temporal test set")
    ax.legend(loc="best",fontsize=8)
    fig.tight_layout()
    save(fig,"fig10_temporal_test_rates_by_year")


_BLUE_DROP_CACHE = None


def load_embedded_blue_drop():
    global _BLUE_DROP_CACHE
    if _BLUE_DROP_CACHE is not None:
        return _BLUE_DROP_CACHE.copy()
    # Reuse the manually transcribed Blue Drop table in the original project notebook.
    nb=json.loads((HERE/"01_data_loading_and_merging.ipynb").read_text())
    namespace={"pd":pd,"np":__import__("numpy"),"plt":plt,"sns":sns,
               "stats":stats,"warnings":warnings}
    with redirect_stdout(io.StringIO()):
        exec("".join(nb["cells"][26]["source"]),namespace)
    _BLUE_DROP_CACHE=namespace["df_bd_clean"]
    return _BLUE_DROP_CACHE.copy()


def blue_drop_province_scores():
    d=load_embedded_blue_drop()
    summary=d.groupby("province").agg(mean_score=("bd_score_2023","mean"),
                                       n=("bd_score_2023","size")).sort_values("mean_score")
    fig,ax=plt.subplots(figsize=(9,6.5))
    colors=["#d55e00" if v<50 else "#e6ab02" if v<80 else "#009e73" for v in summary.mean_score]
    bars=ax.barh(summary.index,summary.mean_score,color=colors)
    ax.axvline(31,color="#a50f15",linestyle="--",linewidth=1,label="Critical threshold (31)")
    ax.axvline(50,color="#d55e00",linestyle=":",linewidth=1,label="Poor-score threshold (50)")
    ax.axvline(80,color="#4d9221",linestyle="--",linewidth=1,label="Good threshold (80)")
    for bar,(_,r) in zip(bars,summary.iterrows()):
        ax.text(r.mean_score+1,bar.get_y()+bar.get_height()/2,
                f"{r.mean_score:.1f}  (n={int(r.n)})",va="center",fontsize=8)
    ax.set(xlim=(0,110),xlabel="Mean Blue Drop score (%)",ylabel="Province",
           title="Blue Drop 2023 scores by province\nAvailable scored WSAs in the notebook table")
    ax.legend(loc="lower right",fontsize=8)
    fig.text(.5,-.015,"Provincial means use only the municipalities with a score entered in the notebook; coverage varies by province.",ha="center",fontsize=9)
    fig.tight_layout()
    save(fig,"fig11_blue_drop_scores_by_province")


def blue_drop_distribution():
    d=load_embedded_blue_drop()
    order=["Critical","Poor","Average","Good","Excellent (Certified)"]
    colors={"Critical":"#b2182b","Poor":"#ef8a62","Average":"#fddbc7",
            "Good":"#a6dba0","Excellent (Certified)":"#2166ac"}
    counts=d.bd_category.value_counts().reindex(order,fill_value=0)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].hist(d.bd_score_2023,bins=[0,31,50,80,95,101],color="#4c78a8",edgecolor="white")
    for x,label in [(31,"31"),(50,"50"),(80,"80"),(95,"95")]:
        axes[0].axvline(x,color="gray",linestyle="--",linewidth=1)
        axes[0].text(x+1,axes[0].get_ylim()[1]*.88,label,fontsize=8,color="gray")
    axes[0].set(xlim=(0,101),xlabel="Blue Drop score (%)",ylabel="Number of WSAs",title="Entered score distribution")
    bars=axes[1].bar(range(len(order)),counts.values,color=[colors[x] for x in order])
    axes[1].set_xticks(range(len(order)),order,rotation=24,ha="right")
    axes[1].set(ylabel="Number of WSAs",title="Score categories")
    for bar,val in zip(bars,counts.values):axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.3,str(val),ha="center",fontsize=9)
    fig.suptitle("Blue Drop 2023 outcomes in the notebook’s manually assembled dataset",y=1.03,fontsize=14,fontweight="bold")
    fig.text(.5,-.07,"The notebook assigns 96% as a representative value to certified WSAs (≥95%); it is not their exact score. Only 90 of the listed WSAs have an entered score.",ha="center",fontsize=9)
    fig.tight_layout()
    save(fig,"fig12_blue_drop_score_distribution")


def blue_drop_paired_change():
    d=load_embedded_blue_drop().dropna(subset=["bd_score_2014","bd_score_2023"]).copy()
    d["change"]=d.bd_score_2023-d.bd_score_2014
    d=d.sort_values("change")
    fig,ax=plt.subplots(figsize=(9,7))
    for y,(_,r) in enumerate(d.iterrows()):
        color="#2166ac" if r.change>=0 else "#b2182b"
        ax.plot([r.bd_score_2014,r.bd_score_2023],[y,y],color="#999999",linewidth=1.2,zorder=1)
        ax.scatter(r.bd_score_2014,y,color="#4c78a8",s=38,zorder=2)
        ax.scatter(r.bd_score_2023,y,color=color,s=48,zorder=3)
    ax.set_yticks(range(len(d)),d.wsa_name,fontsize=8)
    ax.set(xlim=(0,100),xlabel="Blue Drop score (%)",ylabel="Water services authority",
           title="Blue Drop scores in 2014 and 2023\nPaired entries available in the notebook (n=14)")
    ax.scatter([],[],color="#4c78a8",label="2014 score")
    ax.scatter([],[],color="#2166ac",label="2023 higher than or equal to 2014")
    ax.scatter([],[],color="#b2182b",label="2023 lower than 2014")
    ax.legend(loc="lower right",fontsize=8)
    fig.tight_layout()
    save(fig,"fig13_blue_drop_paired_2014_2023")


def main():
    all_rounds_heatmap()
    temporal_holdout()
    threshold_tradeoff()
    rf_by_year()
    rf_confusion_matrices()
    legacy_notebook_runs()
    temporal_test_confusion()
    temporal_test_curves()
    cv_vs_test()
    test_rates_by_year()
    blue_drop_province_scores()
    blue_drop_distribution()
    blue_drop_paired_change()
    print(f"Saved 13 figures as PNG and PDF in {OUT}")


if __name__ == "__main__":
    main()
