"""Rebuild publication figures from saved evidence, without loading held-out images.

Missing experiment artifacts are reported and skipped, never replaced with
illustrative statistics. Run this again after final evaluation to complete all
panels. PNGs are 300 dpi; matching PDFs preserve vector text and geometry.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
import numpy as np
from scipy.stats import spearmanr
import yaml
from bioalgo.benchmarks import FUNCTIONS

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
STYLE = CONFIG["style"]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
LABELS = {"baseline": "Baseline", "de": "DE-optimized", "rs": "Random search"}
COLORS = {"baseline": STYLE["baseline"], "de": STYLE["de"], "rs": STYLE["random_search"]}
GENES = [("learning_rate", "Learning rate"), ("weight_decay", "Weight decay"), ("dropout", "Dropout"),
         ("head_units", "Head units"), ("unfreeze_blocks", "Unfreeze depth"),
         ("label_smoothing", "Label smoothing"), ("augmentation", "Augmentation")]
MANIFEST = {}
plt.rcParams.update({"font.family": STYLE.get("font", "DejaVu Sans"), "font.size": 13,
    "axes.labelsize": 13, "axes.titlesize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "text.color": STYLE["foreground"],
    "axes.labelcolor": STYLE["foreground"], "axes.titleweight": "bold", "figure.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight"})


def read(name, default=None):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def csv_rows(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(figure, name, sources, layout_options=None, *, apply_tight_layout=True):
    if apply_tight_layout:figure.tight_layout(**(layout_options or {}))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(FIGURES / f"{name}.{extension}")
    plt.close(figure)
    MANIFEST[name] = {"status": "generated", "sources": sources}


def skip(names, reason):
    for name in names.split():
        MANIFEST[name] = {"status": "unavailable", "reason": reason}


def short_class(name):
    name = name.replace("Tomato___", "").replace("Tomato_", "").replace("_", " ")
    return name.replace("Spider mites Two-spotted spider mite", "Spider mites").replace("Yellow Leaf Curl Virus", "Yellow leaf curl")


def search_figures():
    history = read("de_history.json")
    if history:
        generation = [r["generation"] for r in history]
        figure, axis = plt.subplots(figsize=(9.5, 5.7))
        mean, spread = np.array([r["mean"] for r in history]), np.array([r["std"] for r in history])
        axis.plot(generation, [r["best"] for r in history], color=STYLE["de"], marker="o", label="Best candidate")
        axis.plot(generation, mean, color=STYLE["pso"], marker="s", label="Population mean")
        axis.fill_between(generation, np.maximum(0, mean - spread), np.minimum(1, mean + spread), color=STYLE["pso"], alpha=.12, label="Population ±1 SD")
        baseline = read("baseline_proxy.json")
        sources = ["de_history.json"]
        if baseline:
            axis.axhline(baseline["validation_macro_f1"], color=STYLE["baseline"], linestyle="--", label="Baseline · same proxy")
            sources.append("baseline_proxy.json")
        axis.set(xlabel="Generation (0 = initialization)", ylabel="Validation macro-F1", title="DE proxy fitness across generations")
        result_axis_style(axis,20,18,16)
        axis.legend(frameon=False,fontsize=16,loc="upper center",bbox_to_anchor=(.5,-.24),ncol=2);axis.grid(alpha=.15)
        save(figure, "fig_de_convergence", sources)
        figure, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
        axes[0].plot(generation, [r["diversity"] for r in history], color=STYLE["de"], marker="o")
        axes[0].set(ylabel="Mean pairwise\ndistance", title="Population diversity and replacement success")
        axes[1].plot(generation, [r["success_rate"] for r in history], color=STYLE["pso"], marker="s")
        axes[1].set(xlabel="Generation", ylabel="Replacement\nrate", ylim=(-.03, 1.03))
        for axis in axes:
            result_axis_style(axis,21,18,18);axis.grid(alpha=.15)
            axis.yaxis.set_major_locator(MaxNLocator(3));axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        save(figure, "fig_de_diversity", ["de_history.json"])
    else:
        skip("fig_de_convergence fig_de_diversity", "DE search has not completed")
    rows = csv_rows("de_evaluations.csv")
    random_rows = csv_rows("rs_evaluations.csv")
    if rows and random_rows:
        figure, axis = plt.subplots(figsize=(12, 6))
        for method, records in (("de", rows), ("rs", random_rows)):
            ordered = sorted(records, key=lambda r: int(r["evaluation"]))
            axis.plot([int(r["evaluation"]) for r in ordered], np.maximum.accumulate([float(r["fitness"]) for r in ordered]),
                      color=COLORS[method], label=LABELS[method], marker="o" if method == "de" else "s",markersize=8,linewidth=2.3,markevery=max(1, len(ordered)//8))
        axis.set(xlabel="Candidate evaluations", ylabel="Best validation\nmacro-F1", title="Equal-budget search on validation data")
        result_axis_style(axis,24,24,22);axis.legend(frameon=False,fontsize=22);axis.grid(alpha=.15)
        save(figure, "fig_de_vs_rs", ["de_evaluations.csv", "rs_evaluations.csv"])
    else:
        skip("fig_de_vs_rs", "Both DE and random-search records are needed")
    if not rows:
        skip("fig_search_landscape fig_parallel_coords fig_gene_importance", "No DE candidate evaluations saved")
        return
    scores = np.asarray([float(r["fitness"]) for r in rows])
    figure, axis = plt.subplots(figsize=(12, 6.6))
    normalizer = matplotlib.colors.Normalize(vmin=float(scores.min()), vmax=max(float(scores.max()), float(scores.min()) + 1e-12))
    # Redundant marker shapes preserve the fitness bands when hues are hard to
    # distinguish. Quantile bands are descriptive views of these saved scores.
    bands=np.digitize(scores,np.quantile(scores,[.25,.5,.75]),right=True)
    x=np.array([np.log10(float(r["learning_rate"])) for r in rows]);y=np.array([int(r["unfreeze_blocks"]) for r in rows])
    sizes=np.array([45+14*int(r["generation"]) for r in rows])
    for band,marker in enumerate(("v","s","D","o")):
        selected=bands==band
        if not np.any(selected):continue
        axis.scatter(x[selected],y[selected],c=scores[selected],norm=normalizer,cmap="viridis",s=sizes[selected],
            marker=marker,edgecolors=STYLE["foreground"],linewidth=.6,
            label=f"F1 {scores[selected].min():.3f}–{scores[selected].max():.3f}")
    best = rows[int(np.argmax(scores))]
    axis.scatter(np.log10(float(best["learning_rate"])), int(best["unfreeze_blocks"]), marker="*", s=300,
                 facecolors="none", edgecolors="#B45F06", linewidths=1.5, label="Best proxy candidate")
    axis.set(xlabel="log₁₀ learning rate", ylabel="Unfrozen backbone blocks", title="Candidate scores in two search dimensions")
    result_axis_style(axis,22,20,20)
    axis.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.26),ncol=3,fontsize=18,
        title="Shape: F1 band · marker area grows with generation",title_fontsize=18)
    colorbar=figure.colorbar(plt.cm.ScalarMappable(norm=normalizer,cmap="viridis"),ax=axis)
    colorbar.set_label("Validation macro-F1",fontsize=20);colorbar.ax.tick_params(labelsize=18)
    save(figure, "fig_search_landscape", ["de_evaluations.csv"])
    encoded = np.asarray([[float(row[f"x{i+1}"]) for i in range(7)] for row in rows])
    figure, axis = plt.subplots(figsize=(13.6, 6.3))
    for vector, score in sorted(zip(encoded, scores), key=lambda item: item[1]):
        axis.plot(range(8), np.r_[vector,score], color=plt.cm.viridis(normalizer(score)), linewidth=1.1,
            path_effects=[path_effects.Stroke(linewidth=1.9,foreground=STYLE["foreground"],alpha=.75),path_effects.Normal()])
    axis.set_xticks(range(8), ["Learning\nrate","Weight\ndecay","Dropout","Head\nunits","Unfreeze\ndepth","Label\nsmooth.","Aug.\nstrength","Val.\nmacro\nF1"])
    axis.axvline(7,color=STYLE["foreground"],linewidth=.8,alpha=.35)
    axis.set(ylabel="Normalized genes /\nvalidation macro-F1", ylim=(0, 1), title="Candidate recipes and their measured fitness")
    result_axis_style(axis,23,22,20)
    colorbar=figure.colorbar(plt.cm.ScalarMappable(norm=normalizer,cmap="viridis"),ax=axis)
    colorbar.set_label("Validation macro-F1",fontsize=22);colorbar.ax.tick_params(labelsize=20)
    save(figure, "fig_parallel_coords", ["de_evaluations.csv"])
    import plotly.graph_objects as go
    interactive = go.Figure(data=go.Parcoords(line=dict(color=scores, colorscale="Viridis", showscale=True),
        dimensions=[dict(range=[0,1], label=label, values=encoded[:, index]) for index, (_, label) in enumerate(GENES)]+
            [dict(range=[0,1],label="Validation macro-F1",values=scores)]))
    interactive.update_layout(title="DE candidate chromosomes · measured validation macro-F1",font_family="Arial",font_size=18)
    interactive.write_html(str(FIGURES / "fig_parallel_coords.html"), include_plotlyjs=True, full_html=True)
    correlations = [float(spearmanr(encoded[:, i], scores).statistic) if np.ptp(encoded[:, i]) > 0 and np.ptp(scores) > 0 else 0.0 for i in range(7)]
    figure, axis = plt.subplots(figsize=(10.5, 6))
    axis.barh([label for _, label in GENES], correlations, color=[STYLE["de"] if v >= 0 else STYLE["random_search"] for v in correlations])
    axis.axvline(0, color=STYLE["baseline"], linewidth=.8)
    axis.set(xlabel="Spearman rank correlation\nwith validation macro-F1",xlim=(-1,1),title="Gene–fitness associations\nExploratory, not causal importance")
    result_axis_style(axis,22,20,18)
    save(figure, "fig_gene_importance", ["de_evaluations.csv"])


def learning_figures():
    curves = {method: [read(f"curves_{method}_s{seed}.json") for seed in CONFIG["seeds"]] for method in LABELS}
    curves = {method: [c for c in rows if c] for method, rows in curves.items() if any(rows)}
    if not curves:
        skip("fig_learning_curves", "No learning curves saved")
        return
    figure, axes = plt.subplots(3, 1, figsize=(8.6, 8.6),sharex=True)
    for method, records in curves.items():
        common = min(map(len, records))
        epochs = np.arange(1, common + 1)
        for axis, key, title in zip(axes, ("train_loss", "val_loss", "val_macro_f1"), ("Training loss", "Validation loss", "Validation macro-F1")):
            values = np.asarray([[r[key] for r in record[:common]] for record in records])
            mean, deviation = values.mean(0), values.std(0, ddof=1) if len(values)>1 else np.zeros(common)
            axis.plot(epochs, mean, color=COLORS[method], label=f"{LABELS[method]} (n={len(records)})", marker={"baseline":"s","de":"o","rs":"^"}[method], markevery=max(1,common//6))
            if len(values)>1: axis.fill_between(epochs, mean-deviation, mean+deviation, color=COLORS[method], alpha=.12)
            axis.set(title=title, ylabel="Cross-entropy" if "loss" in key else "Macro-F1")
            axis.grid(alpha=.15)
    for axis in axes:
        result_axis_style(axis,19,18,16)
        axis.yaxis.set_major_locator(MaxNLocator(3))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xlabel("Epoch",fontsize=18)
    handles,labels=axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,frameon=False,fontsize=15,loc="lower center",ncol=3)
    figure.suptitle("Mean ± sample SD across available seeds\nEach method uses its shared epoch range", fontsize=19)
    save(figure, "fig_learning_curves", [f"curves_{m}_s{s}.json" for m in curves for s in CONFIG["seeds"] if (RESULTS/f"curves_{m}_s{s}.json").exists()],
        {"rect":(0,.055,1,.94),"h_pad":1.15})


def result_axis_style(axis,title_size=24,label_size=24,tick_size=22):
    """Physical source sizes that remain readable after report/slide scaling."""
    axis.title.set_fontsize(title_size)
    axis.xaxis.label.set_fontsize(label_size);axis.yaxis.label.set_fontsize(label_size)
    axis.tick_params(labelsize=tick_size)


def final_figures():
    final = read("final_metrics.json")
    names = "fig_before_after_bars fig_before_after_slide fig_per_class_f1 fig_confusion_side_by_side fig_roc fig_robustness fig_reliability"
    if not final:
        skip(names, "Final held-out evaluation has not completed")
        return
    methods = [m for m in LABELS if m in final["models"]]
    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc"]
    figure, axis = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(metric_keys)); width = .24
    for offset, method in enumerate(methods):
        aggregate = final["models"][method]["aggregate"]
        bars = axis.bar(x+(offset-(len(methods)-1)/2)*width, [100*aggregate[k]["mean"] for k in metric_keys], width,
            yerr=[100*aggregate[k]["std"] for k in metric_keys], capsize=3, color=COLORS[method], label=LABELS[method], hatch={"baseline":"//","de":"","rs":".."}[method], edgecolor="white")
        axis.bar_label(bars, labels=[f"{100*aggregate[k]['mean']:.2f}" for k in metric_keys], fontsize=22, padding=5,rotation=90)
    axis.set_xticks(x, ["Accuracy", "Macro\nprecision", "Macro\nrecall", "Macro-F1", "Macro\nROC-AUC"])
    axis.set(ylabel="Score (%)", ylim=(0,145), title="Held-out performance across every preregistered seed")
    axis.set_yticks([0,25,50,75,100]);result_axis_style(axis)
    axis.legend(frameon=False, loc="upper center",bbox_to_anchor=(.5,-.22),ncol=3,fontsize=22); axis.grid(axis="y", alpha=.13)
    save(figure, "fig_before_after_bars", ["final_metrics.json"])
    # The hero slide also carries four large numeric cards. This shallow
    # companion keeps every metric, method and SD bar readable beneath them.
    figure,axis=plt.subplots(figsize=(18,3.5))
    for offset,method in enumerate(methods):
        aggregate=final["models"][method]["aggregate"]
        axis.bar(x+(offset-(len(methods)-1)/2)*width,[100*aggregate[k]["mean"] for k in metric_keys],width,
            yerr=[100*aggregate[k]["std"] for k in metric_keys],capsize=4,color=COLORS[method],label=LABELS[method],
            hatch={"baseline":"//","de":"","rs":".."}[method],edgecolor="white")
    axis.set_xticks(x,["Accuracy","Macro\nprecision","Macro\nrecall","Macro-F1","Macro\nROC-AUC"])
    highest=max(100*(final["models"][m]["aggregate"][k]["mean"]+final["models"][m]["aggregate"][k]["std"]) for m in methods for k in metric_keys)
    axis.set(ylabel="Score (%)",ylim=(0,max(105,highest+3)));axis.set_yticks([0,50,100]);axis.grid(axis="y",alpha=.13)
    result_axis_style(axis,24,24,24)
    handles,labels=axis.get_legend_handles_labels();figure.legend(handles,labels,frameon=False,fontsize=24,ncol=3,loc="upper center")
    save(figure,"fig_before_after_slide",["final_metrics.json"],{"rect":(0,0,1,.78)})
    MANIFEST["fig_before_after_slide"].update({"selection":"All five metrics and all available methods; identical means and sample SDs to the report hero",
        "metric_keys":metric_keys,"methods":methods,"label_font_points":24,"numeric_labels":"Exact values remain in the report chart/table and slide cards"})
    base = final["models"]["baseline"]["per_seed"]
    optimized = final["models"]["de"]["per_seed"]
    base_f1 = np.mean([[r["f1"] for r in s["per_class"]] for s in base], axis=0)
    de_f1 = np.mean([[r["f1"] for r in s["per_class"]] for s in optimized], axis=0)
    order = np.argsort(de_f1-base_f1)
    classes = final["class_names"]
    figure, axis = plt.subplots(figsize=(12.5,8.5))
    y = np.arange(len(classes))
    axis.barh(y-.17, base_f1[order]*100, height=.32, color=COLORS["baseline"], hatch="//", label="Baseline")
    axis.barh(y+.17, de_f1[order]*100, height=.32, color=COLORS["de"], label="DE-optimized")
    axis.set_yticks(y, [short_class(classes[i]) for i in order]); axis.set(xlim=(0,145), xlabel="Mean per-class F1 (%)", title="Per-class F1, ordered by measured gain")
    axis.set_xticks([0,25,50,75,100]);result_axis_style(axis,30,28,28)
    for position, index in enumerate(order): axis.text(102,position,f"{100*(de_f1[index]-base_f1[index]):+.1f} pp",va="center",fontsize=28)
    axis.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.14),ncol=2,fontsize=28)
    save(figure,"fig_per_class_f1",["final_metrics.json"])
    # Compact stable class codes let both matrices and the full class key keep
    # readable type. Codes follow the saved class order, never model outcomes.
    codes=[chr(65+i) for i in range(len(classes))]
    figure=plt.figure(figsize=(14.9,6.4))
    # Slightly wider cells leave a positive gap between adjacent fractions at
    # 22 pt. Wrapping the long key entries preserves the full report type size.
    axes=[figure.add_axes([left/14.9,.6/6.4,5.3/14.9,5.3/6.4]) for left in (.65,6.6)]
    key_axis=figure.add_axes([12.25/14.9,1.65/6.4,2.5/14.9,4.45/6.4]);key_axis.axis("off")
    scale_axis=figure.add_axes([12.25/14.9,.85/6.4,2.5/14.9,.15/6.4])
    for axis,method in zip(axes,("baseline","de")):
        matrix=np.asarray(final["models"][method]["per_seed"][0]["confusion_matrix"])
        normalized=matrix/np.maximum(matrix.sum(1,keepdims=True),1)
        im=axis.imshow(normalized,cmap="Blues",vmin=0,vmax=1)
        axis.set_xticks(range(len(classes)),codes)
        axis.set_yticks(range(len(classes)),codes)
        axis.set_xticks(np.arange(-.5,len(classes),1),minor=True);axis.set_yticks(np.arange(-.5,len(classes),1),minor=True)
        axis.grid(which="minor",color=STYLE["foreground"],alpha=.2,linewidth=.7)
        axis.tick_params(which="minor",bottom=False,left=False)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if normalized[i,j]>=.005:
                    value=f"{normalized[i,j]:.2f}"
                    label="1" if value=="1.00" else value.lstrip("0")
                    axis.text(j,i,label,ha="center",va="center",fontsize=22,color="white" if normalized[i,j]>.5 else "black")
        axis.set(title=f"{LABELS[method]} · seed 0",xlabel="Predicted class",ylabel="True class" if method=="baseline" else "")
        result_axis_style(axis,24,24,22)
        axis.tick_params(pad=6)
        axis.set_title(f"{LABELS[method]} · seed 0",fontsize=24,pad=3)
        axis.tick_params(axis="x",pad=1);axis.xaxis.labelpad=1
    key_offset=0.
    for code,name in zip(codes,classes):
        lines=textwrap.wrap(short_class(name),14)
        label=f"{code}  {lines[0]}"+"".join(f"\n   {line}" for line in lines[1:])
        key_axis.text(0,1-key_offset/4.45,label,fontsize=22,va="top",linespacing=1.1)
        key_offset+=.34*len(lines)+.015
    figure.text(12.25/14.9,1.4/6.4,"Cells: fractions",fontsize=22)
    scale=figure.colorbar(im,cax=scale_axis,orientation="horizontal",ticks=[0,.5,1])
    scale.set_label("True-class fraction",fontsize=22);scale.ax.tick_params(labelsize=22)
    save(figure,"fig_confusion_side_by_side",["final_metrics.json"],apply_tight_layout=False)
    MANIFEST["fig_confusion_side_by_side"]["class_code_key"]=dict(zip(codes,classes))
    figure,axis=plt.subplots(figsize=(8,6.4))
    for method in ("baseline","de"):
        roc=final["roc"][method]
        for row in roc["per_class"]: axis.plot(row["fpr"],row["tpr"],color=COLORS[method],alpha=.13,linewidth=.8,linestyle="--" if method=="baseline" else "-")
        axis.plot(roc["macro_fpr"],roc["macro_tpr"],color=COLORS[method],linewidth=2,linestyle="--" if method=="baseline" else "-",label=f"{LABELS[method]} · macro ROC")
    axis.plot([0,1],[0,1],"--",color="#A0A0A0",linewidth=.8)
    axis.set(xlabel="False-positive rate",ylabel="True-positive rate",title="One-vs-rest ROC curves · seed 0",xlim=(0,1),ylim=(0,1.02));axis.legend(frameon=False,fontsize=18)
    result_axis_style(axis,20,20,18)
    save(figure,"fig_roc",["final_metrics.json"])
    corruption_names=list(final["robustness"]["baseline"]["results"])
    figure,axes=plt.subplots(2,3,figsize=(13.6,6.8))
    for axis,corruption in zip(axes.flat,corruption_names):
        for method,marker in (("baseline","s"),("de","o")):
            rows=final["robustness"][method]["results"][corruption]
            axis.plot([1,2,3],[100*rows[str(s)]["accuracy"] for s in (1,2,3)],color=COLORS[method],marker=marker,label=LABELS[method])
        axis.set(title="JPEG" if corruption=="jpeg" else corruption.replace("_"," ").title(),xlabel="Severity",ylabel="Accuracy (%)",ylim=(0,102));axis.set_xticks([1,2,3]);axis.set_yticks([0,50,100]);axis.grid(alpha=.15)
        result_axis_style(axis,24,24,24)
    axes.flat[5].axis("off");handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[5].legend(handles,labels,frameon=False,loc="center",fontsize=24)
    save(figure,"fig_robustness",["final_metrics.json"],{"h_pad":1.5,"w_pad":1.4})
    figure,axes=plt.subplots(1,2,figsize=(14,6.6))
    for axis,method in zip(axes,("baseline","de")):
        calibration=final["models"][method]["per_seed"][0]["calibration"]
        rows=[r for r in calibration["bins"] if r["count"]]
        axis.plot([0,1],[0,1],"--",color="#A0A0A0",label="Perfect calibration")
        axis.scatter([r["confidence"] for r in rows],[r["accuracy"] for r in rows],s=[25+100*r["count"]/final["test_images"] for r in rows],color=COLORS[method],zorder=3)
        axis.plot([r["confidence"] for r in rows],[r["accuracy"] for r in rows],color=COLORS[method],label=f"ECE = {100*calibration['ece']:.2f}%")
        axis.set(xlim=(0,1),ylim=(0,1),xlabel="Mean confidence in bin",ylabel="Observed accuracy in bin",title=f"{LABELS[method]} · seed 0");axis.legend(frameon=False,fontsize=22)
        axis.set_xticks([0,.5,1]);axis.set_yticks([0,.5,1]);result_axis_style(axis)
    save(figure,"fig_reliability",["final_metrics.json"])


def gradcam_canvas(case_count,figsize,scale_font=18):
    """Two image/baseline/DE triplets per row, with a shared activation scale."""
    rows=(case_count+1)//2
    figure=plt.figure(figsize=figsize)
    width,height=figsize
    # Fixed physical areas prevent automatic square-image aspect adjustment
    # from moving four-line titles over the preceding row at report scale.
    margin,gap,case_gap=.25,.2,.65
    slot=(width-2*margin-4*gap-case_gap)/6
    row_height=(height-.95)/rows
    title_height=1.42 if rows==1 else 1.2
    side=min(slot,row_height-title_height-.12)
    axes=[]
    for row in range(rows):
        bottom=height-.1-row*row_height-title_height-side
        axes.append([figure.add_axes([(margin+column*(slot+gap)+(case_gap-gap if column>=3 else 0)+(slot-side)/2)/width,
            bottom/height,side/width,side/height]) for column in range(6)])
    axes=np.asarray(axes)
    color_axis=figure.add_axes([margin/width,.57/height,(width-2*margin)/width,.1/height])
    scale=plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap="inferno")
    colorbar=figure.colorbar(scale,cax=color_axis,orientation="horizontal",ticks=[0,.5,1])
    colorbar.set_label("Within-image normalized activation · white contour = 0.5",fontsize=scale_font)
    colorbar.ax.tick_params(labelsize=scale_font)
    return figure,axes


def show_gradcam(axis,pixels,activation):
    """Render the saved map unchanged, with a non-color cue for its 0.5 boundary."""
    axis.imshow(pixels,interpolation="none")
    axis.imshow(activation,cmap="inferno",vmin=0,vmax=1,alpha=.42,interpolation="none")
    if float(np.min(activation))<.5<float(np.max(activation)):
        boundary=axis.contour(activation,levels=[.5],colors="white",linewidths=.9)
        boundary.set_path_effects([path_effects.Stroke(linewidth=1.8,foreground=STYLE["foreground"]),path_effects.Normal()])


def gradcam_case(axes,image,maps,record,classes,index,font_size,wrap_width):
    def label(class_index):
        lines=textwrap.wrap(short_class(classes[class_index]),wrap_width)
        # Reserve the same title height for short and long class names.
        return "\n".join(lines+[""]*max(0,2-len(lines)))
    axes[0].imshow(image,interpolation="none")
    axes[0].set_title(f"True class\n{label(record['true_label'])}\nLeaf {index+1}",fontsize=font_size,pad=9)
    for column,method in enumerate(("baseline","de"),start=1):
        show_gradcam(axes[column],image,maps[column-1])
        prediction=record[method]
        axes[column].set_title(f"{LABELS[method]}\n{label(prediction['predicted_index'])}\nConf. {100*prediction['probability']:.1f}%",
            fontsize=font_size,pad=9)
    for axis in axes:axis.axis("off")


def qualitative_figures():
    path=RESULTS/"qualitative_assets.npz"
    if not path.exists():
        for name in ("fig_gradcam_grid","fig_predictions_grid","fig_corruptions"):
            if (FIGURES/f"{name}.png").exists(): MANIFEST[name]={"status":"preserved","sources":["07_evaluate.py"]}
            else: skip(name,"Qualitative evaluation assets unavailable")
        skip("fig_gradcam_slide","Saved held-out Grad-CAM images and maps are required")
        return
    assets=np.load(path);records=read("qualitative.json",[]);predictions=read("prediction_examples.json",[]);classes=read("class_names.json",[])
    grid_records=records[:6]
    if grid_records:
        figure,axes=gradcam_canvas(len(grid_records),(12,9.5))
        for i,record in enumerate(grid_records):
            gradcam_case(axes[i//2,(i%2)*3:(i%2+1)*3],assets["gradcam_images"][i],assets["gradcam_maps"][i],record,classes,i,18,12)
        for axis in axes.flat:axis.axis("off")
        save(figure,"fig_gradcam_grid",["qualitative_assets.npz","qualitative.json","class_names.json"],apply_tight_layout=False)
        MANIFEST["fig_gradcam_grid"].update({"record_indices":list(range(len(grid_records))),
            "source_paths":[r["source_path"] for r in grid_records],"layout":"Two fixed image/baseline/DE triplets per row, row-major case order",
            "label_font_points":18,"scale_font_points":18,"confidence_label":"Conf. is predicted-class confidence"})
    else:skip("fig_gradcam_grid","Saved held-out Grad-CAM records are required")
    if len(records)>=2:
        # Fixed first two saved examples: presentation selection does not depend on
        # whether DE was correct, more confident, or visually more convincing.
        figure,axes=gradcam_canvas(2,(18,5.3))
        for i,record in enumerate(records[:2]):
            gradcam_case(axes[0,i*3:(i+1)*3],assets["gradcam_images"][i],assets["gradcam_maps"][i],record,classes,i,22,16)
        save(figure,"fig_gradcam_slide",["qualitative_assets.npz","qualitative.json","class_names.json"],apply_tight_layout=False)
        MANIFEST["fig_gradcam_slide"].update({"record_indices":[0,1],"source_paths":[r["source_path"] for r in records[:2]],
            "selection":"First two saved examples in class order, independent of model correctness or confidence",
            "layout":"One row containing two image/baseline/DE triplets","label_font_points":22,"scale_font_points":18,
            "confidence_label":"Conf. is predicted-class confidence"})
    else:skip("fig_gradcam_slide","Two saved held-out examples are required")
    figure,axes=plt.subplots(3,4,figsize=(12,10.2))
    for index,axis in enumerate(axes.flat):
        axis.axis("off")
        if index>=len(predictions):continue
        record=predictions[index];axis.imshow(assets["prediction_images"][index],interpolation="none");status="Correct" if record["correct"] else "Error"
        truth=textwrap.fill("True: "+short_class(classes[record["true_label"]]),18)
        prediction=textwrap.fill("Predicted: "+short_class(classes[record["prediction"]]),18)
        axis.set_title(f"{status}\n{truth}\n{prediction}",fontsize=18,pad=8,color=STYLE["de"] if record["correct"] else "#B43232")
    save(figure,"fig_predictions_grid",["qualitative_assets.npz","prediction_examples.json"],{"h_pad":3.0,"w_pad":1.2})
    MANIFEST["fig_predictions_grid"].update({"record_indices":list(range(min(12,len(predictions)))),
        "layout":"Three rows by four columns, saved example order","label_font_points":18,
        "status_cues":"Explicit Correct/Error words, plus green/red labels"})
    figure,axes=plt.subplots(3,5,figsize=(14.5,10))
    for i,name in enumerate(("Blur","Noise","Brightness","JPEG","Low light")):
        for j in range(3):axes[j,i].imshow(assets["corruption_images"][i,j],interpolation="none");axes[j,i].set_title(f"{name}\nSeverity {j+1}",fontsize=22,pad=10);axes[j,i].axis("off")
    save(figure,"fig_corruptions",["qualitative_assets.npz"],{"h_pad":3.0,"w_pad":1.0})
    MANIFEST["fig_corruptions"].update({"layout":"Three severity rows by five corruption columns",
        "source_indices":[[kind,severity] for severity in range(3) for kind in range(5)],"label_font_points":22})
    assets.close()


def benchmark_surfaces():
    """Evaluate the actual two-dimensional functions; no optimizer run is needed."""
    figure=plt.figure(figsize=(14,9.8))
    for index,(name,spec) in enumerate(FUNCTIONS.items(),start=1):
        axis=figure.add_subplot(2,3,index,projection="3d",computed_zorder=False)
        low,high=spec["bounds"]
        coordinates=np.linspace(low,high,201)
        xx,yy=np.meshgrid(coordinates,coordinates)
        zz=spec["function"](np.stack((xx,yy),axis=-1))
        # Surface geometry and heights use the untransformed objective. A
        # common sequential palette does not imply a common scale across functions.
        axis.plot_surface(xx,yy,zz,cmap="viridis",rstride=2,cstride=2,linewidth=0,antialiased=True,zorder=1)
        optimum=spec["optimum_coordinate"]
        axis.scatter([optimum],[optimum],[float(spec["function"]([optimum,optimum]))],
            marker="*",s=100,color=STYLE["random_search"],edgecolors="white",linewidths=.8,depthshade=False,zorder=5)
        axis.set_title(name,fontsize=24,pad=14)
        axis.set(xlim=(low,high),ylim=(low,high),xlabel="x₁",ylabel="x₂")
        axis.set_box_aspect((1,1,.75));axis.view_init(elev=27,azim=-54)
        axis.xaxis.set_major_locator(MaxNLocator(3,prune="both"));axis.yaxis.set_major_locator(MaxNLocator(3,prune="both"))
        axis.zaxis.set_major_locator(MaxNLocator(3))
        formatter=ScalarFormatter(useMathText=True);formatter.set_powerlimits((-3,3));axis.zaxis.set_major_formatter(formatter)
        if float(zz.max())>=1_000_000:
            axis.zaxis.set_major_formatter(FuncFormatter(lambda value,position:f"{value/1_000_000:g}M"))
        axis.tick_params(labelsize=21,pad=0)
        axis.tick_params(axis="y",pad=6)
        # One explicit height key avoids 3D z-labels being clipped by adjacent
        # panels. The untransformed vertical values remain labeled numerically.
        axis.set_zlabel("")
        axis.set_xlabel("x₁",labelpad=7);axis.set_ylabel("x₂",labelpad=9)
        for coordinate_axis in (axis.xaxis,axis.yaxis,axis.zaxis):coordinate_axis.label.set_size(22);coordinate_axis.offsetText.set_fontsize(21)
    note=figure.add_subplot(2,3,6);note.axis("off")
    note.text(.07,.94,"Analytic landscapes",fontsize=25,fontweight="bold",va="top")
    note.text(.07,.77,"Height: f(x₁,x₂)\nFull 2D domains\nSeparate linear scales\nColor: lower → higher",fontsize=21,va="top",linespacing=1.35)
    note.text(.07,.36,"★  Global minimum: 0\nM denotes one million",fontsize=21,va="top",linespacing=1.35)
    dimensions=CONFIG["benchmarks"]["dimensions"]
    note.text(.07,.15,f"Views: two dimensions\nMain study: {dimensions} dimensions",fontsize=21,va="top",color=STYLE["foreground"],linespacing=1.35)
    save(figure,"fig_benchmark_surfaces",["../bioalgo/benchmarks.py/FUNCTIONS","../config.yaml/benchmarks/dimensions"])
    MANIFEST["fig_benchmark_surfaces"].update({"dimensions":2,"grid_points_per_axis":201,
        "functions":{name:{"bounds":list(spec["bounds"]),"optimum_coordinate":spec["optimum_coordinate"]} for name,spec in FUNCTIONS.items()},
        "vertical_scale":"Untransformed objective values; independent linear axis per function"})


def embedding_figures():
    paths={m:RESULTS/f"predictions_{m}_s0.npz" for m in ("baseline","de")}
    if not all(path.exists() for path in paths.values()):
        skip("fig_embeddings","Saved seed-0 embeddings are unavailable");return
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    projections={}
    for method,path in paths.items():
        source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        cached=read(f"embedding_projection_{method}.json")
        if cached and cached["source_sha256"]==source_hash:projections[method]=cached;continue
        saved=np.load(path);values=saved["embeddings"];labels=saved["labels"]
        if values.ndim!=2 or len(values)<5:skip("fig_embeddings","Saved embedding matrix is missing or too small");return
        reduced=PCA(n_components=min(50,values.shape[1],len(values)-1),random_state=42).fit_transform(values)
        xy=TSNE(n_components=2,random_state=42,init="pca",learning_rate="auto",perplexity=min(30,(len(values)-1)/3)).fit_transform(reduced)
        cached=dict(source_sha256=source_hash,seed=42,coordinates=xy.tolist(),labels=labels.tolist(),method="PCA to at most 50 dimensions then t-SNE; independent layout per model")
        (RESULTS/f"embedding_projection_{method}.json").write_text(json.dumps(cached),encoding="utf-8");projections[method]=cached
    classes=read("class_names.json")
    figure,axes=plt.subplots(1,2,figsize=(13,6))
    for axis,method in zip(axes,("baseline","de")):
        xy=np.asarray(projections[method]["coordinates"]);labels=np.asarray(projections[method]["labels"])
        for index,name in enumerate(classes):
            selected=labels==index;axis.scatter(xy[selected,0],xy[selected,1],s=12,alpha=.8,label=short_class(name),color=plt.cm.tab10(index),edgecolors=STYLE["foreground"],linewidths=.3,marker=["o","s","^","D","v","P","X","<",">","*"][index])
        axis.set(title=f"{LABELS[method]} · seed 0",xlabel="t-SNE component 1",ylabel="t-SNE component 2");axis.set_xticks([]);axis.set_yticks([])
        result_axis_style(axis,24,22,22)
    legend=axes[1].legend(frameon=False,fontsize=22,bbox_to_anchor=(1.03,1),loc="upper left",markerscale=3,handletextpad=.3,labelspacing=.35)
    for handle in legend.legend_handles:handle.set_alpha(1)
    figure.suptitle("Independent t-SNE layouts\nDistances across models are not comparable",fontsize=22)
    save(figure,"fig_embeddings",["predictions_baseline_s0.npz","predictions_de_s0.npz"])


def eda_figures():
    summary=read("data_summary.json");classes=read("class_names.json")
    if summary and classes:
        figure,axis=plt.subplots(figsize=(11,7))
        counts=[summary["original_counts"][name] for name in classes]
        bars=axis.barh([short_class(name) for name in classes],counts,color=STYLE["de"])
        axis.invert_yaxis();axis.bar_label(bars,padding=5,fontsize=20);axis.set_xlim(0,max(counts)*1.16)
        axis.set(xlabel="Images in original color dataset",title="The tomato classes are strongly imbalanced")
        result_axis_style(axis,22,22,20)
        save(figure,"fig_class_distribution",["data_summary.json","class_names.json"])
    path=RESULTS/"eda_assets.npz"
    if not path.exists():
        for name in ("fig_dataset_samples","fig_augmentation_examples"):
            MANIFEST[name]={"status":"preserved" if (FIGURES/f"{name}.png").exists() else "unavailable","sources":["02_eda.py"]}
        skip("fig_dataset_slide","Saved training-image display assets are required")
        return
    assets=np.load(path)
    # Two class groups per row keep all thirty leaves legible on a report page.
    # Native pixels avoid inventing detail and keep embedded PDF images compact.
    figure,axes=plt.subplots((len(classes)+1)//2,6,figsize=(12,12.8),squeeze=False)
    for label,name in enumerate(classes):
        for col in range(3):
            axis=axes[label//2,(label%2)*3+col]
            axis.imshow(assets["sample_images"][label,col],interpolation="none");axis.axis("off")
            if col==1:axis.set_title(textwrap.fill(short_class(name),18),fontsize=20,pad=12)
    save(figure,"fig_dataset_samples",["eda_assets.npz","class_names.json"])
    figure,axes=plt.subplots(2,5,figsize=(12.5,5.9),squeeze=False)
    for label,axis in enumerate(axes.flat):
        axis.axis("off")
        if label>=len(classes):continue
        axis.imshow(assets["sample_images"][label,0],interpolation="none")
        title=textwrap.fill(short_class(classes[label]),15)
        if "\n" not in title:
            title="\n".join(title.rsplit(" ",1)) if " " in title else "\n"+title
        axis.set_title(title,fontsize=24,pad=10)
    save(figure,"fig_dataset_slide",["eda_assets.npz/sample_images","class_names.json"],{"h_pad":3.2,"w_pad":1.5})
    MANIFEST["fig_dataset_slide"].update({"sample_indices":[[label,0] for label in range(min(10,len(classes)))],
        "selection":"First saved training example per class, in fixed class order"})
    figure,axes=plt.subplots(1,4,figsize=(12,3.8))
    titles=("Original\ntraining image","Flips +\nrandom crop","RandAugment\nmagnitude 7","RandAugment\nmagnitude 15")
    for index,axis in enumerate(axes):axis.imshow(assets["augmentation_images"][index],interpolation="none");axis.set_title(titles[index],fontsize=20);axis.axis("off")
    save(figure,"fig_augmentation_examples",["eda_assets.npz"])
    assets.close()


def box(axis,x,y,w,h,title,body,color,title_size=24,body_size=22):
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=0.035",facecolor=color,edgecolor="none",alpha=.12)
    axis.add_patch(patch);axis.text(x+w/2,y+h*.8,title,ha="center",va="center",fontsize=title_size,fontweight="bold",color=STYLE["foreground"])
    axis.text(x+w/2,y+h*.34,body,ha="center",va="center",fontsize=body_size,color=STYLE["foreground"],linespacing=1.12)


def arrow(axis,start,end,color=None):
    axis.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=18,linewidth=1.8,color=color or STYLE["de"]))


def diagrams():
    summary=read("model_summary.json")
    if summary:
        figure=plt.figure(figsize=(14,6.2));axis=figure.add_axes([0,0,1,1]);axis.set(xlim=(0,14),ylim=(0,6.2));axis.axis("off")
        blocks=[r for r in summary if isinstance(r["block"],int)]
        specs=[("Input",f"RGB\n{CONFIG['training']['resolution']} × {CONFIG['training']['resolution']}"),
               ("Stem",str(blocks[0]["output_shape"][1:])+"\nConv · BN · SiLU"),
               ("MBConv 1–3","Channels\n"+" / ".join(str(r["output_shape"][1]) for r in blocks[1:4])),
               ("MBConv 4–7","Channels\n"+" / ".join(str(r["output_shape"][1]) for r in blocks[4:6])+"\n"+" / ".join(str(r["output_shape"][1]) for r in blocks[6:8])),
               ("Final conv.",str(blocks[8]["output_shape"][1:])+"\nFeature map"),
               ("Avg. pooling",f"{blocks[8]['output_shape'][1]} features"),
               ("Evolved head","Dropout → Dense\nSiLU → Dropout\nClass logits"),
               ("Prediction",f"{summary[-1]['output_shape'][-1]} tomato classes\nClass probabilities")]
        positions=[(.2,3.75),(3.75,3.75),(7.3,3.75),(10.85,3.75),(10.85,.75),(7.3,.75),(3.75,.75),(.2,.75)]
        for i,((title,body),(x,y)) in enumerate(zip(specs,positions)):box(axis,x,y,3,1.8,title,body,STYLE["de"] if i in (6,7) else STYLE["pso"])
        for i in range(3):arrow(axis,(positions[i][0]+3,4.65),(positions[i+1][0],4.65))
        arrow(axis,(12.35,3.75),(12.35,2.55))
        for i in range(4,7):arrow(axis,(positions[i][0],1.65),(positions[i+1][0]+3,1.65))
        axis.text(7,3.15,"Baseline: backbone frozen · DE selects last k stages and head width",ha="center",va="center",fontsize=22)
        axis.text(7,.25,"Head width 0 skips the hidden Dense and SiLU layers.",ha="center",va="center",fontsize=22)
        axis.text(7,5.98,"EfficientNet-B0 · measured backbone and evolved head",ha="center",va="center",fontsize=24,fontweight="bold")
        save(figure,"fig_architecture",["model_summary.json","../config.yaml"],apply_tight_layout=False)
    else:skip("fig_architecture","Model summary unavailable")
    data=read("data_summary.json",{});sizes=data.get("sizes",{})
    figure=plt.figure(figsize=(14.5,6.9));axis=figure.add_axes([0,0,1,1]);axis.set(xlim=(0,14.5),ylim=(0,6.9));axis.axis("off")
    nodes=[(.2,4.3,2.8,1.8,"PlantVillage",f"Tomato images\n{data.get('usable_total','')} cleaned"),
           (3.3,4.3,3,1.8,"Grouped split",f"Train {sizes.get('train','')}\nVal {sizes.get('val','')}\nTest {sizes.get('test','')}"),
           (6.8,5.2,3.4,1.2,"Fixed baseline","Preregistered recipe"),
           (6.8,3.4,3.55,1.4,"Recipe search","DE + random search\nValidation macro-F1"),
           (10.7,3.75,3.4,1.8,"Rerank recipes","Higher fidelity\nValidation only"),
           (10.7,.85,3.4,1.8,"Final training",f"Same epoch budget\n{len(CONFIG['seeds'])} training seeds\nFixed recipes"),
           (5.6,.85,4.2,1.8,"Held-out evaluation","Metrics + CIs\nPaired tests\nCorruptions + Grad-CAM"),
           (.2,.85,4.6,1.8,"Evidence + demo","JSON → figures / report\nExact checkpoints\nOffline diagnosis")]
    for x,y,w,h,title,body in nodes:box(axis,x,y,w,h,title,body,STYLE["de"],26,24)
    arrow(axis,(3,5.2),(3.3,5.2));arrow(axis,(6.3,5.55),(6.8,5.8));arrow(axis,(6.3,4.6),(6.8,4.1))
    arrow(axis,(10.35,4.1),(10.7,4.65));arrow(axis,(12.4,3.75),(12.4,2.65))
    # The fixed baseline bypasses recipe selection; it must never be re-ranked.
    axis.plot([10.2,14.35,14.35],[5.8,5.8,1.75],color=STYLE["baseline"],linewidth=1.8)
    arrow(axis,(14.35,1.75),(14.1,1.75),STYLE["baseline"])
    arrow(axis,(10.7,1.75),(9.8,1.75));arrow(axis,(5.6,1.75),(4.8,1.75))
    axis.text(7.25,.25,"Test images never guide search, reranking, early stopping or model selection.",ha="center",va="center",fontsize=24)
    axis.text(7.25,6.65,"From leaf images to a reproducible comparison",ha="center",va="center",fontsize=26,fontweight="bold")
    save(figure,"fig_pipeline",["data_summary.json","../config.yaml"],apply_tight_layout=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--skip-embeddings",action="store_true");args=parser.parse_args()
    FIGURES.mkdir(parents=True,exist_ok=True)
    diagrams();eda_figures();search_figures();learning_figures();final_figures();qualitative_figures();benchmark_surfaces()
    if not args.skip_embeddings:embedding_figures()
    benchmark_path=ROOT/"scripts"/"08_benchmarks.py"
    if (RESULTS/"benchmarks.json").exists():
        spec=importlib.util.spec_from_file_location("benchmark_figures",benchmark_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.render_figures();module.render_operators()
        if (RESULTS/"benchmark_2d.json").exists():module.render_animation()
        for name in ("fig_benchmark_convergence","fig_benchmark_box","fig_de_rastrigin_2d","fig_de_operators_diagram"):
            MANIFEST[name]={"status":"generated","sources":["benchmarks.json","benchmark_2d.json","de_operator_example.json"]}
    (RESULTS/"figure_manifest.json").write_text(json.dumps(MANIFEST,indent=2),encoding="utf-8")
    generated=sum(row["status"]=="generated" for row in MANIFEST.values());missing=[name for name,row in MANIFEST.items() if row["status"]=="unavailable"]
    print(f"Generated {generated} figure families from saved evidence. Missing pending experiments: {', '.join(missing) if missing else 'none'}")


if __name__=="__main__":main()
