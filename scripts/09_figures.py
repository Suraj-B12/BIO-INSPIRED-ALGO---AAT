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


def save(figure, name, sources):
    figure.tight_layout()
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
        figure, axis = plt.subplots(figsize=(9, 5))
        mean, spread = np.array([r["mean"] for r in history]), np.array([r["std"] for r in history])
        axis.plot(generation, [r["best"] for r in history], color=STYLE["de"], marker="o", label="Best validation macro-F1")
        axis.plot(generation, mean, color=STYLE["pso"], marker="s", label="Population mean")
        axis.fill_between(generation, np.maximum(0, mean - spread), np.minimum(1, mean + spread), color=STYLE["pso"], alpha=.12, label="Population ±1 SD")
        baseline = read("baseline_proxy.json")
        sources = ["de_history.json"]
        if baseline:
            axis.axhline(baseline["validation_macro_f1"], color=STYLE["baseline"], linestyle="--", label="Baseline · same proxy fidelity")
            sources.append("baseline_proxy.json")
        axis.set(xlabel="Generation (0 = initialization)", ylabel="Validation macro-F1", title="How the DE population improves its training recipes")
        axis.legend(frameon=False, fontsize=11); axis.grid(alpha=.15)
        save(figure, "fig_de_convergence", sources)
        figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(generation, [r["diversity"] for r in history], color=STYLE["de"], marker="o")
        axes[0].set(ylabel="Mean pairwise distance", title="Exploration and successful replacements during search")
        axes[1].plot(generation, [r["success_rate"] for r in history], color=STYLE["pso"], marker="s")
        axes[1].set(xlabel="Generation", ylabel="Replacement rate", ylim=(-.03, 1.03))
        for axis in axes: axis.grid(alpha=.15)
        save(figure, "fig_de_diversity", ["de_history.json"])
    else:
        skip("fig_de_convergence fig_de_diversity", "DE search has not completed")
    rows = csv_rows("de_evaluations.csv")
    random_rows = csv_rows("rs_evaluations.csv")
    if rows and random_rows:
        figure, axis = plt.subplots(figsize=(9, 5))
        for method, records in (("de", rows), ("rs", random_rows)):
            ordered = sorted(records, key=lambda r: int(r["evaluation"]))
            axis.plot([int(r["evaluation"]) for r in ordered], np.maximum.accumulate([float(r["fitness"]) for r in ordered]),
                      color=COLORS[method], label=LABELS[method], marker="o" if method == "de" else "s", markevery=max(1, len(ordered)//8))
        axis.set(xlabel="Candidate evaluations", ylabel="Best-so-far validation macro-F1", title="DE and random search under the same evaluation budget")
        axis.legend(frameon=False); axis.grid(alpha=.15)
        save(figure, "fig_de_vs_rs", ["de_evaluations.csv", "rs_evaluations.csv"])
    else:
        skip("fig_de_vs_rs", "Both DE and random-search records are needed")
    if not rows:
        skip("fig_search_landscape fig_parallel_coords fig_gene_importance", "No DE candidate evaluations saved")
        return
    scores = np.asarray([float(r["fitness"]) for r in rows])
    figure, axis = plt.subplots(figsize=(9, 5.5))
    scatter = axis.scatter([np.log10(float(r["learning_rate"])) for r in rows], [int(r["unfreeze_blocks"]) for r in rows],
        c=scores, cmap="viridis", s=[45 + 14 * int(r["generation"]) for r in rows], edgecolors="white", linewidth=.5)
    best = rows[int(np.argmax(scores))]
    axis.scatter(np.log10(float(best["learning_rate"])), int(best["unfreeze_blocks"]), marker="*", s=300,
                 facecolors="none", edgecolors="#B45F06", linewidths=1.5, label="Best proxy candidate")
    axis.set(xlabel="log₁₀ learning rate", ylabel="Unfrozen backbone blocks", title="Measured candidate scores across two search dimensions")
    axis.legend(frameon=False); figure.colorbar(scatter, ax=axis, label="Validation macro-F1")
    save(figure, "fig_search_landscape", ["de_evaluations.csv"])
    encoded = np.asarray([[float(row[f"x{i+1}"]) for i in range(7)] for row in rows])
    figure, axis = plt.subplots(figsize=(12, 5))
    normalizer = matplotlib.colors.Normalize(vmin=float(scores.min()), vmax=max(float(scores.max()), float(scores.min()) + 1e-12))
    for vector, score in sorted(zip(encoded, scores), key=lambda item: item[1]):
        axis.plot(range(7), vector, color=plt.cm.viridis(normalizer(score)), alpha=.55, linewidth=1)
    axis.set_xticks(range(7), [label.replace(" ", "\n") for _, label in GENES])
    axis.set(ylabel="Normalized gene value", ylim=(0, 1), title="Every evaluated chromosome, colored by validation score")
    figure.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap="viridis"), ax=axis, label="Validation macro-F1")
    save(figure, "fig_parallel_coords", ["de_evaluations.csv"])
    import plotly.graph_objects as go
    interactive = go.Figure(data=go.Parcoords(line=dict(color=scores, colorscale="Viridis", showscale=True),
        dimensions=[dict(range=[0,1], label=label, values=encoded[:, index]) for index, (_, label) in enumerate(GENES)]))
    interactive.update_layout(title="DE candidate chromosomes · measured validation macro-F1", font_family="Arial")
    interactive.write_html(str(FIGURES / "fig_parallel_coords.html"), include_plotlyjs=True, full_html=True)
    correlations = [float(spearmanr(encoded[:, i], scores).statistic) if np.ptp(encoded[:, i]) > 0 and np.ptp(scores) > 0 else 0.0 for i in range(7)]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.barh([label for _, label in GENES], correlations, color=[STYLE["de"] if v >= 0 else STYLE["random_search"] for v in correlations])
    axis.axvline(0, color=STYLE["baseline"], linewidth=.8)
    axis.set(xlabel="Spearman rank correlation with validation macro-F1", xlim=(-1,1), title="Exploratory gene associations · not causal importance")
    save(figure, "fig_gene_importance", ["de_evaluations.csv"])


def learning_figures():
    curves = {method: [read(f"curves_{method}_s{seed}.json") for seed in CONFIG["seeds"]] for method in LABELS}
    curves = {method: [c for c in rows if c] for method, rows in curves.items() if any(rows)}
    if not curves:
        skip("fig_learning_curves", "No learning curves saved")
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for method, records in curves.items():
        common = min(map(len, records))
        epochs = np.arange(1, common + 1)
        for axis, key, title in zip(axes, ("train_loss", "val_loss", "val_macro_f1"), ("Training loss", "Validation loss", "Validation macro-F1")):
            values = np.asarray([[r[key] for r in record[:common]] for record in records])
            mean, deviation = values.mean(0), values.std(0, ddof=1) if len(values)>1 else np.zeros(common)
            axis.plot(epochs, mean, color=COLORS[method], label=f"{LABELS[method]} · {len(records)} seeds", marker={"baseline":"s","de":"o","rs":"^"}[method], markevery=max(1,common//6))
            if len(values)>1: axis.fill_between(epochs, mean-deviation, mean+deviation, color=COLORS[method], alpha=.12)
            axis.set(title=title, xlabel="Epoch", ylabel="Cross-entropy" if "loss" in key else "Macro-F1")
            axis.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=11)
    figure.suptitle("Mean ± sample SD; epochs shared by each method's available seeds", fontsize=15)
    save(figure, "fig_learning_curves", [f"curves_{m}_s{s}.json" for m in curves for s in CONFIG["seeds"] if (RESULTS/f"curves_{m}_s{s}.json").exists()])


def final_figures():
    final = read("final_metrics.json")
    names = "fig_before_after_bars fig_per_class_f1 fig_confusion_side_by_side fig_roc fig_robustness fig_reliability"
    if not final:
        skip(names, "Final held-out evaluation has not completed")
        return
    methods = [m for m in LABELS if m in final["models"]]
    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc"]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(metric_keys)); width = .24
    for offset, method in enumerate(methods):
        aggregate = final["models"][method]["aggregate"]
        bars = axis.bar(x+(offset-(len(methods)-1)/2)*width, [100*aggregate[k]["mean"] for k in metric_keys], width,
            yerr=[100*aggregate[k]["std"] for k in metric_keys], capsize=3, color=COLORS[method], label=LABELS[method], hatch={"baseline":"//","de":"","rs":".."}[method], edgecolor="white")
        axis.bar_label(bars, labels=[f"{100*aggregate[k]['mean']:.2f}" for k in metric_keys], fontsize=10, padding=4)
    axis.set_xticks(x, ["Accuracy", "Macro precision", "Macro recall", "Macro-F1", "Macro ROC-AUC"])
    axis.set(ylabel="Score (%)", ylim=(0,110), title="Held-out performance across every preregistered seed")
    axis.legend(frameon=False, loc="lower right"); axis.grid(axis="y", alpha=.13)
    save(figure, "fig_before_after_bars", ["final_metrics.json"])
    base = final["models"]["baseline"]["per_seed"]
    optimized = final["models"]["de"]["per_seed"]
    base_f1 = np.mean([[r["f1"] for r in s["per_class"]] for s in base], axis=0)
    de_f1 = np.mean([[r["f1"] for r in s["per_class"]] for s in optimized], axis=0)
    order = np.argsort(de_f1-base_f1)
    classes = final["class_names"]
    figure, axis = plt.subplots(figsize=(11,6.5))
    y = np.arange(len(classes))
    axis.barh(y-.17, base_f1[order]*100, height=.32, color=COLORS["baseline"], hatch="//", label="Baseline")
    axis.barh(y+.17, de_f1[order]*100, height=.32, color=COLORS["de"], label="DE-optimized")
    axis.set_yticks(y, [short_class(classes[i]) for i in order]); axis.set(xlim=(0,112), xlabel="Mean per-class F1 (%)", title="Per-class performance, ordered by measured improvement")
    for position, index in enumerate(order): axis.text(101,position,f"{100*(de_f1[index]-base_f1[index]):+.1f} pp",va="center",fontsize=11)
    axis.legend(frameon=False,loc="lower right"); save(figure,"fig_per_class_f1",["final_metrics.json"])
    figure, axes = plt.subplots(1,2,figsize=(14,6.3),constrained_layout=False)
    for axis,method in zip(axes,("baseline","de")):
        matrix=np.asarray(final["models"][method]["per_seed"][0]["confusion_matrix"])
        normalized=matrix/np.maximum(matrix.sum(1,keepdims=True),1)
        im=axis.imshow(normalized,cmap="Blues",vmin=0,vmax=1)
        axis.set_xticks(range(len(classes)),[short_class(c) for c in classes],rotation=65,ha="right",fontsize=10)
        axis.set_yticks(range(len(classes)),[short_class(c) for c in classes],fontsize=10)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if normalized[i,j]>=.005: axis.text(j,i,f"{normalized[i,j]*100:.0f}",ha="center",va="center",fontsize=9,color="white" if normalized[i,j]>.5 else "black")
        axis.set(title=f"{LABELS[method]} · seed 0",xlabel="Predicted class",ylabel="True class")
        figure.colorbar(im,ax=axis,fraction=.046,pad=.04,label="Fraction of true class")
    save(figure,"fig_confusion_side_by_side",["final_metrics.json"])
    figure,axis=plt.subplots(figsize=(7,6))
    for method in ("baseline","de"):
        roc=final["roc"][method]
        for row in roc["per_class"]: axis.plot(row["fpr"],row["tpr"],color=COLORS[method],alpha=.13,linewidth=.8)
        axis.plot(roc["macro_fpr"],roc["macro_tpr"],color=COLORS[method],linewidth=2,linestyle="--" if method=="baseline" else "-",label=f"{LABELS[method]} · macro ROC")
    axis.plot([0,1],[0,1],"--",color="#A0A0A0",linewidth=.8)
    axis.set(xlabel="False-positive rate",ylabel="True-positive rate",title="One-vs-rest ROC curves · seed 0",xlim=(0,1),ylim=(0,1.02));axis.legend(frameon=False)
    save(figure,"fig_roc",["final_metrics.json"])
    corruption_names=list(final["robustness"]["baseline"]["results"])
    figure,axes=plt.subplots(2,3,figsize=(13,7))
    for axis,corruption in zip(axes.flat,corruption_names):
        for method,marker in (("baseline","s"),("de","o")):
            rows=final["robustness"][method]["results"][corruption]
            axis.plot([1,2,3],[100*rows[str(s)]["accuracy"] for s in (1,2,3)],color=COLORS[method],marker=marker,label=LABELS[method])
        axis.set(title=corruption.replace("_"," ").title(),xlabel="Severity",ylabel="Accuracy (%)",ylim=(0,102));axis.set_xticks([1,2,3]);axis.grid(alpha=.15)
    axes.flat[5].axis("off");handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[5].legend(handles,labels,frameon=False,loc="center")
    figure.suptitle("Field-condition stress test · synthetic corruptions of held-out images",fontsize=14)
    save(figure,"fig_robustness",["final_metrics.json"])
    figure,axes=plt.subplots(1,2,figsize=(11,5))
    for axis,method in zip(axes,("baseline","de")):
        calibration=final["models"][method]["per_seed"][0]["calibration"]
        rows=[r for r in calibration["bins"] if r["count"]]
        axis.plot([0,1],[0,1],"--",color="#A0A0A0",label="Perfect calibration")
        axis.scatter([r["confidence"] for r in rows],[r["accuracy"] for r in rows],s=[25+100*r["count"]/final["test_images"] for r in rows],color=COLORS[method],zorder=3)
        axis.plot([r["confidence"] for r in rows],[r["accuracy"] for r in rows],color=COLORS[method],label=f"ECE = {100*calibration['ece']:.2f}%")
        axis.set(xlim=(0,1),ylim=(0,1),xlabel="Mean confidence in bin",ylabel="Observed accuracy in bin",title=f"{LABELS[method]} · seed 0");axis.legend(frameon=False,fontsize=11)
    save(figure,"fig_reliability",["final_metrics.json"])


def qualitative_figures():
    path=RESULTS/"qualitative_assets.npz"
    if not path.exists():
        for name in ("fig_gradcam_grid","fig_predictions_grid","fig_corruptions"):
            if (FIGURES/f"{name}.png").exists(): MANIFEST[name]={"status":"preserved","sources":["07_evaluate.py"]}
            else: skip(name,"Qualitative evaluation assets unavailable")
        skip("fig_gradcam_slide","Saved held-out Grad-CAM images and maps are required")
        return
    assets=np.load(path);records=read("qualitative.json",[]);predictions=read("prediction_examples.json",[]);classes=read("class_names.json",[])
    figure,axes=plt.subplots(len(records),3,figsize=(10,2.8*len(records)),squeeze=False)
    for i,record in enumerate(records):
        image=assets["gradcam_images"][i]
        axes[i,0].imshow(image);axes[i,0].set_title(short_class(classes[record["true_label"]]),fontsize=12)
        for j,method in enumerate(("baseline","de")):
            axes[i,j+1].imshow(image);axes[i,j+1].imshow(assets["gradcam_maps"][i,j],cmap="jet",alpha=.42)
            row=record[method];axes[i,j+1].set_title(f"{LABELS[method]}: {short_class(classes[row['predicted_index']])}\n{100*row['probability']:.1f}% confidence",fontsize=11)
        for axis in axes[i]:axis.axis("off")
    save(figure,"fig_gradcam_grid",["qualitative_assets.npz","qualitative.json"])
    if len(records)>=2:
        # Fixed first two saved examples: presentation selection does not depend on
        # whether DE was correct, more confident, or visually more convincing.
        figure,axes=plt.subplots(2,3,figsize=(9,7.1),squeeze=False)
        for i,record in enumerate(records[:2]):
            image=assets["gradcam_images"][i]
            truth=textwrap.fill(short_class(classes[record["true_label"]]),22)
            axes[i,0].imshow(image)
            axes[i,0].set_title(f"True class\n{truth}",fontsize=15,pad=9)
            for j,method in enumerate(("baseline","de")):
                axes[i,j+1].imshow(image)
                axes[i,j+1].imshow(assets["gradcam_maps"][i,j],cmap="jet",alpha=.42)
                prediction=record[method]
                label=textwrap.fill(short_class(classes[prediction["predicted_index"]]),22)
                axes[i,j+1].set_title(f"{LABELS[method]}\n{label} · {100*prediction['probability']:.1f}%",fontsize=15,pad=9)
            for axis in axes[i]:axis.axis("off")
        save(figure,"fig_gradcam_slide",["qualitative_assets.npz","qualitative.json","class_names.json"])
        MANIFEST["fig_gradcam_slide"].update({"record_indices":[0,1],"source_paths":[r["source_path"] for r in records[:2]],
            "selection":"First two saved examples in class order, independent of model correctness or confidence"})
    else:skip("fig_gradcam_slide","Two saved held-out examples are required")
    figure,axes=plt.subplots(3,4,figsize=(13,10))
    for index,axis in enumerate(axes.flat):
        axis.axis("off")
        if index>=len(predictions):continue
        record=predictions[index];axis.imshow(assets["prediction_images"][index]);status="Correct" if record["correct"] else "Error"
        axis.set_title(f"{status} · true: {short_class(classes[record['true_label']])}\nPredicted: {short_class(classes[record['prediction']])}",fontsize=10,color=STYLE["de"] if record["correct"] else "#B43232")
    save(figure,"fig_predictions_grid",["qualitative_assets.npz","prediction_examples.json"])
    figure,axes=plt.subplots(5,3,figsize=(9,14))
    for i,name in enumerate(("Blur","Noise","Brightness","JPEG","Low light")):
        for j in range(3):axes[i,j].imshow(assets["corruption_images"][i,j]);axes[i,j].set_title(f"{name} · severity {j+1}",fontsize=10);axes[i,j].axis("off")
    save(figure,"fig_corruptions",["qualitative_assets.npz"])
    assets.close()


def benchmark_surfaces():
    """Evaluate the actual two-dimensional functions; no optimizer run is needed."""
    figure=plt.figure(figsize=(13.5,9.1))
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
        axis.set_title(name,fontsize=18,pad=11)
        axis.set(xlim=(low,high),ylim=(low,high),xlabel="$x_1$",ylabel="$x_2$",zlabel="$f(x_1,x_2)$")
        axis.set_box_aspect((1,1,.75));axis.view_init(elev=27,azim=-54)
        axis.xaxis.set_major_locator(MaxNLocator(3,prune="both"));axis.yaxis.set_major_locator(MaxNLocator(3,prune="both"))
        axis.zaxis.set_major_locator(MaxNLocator(3))
        formatter=ScalarFormatter(useMathText=True);formatter.set_powerlimits((-3,3));axis.zaxis.set_major_formatter(formatter)
        if float(zz.max())>=1_000_000:
            axis.zaxis.set_major_formatter(FuncFormatter(lambda value,position:f"{value/1_000_000:g}M"))
        axis.tick_params(labelsize=12,pad=0)
        for coordinate_axis in (axis.xaxis,axis.yaxis,axis.zaxis):coordinate_axis.label.set_size(15)
    note=figure.add_subplot(2,3,6);note.axis("off")
    note.text(.07,.82,"Analytic landscapes",fontsize=20,fontweight="bold",va="top")
    note.text(.07,.64,"Full two-dimensional domains\nSeparate linear vertical scales\nColor: lower → higher objective",fontsize=15,va="top",linespacing=1.6)
    note.text(.07,.34,"★  Global minimum: f(x) = 0\nM denotes one million",fontsize=14,va="top",linespacing=1.6)
    dimensions=CONFIG["benchmarks"]["dimensions"]
    note.text(.07,.13,f"These views use two dimensions.\nMain study: {dimensions} dimensions.",fontsize=14,va="top",color=STYLE["baseline"],linespacing=1.5)
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
            selected=labels==index;axis.scatter(xy[selected,0],xy[selected,1],s=6,alpha=.65,label=short_class(name),color=plt.cm.tab10(index),marker=["o","s","^","D","v","P","X","<",">","*"][index])
        axis.set(title=f"{LABELS[method]} · seed 0",xlabel="t-SNE component 1",ylabel="t-SNE component 2");axis.set_xticks([]);axis.set_yticks([])
    axes[1].legend(frameon=False,fontsize=10,bbox_to_anchor=(1.03,1),loc="upper left")
    figure.suptitle("Independent t-SNE layouts · distances across panels are not comparable",fontsize=14)
    save(figure,"fig_embeddings",["predictions_baseline_s0.npz","predictions_de_s0.npz"])


def eda_figures():
    summary=read("data_summary.json");classes=read("class_names.json")
    if summary and classes:
        figure,axis=plt.subplots(figsize=(11,6))
        counts=[summary["original_counts"][name] for name in classes]
        bars=axis.barh([short_class(name) for name in classes],counts,color=STYLE["de"])
        axis.invert_yaxis();axis.bar_label(bars,padding=4);axis.set_xlim(0,max(counts)*1.16)
        axis.set(xlabel="Images in original color dataset",title="The tomato classes are strongly imbalanced")
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
            if col==1:axis.set_title(textwrap.fill(short_class(name),18),fontsize=16,pad=12)
    save(figure,"fig_dataset_samples",["eda_assets.npz","class_names.json"])
    figure,axes=plt.subplots(2,5,figsize=(12.5,5.9),squeeze=False)
    for label,axis in enumerate(axes.flat):
        axis.axis("off")
        if label>=len(classes):continue
        axis.imshow(assets["sample_images"][label,0],interpolation="none")
        axis.set_title(textwrap.fill(short_class(classes[label]),18),fontsize=14,pad=8)
    save(figure,"fig_dataset_slide",["eda_assets.npz/sample_images","class_names.json"])
    MANIFEST["fig_dataset_slide"].update({"sample_indices":[[label,0] for label in range(min(10,len(classes)))],
        "selection":"First saved training example per class, in fixed class order"})
    figure,axes=plt.subplots(1,4,figsize=(12,3.5))
    titles=("Original training image","Flips + crop","RandAugment magnitude 7","RandAugment magnitude 15")
    for index,axis in enumerate(axes):axis.imshow(assets["augmentation_images"][index],interpolation="none");axis.set_title(titles[index],fontsize=10);axis.axis("off")
    save(figure,"fig_augmentation_examples",["eda_assets.npz"])
    assets.close()


def box(axis,x,y,w,h,title,body,color):
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=0.035",facecolor=color,edgecolor="none",alpha=.12)
    axis.add_patch(patch);axis.text(x+w/2,y+h*.68,title,ha="center",va="center",fontsize=13,fontweight="bold",color=STYLE["foreground"])
    axis.text(x+w/2,y+h*.28,body,ha="center",va="center",fontsize=11,color=STYLE["foreground"])


def arrow(axis,start,end,color=None):
    axis.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=13,linewidth=1.3,color=color or STYLE["de"]))


def diagrams():
    summary=read("model_summary.json")
    if summary:
        figure,axis=plt.subplots(figsize=(13,5.5));axis.set(xlim=(0,13),ylim=(0,5));axis.axis("off")
        blocks=[r for r in summary if isinstance(r["block"],int)]
        specs=[("Input",f"RGB · {CONFIG['training']['resolution']} × {CONFIG['training']['resolution']}"),
               ("Stem",str(blocks[0]["output_shape"][1:])+"\nConv + BN + SiLU"),
               ("MBConv stages 1–3","Channels "+" / ".join(str(r["output_shape"][1]) for r in blocks[1:4])),
               ("MBConv stages 4–7","Channels "+" / ".join(str(r["output_shape"][1]) for r in blocks[4:8])),
               ("Final convolution",str(blocks[8]["output_shape"][1:])+"\nFeature map"),
               ("Global average pool",f"{blocks[8]['output_shape'][1]} feature values"),
               ("DE-controlled head","Dropout → Dense → SiLU\nDropout → class logits"),
               ("Prediction",f"{summary[-1]['output_shape'][-1]} tomato classes\nSoftmax probabilities")]
        positions=[(.15,3.1),(3.4,3.1),(6.65,3.1),(9.9,3.1),(9.9,.7),(6.65,.7),(3.4,.7),(.15,.7)]
        for i,((title,body),(x,y)) in enumerate(zip(specs,positions)):box(axis,x,y,2.8,1.15,title,body,STYLE["de"] if i in (6,7) else STYLE["pso"])
        for i in range(3):arrow(axis,(positions[i][0]+2.8,3.68),(positions[i+1][0],3.68))
        arrow(axis,(11.3,3.1),(11.3,1.85))
        for i in range(4,7):arrow(axis,(positions[i][0],1.27),(positions[i+1][0]+2.8,1.27))
        axis.text(6.5,2.45,"Baseline: all backbone stages frozen  ·  DE chooses the last k trainable stages and head width",ha="center",fontsize=11)
        axis.text(6.5,.22,"A head width of zero skips the hidden dense and SiLU layers.",ha="center",fontsize=11)
        figure.suptitle("EfficientNet-B0 transfer learning · architecture measured by the model summary",fontsize=16)
        save(figure,"fig_architecture",["model_summary.json","../config.yaml"])
    else:skip("fig_architecture","Model summary unavailable")
    data=read("data_summary.json",{});sizes=data.get("sizes",{})
    figure,axis=plt.subplots(figsize=(14,7));axis.set(xlim=(0,14),ylim=(0,7));axis.axis("off")
    nodes=[(.2,4.9,2.6,1.1,"PlantVillage tomato",f"{data.get('usable_total','')} cleaned images\nColor photographs"),
           (3.2,4.9,2.7,1.1,"Grouped split",f"Train {sizes.get('train','')} · val {sizes.get('val','')}\nHeld-out {sizes.get('test','')}"),
           (6.4,5.45,3.0,1.0,"Preregistered baseline","Fixed transfer-learning recipe"),
           (6.4,3.8,3.0,1.0,"DE + random search","Validation macro-F1 only"),
           (10.1,4.2,3.3,1.1,"Higher-fidelity re-ranking","Select recipes on validation"),
           (10.1,1.7,3.3,1.1,"Final training",f"Same epoch budget · {len(CONFIG['seeds'])} seeds\nFreeze selected protocols"),
           (5.3,1.7,3.5,1.1,"One held-out evaluation","Metrics · CIs · paired statistics\nCorruptions · Grad-CAM"),
           (.2,1.7,3.7,1.1,"Evidence + offline demo","Saved JSON → figures + report\nExact checkpoints → diagnosis")]
    for x,y,w,h,title,body in nodes:box(axis,x,y,w,h,title,body,STYLE["de"])
    arrow(axis,(2.8,5.45),(3.2,5.45));arrow(axis,(5.9,5.65),(6.4,5.95));arrow(axis,(5.9,5.1),(6.4,4.3))
    arrow(axis,(9.4,4.3),(10.1,4.7));arrow(axis,(11.75,4.2),(11.75,2.8))
    # The fixed baseline bypasses recipe selection; it must never be re-ranked.
    axis.plot([9.4,13.75,13.75],[5.95,5.95,2.25],color=STYLE["baseline"],linewidth=1.3)
    arrow(axis,(13.75,2.25),(13.4,2.25),STYLE["baseline"])
    arrow(axis,(10.1,2.25),(8.8,2.25));arrow(axis,(5.3,2.25),(3.9,2.25))
    axis.text(7,0.7,"The test partition never influences candidate fitness, re-ranking, early stopping or model selection.",ha="center",fontsize=11)
    figure.suptitle("From leaf images to a reproducible comparison",fontsize=18)
    save(figure,"fig_pipeline",["data_summary.json","../config.yaml"])


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
