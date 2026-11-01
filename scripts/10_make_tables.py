"""Generate every report number and LaTeX table from recorded evidence.

Unavailable experiments render as the explicit word Pending. A companion JSON
records each macro's source field for the independent numerical audit.
"""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bioalgo.common import revision_provenance
RESULTS=ROOT/"results"
GENERATED=ROOT/"report"/"generated"
CONFIG=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))
NUMBERS={}
TABLES={}


def read(name,default=None):
    path=RESULTS/name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def escape(value):
    replacements={"\\":r"\textbackslash{}","&":r"\&","%":r"\%","$":r"\$","#":r"\#","_":r"\_","{":r"\{","}":r"\}","~":r"\textasciitilde{}","^":r"\textasciicircum{}"}
    return "".join(replacements.get(c,c) for c in str(value))


def scientific(value):
    return "Pending" if value is None else r"\num{"+f"{float(value):.4g}"+"}"


def macro(name,value,source,formatted=None):
    displayed="Pending" if value is None else (formatted if formatted is not None else escape(value))
    NUMBERS[name]={"value":value,"display":displayed,"source":source,"available":value is not None}


def percentage(name,value,source):
    macro(name,value,source,f"{100*value:.2f}" if value is not None else None)


def table(name,headers,rows,sources,alignment=None,notes=None):
    alignment=alignment or ("l"+"r"*(len(headers)-1))
    lines=[r"\begin{tabular}{"+alignment+"}",r"\toprule"," & ".join(headers)+r" \\",r"\midrule"]
    if rows:
        lines.extend(" & ".join(str(cell) for cell in row)+r" \\" for row in rows)
    else:
        lines.append(r"\multicolumn{"+str(len(headers))+r"}{l}{Pending: required experiment evidence is not available.} \\")
    if notes:
        lines.append(r"\midrule")
        lines.extend(r"\multicolumn{"+str(len(headers))+r"}{@{}l@{}}{\footnotesize "+note+r"} \\" for note in notes)
    lines += [r"\bottomrule",r"\end{tabular}"]
    (GENERATED/f"{name}.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    TABLES[name]={"sources":sources,"rows":len(rows),"complete":bool(rows)}
    if notes:TABLES[name]["notes"]=notes


def aggregate_cell(model,key,percent=True):
    row=model["aggregate"][key];scale=100 if percent else 1
    return r"\shortstack{"+f"{row['mean']*scale:.2f}"+r"\\$\pm$ "+f"{row['std']*scale:.2f}"+"}"


def experiment_protocol(data,benchmark):
    """Keep the report's compact protocol tied to configured or frozen settings."""
    frozen=read("frozen_protocol.json",{})
    training=frozen.get("training",CONFIG["training"])
    seeds=frozen.get("seeds",CONFIG["seeds"])
    tier=CONFIG["tiers"][CONFIG["tier"]]
    settings=benchmark.get("settings",{})
    split_sizes=data.get("sizes",{})
    split_text=" / ".join(f"{split_sizes[split]:,}" if split in split_sizes else "Pending" for split in ("train","val","test"))
    rows=[
        ["Data partition",f"Train / validation / test: {split_text}; grouped split seed {data.get('split_seed',CONFIG['split_seed'])}."],
        ["Proxy search",f"{tier['resolution']} px; {100*tier['fraction']:.0f}\\% of training data; {tier['epochs']} epochs; "
            f"{tier['population']*(tier['generations']+1)} candidate evaluations per method."],
        ["Search seeds",f"Proposal generation: DE {CONFIG['de']['seed']}, random search {CONFIG['de']['seed']+1}; "
            f"shared fitness-training seed {read('de_protocol.json',{}).get('seed',CONFIG['split_seed'])}."],
        ["Validation reranking",f"Top {CONFIG['de']['rerank_top']} distinct candidates per method; {training['resolution']} px; "
            f"full training split; {CONFIG['de']['rerank_epochs']} epochs; training seed {CONFIG['split_seed']}."],
        ["Final training",f"{training['resolution']} px; full training split; at most {training['epochs']} epochs; "
            f"patience {training['patience']}; seeds {', '.join(map(str,seeds))}."],
        ["Shared training controls",f"Batch size {training['batch_size']}; warmup {training['warmup_epochs']} epoch(s); "
            f"gradient clipping norm {training['grad_clip']:g}."],
        ["Optimizers",f"Baseline: {escape(CONFIG['baseline']['optimizer'].title())}; searched recipes: {escape(training['optimizer'].replace('adamw','AdamW'))}."],
        ["Analytic benchmarks",f"{settings.get('dimensions',CONFIG['benchmarks']['dimensions'])} dimensions; "
            f"population {settings.get('population',CONFIG['benchmarks']['population'])} for population methods; "
            f"{settings.get('budget',CONFIG['benchmarks']['evaluations']):,} evaluations per run; "
            f"{settings.get('runs',CONFIG['benchmarks']['runs'])} independent runs per method and function."]
    ]
    sources=["../config.yaml/tiers","../config.yaml/de","../config.yaml/training","../config.yaml/seeds",
        "../config.yaml/baseline/optimizer","data_summary.json/sizes,split_seed","benchmark_summary.json/settings",
        "de_protocol.json/seed (config.yaml/split_seed before search)","../scripts/05_random_search.py/config['de']['seed'] + 1"]
    if frozen:sources.append("frozen_protocol.json/training,seeds")
    table("tab_experiment_protocol",["Stage","Configured / frozen protocol"],rows,sources,
        r"@{}>{\raggedright\arraybackslash}p{.22\textwidth}>{\raggedright\arraybackslash}p{.73\textwidth}@{}")


def benchmark_companions(benchmark):
    """Expose distribution summaries and the saved exploratory statistical tests."""
    source="benchmark_summary.json" if (RESULTS/"benchmark_summary.json").exists() else "benchmarks.json"
    algorithms=("DE","jDE","GA","PSO","Random Search")
    rows=[];fields=[]
    for name,function in benchmark.get("functions",{}).items():
        cells=[]
        for algorithm in algorithms:
            values=function["algorithms"][algorithm]
            cells.append(r"\shortstack{"+scientific(values["median"])+r"\\"+scientific(values["best"])+"}")
            fields.extend({"function":name,"algorithm":algorithm,"statistic":key,"value":values[key],
                "source":f"{source}/functions/{name}/algorithms/{algorithm}/{key}"} for key in ("median","best"))
        rows.append([escape(name)]+cells)
    table("tab_benchmark_median_best",["Function","DE","jDE","GA","PSO",r"\shortstack{Random\\search}"],rows,
        [f"{source}/functions/*/algorithms/*/median,best"],"@{}lrrrrr@{}",
        ["Each cell: median (upper), best (lower); lower objective error is better."])
    TABLES["tab_benchmark_median_best"].update({"cell_order":["median","best"],"fields":fields})
    rows=[];fields=[];test_methods=[]
    for name,function in benchmark.get("functions",{}).items():
        cells=[]
        for algorithm in algorithms[1:]:
            result=function.get("de_vs_others_rank_sum",{}).get(algorithm)
            if not result:
                cells.append("Pending");continue
            cells.append(r"\shortstack{$z=$ "+scientific(result["statistic"])+r"\\$p=$ "+scientific(result["p_value"])+"}")
            fields.extend({"function":name,"comparator":algorithm,"statistic":key,"value":result[key],
                "source":f"{source}/functions/{name}/de_vs_others_rank_sum/{algorithm}/{key}"} for key in ("statistic","p_value"))
            test_methods.append({"function":name,"comparator":algorithm,"alternative":result.get("alternative"),
                "correction":result.get("correction"),"source":f"{source}/functions/{name}/de_vs_others_rank_sum/{algorithm}"})
        rows.append([escape(name)]+cells)
    table("tab_benchmark_statistics",["Function","DE vs jDE","DE vs GA","DE vs PSO",r"\shortstack{DE vs\\random search}"],rows,
        [f"{source}/functions/*/de_vs_others_rank_sum/*/statistic,p_value,alternative,correction","../scripts/08_benchmarks.py/ranksums"],"@{}lrrrr@{}",
        ["Exploratory rank-sum tests; no tie correction despite tied errors.",
         "Two-sided p-values, unadjusted for multiple comparisons; negative z favors DE."])
    TABLES["tab_benchmark_statistics"].update({"test":"Wilcoxon rank-sum","fields":fields,"comparisons":test_methods})


def main():
    GENERATED.mkdir(parents=True,exist_ok=True)
    final=read("final_metrics.json");data=read("data_summary.json",{});system=read("system_info.json",{})
    benchmark=read("benchmark_summary.json") or read("benchmarks.json",{});model_summary=read("model_summary.json",[])
    complete=bool(final and final.get("status")=="complete" and all(m in final.get("models",{}) for m in ("baseline","de","rs")))
    macro("ResultsComplete",int(complete),"final_metrics.json/status and models keys")
    for prefix,method in (("Base","baseline"),("DE","de"),("RS","rs")):
        model=final["models"].get(method) if final else None
        for suffix,key in (("Acc","accuracy"),("Precision","macro_precision"),("Recall","macro_recall"),("Fone","macro_f1"),("AUC","roc_auc")):
            row=model["aggregate"][key] if model else None
            percentage(prefix+suffix,row["mean"] if row else None,f"final_metrics.json/models/{method}/aggregate/{key}/mean")
            percentage(prefix+suffix+"Std",row["std"] if row else None,f"final_metrics.json/models/{method}/aggregate/{key}/std")
        percentage(prefix+"Std",model["aggregate"]["accuracy"]["std"] if model else None,f"final_metrics.json/models/{method}/aggregate/accuracy/std")
        seed0=model["per_seed"][0] if model else None
        for suffix,key in (("AccCI","accuracy"),("FoneCI","macro_f1")):
            interval=seed0.get("bootstrap_95_ci",{}).get(key) if seed0 else None
            macro(prefix+suffix,interval,f"final_metrics.json/models/{method}/per_seed/0/bootstrap_95_ci/{key}",
                  f"[{100*interval[0]:.2f}, {100*interval[1]:.2f}]\\%" if interval else None)
        percentage(prefix+"ECE",seed0["calibration"]["ece"] if seed0 else None,f"final_metrics.json/models/{method}/per_seed/0/calibration/ece")
        robust=final.get("robustness",{}).get(method) if final else None
        percentage(prefix+"RobustAcc",robust["mean_accuracy"] if robust else None,f"final_metrics.json/robustness/{method}/mean_accuracy")
        macro(prefix+"Params",seed0["parameters"]["trainable"] if seed0 else None,f"final_metrics.json/models/{method}/per_seed/0/parameters/trainable")
    base_params=sum(row["trainable"] for row in model_summary) if model_summary else None
    macro("BaselineParams",base_params,"model_summary.json/sum(trainable)",f"{base_params:,}" if base_params is not None else None)
    probability=final["mcnemar"]["p_value"] if final else None
    macro("McNemarP",probability,"final_metrics.json/mcnemar/p_value",scientific(probability))
    percentage("ErrorReduction",final.get("relative_error_reduction") if final else None,"final_metrics.json/relative_error_reduction")
    best=read("de_rerank.json",{}).get("best_config")
    macro("BestLR",best.get("learning_rate") if best else None,"de_rerank.json/best_config/learning_rate",scientific(best.get("learning_rate") if best else None))
    state=read("de_search.json",read("de_state.json",{}))
    macro("NumEvals",state.get("evaluations"),"de_search.json/evaluations (de_state.json while search is active)")
    for name,key in (("TotalN","usable_total"),("DataOriginal","original_total"),("DataDuplicates","exact_duplicates_removed"),("DataCorrupt","corrupt_removed")):
        value=data.get(key);macro(name,value,f"data_summary.json/{key}",f"{value:,}" if value is not None else None)
    for name,key in (("TrainN","train"),("ValN","val"),("TestN","test")):
        value=data.get("sizes",{}).get(key);macro(name,value,f"data_summary.json/sizes/{key}",f"{value:,}" if value is not None else None)
    for name,key in (("BenchmarkRuns","runs"),("BenchmarkBudget","budget")):
        value=benchmark.get("settings",{}).get(key);macro(name,value,f"benchmarks.json/settings/{key}",f"{value:,}" if value is not None else None)
    for name,key in (("TotalBenchmarkRuns","run_count"),("TotalBenchmarkEvaluations","total_evaluations")):
        value=benchmark.get(key);macro(name,value,f"benchmarks.json/{key}",f"{value:,}" if value is not None else None)
    seconds=benchmark.get("wall_seconds");macro("BenchmarkSeconds",seconds,"benchmarks.json/wall_seconds; invocation wall time excludes cached trials",f"{seconds:.2f}" if seconds is not None else None)
    sphere=benchmark.get("functions",{}).get("Sphere",{}).get("algorithms",{}).get("DE",{}).get("median")
    macro("SphereDE",sphere,"benchmarks.json/functions/Sphere/algorithms/DE/median",scientific(sphere))
    for name,key in (("DeviceName","device"),("CPUName","cpu"),("TorchVersion","torch"),("PythonVersion","python")):
        macro(name,system.get(key),f"system_info.json/{key}")
    memory=float(system["memory_bytes"])/1024**3 if system.get("memory_bytes") else None
    macro("MemoryGB",memory,"system_info.json/memory_bytes / 2^30 (GiB)",f"{memory:.0f}" if memory is not None else None)
    macro("ComputeTier",CONFIG["tier"],"config.yaml/tier")
    revision=revision_provenance(ROOT)
    commit=revision["revision"][:12] if revision["revision"] else "Unavailable"
    macro("RepoCommit",commit,f"revision_provenance/{revision['source']}; {revision['description']}")
    macro("RepoRevisionSource",{"project_git_head":"Project Git HEAD","delivery_manifest":"Inherited delivery manifest","unavailable":"Unavailable"}[revision["source"]],"revision_provenance/source")
    for name,key in (("FinalEpochs","epochs"),("BatchSize","batch_size"),("Resolution","resolution")):
        macro(name,CONFIG["training"][key],f"config.yaml/training/{key}")
    macro("SeedCount",len(CONFIG["seeds"]),"config.yaml/seeds length")
    (GENERATED/"numbers.tex").write_text("% Generated by scripts/10_make_tables.py; never edit result numbers by hand.\n"+
        "\n".join("\\newcommand{\\"+name+"}{"+row["display"]+"}" for name,row in NUMBERS.items())+"\n",encoding="utf-8")
    rows=[]
    if final:
        for method,label in (("baseline","Baseline"),("rs","Random search"),("de",r"\textbf{DE-optimized}")):
            model=final["models"][method]
            rows.append([label]+[aggregate_cell(model,k) for k in ("accuracy","macro_precision","macro_recall","macro_f1","roc_auc")])
    table("tab_final_metrics",["Model","Accuracy (\\%)","Precision (\\%)","Recall (\\%)","Macro-F1 (\\%)","ROC-AUC (\\%)"],rows,["final_metrics.json"])
    baseline=CONFIG["baseline"];recipes={"Baseline":baseline,"DE":best,"Random search":read("rs_rerank.json",{}).get("best_config")}
    rows=[]
    for key,label in (("learning_rate","Learning rate"),("weight_decay","Weight decay"),("dropout","Dropout"),("head_units","Hidden units"),("unfreeze_blocks","Unfrozen blocks"),("label_smoothing","Label smoothing"),("augmentation","Augmentation magnitude")):
        values=[]
        for recipe in recipes.values():
            value=recipe.get(key) if recipe else None
            values.append("Pending" if value is None else scientific(value) if key in ("learning_rate","weight_decay") else str(value))
        rows.append([label]+values)
    table("tab_best_config",["Parameter"]+list(recipes),rows,["../config.yaml/baseline","de_rerank.json","rs_rerank.json"],"lrrr")
    rows=[]
    for name,count in data.get("original_counts",{}).items():
        label=name.replace("Tomato___","").replace("Tomato_","").replace("_"," ").replace("Spider mites Two-spotted spider mite","Spider mites")
        rows.append([escape(label),f"{count:,}"]+[f"{data['counts'][split][name]:,}" for split in ("train","val","test")])
    if rows:rows.append([r"\textbf{Total}",f"{data['original_total']:,}"]+[f"{data['sizes'][split]:,}" for split in ("train","val","test")])
    table("tab_data_summary",["Class","Original","Train","Validation","Test"],rows,["data_summary.json"])
    algorithms=("DE","jDE","GA","PSO","Random Search")
    rows=[]
    for name,function in benchmark.get("functions",{}).items():
        rows.append([escape(name)]+[r"\shortstack{"+scientific(function["algorithms"][a]["mean"])+r"\\$\pm$ "+scientific(function["algorithms"][a]["std"])+"}" for a in algorithms])
    table("tab_benchmarks",["Function"]+list(algorithms),rows,["benchmarks.json (mean and sample SD of final objective error)"])
    benchmark_companions(benchmark)
    rows=[]
    if final:
        for method,label in (("baseline","Baseline"),("de","DE-optimized")):
            model=final["models"][method];rows.append([label,f"{100*final['robustness'][method]['mean_accuracy']:.2f}",f"{100*model['per_seed'][0]['calibration']['ece']:.2f}",aggregate_cell(model,"latency_ms",False)])
    table("tab_robustness",["Model","Mean corruption acc. (\\%)","ECE (\\%)","Latency (ms)"],rows,["final_metrics.json"])
    rows=[]
    if final:
        for method,label in (("baseline","Baseline"),("de","DE-optimized"),("rs","Random search")):
            seed0=final["models"][method]["per_seed"][0];ci=seed0["bootstrap_95_ci"]
            rows.append([label]+[f"[{100*ci[k][0]:.2f}, {100*ci[k][1]:.2f}]" for k in ("accuracy","macro_f1")])
    table("tab_confidence_intervals",["Model (seed 0)","Accuracy 95\\% CI (\\%)","Macro-F1 95\\% CI (\\%)"],rows,["final_metrics.json"])
    rows=[[escape(row["block"]),escape(" × ".join(map(str,row["output_shape"]))),f"{row['total']:,}",f"{row['trainable']:,}"] for row in model_summary]
    table("tab_model_summary",["Block","Output shape","Parameters","Trainable (baseline)"],rows,["model_summary.json"],"llrr")
    rows=[["CPU",escape(system.get("cpu","Pending"))],["Memory (GiB)",NUMBERS["MemoryGB"]["display"]],["Device",escape(system.get("device","Pending"))],["Python",escape(system.get("python","Pending"))],["PyTorch",escape(system.get("torch","Pending"))],["Operating system",escape(system.get("platform","Pending"))],["Compute tier",escape(CONFIG["tier"])]]
    table("tab_system",["Item","Measured / configured value"],rows,["system_info.json","../config.yaml"],"ll")
    tier=CONFIG["tiers"][CONFIG["tier"]]
    rows=[["Proxy resolution",str(tier["resolution"])],["Training fraction",f"{100*tier['fraction']:.0f}\\%"],["Proxy epochs",str(tier["epochs"])],["DE population",str(tier["population"])],["DE generations",str(tier["generations"])],["Budget per search",str(tier["population"]*(tier["generations"]+1))],["Scale F",str(CONFIG["de"]["F"])],["Crossover CR",str(CONFIG["de"]["CR"])],["Search seed",str(CONFIG["de"]["seed"])]]
    table("tab_search_budget",["Setting","Value"],rows,["../config.yaml"],"lr")
    experiment_protocol(data,benchmark)
    rows=[]
    if final:
        for index,name in enumerate(final["class_names"]):
            values={method:np.mean([seed["per_class"][index]["f1"] for seed in final["models"][method]["per_seed"]]) for method in ("baseline","de","rs")}
            label=name.replace("Tomato___","").replace("Tomato_","").replace("_"," ").replace("Spider mites Two-spotted spider mite","Spider mites")
            rows.append([escape(label)]+[f"{100*values[m]:.2f}" for m in ("baseline","de","rs")]+[f"{100*(values['de']-values['baseline']):+.2f}"])
    table("tab_per_class",["Class","Baseline F1","DE F1","RS F1","DE gain (pp)"],rows,["final_metrics.json"])
    rows=[]
    if final:
        for method,label in (("baseline","Baseline"),("de","DE-optimized"),("rs","Random search")):
            model=final["models"][method];seed0=model["per_seed"][0]
            rows.append([label,f"{seed0['parameters']['trainable']:,}",f"{seed0['model_size_mb']:.2f}",aggregate_cell(model,"training_seconds",False),aggregate_cell(model,"latency_ms",False)])
    table("tab_computational_cost",["Model","Trainable params","Size (MiB)","Training (s)","Latency (ms)"],rows,["final_metrics.json"])
    for alias,source in (("tab_results","tab_final_metrics"),("tab_hyperparameters","tab_best_config"),("tab_dataset","tab_data_summary")):
        (GENERATED/f"{alias}.tex").write_text((GENERATED/f"{source}.tex").read_text(encoding="utf-8"),encoding="utf-8")
    provenance={"results_complete":complete,"revision_provenance":revision,"macros":NUMBERS,"tables":TABLES}
    (RESULTS/"report_numbers.json").write_text(json.dumps(provenance,indent=2,allow_nan=False),encoding="utf-8")
    print(f"Generated {len(NUMBERS)} source-traceable macros and {len(TABLES)} tables. Final evaluation complete: {complete}")


if __name__=="__main__":main()
