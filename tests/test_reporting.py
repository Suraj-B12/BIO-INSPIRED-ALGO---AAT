"""Schema integration checks use explicitly synthetic data in pytest temp paths.

These fixtures never write experiment results or figures into the repository.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
from bioalgo.evaluation import classification_metrics, roc_data

ROOT=Path(__file__).resolve().parents[1]


def load_script(name):
    spec=importlib.util.spec_from_file_location("reporting_"+name,ROOT/"scripts"/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def synthetic_final():
    rng=np.random.default_rng(42);classes=[f"Tomato___Synthetic_class_{i}" for i in range(10)]
    labels=np.tile(np.arange(10),4);raw=rng.random((40,10));probabilities=raw/raw.sum(1,keepdims=True)
    measured=classification_metrics(labels,probabilities,classes)
    per_seed=[]
    for seed in range(3):
        row={**measured,"seed":seed,"latency_ms":1.0,"training_seconds":2.0,"model_size_mb":.1,
             "parameters":{"total":100,"trainable":50},"bootstrap_95_ci":{"samples":1000,"accuracy":[.01,.2],"macro_f1":[.01,.2]}}
        per_seed.append(row)
    keys=("accuracy","macro_precision","macro_recall","macro_f1","roc_auc","latency_ms","training_seconds")
    model={"per_seed":per_seed,"aggregate":{k:{"mean":per_seed[0][k],"std":0.} for k in keys},"config":{}}
    stress={kind:{str(severity):{"accuracy":measured["accuracy"],"macro_f1":measured["macro_f1"]} for severity in (1,2,3)} for kind in ("blur","noise","brightness","jpeg","low_light")}
    return {"status":"complete","class_names":classes,"test_images":40,
            "models":{m:model for m in ("baseline","de","rs")},
            "roc":{m:roc_data(labels,probabilities) for m in ("baseline","de","rs")},
            "robustness":{m:{"results":stress,"mean_accuracy":measured["accuracy"]} for m in ("baseline","de")},
            "mcnemar":{"p_value":1.},"relative_error_reduction":0.}


def test_final_figure_schema_renders_without_raw_dataset(tmp_path,monkeypatch):
    module=load_script("09_figures.py");module.RESULTS=tmp_path/"results";module.RESULTS.mkdir();module.FIGURES=tmp_path/"figures"
    (module.RESULTS/"final_metrics.json").write_text(json.dumps(synthetic_final()))
    original_save=module.save;adjacent_gaps=[]
    def check_cell_spacing(figure,name,*args,**kwargs):
        if name=="fig_confusion_side_by_side":
            figure.canvas.draw();renderer=figure.canvas.get_renderer()
            for axis in figure.axes[:2]:
                rows={}
                for label in axis.texts:rows.setdefault(label.get_position()[1],[]).append(label)
                for row in rows.values():
                    ordered=sorted(row,key=lambda label:label.get_position()[0])
                    for left,right in zip(ordered,ordered[1:]):
                        if right.get_position()[0]-left.get_position()[0]==1:
                            gap=(right.get_window_extent(renderer).x0-left.get_window_extent(renderer).x1)*72/figure.dpi
                            adjacent_gaps.append(gap)
        return original_save(figure,name,*args,**kwargs)
    monkeypatch.setattr(module,"save",check_cell_spacing)
    module.final_figures()
    assert adjacent_gaps and min(adjacent_gaps)>1.0
    assert len(module.MANIFEST)==7
    assert module.MANIFEST["fig_before_after_slide"]["metric_keys"]==["accuracy","macro_precision","macro_recall","macro_f1","roc_auc"]
    assert module.MANIFEST["fig_before_after_slide"]["methods"]==["baseline","de","rs"]
    for name in module.MANIFEST:
        assert (module.FIGURES/f"{name}.png").stat().st_size>1000
        assert (module.FIGURES/f"{name}.pdf").read_bytes().startswith(b"%PDF")


def test_report_macros_trace_actual_metrics_and_pending_is_explicit(tmp_path,monkeypatch):
    module=load_script("10_make_tables.py");module.RESULTS=tmp_path/"results";module.RESULTS.mkdir();module.GENERATED=tmp_path/"generated"
    final=synthetic_final();(module.RESULTS/"final_metrics.json").write_text(json.dumps(final))
    monkeypatch.setattr(module,"revision_provenance",lambda *args,**kwargs:{"revision":"a"*40,"source":"delivery_manifest","description":"Explicit synthetic fixture provenance."})
    module.main()
    provenance=json.loads((module.RESULTS/"report_numbers.json").read_text())
    assert provenance["results_complete"]
    macro=provenance["macros"]["DEAcc"]
    assert macro["value"]==final["models"]["de"]["aggregate"]["accuracy"]["mean"]
    assert macro["display"]==f"{100*macro['value']:.2f}"
    assert provenance["macros"]["DataOriginal"]["display"]=="Pending"
    assert provenance["macros"]["DataOriginal"]["value"] is None
    assert "Pending" not in (module.GENERATED/"tab_final_metrics.tex").read_text()
    assert "Pending" in (module.GENERATED/"tab_data_summary.tex").read_text()
