"""Generate the README's measured overview from the same experiment evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bioalgo.common import ROOT,read_json,revision_provenance


def main():
    metrics=read_json('results/final_metrics.json');data=read_json('results/data_summary.json');bench=read_json('results/benchmark_summary.json')
    test=read_json('results/test_report.json',{})
    revision=revision_provenance(ROOT)
    commit=revision['revision'] or 'unavailable'
    if metrics:
        rows=[]
        for key,label in [('baseline','Baseline (pre-registered)'),('rs','Random search'),('de','DE-optimized')]:
            a=metrics['models'][key]['aggregate'];rows.append('| '+label+' | '+' | '.join(f'{a[k]["mean"]*100:.2f} ± {a[k]["std"]*100:.2f}' for k in ['accuracy','macro_precision','macro_recall','macro_f1','roc_auc'])+' |')
        hero='| Model | Accuracy (%) | Precision (%) | Recall (%) | Macro-F1 (%) | ROC-AUC (%) |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
        hero+=f'\n\nMean ± sample standard deviation across all {len(metrics["seeds"])} seeds; {metrics["test_images"]:,} held-out images. Exact seed-zero McNemar p = {metrics["mcnemar"]["p_value"]:.6g} compares baseline with DE; it does not test DE against random search.'
        base=metrics['models']['baseline']['aggregate'];de=metrics['models']['de']['aggregate'];rs=metrics['models']['rs']['aggregate']
        ece={key:metrics['models'][key]['per_seed'][0]['calibration']['ece'] for key in ('baseline','de')}
        robust=metrics['robustness']
        tied=abs(de['accuracy']['mean']-rs['accuracy']['mean'])<1e-12
        comparison=(f"DE and random search tied at **{de['accuracy']['mean']*100:.3f}% mean clean accuracy**" if tied else
                    f"DE reached **{de['accuracy']['mean']*100:.3f}%** and random search **{rs['accuracy']['mean']*100:.3f}%** mean clean accuracy")
        hero+=(f"\n\n{comparison}, versus {base['accuracy']['mean']*100:.3f}% for the fixed baseline; this study does not establish DE superiority over random search. "
               f"The improvement came with tradeoffs: seed-zero DE mean accuracy across the 15 corruption conditions was **{robust['de']['mean_accuracy']*100:.2f}%**, below the baseline's **{robust['baseline']['mean_accuracy']*100:.2f}%**. "
               f"Its seed-zero expected calibration error was also worse: **{ece['de']*100:.2f}% versus {ece['baseline']*100:.2f}%** (lower is better). These controlled-condition results do not establish field reliability."
               '\n\n![Measured comparison](figures/fig_before_after_bars.png)')
    else:hero='Final CNN evaluation is in progress. Accuracy and F1 will appear only after the frozen protocol is evaluated. Optimizer benchmarks below are complete.'
    text=f'''# Plantwise · Intelligence, naturally.

**Plant Disease Detection using Transfer-Learning CNN Optimized with Differential Evolution**

Bio-Inspired Algorithms · **23DS7BSBIO** · **Academic Year 2026–2027** · Team 10

A complete, offline tomato-leaf research studio: ImageNet-pretrained EfficientNet-B0, a hand-written NumPy DE/rand/1/bin optimizer, equal-budget random-search control, leakage-aware data preparation, repeated training, statistical evaluation, Grad-CAM and an animated botanical interface. The fixed task is the ten-class PlantVillage tomato subset.

[![Tests](https://github.com/Suraj-B12/BIO-INSPIRED-ALGO---AAT/actions/workflows/ci.yml/badge.svg)](https://github.com/Suraj-B12/BIO-INSPIRED-ALGO---AAT/actions/workflows/ci.yml)

![Plantwise studio](demo/screenshot.jpg)

## Measured results

{hero}

Optimizer study: **{bench['run_count']:,} trials**, **{bench['total_evaluations']:,} objective evaluations**; five methods, five functions, {bench['settings']['runs']} seeds and {bench['settings']['budget']:,} evaluations per trial. All raw trial records and summaries are saved. Different algorithms lead on different functions; DE is not presented as a universal winner.

![Optimizer convergence](figures/fig_benchmark_convergence.png)

## Start the offline demo

Use Python 3.11 and Git. Node.js 22 or later is needed only for the frontend regression tests. Dependencies and ImageNet initialization are downloaded during setup/training; the prepared demo then runs without an internet connection. Model inference uses the exact baseline/de seed-zero checkpoints. Images never leave the local server.

For a GitHub checkout:

```bash
git clone https://github.com/Suraj-B12/BIO-INSPIRED-ALGO---AAT.git
cd BIO-INSPIRED-ALGO---AAT
```

For the complete delivery ZIP, extract it and open a terminal in its containing folder. Skip the clone commands above and enter the extracted project:

```bash
cd Plantwise
```

Then install and start the demo from that project folder:

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo/serve.py
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe demo/serve.py
```

The Windows commands use the virtual environment directly, so PowerShell activation is unnecessary. For the later reproduction commands, use `.\\.venv\\Scripts\\python.exe` wherever `python` appears.

Open **http://127.0.0.1:8000**. Press **P** for presentation mode. In Evolution Lab, **Space** pauses/resumes the animation; the timeline lets you inspect any generation. Upload JPEG, PNG or WebP images up to 12 MB / 24 megapixels. Corrupt and unsupported inputs return readable errors. The research model does not detect whether an upload is actually a tomato leaf.

For CUDA, first use the matching official PyTorch wheel command from [pytorch.org](https://pytorch.org/get-started/locally/). Device selection is CUDA → Apple MPS → CPU. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before torch imports. Data-loader workers default to zero for Windows/macOS compatibility. Offline inference always uses CPU and supports concurrent users through a serialized explanation step.

![Demo fallback](demo/demo.gif)

## Reproduce without retraining

```bash
python scripts/09_figures.py
python scripts/10_make_tables.py
python scripts/11_build_pptx.py
python scripts/12_export_pdfs.py --offline-citations
```

All charts are regenerated from committed `results/`, including saved qualitative pixels. Raw images and training are unnecessary for figure regeneration. Report compilation requires Tectonic; slide PDF export requires LibreOffice. Make both programs available on PATH. On Windows, add the installed LibreOffice `program` folder (typically `C:\\Program Files\\LibreOffice\\program`) to PATH if its command is not found. `--offline-citations` rechecks current bibliography values against previously verified source records; omit it for fresh network verification.

## Reproduce the full experiment

```bash
python run_all.py
python run_all.py --from 04 --to 07
python run_all.py --smoke
python -m pytest -q
node --test tests/frontend.test.cjs
python scripts/run_checks.py
```

`scripts/run_checks.py` runs the Python tests, frontend tests and offline smoke check together, retaining their actual counts and logs in `results/test_report.json` and `results/checks/`. It refuses a passing summary if any phase fails or the source changes during verification.

The full study is resumable and takes hours, depending on hardware. `--smoke` (also `--quick`) uses synthetic images and never creates final scientific metrics. The system check measures training throughput and selects a proxy tier. `compute.allow_extended_final_training` records the approved longer final-training window. All three final recipes receive the same full-resolution epoch budget and early-stopping rule. On a new machine, every required model is trained or reused only after its recipe/protocol fingerprint matches.

The dataset source is [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset), color variant. Its recorded source commit is `{data['source_commit']}`. Preparation verified {data['original_total']:,} files, removed {data['exact_duplicates_removed']} exact duplicates, retained {data['usable_total']:,} images and grouped perceptually similar leaves before a deterministic 70/15/15 split. Split CSVs are versioned; raw data and caches are excluded.

## Research safeguards

- The baseline recipe is pre-registered before optimization: frozen backbone, a 256-unit head, dropout 0.2, Adam at 1e-3, zero weight decay and label smoothing, and the common random-crop/flip pipeline without RandAugment.
- Search and higher-fidelity re-ranking use validation data only. Only `07_evaluate.py` opens held-out image data for final assessment under a frozen protocol. An evaluation lock prevents silently replacing checkpoints after seeing test outcomes.
- Every evaluation checkpoints state and random generators. Crashes between log writes reconcile from the durable event ledger.
- All seeds are reported. Bootstrap intervals, exact McNemar comparison, calibration and corruption robustness are saved alongside clean accuracy.
- PlantVillage is a controlled-condition dataset. These results do not establish field reliability. Grad-CAM shows contributing regions, not causal proof. Consult an agricultural extension officer for crop-health decisions.

## Evidence and artifacts

| Artifact | Location |
|---|---|
| Full LaTeX report + PDF | `report/` |
| Presentation + PDF, with speaker notes | `slides/` |
| Measured JSON/CSV, model predictions and source records | `results/` |
| Reproducible PNG/vector figures | `figures/` |
| Offline application and example leaves | `demo/` |
| Optimizers, data, model and training helpers | `bioalgo/` |
| Numbered end-to-end pipeline | `scripts/`, `run_all.py` |
| Operator, model, data, API and client regression tests | `tests/` |
| Literature review, presentation script, viva answers, independent audit | `docs/` |

`final_metrics.json` schema: `models.<baseline|de|rs>.aggregate.<metric>.{{mean,std}}`, `per_seed`, recipe `config`, plus `class_names`, `test_images`, `seeds`, `mcnemar`, `relative_error_reduction`, `robustness`, `roc`, and the frozen `protocol`. `report_numbers.json` maps generated numeric macros back to their evidence sources. The test count and actual environment are in `test_report.json` and `system_info.json`.

CI is configured for Ubuntu, macOS and Windows. Local checks are recorded separately; a configured workflow is not a claim that remote CI has run. Recorded source revision: `{commit}` (provenance: `{revision['source']}`). {revision['description']} This revision does not substitute for artifact hashes: the exact delivered files are identified in `results/delivery_manifest.json`.

## Team and academic details

B.M.S. College of Engineering, Bull Temple Road, Basavangudi, Bangalore - 560019 (Autonomous College under VTU). Department of Computer Science & Engineering (Data Science).

| Name | USN |
|---|---|
| Aryanraj S Singh | 1BM23CD008 |
| Richik Chaudhuri | 1BM23CD050 |
| Suraj B | 1BM23CD061 |
| Sushant Deo | 1BM23CD062 |

Guide: **Manoj Kumar S, Assistant Professor**. HOD: **Dr. Shyamala G, Professor & HOD**. **Academic Year 2026–2027**. Course code: **23DS7BSBIO**.

Code is MIT-licensed. Dataset and cited publications retain their own terms; source attribution is preserved. No credentials, raw dataset archives or private files are included.
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8');print('README generated from saved evidence.')

if __name__=='__main__':main()
