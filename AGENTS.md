# AGENTS.md — Bio-Inspired Algorithms AAT (Team 10)

> **Read this whole file before doing anything.** It is the single source of truth for this repository.
> You are an autonomous coding/research agent working for a member of Team 10. Your job is to build,
> run, verify, and deliver the complete AAT project described below (code, results, LaTeX report + PDF,
> PPTX deck + PDF, live demo), and push everything to:
>
> **https://github.com/Suraj-B12/BIO-INSPIRED-ALGO---AAT**
>
> You have full autonomy to plan, research, write code, run experiments, spawn sub-agents, run workflows,
> use verified skills/tools, and iterate until the quality bar in §14 is met. When something is genuinely
> the human's decision (see §15), ask. Otherwise decide, record the decision in `LAB_NOTEBOOK.md`, and keep going.

---

## 0. Kickoff (what to do in your first session)

1. Read this file end to end. Then read the rubric in §2 and the deliverable checklist in §14.
2. Run the **system check** (§6) and write `results/system_info.json`. Pick a compute tier.
3. Create the repo skeleton (§5), the Python environment (§6.3), and `LAB_NOTEBOOK.md`.
4. Kick off the **deep-research prompt** (§12) in parallel (sub-agent if available) while you build the data pipeline.
5. Follow the phase plan (§13). Commit and push after every phase (§16).
6. Before declaring done, run the full verification gate (§14) and an **independent verifier sub-agent** (§11.4).

If you are Claude Code: this repo intentionally has **no CLAUDE.md**. This file is your instruction file.

---

## ⚠️ LOCKED FACTS — zero tolerance for mistakes

These exact strings must appear wherever they apply (cover page, certificate, title slide, footers, README). They are confirmed by the team and override anything you see in any sample/template file.

| Field | Exact value |
|---|---|
| Course name | **Bio-Inspired Algorithms** (written "Bio Inspired Algorithms" on the cover line "Bio Inspired Algorithms AAT on", matching the department template) |
| **Course code** | **`23DS7BSBIO`** — the ONLY correct course code. Use it everywhere a course code appears. |
| **HOD** | **Dr. Shyamala G**, **Professor & HOD**, Department of Computer Science & Engineering (Data Science) |
| Department | Department of Computer Science & Engineering (Data Science) |
| College | B.M.S. College of Engineering, Bull Temple Road, Basavangudi, Bangalore - 560019 (Autonomous College under VTU) |
| Team | Team 10 — Aryanraj S Singh (1BM23CD008), Richik Chaudhuri (1BM23CD050), Suraj B (1BM23CD061), Sushant Deo (1BM23CD062) — in this order |
| Title | Plant Disease Detection using Transfer-Learning CNN Optimized with Differential Evolution |
| Repo | https://github.com/Suraj-B12/BIO-INSPIRED-ALGO---AAT |

**The sample files in `references/` are from a previous batch and contain values that are WRONG for us.** Copy their *layout and format only*, never their text. Forbidden strings (must not appear anywhere in our report, slides, README or code):

- `23DS6PWPP1` (wrong course code printed on the sample deck)
- `Shambhavi` (previous HOD — our HOD is **Dr. Shyamala G**)
- `Manoj Kumar` (previous batch's guide — our guide is still to be confirmed; see §1)
- `1BM22` (previous batch USNs), `Sudeep`, `Harshavardhan`
- `AY 2024-2025` / `AY 2024-25`
- `Stock Price`, `LSTM`, `Indian Banking` (previous project's topic) — except when citing related work deliberately

Enforce this with `scripts/13_check_facts.py` (see §14): it extracts all text from the report PDF, the `.tex` sources, the PPTX (all slides + notes), the slide PDF and README, then **fails** if `23DS7BSBIO` is missing from the cover page, certificate or title slide, if `Dr. Shyamala G` / `Professor & HOD` is missing from the certificate, if any team name/USN is misspelled, or if any forbidden string is present. Run it before every push of report/slides and as the last step of `12_export_pdfs.py`.

---

## 1. Project at a glance

| Item | Value |
|---|---|
| Course | **Bio-Inspired Algorithms — course code `23DS7BSBIO`** (this is the ONLY correct code; never use any other code, even if a sample document shows one) |
| Assessment | AAT (Alternative Assessment Tool), 30 marks; report + PPT are also mandatory for SEE |
| Institution | B.M.S. College of Engineering, Bengaluru-19 (Autonomous Institute, Affiliated to VTU) |
| Department | Department of Computer Science & Engineering (Data Science) |
| Team | Team **10** |
| Title | **Plant Disease Detection using Transfer-Learning CNN Optimized with Differential Evolution** |
| Dataset | **PlantVillage (Tomato subset, 10 classes)** |
| ML/DL model | **CNN — EfficientNet-B0 (ImageNet-pretrained, transfer learning)** |
| Nature-inspired algorithm | **Differential Evolution (DE)** — DE/rand/1/bin, implemented from scratch in NumPy |
| What DE optimizes | Learning rate, weight decay, dropout, dense head units, number of unfrozen backbone stages, label smoothing, augmentation strength (hyperparameters **and** architecture parameters) |
| Metrics | Accuracy, Precision, Recall, F1 (macro + per-class), ROC-AUC (macro OvR), plus McNemar test, bootstrap CIs, robustness accuracy |
| Repo | https://github.com/Suraj-B12/BIO-INSPIRED-ALGO---AAT |

**These fields were already submitted to the department sheet. Do not change the title, dataset, model, or algorithm.**

### Team members (use exactly these names/USNs, in this order)

| Sl. No. | Name | USN |
|---|---|---|
| 1 | Aryanraj S Singh | 1BM23CD008 |
| 2 | Richik Chaudhuri | 1BM23CD050 |
| 3 | Suraj B | 1BM23CD061 |
| 4 | Sushant Deo | 1BM23CD062 |

### Unknowns — ask the human once, early, then fill in

- **Guide name + designation** (for cover page, certificate, title slide). Placeholder: `\GuideName{<<GUIDE NAME>>}`.
- ~~HOD name~~ — **resolved: Dr. Shyamala G, Professor & HOD** (see LOCKED FACTS). Do not ask again; do not use any other name.
- **Academic year** (expected: AY 2026-27, 7th semester — confirm).
- **BMSCE logo PNG** (needed for cover page and slides). Extract it from the sample report in `references/` (`pdfimages -png` or PyMuPDF `page.get_images()`), pick the highest-resolution copy, save to `assets/bmsce_logo.png`, and check it visually. If extraction quality is poor, ask the human for a clean PNG. Do not download random logos from the web.
- Whether the human wants the optional stretch experiments (§9.6).

Keep placeholders as clearly greppable `<<...>>` tokens. The final gate (§14) fails if any `<<` remains.

---

## 2. Grading rubric (verbatim structure, 30 marks) and where we earn each mark

| # | Component | Criterion | Marks | Our evidence (file / figure / section) |
|---|---|---|---|---|
| 1 | Dataset & Problem (4) | Relevant dataset & problem clearly defined | 2 | Report Ch. Introduction + Methodology; slide "Problem Statement"; `fig_dataset_samples.png`, `fig_class_distribution.png` |
| | | Data preprocessing (cleaning, splitting, transformations) | 2 | `scripts/01_prepare_data.py`; duplicate removal log; stratified split CSVs; augmentation description; `fig_augmentation_examples.png` |
| 2 | ML/DL Model (6) | Suitable model selected | 2 | EfficientNet-B0 justification (accuracy/params trade-off, mobile deployability) |
| | | Correct architecture/setup & justification | 2 | Architecture diagram `fig_architecture.png`; table of layers/params; unfreezing scheme |
| | | Baseline results without optimization | 2 | Pre-registered baseline config (§8.3); `results/baseline_*.json`; baseline learning curves |
| 3 | Nature-Inspired Algorithm (10) | Correct choice of algorithm | 2 | "Why DE" section: continuous/mixed space, few control params, robust to noisy fitness |
| | | Operators implemented (mutation, crossover, selection…) | **4** | `bioalgo/de.py` hand-written mutation, binomial crossover, greedy selection, bound handling; unit tests; pseudocode in report |
| | | Optimization applied to model params (hyper/arch) | 2 | 7-gene chromosome incl. architecture genes (head units, unfreeze depth) |
| | | Improvement over baseline | 2 | Final test table, mean ± std over 3 seeds, McNemar p-value, robustness gains |
| 4 | Results & Visualization (6) | Clear baseline vs optimized comparison | 2 | Hero comparison table + `fig_before_after_bars.png` |
| | | Proper metrics | 2 | Acc, P, R, macro-F1, per-class F1, ROC-AUC, CIs |
| | | Convergence/learning curves, comparisons | 2 | `fig_de_convergence.png`, `fig_learning_curves.png`, `fig_confusion_side_by_side.png`, `fig_roc.png`, benchmark convergence plots |
| 5 | Report/Presentation/Code (4) | Well-organized report | 2 | LaTeX report with all required chapters (§10) |
| | | Clean, commented, modular, reproducible code | 2 | Flat numbered scripts, tests, CI on Win/macOS/Linux, pinned env, seeds, one-command run |

Aim to over-deliver on rows 3 and 4 — that is 16 of 30 marks.

---

## 3. The story (keep every artifact consistent with this narrative)

1. **Problem.** Tomato is a major crop in India; leaf diseases cause large yield losses; early visual diagnosis is expensive and needs experts. CNNs can classify leaf images, but their accuracy depends heavily on hyperparameters and on *how much* of a pretrained network you fine-tune — choices that are usually made by trial and error.
2. **Idea.** Treat the training recipe itself (learning rate, regularization, head size, fine-tuning depth, augmentation) as a black-box optimization problem and solve it with **Differential Evolution**, a population-based evolutionary algorithm that needs no gradients of the validation score.
3. **Before.** A sensible, pre-registered default recipe (frozen backbone, Adam 1e-3, dense-256 head, dropout 0.2).
4. **After.** The DE-discovered recipe, trained under the *same* epochs, resolution and data.
5. **Proof.** Clean test metrics (3 seeds), statistical significance, per-class gains, robustness under field-like corruptions, Grad-CAM showing the model looks at lesions, and DE beating random search at an equal evaluation budget.
6. **Validation of the optimizer itself.** DE vs GA vs PSO vs Random Search on standard benchmark functions (Sphere, Rosenbrock, Rastrigin, Ackley, Griewank).
7. **Demo.** A polished, custom live demo (UI designed with the human, §11.3): diagnose a leaf with baseline vs DE-optimized side by side with Grad-CAM, watch DE evolve live, replay our real search.

---

## 4. Ground rules (non-negotiable)

1. **Never fabricate.** Every number in the report, slides, README and demo must come from a file in `results/` produced by code in this repo. Generate LaTeX tables and numeric macros from JSON (§10.4). No "conceptual" figures, no made-up statistics, no invented runtimes. (A sample report from a previous batch contained unverifiable numbers and broken `[?]` citations — do not imitate that.)
2. **Test set is touched once.** Search and model selection use the validation split only. The test split and the robustness set are evaluated only in the final evaluation script, after the configs are frozen.
3. **Baseline is pre-registered.** The baseline config in §8.3 is fixed *before* any search runs. Never weaken it after seeing results. If you change it for a legitimate bug fix, log why in `LAB_NOTEBOOK.md` and re-run everything downstream.
4. **No cherry-picking.** Report mean ± std over all seeds you ran. Negative or neutral results are reported honestly and discussed.
5. **Every citation has a verified DOI.** Each entry in `report/references.bib` must carry a `doi` field that (a) resolves at `https://doi.org/<doi>` and (b) whose registered metadata (Crossref `https://api.crossref.org/works/<doi>`, or DataCite for arXiv/Zenodo DOIs) matches the entry's title, first author and year. arXiv-only papers use their DataCite DOI `10.48550/arXiv.<id>`. If a source genuinely has no DOI (e.g. some JMLR papers, older books, web pages), replace it with an equivalent DOI-bearing source; keep a non-DOI source only when it is indispensable (official dataset/government pages), and then list it with a URL + access date and flag it in `results/citation_check.json`. This is enforced by `scripts/14_check_citations.py` (§14). No placeholders, no `[?]`, no unverified entries in the final PDF or slides.
6. **Course code is `23DS7BSBIO`** everywhere (cover, certificate, footer, title slide).
7. **Cross-platform.** Code must run on macOS (Apple Silicon, MPS), Windows (CUDA or CPU) and Linux with the same commands (§6.4).
8. **Code style (team preference):** minimal, flat, procedural Python. Numbered top-level scripts that read top-to-bottom; one small shared helper module; no class hierarchies, factories, registries or config frameworks beyond a single YAML. Comment the *why*, especially in the DE operators. Each script prints what it is doing and saves its outputs.
9. **Don't commit data or large binaries.** Dataset, caches and checkpoints > 50 MB stay out of git (§16).
10. **Secrets.** Never commit tokens, `kaggle.json`, or anything from `~/.ssh`.

---

## 5. Repository layout (create this)

```
BIO-INSPIRED-ALGO---AAT/
├── AGENTS.md                  # this file (do not delete)
├── README.md                  # human-facing: overview, hero result, quickstart, figures, demo GIF, repo map
├── LAB_NOTEBOOK.md            # running log: decisions, experiments, failures, lessons learned
├── LICENSE                    # MIT
├── config.yaml                # ALL tunables: paths, seeds, tiers, DE params, search space, baseline
├── requirements.txt           # loose, human-readable deps
├── requirements-lock.txt      # exact pins from the machine that produced final results (pip freeze)
├── run_all.py                 # cross-platform orchestrator: runs scripts 00..09 in order, resumable
├── bioalgo/
│   ├── __init__.py
│   ├── common.py              # seed_everything, pick_device, load_config, save_json, timers, logging
│   ├── de.py                  # Differential Evolution (from scratch, NumPy) — the star of the project
│   ├── ga.py                  # simple real-coded GA (for benchmark comparison)
│   ├── pso.py                 # simple global-best PSO (for benchmark comparison)
│   ├── random_search.py       # equal-budget control
│   ├── benchmarks.py          # Sphere, Rosenbrock, Rastrigin, Ackley, Griewank
│   ├── data.py                # dataset/dataloader helpers, transforms, corruptions
│   └── model.py               # build EfficientNet-B0 + head, freeze/unfreeze, Grad-CAM helper
├── scripts/
│   ├── 00_system_check.py     # hardware/OS/torch report + throughput micro-benchmark → tier
│   ├── 01_prepare_data.py     # download/verify, dedup, stratified split, resize cache
│   ├── 02_eda.py              # dataset figures
│   ├── 03_train_baseline.py   # pre-registered baseline, N seeds
│   ├── 04_de_search.py        # DE hyperparameter search (resumable, logs every evaluation)
│   ├── 05_random_search.py    # equal-budget random search control
│   ├── 06_train_final.py      # retrain DE-best (and RS-best) at full fidelity, N seeds
│   ├── 07_evaluate.py         # test + robustness + stats (only script that touches test)
│   ├── 08_benchmarks.py       # DE vs GA vs PSO vs RS on test functions (+ 2-D animation)
│   ├── 09_figures.py          # every figure for report/slides from results/
│   ├── 10_make_tables.py      # LaTeX tables + \newcommand numeric macros from results/
│   ├── 11_build_pptx.py       # builds slides/BIA_AAT_Team10.pptx from results/ + figures/
│   ├── 12_export_pdfs.py      # compiles LaTeX → PDF, pptx → PDF (LibreOffice), sanity checks
│   ├── 13_check_facts.py      # LOCKED FACTS checker: course code, HOD, names/USNs, forbidden strings
│   └── 14_check_citations.py  # every bib entry has a DOI that resolves and matches (Crossref/DataCite)
├── demo/                      # live demo — stack and UI chosen with the human (§11.3)
│   └── examples/              # 10–20 small example leaf images (one+ per class, from TEST split, < 100 KB each)
├── tests/
│   ├── test_de.py             # operator-level unit tests
│   ├── test_benchmarks.py
│   ├── test_data.py           # split integrity, no leakage, label mapping
│   ├── test_model.py          # freeze/unfreeze counts, forward pass shapes
│   └── test_smoke.py          # 1-batch end-to-end on synthetic images, CPU
├── results/                   # JSON/CSV outputs (committed)
├── figures/                   # PNG (300 dpi) + PDF vector versions (committed)
├── models/                    # final weights if < 50 MB each, else GitHub Release
├── report/
│   ├── main.tex
│   ├── chapters/*.tex
│   ├── generated/             # auto-generated tables + numbers.tex (from 10_make_tables.py)
│   ├── references.bib
│   └── BIA_AAT_Team10_Report.pdf
├── slides/
│   ├── BIA_AAT_Team10.pptx
│   └── BIA_AAT_Team10.pdf
├── docs/
│   ├── research/literature_review.md   # output of the deep-research prompt
│   ├── speaker_script.md               # who says what, timings
│   └── viva_qa.md                       # anticipated viva questions with answers
├── assets/                    # logo, architecture diagram sources
├── references/                # PROVIDED: rubric + department templates (see §5.1) — format only, never copy text
└── .github/workflows/ci.yml   # pytest + smoke run on ubuntu, macos, windows
```

Keep scripts flat and readable; `bioalgo/` holds only shared functions.

### 5.1 Provided reference files (already in the repo)

| File | What it is | How to use it |
|---|---|---|
| `references/BIA_AAT_Rubric_23DS7BSBIO.pdf` | Official 30-mark AAT rubric | Ground truth for §2 |
| `references/TEMPLATE_report_sample_previous_batch.pdf` | A previous batch's BIA AAT report (department format) | Copy **layout only**: cover page (blue college name, red title, logo, rounded page border), certificate page (candidate table, guide/HOD signatures, examiners), acknowledgement style, footer "Department of CSE(Data Science)" + page number. Also note what it did badly and do better: broken `[?]` citations, "conceptual" figures, unverifiable numbers, empty conclusion |
| `references/TEMPLATE_ppt_sample_previous_batch.pdf` | The same batch's slide deck (department format) | Copy **layout only**: title slide arrangement, logo + "B.M.S. College of Engineering" top-left, navy titles, footer "Department of Computer Science Engineering(Data Science)" + slide number. It prints the wrong course code `23DS6PWPP1` — **never copy it**; ours is `23DS7BSBIO` |
| `references/team10_sheet_entry.png` | Our row in the department topic sheet | Confirms team number, member names/USNs, dataset |

Our deliverables must look like they belong to the same department series, but be visibly more polished (consistent design chosen with the human, real figures, DOI-verified references).

---

## 6. System analysis & environment (do this first)

### 6.1 Inspect the machine

Write `scripts/00_system_check.py` that collects the following and saves `results/system_info.json`, then prints a one-paragraph summary. Run the shell commands too and paste the summary into `LAB_NOTEBOOK.md`.

macOS (teammate's MacBook):
```bash
sw_vers; uname -m
sysctl -n machdep.cpu.brand_string; sysctl -n hw.ncpu; sysctl -n hw.memsize
system_profiler SPHardwareDataType SPDisplaysDataType | sed -n '1,60p'
df -h .
python3 --version
```
Windows (PowerShell):
```powershell
Get-CimInstance Win32_Processor | Select Name,NumberOfCores
Get-CimInstance Win32_ComputerSystem | Select TotalPhysicalMemory
nvidia-smi
Get-PSDrive C
python --version
```
In Python: `platform.platform()`, `torch.__version__`, `torch.cuda.is_available()`, `torch.backends.mps.is_available()`, GPU name/VRAM if CUDA, free disk.

### 6.2 Throughput micro-benchmark → compute tier

In `00_system_check.py`, time 30 training steps of EfficientNet-B0 (batch 32) at 128 px and 160 px, and 30 at 224 px, on the picked device (fp32; then try mixed precision and keep it only if it's stable and faster). Compute `sec_per_image_train`. From that, estimate seconds per DE evaluation for each tier and pick the largest tier whose **total search (DE + random search) ≤ ~3 hours** and final training ≤ ~1.5 hours.

| Tier | Typical hardware | Proxy res | Proxy subset of train | Proxy epochs | DE NP × G | RS budget |
|---|---|---|---|---|---|---|
| A | CUDA ≥ 6 GB (e.g. RTX 4050), or Apple M-Pro/Max | 160 | 30% | 3 | 10 × 10 (+10 init = 110 evals) | 110 |
| B | Apple M1/M2/M3/M4 base, 8–24 GB | 128 | 20% | 3 | 8 × 8 (= 72 evals) | 72 |
| C | CPU only / very slow | 112 | 10% | 2 | 6 × 6 (= 42 evals) | 42 |

Write the chosen tier into `results/system_info.json` and `config.yaml` (`tier:`). The DE search must be **resumable** (§8.5) so a laptop can sleep or close the lid without losing work. If even Tier C is too slow, stop and tell the human: Suraj has a Windows laptop with an RTX 4050 that can run `04_de_search.py` with the same commands (this also proves Windows reproducibility).

Device selection (in `common.pick_device`): `cuda` → `mps` → `cpu`. On MPS set `PYTORCH_ENABLE_MPS_FALLBACK=1` (set it in `common.py` *before* importing torch, and document it). Use `pin_memory=False` on MPS/CPU. Try `channels_last` and keep it if faster.

### 6.3 Python environment

- Python **3.11** (best wheel coverage on all three OSes). Use `venv` (built-in) — `uv` is fine if present, but commands in README must work with plain `python -m venv`.
- Core deps (check pytorch.org for the current install command per OS before installing; Windows+CUDA needs the CUDA wheel index URL):
  `torch torchvision numpy scipy pandas scikit-learn statsmodels matplotlib seaborn plotly pillow imagehash pyyaml tqdm grad-cam python-pptx pytest requests`
  Optional: `timm` (only if torchvision weights are unavailable), `kaleido` (static plotly export), `umap-learn`, `onnx`/`onnxruntime` (if the demo runs the model in-browser or via ONNX). Demo-specific dependencies follow the stack the human picks (§11.3); keep them in a separate `demo/requirements.txt` or `package.json`.
- After the final results are produced: `pip freeze > requirements-lock.txt` and record OS/torch/device in `results/system_info.json`.

### 6.4 Cross-platform rules (Windows must work)

- Use `pathlib.Path` everywhere; no hard-coded `/` paths, no `~` expansion assumptions, no symlinks.
- Every script has `if __name__ == "__main__": main()` (required for DataLoader workers on Windows/macOS spawn).
- `num_workers`: default 0 on Windows and macOS (spawn overhead often makes >0 slower for cached small images); make it a config value and benchmark 0 vs 2 vs 4.
- Provide **one entry point** that works everywhere: `python run_all.py` (plus `python run_all.py --from 04 --to 07`, `--quick` for a 5-minute smoke version). Do not rely on bash-only scripts; a convenience `Makefile` is optional.
- Open text files with `encoding="utf-8"`. Class folder names may contain spaces — quote paths and test that.
- LaTeX build: prefer **Tectonic** (single binary, macOS/Windows/Linux, auto-fetches packages); fallback `latexmk -pdf` with MacTeX/MiKTeX/TeX Live.
- PPTX → PDF: LibreOffice headless (`soffice --headless --convert-to pdf`); on macOS the binary is usually `/Applications/LibreOffice.app/Contents/MacOS/soffice`. If unavailable, ask the human to export from PowerPoint/Keynote.
- CI (`.github/workflows/ci.yml`): matrix `ubuntu-latest, macos-latest, windows-latest`, Python 3.11, CPU torch, run `pytest -q` and `python run_all.py --smoke` (synthetic data, tiny model run). A green Windows CI badge in the README is the proof of Windows reproducibility.

---

## 7. Data pipeline (`01_prepare_data.py`, `02_eda.py`)

### 7.1 Source (try in this order; verify whichever you use)

1. **GitHub mirror, no credentials:** `https://github.com/spMohanty/PlantVillage-Dataset` — sparse-checkout only `raw/color/Tomato___*` folders:
   ```bash
   git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset data/raw/PlantVillage-Dataset
   cd data/raw/PlantVillage-Dataset && git sparse-checkout set raw/color   # then keep only Tomato___* (or set each Tomato folder explicitly)
   ```
   List the folders and use the exact names you find.
2. **Kaggle** (needs the human's `kaggle.json`): `abdallahalidev/plantvillage-dataset` (use the `color` variant).
3. **TFDS / Hugging Face mirrors** — only if 1 and 2 fail; verify class names and counts.

Use the **color** images only (not grayscale/segmented). The 10 tomato classes: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (two-spotted), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, Healthy. Expect roughly 18k images total with strong imbalance (YLCV largest, mosaic virus smallest). Log the actual per-class counts; if the total differs a lot from ~18k, investigate before continuing.

### 7.2 Cleaning & splitting

- Verify every file opens (PIL `verify()`), drop corrupt ones, log count.
- **Duplicates:** compute perceptual hashes (`imagehash.phash`); drop exact duplicates and group near-duplicates (Hamming distance ≤ 4) so a group never spans splits. Log how many. This is a real, reportable preprocessing step (PlantVillage is known to contain near-identical shots).
- **Stratified split 70/15/15** (train/val/test) with `seed=42`, done on groups. Save `data/splits/{train,val,test}.csv` (relative path, label, class_name, phash) and **commit the split CSVs** (small) so every machine uses identical splits.
- Label mapping saved to `results/class_names.json` (fixed alphabetical order).
- **Resize cache:** write resized copies (short side 256, center region kept, JPEG q=95) to `data/cache_256/`, and an optional uint8 `.npy` per split at the proxy resolution for fast search epochs. Benchmark which is faster on this machine and record the choice.

### 7.3 Transforms

- Train: RandomResizedCrop(res, scale=(0.7,1.0)), horizontal + vertical flip, **RandAugment with magnitude from the `aug` gene** (baseline: flips only), ToTensor, ImageNet normalize.
- Val/Test: Resize(res·1.14) + CenterCrop(res), normalize.
- **Corruption robustness set** (`bioalgo/data.py`, deterministic with fixed seeds, applied to test images only in `07_evaluate.py`): Gaussian blur, Gaussian noise, brightness/contrast shift, JPEG compression, low-light (gamma), each at 3 severities. Name it the "field-condition stress test" and explain it simulates phone photos in the field.

### 7.4 EDA figures (`02_eda.py`)

`fig_class_distribution.png` (horizontal bars with counts), `fig_dataset_samples.png` (grid: 3 samples per class, labeled), `fig_augmentation_examples.png` (one leaf, baseline aug vs strong RandAugment), `fig_corruptions.png` (one leaf × 5 corruptions × 3 severities), and `results/data_summary.json` (counts per split per class, duplicates removed, corrupt removed).

---

## 8. Model, baseline and the Differential Evolution search

### 8.1 Model (`bioalgo/model.py`)

- `torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)`.
- Replace classifier with head: `Dropout(p) → Linear(1280, units) → SiLU → Dropout(p) → Linear(units, 10)`; if `units == 0`, head is `Dropout(p) → Linear(1280, 10)`.
- **Unfreeze depth `k` ∈ {0..9}:** torchvision's `efficientnet_b0.features` has 9 children (indices 0–8). `k=0` → backbone fully frozen (only head trains); `k=j` → the last `j` feature blocks are trainable; `k=9` → full fine-tune. BatchNorm in frozen blocks stays in eval mode. Unit-test the trainable-parameter count for several `k`.
- Optimizer AdamW, cosine LR schedule with 1-epoch linear warmup, batch size 32 (fixed), grad clipping 1.0, cross-entropy with label smoothing gene.
- Export a model summary table (block, output shape, params, trainable?) to `results/model_summary.json` and draw `fig_architecture.png` (clean block diagram: input → stem → MBConv stages → head; highlight the trainable region and the DE-controlled parts). Draw it with matplotlib or TikZ, not a screenshot.

### 8.2 Search space — the chromosome (7 genes, encoded in [0,1]^7)

DE works in the normalized unit hypercube; `decode()` maps to real hyperparameters. Integer genes are rounded at decode time (standard practice; keeps DE purely continuous).

| Gene | Hyperparameter | Decoded range | Scale | Type |
|---|---|---|---|---|
| x1 | learning rate | 10^[-4.5, -2.0] | log | hyper |
| x2 | weight decay | 10^[-6, -2] | log | hyper |
| x3 | dropout | [0.0, 0.6] | linear | hyper |
| x4 | head hidden units | {0, 128, 256, 512} | categorical (index) | **architecture** |
| x5 | unfrozen backbone blocks k | {0, …, 9} | integer | **architecture** |
| x6 | label smoothing | [0.0, 0.2] | linear | hyper |
| x7 | RandAugment magnitude | {0, …, 15} (0 = flips only) | integer | hyper |

### 8.3 Pre-registered baseline (fixed now; do not change after search)

EfficientNet-B0 ImageNet weights, **k = 0 (frozen backbone)**, head units 256, dropout 0.2, **Adam lr 1e-3**, weight decay 0, label smoothing 0, flips-only augmentation, batch 32, 224 px, same epoch budget and early-stopping rule as the final DE model (§8.6). This mirrors the most common "default transfer learning" tutorial recipe and matches what the team declared. Store it in `config.yaml` under `baseline:` and copy it verbatim into the report.

### 8.4 Differential Evolution (`bioalgo/de.py`) — implement from scratch

Use NumPy only (no `scipy.optimize.differential_evolution` in the main pipeline — the rubric rewards hand-written operators; you may use SciPy's DE once in a test as a sanity cross-check on benchmark functions).

For each generation g and each target vector **x_i**:

1. **Mutation (DE/rand/1):** pick distinct r1, r2, r3 ≠ i uniformly; **v_i = x_r1 + F · (x_r2 − x_r3)**.
2. **Bound handling:** if a component leaves [0,1], use *bounce-back*: `v_j = x_ij + rand·(bound − x_ij)` (move randomly between the parent and the violated bound). Document why (keeps diversity, avoids piling up at the boundary like clipping does).
3. **Binomial crossover:** pick j_rand; `u_j = v_j if (rand_j < CR or j == j_rand) else x_ij` — guarantees at least one gene from the mutant.
4. **Selection (greedy, maximizing fitness):** `x_i ← u_i if f(u_i) ≥ f(x_i)`.

Defaults (in `config.yaml`): **F = 0.6** with optional per-generation dither F ~ U(0.5, 0.9), **CR = 0.9**, NP and G from the tier. Initialization: **Latin Hypercube** sampling (`scipy.stats.qmc.LatinHypercube`) for even coverage. Optional warm start: include the baseline config as one initial individual — if you do, state it in the report (it guarantees the curve starts at the baseline, which is a legitimate and common choice).

Track per generation: best, mean, worst, std of fitness, **population diversity** (mean pairwise distance in [0,1]^7), success rate (fraction of trials that replaced their parent), and wall-clock time. Save `results/de_history.json`.

Also implement in the same file a tiny **jDE** variant (self-adaptive F and CR, Brest et al. 2006) behind a flag and compare it with classic DE **on benchmark functions only** — a cheap, impressive ablation for the report.

### 8.5 Fitness function (`04_de_search.py`)

`fitness(genes)` = **validation macro-F1** after training a fresh model with the decoded config at proxy fidelity (tier's resolution, subset, epochs). Rules:
- Same fixed training subset (stratified) and same seed for every evaluation → less noise, fair comparison.
- Validation set at the proxy resolution, full val split (not subset).
- Log **every** evaluation to `results/de_evaluations.csv`: generation, individual, all 7 raw genes, decoded hparams, fitness, val accuracy, val loss, train time, timestamp, device.
- **Cache** by rounded decoded config — never re-train an identical config.
- **Resumable:** after each evaluation, atomically write `results/de_state.json` (population, fitnesses, RNG states, generation index). On start, resume if the state file exists (`--fresh` to restart).
- Guard against failures: NaN loss or OOM → fitness = 0, log it, continue.
- **Re-rank at higher fidelity:** after the search, re-train the top-3 distinct configs at full resolution (224) on the full train set for a short budget and pick the final winner by validation macro-F1. Log it to `results/de_rerank.json`. This counters the noise of low-fidelity fitness.

### 8.6 Final training (`03_train_baseline.py`, `06_train_final.py`)

Train **baseline**, **DE-best** and **RS-best** at 224 px on the full train split, up to 15 epochs with early stopping on val macro-F1 (patience 4, restore best), **3 seeds each (0, 1, 2)**. Save per-epoch train/val loss & accuracy & macro-F1 (`results/curves_<name>_s<seed>.json`), best checkpoint, and timing. Save the seed-0 checkpoints of baseline and DE-best for the demo (`models/`).

### 8.7 Random search control (`05_random_search.py`)

Same search space, same fitness function, same number of evaluations as DE, same proxy fidelity. This answers the obvious viva question "would random tuning have done just as well?". Report the best-so-far curve vs number of evaluations for DE and RS on one plot.

---

## 9. Evaluation, statistics, figures and the before/after story

### 9.1 `07_evaluate.py` (the only script that reads the test set)

For each model × seed: accuracy, macro/weighted precision, recall, F1, per-class P/R/F1, macro ROC-AUC (one-vs-rest), confusion matrix, top-3 accuracy, inference latency (ms/image, batch 1, on this device), parameter count, trainable params, model size (MB). Aggregate mean ± std across seeds. Also:
- **Bootstrap 95% CIs** (1000 resamples) for accuracy and macro-F1 of seed-0 models.
- **McNemar's test** (`statsmodels.stats.contingency_tables.mcnemar`) baseline vs DE-best on seed-0 test predictions; report the p-value and the discordant counts.
- **Error-rate reduction**: (err_base − err_DE) / err_base. If accuracies are both very high (e.g. 98–99.5%), this relative number is the honest way to express the gain — use it.
- **Robustness**: accuracy for every corruption × severity, for baseline and DE-best; mean corruption accuracy.
- **Calibration**: expected calibration error (15 bins) + reliability diagram.
Write `results/final_metrics.json` (one clear schema; document it in the README).

### 9.2 Required figures (`09_figures.py`) — 300 dpi PNG + vector PDF, consistent style

Use **one consistent palette and font** across figures, slides and the demo, chosen together with the human (§11.1/§11.3 interview); define it once in `config.yaml` (`style:` block) and have every plotting script read it. Each method (baseline, DE, random search, GA, PSO, jDE) keeps the same colour everywhere. Whatever the palette, it must pass a colour-blind-safety check, and colour must never be the only cue (add markers/labels). Readable at slide size, no chart junk, titles that state the finding ("DE converges by generation 6"), axes labelled with units.

| File | What it shows |
|---|---|
| `fig_de_convergence.png` | Best / mean fitness per generation with ±1 std band of the population; annotate baseline fitness as a dashed line |
| `fig_de_vs_rs.png` | Best-so-far validation macro-F1 vs number of evaluations: DE vs RS (equal budget) |
| `fig_de_diversity.png` | Population diversity and success rate per generation |
| `fig_search_landscape.png` | Scatter of all evaluations in (log lr, unfreeze k) colored by fitness, generation as marker size; best marked with a star |
| `fig_parallel_coords.png` | Parallel coordinates of all evaluated configs colored by fitness (also an interactive Plotly HTML for the demo) |
| `fig_gene_importance.png` | Spearman correlation (or a small random-forest importance) of each gene with fitness |
| `fig_learning_curves.png` | Train/val loss and val macro-F1 vs epoch: baseline vs DE-best (mean ± std over seeds) |
| `fig_before_after_bars.png` | **Hero chart**: Acc, Precision, Recall, macro-F1, ROC-AUC — baseline vs DE-best with error bars and value labels |
| `fig_per_class_f1.png` | Per-class F1, baseline vs DE-best, sorted by gain |
| `fig_confusion_side_by_side.png` | Row-normalized confusion matrices, baseline \| DE-best, same color scale, short class names |
| `fig_roc.png` | Macro-average ROC (plus per-class thin lines) for both models |
| `fig_robustness.png` | Accuracy under each corruption (severity on x-axis), baseline vs DE-best |
| `fig_reliability.png` | Reliability diagrams with ECE in the legend |
| `fig_gradcam_grid.png` | Same 6 test leaves: image \| baseline Grad-CAM \| DE Grad-CAM, with predicted label + confidence (use `pytorch-grad-cam`, target the last conv block) |
| `fig_predictions_grid.png` | 12 test images with true/pred labels (green tick / red cross), include 2–3 honest failure cases |
| `fig_embeddings.png` | t-SNE (or UMAP) of penultimate features, baseline vs DE-best, colored by class |
| `fig_benchmark_convergence.png` | Median best-so-far (with IQR band) vs evaluations for DE, jDE, GA, PSO, RS on each test function |
| `fig_benchmark_box.png` | Box plots of final best values over 30 runs per function/algorithm (log scale) |
| `fig_de_rastrigin_2d.gif` (+ 4-panel PNG snapshot) | DE population converging on the 2-D Rastrigin contour — great for the "how DE works" slide and the demo |
| `fig_de_operators_diagram.png` | Didactic diagram of mutation vector construction + binomial crossover (drawn with matplotlib arrows on a 2-D contour) |
| `fig_architecture.png` | Model + pipeline diagram (§8.1) |
| `fig_pipeline.png` | End-to-end system flow: data → preprocessing → baseline training / DE loop (fitness = proxy training) → re-rank → final training → evaluation → demo |

### 9.3 Hero table (report Table + slide)

| Model | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC | Trainable params | Train time |
|---|---|---|---|---|---|---|---|
| Baseline (pre-registered) | mean ± std | … | … | … | … | … | … |
| Random-search best | … | … | … | … | … | … | … |
| **DE-optimized (ours)** | **…** | … | … | **…** | … | … | … |

Plus a second table: best hyperparameters found by DE vs baseline values (gene by gene), and a third: robustness mean accuracy + ECE + McNemar p.

### 9.4 Benchmark-function study (`08_benchmarks.py`, feeds "Experimental Setup and Test Functions")

Functions (D = 10, plus D = 2 for visuals): Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, with standard bounds and known optimum 0. Algorithms: DE/rand/1/bin, jDE, real-coded GA (tournament selection, SBX or blend crossover, Gaussian mutation, elitism), global-best PSO (inertia 0.7298, c1 = c2 = 1.49618), Random Search. **Equal budget: 20,000 function evaluations; 30 independent runs each** with seeds 0–29. Report mean ± std, median, best of final error; **Wilcoxon rank-sum** DE vs each other algorithm with p-values. Runs on CPU in minutes. Unit-test that DE reaches < 1e-6 on 10-D Sphere within budget.

### 9.5 Expectations (set them honestly)

PlantVillage images are taken in controlled conditions, so well-tuned fine-tuned CNNs typically exceed 98–99% on its own test split, while a frozen-backbone default recipe usually sits several points lower. So the main gain usually comes from DE discovering the right unfreezing depth and learning rate. Your job is to **measure** it, not assume it. If the clean-test gap turns out small, emphasize (honestly) error-rate reduction, per-class gains on the hardest classes, robustness under corruptions, calibration, and DE vs RS efficiency. Discuss PlantVillage's lab-condition bias as a limitation.

### 9.6 Stretch experiments (only after the core is done and verified)

In priority order; each must be logged and only included if complete:
1. **Field-image generalization** on PlantDoc (`https://github.com/pratikkayal/PlantDoc-Dataset`) tomato classes — map PlantDoc class names to our 10 classes (Target Spot has no PlantDoc counterpart; verify the mapping by listing folders). Evaluate baseline vs DE-best zero-shot. Expect much lower accuracy — that's a valuable honest finding for "Limitations/Future work".
2. GA and PSO also tuning the CNN at a reduced equal budget (e.g. 40 evals each) → DE vs GA vs PSO vs RS on the real task.
3. Multi-objective variant: fitness = macro-F1 − λ·(latency) to show DE can trade accuracy for mobile speed.
4. Export DE-best to ONNX / TorchScript and report CPU latency (deployment story).

---

## 10. Report (LaTeX → PDF)

### 10.1 Build

`report/main.tex`, `report` (or `book`) class, 12 pt, A4, 1-inch margins, `\linespread{1.15}`, packages: `graphicx, booktabs, siunitx, amsmath, amssymb, algorithm, algpseudocode, hyperref, cleveref, xcolor, tikz, caption, subcaption, float, fancyhdr, tocloft`, BibLaTeX (biber) or BibTeX with an IEEE style. Build with `tectonic report/main.tex` (or `latexmk -pdf`) from `12_export_pdfs.py`. Output: `report/BIA_AAT_Team10_Report.pdf`. Target 30–45 pages including figures; no filler.

Header/footer on every body page (like the department's previous AAT reports): footer left "Department of CSE (Data Science)", right page number; thin rounded page border on cover and certificate pages (TikZ `remember picture, overlay`).

### 10.2 Required front matter (exact order)

1. **Cover Page** — "B.M.S. COLLEGE OF ENGINEERING" (blue, large), "(Autonomous College under VTU)", "Bull Temple Road, Basavangudi, Bangalore - 560019", logo, "Bio Inspired Algorithms AAT on", **title in red**, "in fulfillment of the requirements for the award of degree / BACHELOR OF ENGINEERING / in / Computer Science & Engineering (Data Science)", "by" + four names with USNs, "Under the Guidance of" `<<GUIDE NAME>>`, designation, "Department of CSE (Data Science), B.M.S. College of Engineering", `AY <<2026-27>>`.
2. **Certificate Page** — department header + logo, "CERTIFICATE", paragraph: *This is to certify that the project entitled "<title>" is a bona fide work carried out by the following students for the course Bio Inspired Algorithms (Course Code: **23DS7BSBIO**). It is certified that all corrections and suggestions indicated during the internal assessments have been incorporated in the final report submitted to the departmental library. The project report has been approved as it meets the academic requirements prescribed for the Bachelor of Engineering Degree in respect of project work.* Candidate table (Sl. No., Student Name, USN, Student's Signature — 4 rows), Signature of the Guide (name, designation), Signature of the HOD — **Dr. Shyamala G**, **Professor & HOD** — Examiners block (Name of Examiner / Signature, rows 1–2).
3. **Acknowledgement** — thank the Principal, management (BMSET), **Dr. Shyamala G, Professor & Head of the Department, Computer Science and Engineering (Data Science)**, the guide, and department faculty. Verify the Principal/management names with the human before printing them; do not copy names from the sample report and do not invent names.
4. **Abstract** — ~200–250 words, with the real headline numbers pulled from `numbers.tex` macros.
5. **Table of Contents**, List of Figures, List of Tables.

### 10.3 Required chapters (exact titles and order) and what goes in each

1. **Introduction** — background & motivation (tomato's importance, disease losses — cite verified sources), problem statement, objectives (numbered, measurable), contributions, report organization.
2. **Literature Review** — CNNs for plant disease (Mohanty et al. 2016; Ferentinos 2018; Too et al. 2019; PlantDoc 2020), transfer learning & EfficientNet, hyperparameter optimization (grid, random — Bergstra & Bengio 2012, Bayesian), evolutionary/swarm HPO for CNNs, DE in ML. End with a comparison table of prior work (dataset, model, optimizer, accuracy) and a clear **research gap** paragraph. Only verified citations (§12).
3. **Theoretical Background** — CNN basics (convolution, pooling, activation), MBConv + squeeze-and-excitation + compound scaling (EfficientNet), transfer learning & fine-tuning depth, regularization (dropout, weight decay, label smoothing, augmentation), optimization (AdamW, cosine schedule), evaluation metrics with formulas (accuracy, precision, recall, F1, macro averaging, ROC-AUC, ECE, McNemar statistic).
4. **Nature-Inspired Algorithms: Detailed Study** — what metaheuristics are; classification (evolutionary: GA, DE, ES; swarm: PSO, ACO, ABC, Firefly, GWO; physics-based: SA, GSA; human/other: BBO, TLBO) with a taxonomy figure; concise equations for GA, PSO, ACO, SA, Firefly, BBO; then **DE in depth**: biological/evolutionary intuition, all operator equations, strategy variants (rand/1, best/1, current-to-best/1, rand/2), control parameters F, CR, NP and their effects, bound handling, jDE/SHADE self-adaptation, convergence behavior; **why DE for this problem** (continuous mixed search space after encoding, few control parameters, rotation-invariant mutation via difference vectors, robust with noisy expensive fitness, easy parallelism); pseudocode (algorithmicx).
5. **Methodology and Implementation** — system architecture (`fig_pipeline.png`), dataset & preprocessing (dedup, split, cache, augmentation; data summary table), model architecture (`fig_architecture.png`, params table), baseline definition, chromosome encoding table, fitness function design (proxy fidelity and why), DE configuration table, re-ranking, final training protocol, random-search control, implementation details (tech stack, hardware from `system_info.json`, reproducibility measures: seeds, deterministic flags, pinned env, committed splits, CI).
6. **Experimental Setup and Test Functions** — hardware/software table; all hyperparameters of every experiment; benchmark functions with formulas, bounds, optimum, 2-D surface plots; algorithm settings for DE/jDE/GA/PSO/RS; evaluation protocol (30 runs, equal FEs, Wilcoxon); CNN experimental protocol (splits, seeds, early stopping, metrics, statistical tests, robustness protocol).
7. **Results and Discussion** — benchmark results (tables + convergence + box plots + 2-D animation snapshots); DE search results (convergence, diversity, landscape, parallel coords, best config table, gene importance); final test results (hero table, before/after bars, learning curves, confusion matrices, per-class F1, ROC, CIs, McNemar); robustness and calibration; Grad-CAM interpretation; error analysis (which classes get confused and why, e.g. Early blight vs Target Spot); discussion of what DE learned (e.g. optimal unfreezing depth) and why it helps.
8. **Comparative Performance Analysis** — DE vs RS (equal budget) on the CNN; DE vs jDE vs GA vs PSO vs RS on benchmarks; our results vs published PlantVillage results (with caveats about different splits); compute cost comparison (GPU-hours / evaluations to reach X).
9. **Applications and Case Studies** — farmer-facing mobile diagnosis (offline on-device models), greenhouse/drone monitoring, agricultural extension services, early-warning/outbreak mapping; real-world case studies to research and cite (e.g. PlantVillage Nuru, Plantix — verify claims); a case study from our own results (walk through 2–3 test images with Grad-CAM); DE applications beyond this project (engineering design, power-system dispatch, neural architecture search) — cited.
10. **Conclusion** — what was done, headline numbers, whether objectives were met (one line per objective).
11. **Future Work** — field data (PlantDoc), multi-crop, multi-objective DE (accuracy vs latency), successive-halving/multi-fidelity DE, SHADE/L-SHADE, on-device deployment, active learning with farmers.
12. **References** — IEEE style, only verified entries.
13. **GitHub Repo Link** — a short chapter: repo URL as hyperlink + QR code (generate with `qrcode` into `figures/repo_qr.png`), repo map, one-command reproduction instructions, CI badge status, commit hash of the results.

### 10.4 Numbers come from code

`10_make_tables.py` writes:
- `report/generated/numbers.tex` with macros like `\newcommand{\BaseAcc}{96.42}`, `\newcommand{\DEAcc}{99.31}`, `\newcommand{\DEStd}{0.08}`, `\newcommand{\McNemarP}{...}`, `\newcommand{\BestLR}{...}`, `\newcommand{\NumEvals}{...}`, etc.
- `report/generated/tab_*.tex` (booktabs) for every table.
Prose references macros (`accuracy rose from \BaseAcc\% to \DEAcc\%`). The PPT builder reads the same JSON. **Zero hand-typed result numbers anywhere.**

---

## 11. Presentation (PPTX → PDF) and live demo

### 11.1 Build

`scripts/11_build_pptx.py` (python-pptx) generates `slides/BIA_AAT_Team10.pptx` from results + figures; `12_export_pdfs.py` converts it to `slides/BIA_AAT_Team10.pdf`. If a `pptx` skill/tool is available to you, read it and use its guidance for layout quality; still keep the generator script in the repo so the deck is reproducible.

**Visual design is the human's call — ask, don't assume.** Before building the deck, ask the human (in the same UI interview as §11.3 if convenient) for: theme/palette, fonts, layout style, how closely to follow the department template in `references/`, animation/transitions, and whether they want to design the master themselves (e.g. hand you a `.potx`/`.pptx` template to fill). Show 1–2 sample slides in their chosen direction for approval before generating the full deck.

Fixed requirements regardless of design: 16:9; BMSCE logo and "B.M.S. College of Engineering" on content slides and a department footer + slide number (department convention seen in the template); required titles in order (§11.2); the course code `23DS7BSBIO` on the title slide; text readable from the back of a classroom; one idea per slide; figures large. Put a full **speaker script in the notes** of every slide. Check every slide for overflow/overlap by rendering the PDF to images and looking at them; fix and re-render.

### 11.2 Required slide titles (exact, in this order). A section may span several slides — keep the exact title and add "(1/3)" etc.

1. **Cover Slide/Main Slide** — BMSCE logo, "Bio Inspired Algorithms", **23DS7BSBIO**, project title, 4 names + USNs, guide name/designation, department, BMSCE Bengaluru, AY. (Imagery and layout per the human's design direction.)
2. **Introduction to Optimization Problems** — what optimization is (objective, decision variables, constraints); continuous / discrete / mixed / multi-objective; black-box & expensive objectives; why hyperparameter tuning is one (tiny diagram: config → train → val score).
3. **What are Nature-Inspired Algorithms?** — definition, inspiration sources, exploration vs exploitation, no gradients needed, population-based.
4. **Classification of Nature-Inspired Algorithms** — taxonomy diagram (evolutionary / swarm / physics-based / human-based) with examples; DE visibly highlighted.
5. **Problem Statement** — tomato disease impact; why automated diagnosis; why tuning matters; precise statement: *maximize validation macro-F1 of an EfficientNet-B0 classifier over a 7-D mixed hyperparameter/architecture space using DE*; dataset card (10 classes, counts, split) with sample image grid.
6. **Implementation and Results** (expect ~5–7 slides):
   - System pipeline (`fig_pipeline.png`)
   - Model architecture + baseline (pre-registered)
   - DE: chromosome table + operators diagram (`fig_de_operators_diagram.png`) + config
   - DE search: convergence + DE vs RS
   - **Before vs After hero slide**: two big numbers (baseline → DE) for accuracy and macro-F1, error-rate reduction, hero bar chart
   - Confusion matrices + per-class F1
   - Grad-CAM + robustness
7. **Performance Metrics and Benchmarking** — metric definitions (1 slide, formulas), statistical significance (McNemar, CIs), benchmark functions DE vs jDE vs GA vs PSO vs RS (convergence + table), compute cost.
8. **Advantages and Limitations** — two columns; honest limitations (lab-condition dataset, proxy-fidelity noise, compute cost, single crop, results on one split).
9. **Future Scope** — field data, multi-objective DE for mobile latency, SHADE, on-device app, multi-crop.
10. **References** — 8–12 key references, IEEE style, each **with its verified DOI** (from the checked `references.bib`), plus repo link + QR code.

Add a final "Thank You / Questions" slide with the repo QR code. Optional: a "Live Demo" divider slide before section 7.

### 11.3 Live demo (`demo/`) — the showpiece; the human owns the UI

The live demo is the centrepiece of the presentation, not an afterthought. **The teammate running you is a UI expert, so every UI/visual decision is theirs.** A stock Gradio/Streamlit form is *not* the default. A custom HTML/CSS/JS experience (served by a small local Python backend for inference, or running the model in-browser via ONNX Runtime Web / WebGPU / TF.js), a desktop app, or anything else is fine — whatever the human chooses.

**Before building, interview the human** (one structured message, with options and your recommendation for each, plus "your call"):
- Format & stack: custom web app (framework or vanilla?), in-browser inference vs local Python API, desktop app, other.
- Visual direction: mood/aesthetic, colour palette, typography, light/dark, motion/animation level, references they like.
- Layout & flow: single-page story/scrollytelling vs dashboard vs presenter mode; what the audience sees first.
- Which experiences to include (menu below) and which one is the "wow" moment.
- Presentation constraints: projector resolution, keyboard-driven presenter controls, offline requirement, which laptop it runs on.
Then show a quick prototype/wireframe of their direction early, iterate with their feedback, and build what fits best. Record their decisions in `LAB_NOTEBOOK.md`.

**Functional menu** (content the demo can offer — the human decides which, and how it looks):
1. **Diagnose a leaf** — upload / drag-drop / webcam / example gallery (`demo/examples/`, test-split images). Baseline vs DE-optimized side by side: predicted class, top-3 probabilities, Grad-CAM overlay, latency. Optional short class description from a cited, verified source (no treatment advice beyond "consult a local agricultural extension officer").
2. **Watch DE evolve** — pick a 2-D test function, NP, F, CR; animate the population converging (mutation vectors and crossover visible step by step if the human wants an explanatory mode).
3. **Our search, replayed** — the real `results/de_evaluations.csv` and `de_history.json` animated: convergence, population over generations, landscape, best config vs baseline.
4. **Before / After** — the headline metrics and per-class gains, read live from `results/final_metrics.json`.
5. **Field stress test** — apply a corruption + severity to a leaf and watch both models' predictions change.
6. Anything else the human proposes.

**Non-negotiable engineering requirements** (independent of the UI):
- Runs **fully offline**; starts with one documented command on macOS and Windows (e.g. `python demo/serve.py` or opening `demo/index.html`); works on a CPU-only laptop; loads fast; never crashes on bad input (wrong file type, huge image, no webcam).
- Every number shown comes from `results/` or live model inference — nothing hard-coded.
- Models used are the exact seed-0 checkpoints from `models/` (or an ONNX export of them, verified to give identical top-1 predictions on the test set — log the agreement rate).
- A fallback recording (`demo/demo.gif` or `demo/demo.mp4`, ask the human to screen-record if needed) in case the live demo fails during the talk.
- Document start-up and the demo script in `docs/speaker_script.md`.

### 11.4 Speaker plan & viva prep

`docs/speaker_script.md`: 10–12 minute talk split among the 4 members (suggested: Aryanraj — intro & nature-inspired algorithms; Richik — problem, data, model, baseline; Suraj — DE design & search results; Sushant — results, benchmarking, demo, future scope — the human can reassign). Timings per slide.
`docs/viva_qa.md`: 25+ likely questions with crisp answers, e.g. why DE over GA/PSO; what F and CR do; why macro-F1 as fitness; how you avoided test leakage; why proxy fidelity is valid; what if DE only matched random search; why EfficientNet-B0; what Grad-CAM shows; limitations of PlantVillage; complexity of DE per generation; how bound handling works; how results are reproducible.

---

## 12. Deep-research prompt (run early, in parallel; sub-agent or deep-research skill if available)

Copy this prompt verbatim into a research sub-agent (or your deep-research tool). Output to `docs/research/literature_review.md` + `report/references.bib`.

> **Research task — literature & facts for a BMSCE Bio-Inspired Algorithms (23DS7BSBIO) AAT report: "Plant Disease Detection using Transfer-Learning CNN Optimized with Differential Evolution" (PlantVillage tomato, 10 classes, EfficientNet-B0, DE/rand/1/bin tuning 7 hyperparameters/architecture genes).**
>
> Find, read and summarize authoritative sources (peer-reviewed papers, official dataset pages, FAO/ICAR/government reports). For **every** source record: full citation (IEEE style), its **DOI** (mandatory — look it up on the publisher page or Crossref; arXiv papers use `10.48550/arXiv.<id>`), the specific claim(s) we will cite, and a verification note (you resolved the DOI, the Crossref/DataCite metadata matches title/authors/year, and you opened the paper/abstract and confirmed the claim). Never guess or construct a DOI — copy it from an authoritative record. If a source has no DOI, find an equivalent peer-reviewed source that does; only official dataset/government statistics pages may be kept without a DOI (URL + access date, flagged). Produce BibTeX entries with `doi` fields. Reject anything you could not verify. Do not invent page numbers, figures, or quotes; paraphrase.
>
> Cover:
> 1. Crop-disease impact: global and Indian estimates of yield loss from plant diseases, tomato production importance in India (FAO / ICAR / Ministry of Agriculture data), with years.
> 2. PlantVillage: Hughes & Salathé (2015, arXiv:1511.08060); Mohanty, Hughes & Salathé (2016, Frontiers in Plant Science) — their accuracy and the lab-condition/generalization caveat.
> 3. CNN plant-disease classification surveys and benchmarks: Ferentinos (2018, Computers and Electronics in Agriculture), Too et al. (2019, same journal), Kamilaris & Prenafeta-Boldú (2018) deep learning in agriculture survey, PlantDoc (Singh et al., 2020, CODS-COMAD) and field-domain generalization results. Collect a table of reported tomato/PlantVillage accuracies with model and split.
> 4. EfficientNet (Tan & Le, ICML 2019), MobileNetV2/V3, transfer learning & fine-tuning depth (Yosinski et al., 2014 NeurIPS "How transferable are features").
> 5. Differential Evolution: Storn & Price (1997, J. Global Optimization); Das & Suganthan (2011, IEEE TEVC survey); Das, Mullick & Suganthan (2016, Swarm and Evolutionary Computation) update; jDE (Brest et al., 2006, IEEE TEVC); SHADE (Tanabe & Fukunaga, 2013, CEC) / L-SHADE (2014); bounce-back/boundary handling discussion; parameter guidelines for F, CR, NP.
> 6. Other nature-inspired algorithms for the "classification" chapter: GA (Holland 1975; Goldberg 1989), PSO (Kennedy & Eberhart 1995), ACO (Dorigo et al. 1996), SA (Kirkpatrick et al. 1983, Science), Firefly (Yang 2009), BBO (Simon 2008, IEEE TEVC), GWO (Mirjalili et al. 2014), ABC (Karaboga & Basturk 2007) — plus a recognized taxonomy/survey of nature-inspired metaheuristics. Also note the critique literature on "novel metaphor" algorithms (Sörensen 2015, ITOR) for a balanced view.
> 7. Hyperparameter optimization: Bergstra & Bengio (2012, JMLR random search), Snoek et al. (2012 Bayesian), Hyperband (Li et al., 2018 JMLR), evolutionary/swarm HPO of CNNs (e.g. Real et al. 2019 regularized evolution; papers using DE or PSO to tune CNNs — find 3–5 credible ones, ideally including plant disease).
> 8. Benchmark functions definitions (Sphere, Rosenbrock, Rastrigin, Ackley, Griewank) — a standard reference (e.g. Jamil & Yang 2013 survey of test functions; CEC benchmark reports).
> 9. Evaluation methodology: McNemar test for classifier comparison (Dietterich 1998, Neural Computation), bootstrap CIs, ECE / calibration (Guo et al., 2017 ICML), Grad-CAM (Selvaraju et al., ICCV 2017), image-corruption robustness (Hendrycks & Dietterich, ICLR 2019).
> 10. Real-world applications/case studies: PlantVillage Nuru, Plantix, other deployed diagnosis apps; DE applications in engineering/power systems/NAS. Verify every factual claim about them.
>
> Deliverables: (a) `literature_review.md` with a 2–3 paragraph synthesis per theme, a related-work comparison table, and an explicit research-gap statement; (b) `references.bib` (only verified entries, consistent keys like `storn1997de`); (c) a "claims we can safely make" list with the supporting key for each.

After it finishes, run `scripts/14_check_citations.py` (below) and also manually open at least 5 random DOIs yourself to confirm the claims.

`scripts/14_check_citations.py`: parse `report/references.bib`; for each entry, query `https://api.crossref.org/works/<doi>` (fall back to `https://api.datacite.org/dois/<doi>` for arXiv/DataCite DOIs; send a polite `User-Agent` with a contact email and rate-limit requests); compare normalized title (fuzzy ratio ≥ 0.9), first-author surname and year; write `results/citation_check.json` with status per entry; **exit non-zero** if any entry lacks a DOI (unless explicitly flagged as an allowed official page), fails to resolve, or mismatches. Also check every `\cite{}` key in the `.tex` files exists in the bib and every bib entry is cited.

---

## 13. Phase plan (do in order; commit + push after each)

| Phase | Work | Exit criterion |
|---|---|---|
| P0 Setup | system check, env, skeleton, config, CI, LAB_NOTEBOOK, research sub-agent started | `pytest` (empty/smoke) green locally; CI file present |
| P1 DE core | `de.py`, `ga.py`, `pso.py`, `random_search.py`, `benchmarks.py` + tests; run `08_benchmarks.py` | DE < 1e-6 on 10-D Sphere; tests green; benchmark figures exist |
| P2 Data | `01_prepare_data.py`, `02_eda.py`, data tests | split CSVs committed; no group spans splits; EDA figures |
| P3 Model + baseline | `model.py`, `03_train_baseline.py` (3 seeds) | baseline JSON + curves; sane accuracy |
| P4 Search | `04_de_search.py` (resumable), `05_random_search.py`, re-rank | `de_history.json`, `de_evaluations.csv`, `rs_evaluations.csv`, `de_rerank.json` |
| P5 Final | `06_train_final.py` (DE-best, RS-best; 3 seeds), `07_evaluate.py` | `final_metrics.json` complete |
| P6 Visuals | `09_figures.py`, `10_make_tables.py` | every figure in §9.2 exists and passes visual review |
| P7 Demo | UI interview with the human → prototype → build in `demo/`, examples, fallback recording | human approves the look; demo runs offline on CPU on macOS + Windows; screenshot/GIF in README |
| P8 Report | LaTeX chapters, build PDF | PDF compiles, all chapters, no `<<`, no `??`, no `[?]` |
| P9 Slides | `11_build_pptx.py`, export PDF, speaker notes, speaker script, viva Q&A | rendered slides visually checked |
| P10 Verify & ship | §14 gate, verifier sub-agent, README polish, `requirements-lock.txt`, tag `v1.0` | all checks pass; pushed |

---

## 14. Quality gate — done means ALL of these are true

Code & reproducibility
- [ ] `python run_all.py --smoke` passes on this machine in < 10 min; CI green on ubuntu/macos/windows.
- [ ] `pytest -q` green. Tests cover at minimum: r1,r2,r3 distinct and ≠ i; crossover takes ≥ 1 mutant gene (j_rand) and respects CR statistically; bounce-back keeps all values in [0,1]; greedy selection never decreases best fitness; DE solves 10-D Sphere; same seed ⇒ identical DE trajectory; `decode()` maps corners of [0,1]^7 to valid configs (and integer genes cover all values); trainable-param counts for k ∈ {0, 3, 9}; split CSVs disjoint and stratified (class proportions within ±1%); no phash group across splits; forward pass shape (B,10); end-to-end 1-batch smoke on synthetic images.
- [ ] Fresh clone + README quickstart reproduces figures from committed `results/` without retraining (`python scripts/09_figures.py`), and full pipeline reproduces with `python run_all.py`.
- [ ] Seeds fixed everywhere; `requirements-lock.txt` and `results/system_info.json` committed.

Results integrity
- [ ] Baseline config in report == `config.yaml` == what was trained (assert in `07_evaluate.py`).
- [ ] Test set read only by `07_evaluate.py` (grep check).
- [ ] Every number in report/slides/README traced to `results/*.json` (verifier sub-agent checks this, §14 last item).
- [ ] Mean ± std over 3 seeds for all three final models; McNemar p reported; CIs reported.

Deliverables
- [ ] `report/BIA_AAT_Team10_Report.pdf` + all `.tex` sources; every required chapter in the required order; cover + certificate match §10.2; course code 23DS7BSBIO; no `<<`, `??`, `[?]`, overfull boxes that spill into margins.
- [ ] `slides/BIA_AAT_Team10.pptx` + `.pdf`; all 10 required titles present in order; speaker notes on every slide; no text overflow (checked visually from rendered images).
- [ ] Every figure in §9.2 present (PNG + PDF), consistent palette, readable at slide size.
- [ ] Demo built in the UI direction the human approved; runs offline on CPU on macOS and Windows with one documented command; numbers come from `results/`/live inference; fallback recording present.
- [ ] `python scripts/14_check_citations.py` passes: every reference has a DOI that resolves and matches its metadata; no uncited/unknown keys.
- [ ] `README.md`: title, one-paragraph summary, hero result table + hero figure, demo GIF, quickstart (macOS / Windows / Linux), repo map, results reproduction, team, course code, license, CI badge.
- [ ] `docs/speaker_script.md`, `docs/viva_qa.md`, `docs/research/literature_review.md` complete.

Locked facts
- [ ] `python scripts/13_check_facts.py` passes: `23DS7BSBIO` on cover page, certificate, title slide and in report/slide footers or title blocks where a course code is shown; `Dr. Shyamala G` + `Professor & HOD` on the certificate; all 4 names + USNs exact and in order on cover, certificate and title slide; zero forbidden strings from the LOCKED FACTS list in any deliverable.

Independent verification
- [ ] Spawn a **separate verifier sub-agent that did not write the code or documents**. Give it only the repo. It must: (1) re-run `pytest` and `python scripts/09_figures.py`; (2) extract every number from the report PDF and slides and match it to `results/`; (3) check citations resolve; (4) check the rubric table in §2 item by item and point to evidence; (5) open the PDF pages and slide images and list visual defects. Fix everything it reports, then re-run it until clean. Save its final report to `docs/verification_report.md`.

---

## 15. Autonomy, self-improvement and when to ask

### Work loop (repeat per phase)
**Plan → implement → run → measure → review → reflect → improve.** After each run, write a short `LAB_NOTEBOOK.md` entry: date, what you tried, command, key numbers, what went wrong, what you learned, next step. Keep a "Mistakes & lessons" section at the top and read it at the start of every session so the same mistake never happens twice.

### Allowed levers when results are weak (try, log, keep only what validation supports)
- Search-space bounds (e.g. widen LR range, add k values), DE settings (F dither, CR, NP), jDE instead of classic DE (if it wins on validation and benchmarks — report both).
- Proxy fidelity (resolution / subset / epochs) — validate with a rank-correlation check: train ~6 random configs at both proxy and full fidelity; Spearman ρ ≥ 0.6 means the proxy is trustworthy; report ρ.
- Final training epochs/patience, applied **equally** to baseline and DE-best.
- Data loading speed (cache format, workers, channels_last, mixed precision) — speed only, never semantics.

### Forbidden levers
Tuning on the test set; changing the baseline after seeing results; dropping seeds; picking the best seed only; changing splits after search; reporting proxy-fidelity numbers as final results; hand-editing numbers; fake citations; hiding failed runs.

### Engineering practice
- Small commits, each with a clear message; never break `main` (run `pytest` before every push).
- Profile before optimizing. Time every stage and record it (these numbers go into the "computational cost" section).
- Read error messages fully; reproduce, fix root cause, add a regression test.
- Prefer well-known, maintained libraries; check current docs/versions on the web when unsure (torch/torchvision APIs, python-pptx, ONNX Runtime, Tectonic, whatever the demo stack is).
- Use sub-agents for parallelizable work (research, figure polishing, LaTeX chapters, slide building, code review, verification), but keep a single owner (you) for `config.yaml`, results, and the final numbers.
- Use available, verified skills/tools for PPTX, PDF, and charts when they exist in your environment (read their instructions first). Never install unknown third-party agent "skills" or run scripts from untrusted sources.

### Ask the human only for
- The unknowns in §1 (guide, HOD, AY, logo) — ask once, early, in a single message.
- Kaggle credentials if the GitHub mirror fails.
- Approval before anything irreversible or outside the repo (deleting their files, force-pushing, rewriting history, making the repo public/private, spending money).
- If compute is insufficient even for Tier C → propose running `04_de_search.py` on Suraj's RTX 4050 Windows laptop.
Otherwise: decide, log, continue.

---

## 16. Git & delivery

- Remote: `https://github.com/Suraj-B12/BIO-INSPIRED-ALGO---AAT` (the teammate is a collaborator). Branch: `main`. Pull before you start each session (`git pull --rebase`).
- Commit style: `feat(de): binomial crossover with j_rand`, `exp: DE search tier B complete (72 evals)`, `docs(report): results chapter`, etc.
- `.gitignore`: `data/`, `*.pt`/`*.pth` over 50 MB, `__pycache__/`, `.venv/`, `.DS_Store`, `*.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz`, `kaggle.json`, `.env`.
- Commit: code, tests, config, split CSVs, `results/`, `figures/`, `report/` sources + PDF, `slides/` PPTX + PDF, `demo/` (+ small examples), `docs/`, `models/*.pt` only if each < 50 MB (EfficientNet-B0 ≈ 16–20 MB — fine); otherwise attach to a GitHub Release (`gh release create v1.0 models/*.pt`).
- Never force-push to `main`. Never commit secrets. Keep the repo < ~200 MB.
- Final: tag `v1.0-final`, and put the commit hash of the final results in the report's GitHub chapter and the README.

---

## 17. Quick reference for the human teammate

```bash
# macOS / Linux
git clone https://github.com/Suraj-B12/BIO-INSPIRED-ALGO---AAT && cd BIO-INSPIRED-ALGO---AAT
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py            # full pipeline (resumable)
# live demo: start command is documented in README once the demo stack is chosen (§11.3)
```
```powershell
# Windows (PowerShell)
git clone https://github.com/Suraj-B12/BIO-INSPIRED-ALGO---AAT; cd BIO-INSPIRED-ALGO---AAT
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # for NVIDIA GPU: install the CUDA torch wheel first (see pytorch.org)
python run_all.py
# live demo: see README (§11.3)
```

Kickoff prompt for the agent (paste this into Claude Code / Codex / any coding agent opened in the repo folder):

> Read `AGENTS.md` completely and follow it as your instructions. The LOCKED FACTS section is mandatory: course code is 23DS7BSBIO and HOD is Dr. Shyamala G, Professor & HOD — never copy names, codes or numbers from the sample files in `references/`. Start with §0: run the system check, set up the environment, ask me the §1 unknowns in one message, interview me on UI/design choices for the demo and slides before building them (§11.1, §11.3), start the §12 research in parallel, then execute the §13 phases in order, committing and pushing to `main` after each phase. Do not stop until every box in §14 is checked and the verifier sub-agent's report is clean.
