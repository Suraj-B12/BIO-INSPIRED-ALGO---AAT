# Independent verification report

**Status: interim audit; final quality gate not passed.**

Audit snapshot: 1 November 2026, 18:14 UTC. The independent verifier read the complete `AGENTS.md`, reviewed implementation and tests, and did not author the project implementation or documents. Training and document production were still running. No held-out image was opened by this verifier, no experiment was retrained, and no synthetic measurements were substituted for research results.

## Checks independently executed

| Check | Observed result |
|---|---|
| Initial `.venv/bin/python -m pytest -q` | 2,268 passed; one dependency deprecation warning; 21.64 seconds. |
| Repeat after audit corrections | 2,271 passed, one skipped, one warning; 27.92 seconds. The skip is the MPS-specific training-resume test because this invocation reported MPS unavailable. |
| `node --test tests/frontend.test.cjs` | Six controller tests passed, including stale responses after image/corruption/severity changes. |
| `.venv/bin/python run_all.py --smoke` | Passed in 2.2 seconds, with no dataset or pretrained-weight download. This is explicitly a synthetic engineering check, not model performance. |
| `.venv/bin/python scripts/09_figures.py` | Exit 0; ten currently available figure families reproduced; sixteen families correctly marked unavailable pending experiments. This is not a claim that the complete figure gate passes. |
| Independent manifest regrouping | Recomputed radius-four perceptual-hash groups from all 18,146 manifest rows: 18,094 groups; zero groups span splits; all paths and pixel hashes unique. No image decoding was involved. |
| Independent benchmark arithmetic | All 750 recorded trials checked: 30 seeds per algorithm/function, 20,000 evaluations each, 15,000,000 total evaluations. Recomputed final objectives from saved best positions and recomputed mean/sample SD/median/best/worst. No mismatches. All saved best-error trajectories are nonincreasing. |
| DE Sphere requirement | Worst final DE Sphere error across the 30 saved trials is `2.919229217310165e-14`, below the required `1e-6`. |
| Local browser behavior | Opened Overview, Evolution Lab and Research Results through the supported browser tool. The default live 2-D Rastrigin run exposed generation 70/70 and 1,988 evaluations. Results view correctly reported CNN evaluation as pending and displayed the saved benchmark summary. No browser warnings/errors were observed in the captured log. |

Direct localhost HTTP requests from the restricted shell were unavailable; browser inspection and the FastAPI test client covered the interface instead. The browser examination did not establish Windows compatibility or successful inference using both final checkpoints.

## Findings and corrections

1. **Search log crash recovery defect — corrected by the pipeline owner.** The callback originally saved its JSON ledger before replacing the CSV, then skipped already-recorded evaluations on replay. A crash between the writes could therefore leave a missing or stale CSV permanently. The callback now always rebuilds the CSV from the authoritative ledger. Missing/stale-CSV regression coverage was added and passed on independent rerun.
2. **Incomplete training provenance — corrected by the pipeline owner.** Training signatures originally omitted warmup, gradient clipping and worker controls, and final evaluation did not compare the complete expected training protocol and split fingerprint. New identities include these controls; final evaluation checks full protocol/signature consistency before accessing the held-out split. The initially running baseline has an explicit, narrowly scoped compatibility path tied to its immutable original preregistration; the verifier inspected this exception. It must not be generalized to altered experiments.
3. **Unwired DE configuration flags — corrected by the pipeline owner.** CNN search now applies the configured dither setting and explicitly rejects adaptive jDE for the CNN experiment, where the required strategy is classic DE/rand/1/bin. Current false defaults were unaffected.
4. **Stale prediction response after input changes — corrected by the primary owner.** Replacing the leaf or changing corruption/severity now invalidates the request generation. Independently reviewed the real controller and ran all six asynchronous controller regressions successfully.

No remaining substantive defect was identified in the reviewed operator implementation. Mutation excludes the target and uses distinct donors; deferred donor populations, bounce-back repair, forced crossover coordinate, greedy selection, seeded Latin-hypercube initialization and exact optimizer checkpoint restoration are implemented and tested. The GA, PSO and random-search controls use the documented equal evaluation budgets. The benchmark results do not establish DE superiority on every function, and the interface correctly avoids that claim.

## Evaluation integrity review

`train_recipe`, proxy fitness and re-ranking load only training and validation rows. The test-row loader refuses access without an explicit final-evaluation flag. Data preparation necessarily validates and groups the entire source dataset before splitting; this preprocessing is distinct from using held-out outcomes for selection. The independent manifest audit found no recorded near-duplicate group leakage.

The final-evaluation implementation requires the frozen recipes, all nine checkpoints, class mapping, full training budget, split fingerprint and compatible protocol signatures before held-out access. A persisted evaluation lock binds checkpoint hashes and held-out manifest hash. Completed evaluation components are resumable. Metric code implements macro/weighted precision, recall and F1, macro OvR ROC-AUC, top-three accuracy, confusion matrices, calibration bins/ECE, 1,000-resample bootstrap intervals and the exact paired McNemar test. These are implementation checks only until real final predictions exist and are independently recalculated.

The baseline remains Adam with the mandated fixed recipe; searched recipes use AdamW as prescribed in the project instructions. Any scientific discussion of the comparison should acknowledge that optimizer choice is part of the overall recipe difference. Synthetic corruption results must be described as controlled stress tests, not field validation. A three-candidate higher-fidelity re-ranking reduces proxy-selection risk but does not itself prove global proxy/full-fidelity rank correlation.

## Rubric evidence and current gaps

| Rubric criterion | Evidence reviewed | Current assessment |
|---|---|---|
| Relevant dataset/problem (2) | `results/data_summary.json`, class mapping, introduction/methodology sources | Dataset evidence present: 18,160 original images, 14 exact duplicates removed, 18,146 retained, ten tomato classes. Final document wording still to audit. |
| Cleaning/splitting/transforms (2) | `01_prepare_data.py`, `bioalgo/data.py`, split CSVs, cleaning log, data tests | Implementation and manifest integrity pass. Split sizes: 12,701 train / 2,723 validation / 2,722 test. Class-wise split proportions pass the specified tolerance. |
| Suitable model (2) | `bioalgo/model.py`, model summary, architecture figure | EfficientNet-B0 ImageNet transfer model is implemented. Final report justification pending document audit. |
| Architecture/setup (2) | Frozen BatchNorm handling, head construction, unfreeze tests, `model_summary.json` | Implementation passes inspected tests, including forward shape and trainable parameter counts. |
| Unoptimized baseline results (2) | Immutable preregistration, baseline training artifacts | Baseline training still incomplete at snapshot; three-seed held-out baseline table unavailable. |
| Nature-inspired algorithm choice (2) | Classic DE implementation; algorithm/methodology sources | Suitable implementation present; final explanation/pseudocode will be checked in document audit. |
| Operators implemented (4) | `bioalgo/de.py`, operator tests, saved operator example, benchmark outcomes | Pass for the reviewed code and benchmark evidence. |
| Applied hyper/architecture optimization (2) | Seven-gene decode, search fitness, re-ranking scripts | Correctly connected in code; actual CNN search completion pending. |
| Improvement over baseline (2) | Required final metrics, statistics and robustness | **Unmet:** no completed final comparison. No improvement claim is verified. |
| Clear baseline/optimized comparison (2) | Results-to-figures/table generators | Schema tests pass; real comparison figures/tables pending experiments. |
| Proper metrics (2) | `bioalgo/evaluation.py`, `07_evaluate.py`, evaluation tests | Numerical helpers reviewed; final measured metrics and seed aggregation not yet available. |
| Convergence/learning/comparison figures (2) | Benchmark figures and generated manifest | Benchmark figures reproduce. CNN search/final figures incomplete. |
| Organized report (2) | LaTeX chapter sources | Final PDF/page order, all numerical tokens, citations and visual defects are not yet audited. |
| Clean/reproducible code (2) | Modular source, numbered scripts, tests, smoke, CI workflow | Local tests pass with disclosed skip/warning. Fresh-clone proof, dependency lock and actual three-platform CI success remain unverified. |

## Unmet final gates

- Complete baseline, DE and random-search training over seeds 0, 1 and 2; freeze configurations and produce all final evaluation artifacts.
- Independently recalculate final metrics from saved predictions, aggregate every seed, compare all exported numerical claims with source results, and inspect checkpoint provenance.
- Re-run figure generation after final evaluation and verify every required PNG/PDF family, including Grad-CAM, honest prediction failures, ROC, robustness and embeddings.
- Audit final report PDF and slide deck/PDF: every required chapter/title in order, identity facts, every result number traced to evidence, no unresolved citations/placeholders, and visual inspection of every page/slide.
- Independently verify citation resolution and metadata on the final bibliography; current citation evidence was inspected but its final gate has not been independently rerun in this pass.
- Verify final seed-zero baseline/DE inference and bad-input behavior in the offline demo, review fallback recording, and document actual supported-platform evidence.
- Finish README, speaker script and exact dependency lock; verify fresh-clone reproduction from saved evidence.
- Obtain successful publication and CI evidence. The primary owner reports the configured GitHub accounts have read-only repository access; **no push, tag or green remote CI is claimed by this verifier**.

This report deliberately withholds a clean final sign-off. It will be updated when the experiment and document artifacts are complete.
