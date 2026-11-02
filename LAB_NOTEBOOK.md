# Laboratory notebook

## Mistakes and lessons

- Results must be measured, never invented. Missing experimental outputs remain explicitly unavailable.
- The baseline uses Adam; searched recipes use AdamW. Preserve the pre-registered baseline verbatim.
- Test data is isolated until configuration freeze and the final evaluation.

## 2026-11-01 — Setup

Read all repository instructions and official rubric. Cloned the provided repository into this task's work directory. The repository initially contains instructions and reference files only.

User requested a complete system, extensive testing, and polished Apple-inspired animation. Proposed a light botanical interface with warm white, forest green, glass panels, subtle motion and an offline Python API. Design question offered asynchronously; proceed using that direction unless steered otherwise. No stretch experiments until core verification is complete. Acknowledgement uses Principal and management titles without inventing personal names.

Created isolated Python 3.11 environment. Parallel owners: optimizer/testing, ML/data, DOI-backed research; primary owns integration, UI, figures and documents. Hardware benchmark determines compute feasibility before expensive training.

Downloaded no datasets or model weights into version control. GitHub authentication initially could not be validated under restricted network; recheck before publication. No claim of completion until independently verified.

## Hardware and plan

Measured Apple M4 Pro / 48 GiB with MPS: 30 steps per resolution, batch 32. Full-fine-tune time at224px 0.007124 seconds/image. Tier B proxy searches throughput-based estimate 1.063 hours; all nine final trainings at 15 epochs estimate 3.607 hours, excluding data loading and model construction. The original1.5h final target is exceeded even at TierC because final fidelity is constant. The user's explicit instruction to take time permits a longer current-machine run; preserving the complete experiment instead of reducing seeds or inventing outputs. Chosen TierB. Channels-last was slower at128px; fp16 was not faster, so retain fp32 contiguous.

UI catalog discovery failed (package registry socket unavailable). The required offline custom experience will use dependency-free HTML/CSS/JavaScript with a local FastAPI backend. User's Apple-inspired design takes precedence over unrelated Oracle branding defaults in the presentation skill. Logo extracted220x220 from template and visually checked.

## UI direction confirmed

User approved the light botanical recommendation. Interface implemented dependency-free and offline with five views: Overview, Leaf diagnosis, Evolution Lab, Research results, Project/resources. Desktop and390px responsive layout inspected in the live browser. Evolution is actual NumPy DE through the local API; medical-style diagnosis claims are avoided because this is a controlled-dataset crop research model.

## Verification checkpoint

Combined suite passed2266 tests, zero failures/skips/errors (results/test_suite.xml). Independent verifier identified narrow crash-recovery and protocol-fingerprint gaps; ML owner is fixing with regressions while preserving legitimate ongoing experiment provenance. GitHub accounts vsxye and richikchaudhuri both report READ permission on the requested public repository. Remote publication therefore requires collaborator access; local construction and experiments continue.

## 2026-11-02 — Independent review and reproducibility fixes

All three baseline runs completed under the pre-registered recipe. DE search is in progress; no held-out images have been opened for model selection. The background tool session ended after three completed candidates without a Python failure. The experiment resumed from the durable ledger and was moved to a detached process with explicit log files so it survives tool-session endings. Completed evaluations were preserved.

Independent review found fresh-clone example-directory handling, unpinned upstream dataset checkout, missing quantitative slide panels, and incomplete artifact binding in the delivery gate. Fixed the directory initialization, pinned source revision, explicit CPU installation in CI, quickstart prerequisites, and added readable numerical evidence panels. The final gate now binds test and audit evidence to source/artifact hashes, and the final-result validator recomputes statistics from saved predictions without reopening held-out images.

The common training pipeline is random-resized-crop plus horizontal/vertical flips. Augmentation magnitude zero disables extra RandAugment; it does not remove the common crop. Documentation now makes this distinction explicit without changing the baseline or training.

Verification: `python -m pytest -q --junitxml=results/test_suite.xml` passed 2,289 tests with one MPS-only skip under the restricted CPU test process. `node --test tests/frontend.test.cjs` passed all six controller regressions. GPU-specific verification remains scheduled after training to avoid resource contention. The final slide/report checks remain pending actual held-out results.

## Explanations, accessibility and packaging

The explanation helper now uses the required pytorch-grad-cam package with the final convolution, including a frozen backbone. CPU tests and a tiny MPS smoke verified finite maps, unchanged probabilities/weights and hook cleanup. The desktop and phone UI were visually checked. Numerical labels, distinct marker shapes, line patterns and a sequential attention scale complement color; the style audit records its primary sources and simulation limitations.

The combined CPU suite passed 2,296 cases, with two MPS-specific skips; all nine frontend regression cases passed. Final MPS-inclusive execution and final-document review remain scheduled after the experiments. A packaging command excludes raw data, environments and caches and refuses unverified artifacts. The live server now runs independently of the tool session; temporary idle-sleep prevention is tied only to the experiment process.

The GitHub Actions setup was checked against the official actions/checkout, actions/setup-python and actions/setup-node repositories on 2026-11-02; workflow uses their documented v7 actions with read-only contents permission and explicit CPU PyTorch on Linux/Windows. The workflow remains unexecuted remotely until authorized repository access is available.

## Final-training checkpoint and presentation review

Both Tier B searches completed their 72 candidate evaluations and validation-only top-three reranking. The frozen selected recipes are committed before held-out evaluation. All three baseline runs are complete; the first DE final run is in progress. Intermediate best-checkpoint files are not completion markers: the demo now requires the matching completed training record before offering a model.

The report's interim layout is 39 pages after removing repeated prose while preserving the required settings, figures and limitations. The final page plan reserves space for the missing measured comparisons and concrete image cases. Report figures and slide-specific variants were checked at their actual placement sizes; search curves use all 72 recorded evaluations, and benchmark rerendering preserved the measured raw data. All 26 preview slides have speaker notes and manual fade transitions. The final deck is still withheld until the held-out evaluation completes.

A fresh read-only preflight review inspected training/resume identities, evaluation locking and aggregation, search persistence, data handling, offline demo loading and packaging gates. It reported no actionable findings. The 13 frontend regression cases passed; `uv pip check` found all 85 installed packages compatible. These checks do not replace the independent review of the completed artifacts. The final combined CPU/MPS test run, genuine checkpoint inference, full PDF/deck review and remote CI remain scheduled after training.

The local Python runtime was copied into the workspace to avoid depending on a temporary-directory installation after restart. The active environment will be repointed only after the scientific process exits, then checked before delivery; no running training environment was changed.

## 2026-11-02 — Windows CI text decoding

GitHub Actions run 41373000140 failed only on Windows in `test_no_network_asset_dependency`: Python 3.11 decoded the UTF-8 demo HTML with the system cp1252 codec. Read both HTML and CSS explicitly as UTF-8, matching the application assets and the repository cross-platform rule. Keep the Windows test enabled; do not mask the failure with a process-wide encoding override. The fix is isolated from ongoing local research and interface edits.

Validation before push: 2,316 Python tests passed with two hardware-specific skips; all 13 frontend tests passed; synthetic pipeline smoke passed. Independent review reproduced the CP1252 failure and confirmed the two explicit UTF-8 reads address it. GitHub matrix confirmation follows publication.
