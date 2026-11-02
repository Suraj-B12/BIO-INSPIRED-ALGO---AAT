# Report review record

Reviewed 2026-11-01. This is a pre-final experiment build, not a finished deliverable.

- All 36 rendered draft pages inspected as six contact sheets; cover, certificate, mathematical layout, results plots, references, and repository page checked.
- Correct transparent college logo; locked title, names, USNs, guide, HOD, course code, and academic year. Certificate signature cells remain blank.
- Exact numbered chapters 1–13, including References and GitHub Repo Link.
- Required 12pt A4 report, 1-inch margins, 1.15 line spread.
- No overfull boxes, undefined references, or missing citation markers in the inspected compile. Third-party algorithm.sty emits a UTF-8 warning in its package source; no replacement characters appear in document text.
- Bibliography uses 40 verified DOI sources and one documented government exception; every key is cited in relevant prose. Fresh online full citation gate passed, followed by cached recheck after editorial changes.
- Missing experiment figures and metrics are explicitly pending. No predicted numerical outcome is presented as measured.
- Final work: regenerate results/tables/plots; add actual class-confusion and 2–3-image case observations; check sign-aware comparisons; rebuild and inspect every final page; ensure 30–45 pages.
- Figure owner is improving benchmark label readability and adding five benchmark surfaces plus a compact experimental-protocol table.

## Integration update — 2026-11-02

- Updated rehearsal script and machine-readable speaker notes to all 25 exact deck titles; validated total 660 seconds and four-member assignments. Grad-CAM appears automatically.
- Added generated median/best benchmark and rank-sum tables to Chapter 8, with references from Chapter 6. Qualitative median rankings were recomputed directly from benchmark_summary.json.
- Explicit caveats: exploratory, two-sided, unadjusted for multiple comparisons, and not tie-corrected despite some ties. No confirmatory significance claim.
- Combined hardware and search-setting tables; removed an algorithm-chapter orphan page through concise prose. Latest interim build: 38 pages, required body formatting retained.
- Inspected updated benchmark tables, protocol table, landscape page, and compact algorithm/environment pages. Paragraph boundaries prevent floats from splitting sentences.
- All 41 bibliography keys remain cited; no unknown keys. No overfull, underfull, or undefined-reference warnings in the integration compile.
- Final CNN results, actual class/error cases, final figure pairing, and final full-document inspection remain deferred until the measured artifacts exist.

## Deck and curve clarification — 2026-11-02

- Speaker script and JSON now match all 26 manifest titles and total 670 seconds. Grad-CAM and corruption robustness have separate 20-second slots; Live Demo is slide 16. Four-presenter ownership is preserved.
- Learning-curve caption and discussion state that each method uses epochs shared by all seed runs, with no stopped-run extrapolation. Recipe-specific label smoothing makes raw loss objectives different; cross-recipe comparisons use macro-F1/accuracy.
- Throughput description now identifies synthetic-tensor measurements and excludes loading/model-construction overhead; estimates are not wall-clock upper bounds. Matching viva answers added.
- Updated report compiles to 38 pages. Inspected the figure list, environment page, and learning-curve caption at rendered size. No overflow, undefined references, or missing-citation markers. Final CNN analysis remains pending.


## Completed-search pagination pass — 2026-11-02

- Refreshed generated tables/macros from completed search and reranking artifacts. Final evaluation remains absent and ResultsComplete remains zero; no final CNN result is claimed.
- Rebuilt the learning chart only from the three completed baseline seed histories using the real revised vertical layout. No held-out images, fabricated measurements, or substitute plots were used. Its source manifest identifies this interim scope; final script 09 regeneration adds all completed methods.
- Current ignored report/main.pdf has 39 pages, down from the initial refreshed 41. Setup now occupies three pages and comparison two, through concise prose and moving the timing definition into the neural protocol. Required 12pt/1in/1.15 formatting, figures, all algorithm settings, pHash identity caveat, and single-CNN-search limitation remain.
- Inspected the refreshed List of Figures, benchmark pair, 2×2 Rastrigin snapshots, DE convergence/diversity pair, landscape/parallel pair, selected-recipe/gene-association page, vertical learning curve, and revised setup/comparison pages. New actual chart labels are readable; no clipping or overlap found.
- Current compile has no overfull, underfull, undefined-reference or citation warnings. The known third-party algorithm.sty UTF-8 comment warning and redundant inputenc warning persist; neither produces corrupted PDF text.
- Offline full citation gate passed all 40 DOI records plus one official exception, with zero unknown or uncited keys.
- Advantages/limitations notes explicitly state one CNN search per method and distinguish conditional final-training variation. Markdown/JSON match, all 26 titles align with the slide manifest, and timing remains 670 seconds.
- Final page plan is 44–45: roughly four additional Results pages, one case-study page, and at most one comparison-table page. Final held-out analysis and full-document inspection remain deferred until stage 07.

## Completed scientific report — 2026-11-02

- Final local build: `report/main.pdf`, **44 A4 pages**, with the required 12-point body, one-inch margins, and 1.15 line spread. The abstract contains 220 words. The earlier entries above describe superseded interim builds.
- Final analysis reads the completed `results/final_metrics.json` and saved qualitative artifacts. The abstract, results, comparison, conclusion, notes, and viva explicitly distinguish the clean-accuracy tie between DE and random search from improvement over the registered baseline. Worse DE corruption accuracy and calibration error are prominent. No superiority over random search or field-reliability claim is made.
- The results identify actual classwise gains, remaining Early/Late blight errors, underconfidence, and corruption-specific reversals. Three concrete Grad-CAM cases describe visible activation and background effects without treating the maps as lesion masks. The fixed six-map selection and its all-correct status are disclosed; the separate gallery includes actual saved errors.
- The completed first build was rendered and reviewed across every page; the condensed 44-page build and all subsequently changed pages were checked again. Detailed inspections covered the confusion/per-class pair, robustness/reliability pair, six-case maps, model table, comparison tables, abstract, case studies, and conclusion. A confusion-axis label collision was fixed in the figure generator; the corrected asset is embedded. Model output shapes now render the multiplication glyph correctly. A paragraph boundary was adjusted to avoid splitting a chapter reference across a float page.
- All required main-report evidence remains; the supplementary prediction gallery and embedding projection remain standalone artifacts. Figure labels are readable at their final report sizes. No clipped text, overlapping labels, unresolved citation markers, pending-result text, replacement characters, overfull boxes, underfull boxes, or undefined-reference warnings remain in the checked PDF/build.
- Tectonic still reports a UTF-8 comment warning in the third-party `algorithm.sty` source, and its log notes the redundant `inputenc` package under the Unicode engine. Neither corrupts the PDF. These are the only known harmless package warnings.
- The cached full citation gate passes all 40 DOI entries and the documented official-page exception, with no unknown or uncited keys. No new references were added during final result analysis; the earlier live DOI verification remains recorded.
- Speaker Markdown and JSON use the exact 26 slide titles, total **670 seconds**, and preserve four-member ownership. The notes record the SHA-256 of the completed metrics source; title alignment and source hash were checked. The deck owner confirmed the exact scripts are embedded. Viva preparation now contains **52** questions and answers, including the neutral/negative findings and the one-search-per-method limitation.
- `git diff --check` passes for the report/document edits. Final named PDF export, locked-fact checks on the delivery artifacts, and the independent project-wide verification gate remain the coordinating agent's delivery steps; this record is the document author's review, not that independent audit.
