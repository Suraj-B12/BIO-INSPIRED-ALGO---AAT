# Final report page budget and layout

The completed scientific report compiles to **44 pages** including front matter, figures, references, and the repository chapter. It retains the required 12-point A4 report format, one-inch margins, and 1.15 line spread. All measurements come from saved experiment artifacts; no synthetic result pages or substitute plots were used for pagination.

## Required evidence retained

AGENTS.md §10.3 placements remain: analytic benchmark tables, convergence, box plots and population snapshots; CNN DE convergence, diversity, landscape, parallel coordinates, gene associations and selected recipes; final metrics and bars, learning curves, confusion matrices, per-class F1, ROC, confidence intervals and McNemar; robustness, calibration, Grad-CAM and error analysis. The dataset, pipeline, architecture, protocol, algorithm settings, function definitions/surfaces, benchmark median/best and exploratory tests also remain.

The 12-image prediction gallery and embedding projection required by §9.2 remain standalone repository artifacts. The main report explains their supplementary role. The prediction gallery includes actual DE errors; the main report includes all six fixed Grad-CAM cases and discusses three concrete leaves.

## Rendered allocation

| Section | Pages |
|---|---:|
| Front matter | 7 |
| Introduction | 1 |
| Literature Review | 2 |
| Theoretical Background | 2 |
| Nature-Inspired Algorithms | 3 |
| Methodology | 4 |
| Experimental Setup | 3 |
| Results and Discussion | 12 |
| Comparative Performance | 2 |
| Applications and Case Studies | 2 |
| Conclusion | 1 |
| Future Work | 1 |
| References | 3 |
| GitHub Repo Link | 1 |
| **Total** | **44** |

## Readability decisions

Five complementary pairs retain separate numbered captions: benchmark convergence/box, DE fitness/diversity, landscape/parallel coordinates, confusion/per-class F1, and corruption/reliability. The pair helper uses full text width and a maximum height of 0.42 text height per plot, preserving aspect ratios. The final confusion figure must stay full width.

The three-row learning chart is a single figure, with every registered seed retained through the method’s shared epoch range. Its caption distinguishes recipe-specific label-smoothed losses. The six-case Grad-CAM grid uses full width and identifies within-image activation normalization. ROC uses 80% width with large source labels. Concise figure-list entries keep the front matter readable.

Repeated prose was shortened and the timing definition moved beside the neural protocol. The final cost table fits with the other comparison tables, removing a sparse float page. No required body typography was reduced. The clean-accuracy tie and the worse DE corruption accuracy/ECE are prominent in the abstract, results and conclusion.

The final export should be checked again if figure geometry, table content, or report prose changes. Rebuilding only a source-revision macro of the same width does not alter the scientific page allocation.
