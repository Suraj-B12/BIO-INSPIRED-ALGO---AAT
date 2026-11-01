# Final report page budget

This is a layout plan, not a completed experiment or a simulated final PDF. It uses the actual current report and the aspect ratios declared in the real figure generator. No synthetic measurements, substitute plots, or dummy result pages were generated.

## Requirements retained

AGENTS.md §10.1 requires 12-point A4 report, one-inch margins, 1.15 line spread, and 30–45 pages. §10.3 explicitly names benchmark convergence/box/snapshots, DE convergence/diversity/landscape/parallel coordinates/gene associations, the final metric table and bars, learning curves, confusion matrices, per-class F1, ROC, uncertainty, robustness, calibration, and Grad-CAM. All these main-report placements remain. The dataset table, pipeline, architecture, model parameters, baseline, chromosome, protocol, benchmark formulas/surfaces, median/best table, and exploratory statistics also remain.

§9.2 requires the 12-image prediction grid and embedding projection as standalone figure artifacts, but §10.3 does not specifically require reproducing those two images in the main report. Their supplementary location is now explained in Results. The required confusion matrices, per-class scores, and six-image Grad-CAM evidence remain in the main report and support the concrete final error/case analysis. No standalone figure is removed or reduced.

## Layout decisions

- Five complementary plot pairs share floats, retaining separate numbered captions: benchmark convergence + box plots; DE fitness + diversity; candidate landscape + parallel coordinates; confusion matrices + per-class F1; corruption response + reliability diagrams.
- Paired panels use full text width and a maximum height of 0.42 text height each, preserving aspect ratios. This makes each pair fit a page without shrinking body text. The actual available plots are checked at rendered size; missing pairs continue to show honest pending states.
- Grad-CAM uses full text width. Its revised 12.1 × 9.72-inch source becomes approximately 6.27 × 5.04 inches before caption, keeping the six cases legible.
- ROC uses 80% text width. The revised 8 × 6.4-inch source is approximately 5.02 × 4.01 inches before tight cropping; its 18-point minimum labels remain above 11 points at that scale. This leaves room for interval/statistical discussion.
- Learning curves use the revised three-row layout as one figure, never a paired-panel height cap. The current interim plot contains only the three completed baseline runs, using actual saved curves. Final regeneration will add all methods in the same geometry; no synthetic curves or test images were used for layout.
- Concise List-of-Figures entries retain the full scientific caveats in the actual captions. This avoids repeating long loss/uncertainty caveats in the front matter.
- No main-body font, margin, or line-spacing requirement is relaxed.

## Expected chapter envelope

| Section | Planned pages | Notes |
|---|---:|---|
| Front matter | 7 | Cover, certificate, acknowledgement, abstract, contents, figures, tables |
| Introduction | 1 | Fixed |
| Literature Review | 2 | Fixed |
| Theoretical Background | 2 | Fixed |
| Nature-Inspired Algorithms | 3 | Orphan page already removed |
| Methodology | 4 | Pipeline, architecture, required tables |
| Experimental Setup | 3 | Verified after moving the timing definition beside the neural protocol |
| Results and Discussion | 12 | All required plots plus final measured interpretation; currently 8 pages |
| Comparative Performance | 2–3 | Two pages now; allow one extra if the final cost table requires it |
| Applications and Case Studies | 2 | Reserve space for 2–3 actual held-out cases |
| Conclusion | 1 | Generated final metrics |
| Future Work | 1 | Fixed |
| References | 3 | All verified cited sources |
| GitHub Repo Link | 1 | Fixed |
| **Total** | **44–45** | Final rendered check is mandatory |

The latest interim build is **39 pages**, including the completed DE/random-search figures, selected-recipe table, larger benchmark plots, and vertical baseline learning curves. Setup was reduced from four to three pages and comparison from three to two by removing repeated prose and placing definitions together; required settings, figures, body size, margins, and spacing are unchanged. The pHash identity limitation and single-CNN-search limitation remain.

Remaining plots are the final metric bars, confusion/per-class pair, ROC, robustness/reliability pair, and full-width six-case Grad-CAM grid. Their current source geometries total roughly three pages of image area, with captions and float packing bringing the expected net Results growth to about four pages. Replace pending/generic discussion with concrete results rather than appending parallel prose. One additional case-study page and, if necessary, one comparison-table page leave a target of 44–45 pages. These are geometric estimates, not a simulated final PDF or a guarantee about unrendered results.

The refreshed List of Figures, benchmark pair, Rastrigin snapshots, DE convergence/diversity pair, landscape/parallel pair, gene-association/configuration page, learning curve, setup pages, and comparison pages were inspected at rendered size. Existing search/benchmark labels are readable. The final confusion pair must retain full text width to keep its smallest source annotations at about 9 points; the final learning plot must retain its single-figure geometry. No outstanding font issue was found in the newly rendered actual plots, but final figures still require inspection after evaluation.

## Final gate after evaluation

1. Regenerate real figures and tables from the locked evaluation.
2. Add measured class-confusion observations, selected-recipe interpretation, and 2–3 actual Grad-CAM cases without hand-typing result numbers.
3. Compile and inspect every page; confirm 30–45 pages.
4. Check paired plots and legends at normal reading scale. Confirm that the final three-method learning legend fits, and that confusion annotations, class key, and Grad-CAM labels remain legible.
5. If a paired panel becomes unreadable, expand/reflow the source layout; recover space from repeated discussion or secondary supplementary placements, not smaller body text.
6. Confirm the supplementary prediction grid and embedding files exist in the delivered repository.
