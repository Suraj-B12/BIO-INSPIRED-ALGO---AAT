# Final report page budget

This is a layout plan, not a completed experiment or a simulated final PDF. It uses the actual current report and the aspect ratios declared in the real figure generator. No synthetic measurements, substitute plots, or dummy result pages were generated.

## Requirements retained

AGENTS.md §10.1 requires 12-point A4 report, one-inch margins, 1.15 line spread, and 30–45 pages. §10.3 explicitly names benchmark convergence/box/snapshots, DE convergence/diversity/landscape/parallel coordinates/gene associations, the final metric table and bars, learning curves, confusion matrices, per-class F1, ROC, uncertainty, robustness, calibration, and Grad-CAM. All these main-report placements remain. The dataset table, pipeline, architecture, model parameters, baseline, chromosome, protocol, benchmark formulas/surfaces, median/best table, and exploratory statistics also remain.

§9.2 requires the 12-image prediction grid and embedding projection as standalone figure artifacts, but §10.3 does not specifically require reproducing those two images in the main report. Their supplementary location is now explained in Results. The required confusion matrices, per-class scores, and six-image Grad-CAM evidence remain in the main report and support the concrete final error/case analysis. No standalone figure is removed or reduced.

## Layout decisions

- Five complementary plot pairs share floats, retaining separate numbered captions: benchmark convergence + box plots; DE fitness + diversity; candidate landscape + parallel coordinates; confusion matrices + per-class F1; corruption response + reliability diagrams.
- Paired panels use full text width and a maximum height of 0.42 text height each, preserving aspect ratios. This makes each pair fit a page without shrinking body text. The actual available plots are checked at rendered size; missing pairs continue to show honest pending states.
- Grad-CAM uses full text width. Its revised 12.1 × 9.72-inch source becomes approximately 6.27 × 5.04 inches before caption, keeping the six cases legible.
- ROC uses 80% text width. Its 7 × 6-inch source becomes approximately 5.02 × 4.30 inches; the source's 12-point tick text remains about 8.6 points in print. This leaves room for interval/statistical discussion.
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
| Experimental Setup | 3–4 | Current setup may retain four pages |
| Results and Discussion | 10–11 | All required plots plus final measured interpretation |
| Comparative Performance | 3 | Equal-budget curve, benchmark tables, actual cost |
| Applications and Case Studies | 2 | Reserve space for 2–3 actual held-out cases |
| Conclusion | 1 | Generated final metrics |
| Future Work | 1 | Fixed |
| References | 3 | All verified cited sources |
| GitHub Repo Link | 1 | Fixed |
| **Total** | **43–45** | Final rendered check is mandatory |

The added final classifier plots contribute roughly four to five pages after replacing current placeholders and reusing the existing learning-curve/table space. The DE convergence/diversity pair and DE-versus-random curve add about two pages; final tables and concrete interpretation use remaining space. These are geometric estimates, not a promise about unrendered artifacts.

## Final gate after evaluation

1. Regenerate real figures and tables from the locked evaluation.
2. Add measured class-confusion observations, selected-recipe interpretation, and 2–3 actual Grad-CAM cases without hand-typing result numbers.
3. Compile and inspect every page; confirm 30–45 pages.
4. Check paired plots and legends at normal reading scale. Learning-curve and confusion-matrix source fonts need particular attention because wide canvases shrink text in print.
5. If a paired panel becomes unreadable, expand/reflow the source layout; recover space from repeated discussion or secondary supplementary placements, not smaller body text.
6. Confirm the supplementary prediction grid and embedding files exist in the delivered repository.
