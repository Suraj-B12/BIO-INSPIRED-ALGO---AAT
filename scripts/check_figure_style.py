"""Reproduce the figure color-vision audit without changing experiment artifacts.

Machado et al. (2009) endpoint matrix data are reproduced from colorspacious
1.1.2, which attributes them to the authors' published supplement. The source
URLs and exact matrices are retained in results/style_check.json. This is an
engineering review, not certification or a study with human participants.
"""
from pathlib import Path
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"data"/".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb,to_hex
import numpy as np
import pymupdf
from pptx import Presentation
import yaml

# Severity 100 in colorspacious corresponds to complete dichromacy. These are
# the published matrices, not hand-tuned palette transformations.
MATRICES={
    "protanomaly": [[.152286,1.052583,-.204868],[.114503,.786281,.099216],[-.003882,-.048116,1.051998]],
    "deuteranomaly": [[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.011820,.042940,.968881]],
    "tritanomaly": [[1.255528,-.076749,-.178779],[-.078411,.930809,.147602],[.004733,.691367,.303900]],
}
SIMULATION_NAMES={"protanomaly":"Protanopia","deuteranomaly":"Deuteranopia","tritanomaly":"Tritanopia"}
SOURCES={
    "paper":"https://doi.org/10.1109/TVCG.2009.113",
    "authors_supplement":"https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html",
    "reference_implementation":"https://github.com/njsmith/colorspacious/blob/v1.1.2/colorspacious/cvd.py",
    "simulation_direction":"https://colorspacious.readthedocs.io/en/stable/tutorial.html#simulating-colorblindness",
    "color_cues":"https://www.w3.org/WAI/WCAG21/Understanding/use-of-color.html",
    "text_contrast":"https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html",
    "graphic_contrast":"https://www.w3.org/WAI/WCAG21/Understanding/non-text-contrast.html",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linear_rgb(rgb):
    rgb=np.asarray(rgb,dtype=float)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def encoded_rgb(linear):
    return np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055)


def simulate(rgb,kind):
    return np.clip(encoded_rgb(linear_rgb(rgb)@np.asarray(MATRICES[kind]).T),0,1)


def luminance(rgb):
    return linear_rgb(rgb)@np.array([.2126,.7152,.0722])


def contrast(a,b):
    first,second=float(luminance(a)),float(luminance(b))
    return (max(first,second)+.05)/(min(first,second)+.05)


def actual_preview(path,page=0):
    document=pymupdf.open(path)
    pixmap=document[page].get_pixmap(matrix=pymupdf.Matrix(1,1),alpha=False)
    return np.frombuffer(pixmap.samples,dtype=np.uint8).reshape(pixmap.height,pixmap.width,3)/255


def contact(rgb,path):
    figure,axes=plt.subplots(2,2,figsize=(15,9))
    views=[("Original",rgb)]+[(SIMULATION_NAMES[kind]+" · simulated",simulate(rgb,kind)) for kind in MATRICES]
    for axis,(name,pixels) in zip(axes.flat,views):
        axis.imshow(pixels);axis.set_title(name,fontsize=14);axis.axis("off")
    figure.tight_layout();figure.savefig(path,dpi=150,bbox_inches="tight");plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-library",type=Path,help="Optional temporary colorspacious 1.1.2 install for a numerical cross-check")
    parser.add_argument("--record-visual-review",action="store_true",help="Record review only after inspecting the current contact sheets and layouts")
    args=parser.parse_args()
    style=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))["style"]
    output=ROOT/"results"/"style_review";output.mkdir(parents=True,exist_ok=True)
    keys=["baseline","de","random_search","ga","pso","jde"]
    colors=np.array([to_rgb(style[key]) for key in keys])
    ratios={key:{"hex":style[key],"against_white":contrast(to_rgb(style[key]),[1,1,1]),
        "against_botanical_background":contrast(to_rgb(style[key]),to_rgb(style["background"]))} for key in ["foreground"]+keys}
    assert all(ratios[key]["against_white"]>=3 for key in keys)
    assert ratios["foreground"]["against_white"]>=4.5
    cnn_markers={"baseline":"s","de":"o","rs":"^"}
    benchmark_tree=ast.parse((ROOT/"scripts/08_benchmarks.py").read_text(encoding="utf-8"))
    benchmark_markers=next(ast.literal_eval(node.value) for node in benchmark_tree.body
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=="MARKERS" for target in node.targets))
    benchmark_lines=next(ast.literal_eval(node.value) for node in benchmark_tree.body
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=="LINE_STYLES" for target in node.targets))
    assert len(set(cnn_markers.values()))==len(cnn_markers)
    assert len(set(benchmark_markers.values()))==len(benchmark_markers)
    assert len(set(map(repr,benchmark_lines.values())))==len(benchmark_lines)
    source=(ROOT/"scripts/09_figures.py").read_text(encoding="utf-8")
    checks={"parallel_coordinates_explicit_fitness_axis":"np.r_[vector,score]" in source,
        "landscape_shape_bands":"bands=np.digitize" in source,
        "gradcam_sequential_map_and_scale":'cmap="inferno"' in source and 'Normalize(0,1)' in source,
        "gradcam_numeric_boundary":"levels=[.5]" in source and "white contour = 0.5" in source,
        "surface_note_foreground":'fontsize=14,va="top",color=STYLE["foreground"]' in source,
        "preserved_native_eda_pixels":'interpolation="none"' in source,
        "dataset_source_labels_24pt":"fontsize=24,pad=10" in source}
    assert all(checks.values()),checks
    reference={"status":"not requested"}
    if args.reference_library:
        sys.path.insert(0,str(args.reference_library))
        import colorspacious
        reference={"version":colorspacious.__version__,"random_srgb_samples":1000,"seed":42,"max_absolute_error":{},
            "source_sha256":sha(Path(colorspacious.cvd.__file__))}
        sample=np.random.default_rng(42).random((1000,3))
        for kind,matrix in MATRICES.items():
            np.testing.assert_array_equal(matrix,colorspacious.cvd.machado_et_al_2009_matrix(kind,100))
            expected=np.clip(colorspacious.cspace_convert(sample,{"name":"sRGB1+CVD","cvd_type":kind,"severity":100},"sRGB1"),0,1)
            error=float(np.max(np.abs(simulate(sample,kind)-expected)))
            assert error<1e-12
            reference["max_absolute_error"][kind]=error
        reference["status"]="passed"
    figure,axes=plt.subplots(4,1,figsize=(12,6.5))
    views=[("Original",colors)]+[(SIMULATION_NAMES[kind]+" · simulated",simulate(colors,kind)) for kind in MATRICES]
    for axis,(label,palette) in zip(axes,views):
        for position,color in enumerate(palette):
            axis.plot([position-.3,position+.3],[0,0],color=color,linewidth=8)
            axis.text(position,-.16,keys[position].replace("_"," "),ha="center",fontsize=11)
        axis.set(xlim=(-.5,5.5),ylim=(-.3,.28));axis.set_title(label,loc="left",fontsize=12);axis.axis("off")
    figure.tight_layout();figure.savefig(output/"palette_cvd.png",dpi=180,bbox_inches="tight");plt.close(figure)
    actual_files={}
    benchmark=ROOT/"figures/fig_benchmark_convergence.pdf"
    if benchmark.exists():
        contact(actual_preview(benchmark),output/"benchmark_cvd.png");actual_files[str(benchmark.relative_to(ROOT))]=sha(benchmark)
    candidates=[ROOT/"slides/BIA_AAT_Team10.pptx",ROOT/"slides/Plantwise_Design_Preview.pptx"]
    deck=next((path for path in candidates if path.exists() and path.with_suffix(".pdf").exists()),candidates[-1])
    deck_pdf=deck.with_suffix(".pdf")
    dataset=ROOT/"figures/fig_dataset_slide.png";dataset_pdf=dataset.with_suffix(".pdf")
    slide_check={"status":"unavailable"}
    if deck.exists() and deck_pdf.exists() and dataset.exists():
        slides=Presentation(deck);match=hashlib.sha1(dataset.read_bytes()).hexdigest()
        for slide_number,slide in enumerate(slides.slides,start=1):
            for shape in slide.shapes:
                if hasattr(shape,"image") and shape.image.sha1==match:
                    page=pymupdf.open(dataset_pdf)[0]
                    spans=[span for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"] for span in line["spans"]]
                    source_size=max(span["size"] for span in spans)
                    slide_check={"status":"measured; visually review contact image","slide":slide_number,"slide_count":len(slides.slides),
                        "deck_path":str(deck.relative_to(ROOT)),"deck_kind":"final" if deck==candidates[0] else "draft",
                        "image_height_inches":shape.height/914400,"image_width_inches":shape.width/914400,
                        "source_label_points":source_size,"effective_label_points":source_size*(shape.height/914400)/(page.rect.height/72),
                        "deck_sha256":sha(deck),"pdf_sha256":sha(deck_pdf),"current_figure_matches_embedded_image":True}
                    contact(actual_preview(deck_pdf,slide_number-1),output/"dataset_slide_cvd.png")
        actual_files[str(deck_pdf.relative_to(ROOT))]=sha(deck_pdf)
        if args.record_visual_review:
            assert slide_check.get("current_figure_matches_embedded_image"),"The deck does not contain the current dataset figure; rebuild and inspect it before recording review."
    families={
        "fig_de_convergence":"Distinct circle/square markers and a dashed baseline, with text legend.",
        "fig_de_diversity":"Separate titled panels and labeled axes.",
        "fig_de_vs_rs":"Circle/square markers and named legend.",
        "fig_search_landscape":"Fitness-band marker shapes, numeric band labels, dark edges, and a star for the best candidate.",
        "fig_parallel_coords":"An eighth numeric macro-F1 axis; dark outlines keep bright trajectories visible.",
        "fig_gene_importance":"Named gene rows; signed bar position and a numeric axis.",
        "fig_learning_curves":"Square/circle/triangle markers and named legend.",
        "fig_before_after_bars":"Hatches, fixed grouped positions, numeric bar labels, and named legend.",
        "fig_per_class_f1":"Baseline hatch, row labels, bar lengths, and numeric gain labels.",
        "fig_confusion_side_by_side":"Named model panels, class axes, and numeric non-negligible cell percentages.",
        "fig_roc":"Dashed baseline and solid DE curves, including per-class curves; named macro legend.",
        "fig_robustness":"Square/circle markers, legend, and titled corruption panels.",
        "fig_reliability":"Separate named panels, numeric ECE, points and a diagonal reference.",
        "fig_gradcam_grid":"Named model/class panels, numeric confidence, sequential activation scale, and a labeled 0.5 boundary.",
        "fig_gradcam_slide":"Same redundant cues as the report grid; fixed first two examples.",
        "fig_predictions_grid":"Explicit Correct/Error words and true/predicted labels.",
        "fig_embeddings":"Ten distinct class marker shapes with a text legend.",
        "fig_benchmark_convergence":"Five distinct markers and line patterns with a named legend.",
        "fig_benchmark_box":"Algorithm labels directly beneath boxes.",
        "fig_de_rastrigin_2d":"Population points and a distinct optimum star; generation titles and numeric coordinates.",
        "fig_de_operators_diagram":"Explicit donor labels, arrow geometry, different crossover markers, and a named legend.",
        "fig_benchmark_surfaces":"Function titles, spatial height/axes, and optimum stars.",
        "fig_architecture":"Named nodes and directed connectors.","fig_pipeline":"Named nodes and directed connectors.",
        "fig_class_distribution":"Class labels, numeric counts, and bar length.",
        "fig_dataset_samples":"Class labels on real image groups.","fig_dataset_slide":"Large wrapped class labels on ten real images.",
        "fig_augmentation_examples":"Named augmentation panels.","fig_corruptions":"Named corruption and severity panels."}
    cue_audit=[{"figure":name,"non_color_cues":cue,"actual_artifact_available":(ROOT/"figures"/f"{name}.png").exists()} for name,cue in families.items()]
    pending=[row["figure"] for row in cue_audit if not row["actual_artifact_available"]]
    spec=importlib.util.spec_from_file_location("figure_style_validation",ROOT/"scripts/09_figures.py")
    rendering=importlib.util.module_from_spec(spec);spec.loader.exec_module(rendering)
    for activation,expected_contours in ((np.zeros((24,24)),0),(np.ones((24,24)),0),
            (np.tile(np.linspace(0,1,24),(24,1)),1)):
        before=activation.copy();figure,axis=plt.subplots()
        rendering.show_gradcam(axis,np.zeros((24,24,3),dtype=np.uint8),activation)
        np.testing.assert_array_equal(before,activation)
        assert len(axis.collections)==expected_contours
        plt.close(figure)
    result={"schema_version":1,"timestamp":datetime.now(timezone.utc).isoformat(),
        "status":("passed_available_artifacts; final experiment figure review remains pending" if pending else "passed_available_artifacts")
            if args.record_visual_review else "numerical_and_source_checks_passed; visual_review_not_recorded",
        "scope":"Engineering simulation, source-level redundant-cue audit, and actual current-draft slide inspection; not universal accessibility certification.",
        "method":{"name":"Machado, Oliveira and Fernandes (2009)","severity":100,"matrices":MATRICES,
            "transform":"sRGB decode -> published 3x3 linear-RGB matrix -> sRGB encode -> clip [0,1] for display",
            "sources":SOURCES,"reference_crosscheck":reference},
        "palette":{key:style[key] for key in keys},"contrast":ratios,
        "contrast_interpretation":"All opaque method colors exceed 3:1 against white; text uses foreground with >4.5:1. Pairwise color distinction is not assumed. Low-opacity uncertainty bands and photographic content are not certified by these swatch ratios.",
        "simulated_palette":{kind:{key:to_hex(value) for key,value in zip(keys,simulate(colors,kind))} for kind in MATRICES},
        "marker_maps":{"cnn":cnn_markers,"benchmarks":benchmark_markers},"benchmark_line_styles":benchmark_lines,"source_checks":checks,"figure_cues":cue_audit,"pending_figures":pending,
        "gradcam_rendering_checks":{"status":"passed","synthetic_in_memory_cases":3,"cases":["constant zero: no contour","constant one: no contour","gradient crossing 0.5: one contour"],
            "input_activation_arrays_unchanged":True,"synthetic_assets_saved_to_results":False},
        "visual_review":{"recorded":args.record_visual_review,"reviewer":"assistant visual inspection" if args.record_visual_review else None,
            "scope":"Generated simulation contact sheets, current draft dataset slide, changed figure layouts; final-data plots excluded while unavailable"},
        "dataset_slide":slide_check,"review_artifacts":[str(path.relative_to(ROOT)) for path in sorted(output.glob('*.png'))],
        "actual_source_hashes":actual_files,"plot_generator_sha256":sha(ROOT/'scripts/09_figures.py'),"config_sha256":sha(ROOT/'config.yaml'),
        "limitations":["Simulation approximates selected color-vision deficiencies; it is not user testing or a guarantee for every observer/projector.",
            "Some method colors become similar. Distinct markers, hatches, geometry and labels provide identification independently of color.",
            "Grad-CAM values are normalized within each image and are not comparable causal evidence or proof that lesions drove a prediction."]+
            (["Final held-out plots and genuine Grad-CAM examples require another visual review after evaluation creates those artifacts."] if pending else [])}
    (ROOT/"results/style_check.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Style audit saved; {sum(row['actual_artifact_available'] for row in cue_audit)} actual figure families available; reference cross-check {reference['status']}.")


if __name__=="__main__":main()
