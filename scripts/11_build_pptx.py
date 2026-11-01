"""Rebuild the botanical Team10 deck from measured JSON and figures."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import importlib.util
import json
import re
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from bioalgo.common import ROOT,load_config,read_json,save_json

INK='193B2B';GREEN='28734E';MUTED='75866B';PALE='EAF1E0';BG='F8FAF4';LINE='DDE6D3';WHITE='FFFFFF'
TEAM=[('Aryanraj S Singh','1BM23CD008'),('Richik Chaudhuri','1BM23CD050'),('Suraj B','1BM23CD061'),('Sushant Deo','1BM23CD062')]
TITLE='Plant Disease Detection using Transfer-Learning CNN Optimized with Differential Evolution'
MANIFEST=[]


def color(h):return RGBColor.from_string(h)

def text(slide,value,x,y,w,h,size=20,bold=False,ink=INK,align=None):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame
    tf.clear();tf.word_wrap=True;tf.margin_left=tf.margin_right=Inches(.01);tf.margin_top=tf.margin_bottom=0
    for i,line in enumerate(str(value).split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color(ink);p.space_after=Pt(6);p.line_spacing=1.13
        if align is not None:p.alignment=align
    return box

def rect(slide,x,y,w,h,fill=WHITE,radius=True):
    s=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));s.fill.solid();s.fill.fore_color.rgb=color(fill);s.line.fill.background()
    if radius:s.adjustments[0]=.12
    return s

def picture(slide,path,x,y,w,h):
    path=ROOT/path
    if not path.exists():return text(slide,'Figure becomes available after its experiment completes.',x,y,w,h,20,ink=MUTED)
    with Image.open(path) as im:iw,ih=im.size
    ratio=min(w/iw,h/ih);ww,hh=iw*ratio,ih*ratio
    slide.shapes.add_picture(str(path),Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))

def new_slide(prs,title,kicker,notes):
    s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=color(BG)
    picture(s,Path('assets/bmsce_logo.png'),.45,.24,.44,.44)
    text(s,'B.M.S. College of Engineering',1.02,.32,7,.25,10,ink=MUTED)
    text(s,kicker.upper(),.56,.99,12,.25,10,ink=GREEN)
    text(s,title,.55,1.42,12.1,.85,29,bold=False)
    rect(s,.56,6.96,12.2,.012,LINE,False)
    text(s,'Department of Computer Science & Engineering (Data Science)',.56,7.12,10,.18,8,ink=MUTED)
    text(s,str(len(prs.slides)),12.05,7.09,.7,.25,9,ink=MUTED,align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text=notes
    MANIFEST.append({'slide':len(prs.slides),'title':title,'notes':notes})
    return s

def cards(s,items,y=2.65,h=3.45):
    gap=.22;w=(12.2-gap*(len(items)-1))/len(items)
    for i,(heading,body) in enumerate(items):
        x=.56+i*(w+gap);rect(s,x,y,w,h,PALE if i==0 else WHITE)
        text(s,f'{i+1:02d}',x+.22,y+.22,w-.44,.32,12,ink=GREEN)
        text(s,heading,x+.22,y+.91,w-.44,.65,23)
        text(s,body,x+.22,y+1.72,w-.44,h-1.9,17,ink=MUTED)

def chart(s,name,caption=''):
    picture(s,Path('figures')/name,.65,2.35,12.0,4.15)
    if caption:text(s,caption,.72,6.61,11.9,.24,9,ink=MUTED)

def pct(v):return f'{100*v:.2f}%'

def measured(path,value):
    MANIFEST[-1].setdefault('measured_values',[]).append({'source':path,'value':value});return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');args=parser.parse_args()
    cfg=load_config();data=read_json('results/data_summary.json');metrics=read_json('results/final_metrics.json');bench=read_json('results/benchmark_summary.json')
    if not args.draft and metrics is None:raise SystemExit('Final evaluation required. Use --draft only for a clearly marked design preview.')
    prs=Presentation();prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
    s=new_slide(prs,'Cover Slide/Main Slide','Bio Inspired Algorithms · 23DS7BSBIO',
        'Aryanraj, 25 seconds. Introduce Team 10, the fixed project title, and the question: can an evolutionary search improve the recipe of a pretrained plant-disease classifier? Acknowledge Manoj Kumar S, Assistant Professor. All results in this deck are generated from repository experiment files.')
    # Exact required title is the section label; the project title is the hero.
    for shape in list(s.shapes):
        if shape.has_text_frame and shape.text=='Cover Slide/Main Slide':shape.text_frame.paragraphs[0].font.size=Pt(11);shape.text_frame.paragraphs[0].font.color.rgb=color(MUTED)
    text(s,TITLE,.56,2.05,8.5,1.66,32)
    text(s,'INTELLIGENCE, NATURALLY.',.58,4.03,8,.28,11,ink=GREEN)
    rect(s,9.45,1.55,3.25,4.96,PALE)
    picture(s,Path('assets/bmsce_logo.png'),10.34,1.88,1.4,1.4)
    text(s,'TEAM 10',9.7,3.59,2.7,.3,12,ink=GREEN)
    for i,(name,usn) in enumerate(TEAM):text(s,f'{name}\n{usn}',9.7,4.05+i*.54,2.7,.5,11)
    text(s,'Under the guidance of\nManoj Kumar S, Assistant Professor',.58,4.83,8,1,18)
    text(s,'Academic Year 2026–2027',.58,6.24,8,.35,15,ink=GREEN)
    s=new_slide(prs,'Introduction to Optimization Problems','Find the best recipe',
        'Aryanraj, 30 seconds. An optimization problem has decision variables, an objective and constraints. Our objective is validation macro-F1. Variables include continuous learning rate and discrete head size and unfreezing depth. Each candidate costs an actual training run; gradients through this full recipe are unavailable. We therefore need a method for expensive mixed black-box optimization.')
    cards(s,[('Choose','Seven decisions define a training recipe: learning, regularization, architecture and augmentation.'),('Measure','Train on the training split. Measure macro-F1 on validation images only.'),('Improve','Search for a better recipe under a fixed evaluation budget.')])
    s=new_slide(prs,'What are Nature-Inspired Algorithms?','Variation · exploration · selection',
        'Aryanraj, 30 seconds. Nature-inspired algorithms borrow computational mechanisms such as population variation and competitive selection. They do not guarantee global optima. Exploration seeks new regions; exploitation refines promising ones. DE uses numerical differences between candidates to determine steps. The value lies in operators and measured behavior, not in the metaphor.')
    cards(s,[('A population','Maintain several possible solutions instead of one guess.'),('Useful variation','Create trials by mutation and recombination.'),('Evidence-led selection','Keep a candidate when its measured fitness is no worse.')])
    s=new_slide(prs,'Classification of Nature-Inspired Algorithms','A map of the field',
        'Aryanraj, 35 seconds. Evolutionary methods include GA, DE and evolution strategies. Swarm methods include PSO, ACO, ABC and Firefly. Physics-based methods include simulated annealing. Biogeography is a biological metaphor; TLBO is human-inspired. Explain that taxonomy boundaries vary across surveys, and that binomial crossover makes full DE coordinate-dependent, despite the useful geometry of difference-vector mutation.')
    cards(s,[('Evolutionary','Genetic Algorithm\nDifferential Evolution\nEvolution Strategies'),('Swarm','Particle Swarm\nAnt Colony\nArtificial Bee Colony'),('Other families','Simulated Annealing\nBiogeography\nTeaching–Learning')])
    s=new_slide(prs,'Problem Statement','Tomato leaf health',
        'Richik, 35 seconds. The fixed task is PlantVillage tomato classification with EfficientNet-B0 and DE. Controlled leaf images make a reproducible benchmark but are not the same as field photographs. We maximize validation macro-F1 over the mixed recipe while leaving test images unopened until the final protocol is frozen. The dataset counts on this slide come from the cleaning output.')
    text(s,'Maximize validation macro-F1 by evolving the training recipe.',.6,2.5,6.0,1.1,28)
    text(s,f"{measured('data_summary.json/usable_total',data['usable_total']):,} usable tomato images\n{len(data['original_counts'])} classes · group-aware 70 / 15 / 15 split\n{data['exact_duplicates_removed']} exact duplicates removed",.62,4.0,5.8,1.3,19,ink=MUTED)
    picture(s,Path('figures/fig_dataset_samples.png'),6.8,2.4,5.7,4.2)
    s=new_slide(prs,'Implementation and Results (1/7)','The full system',
        'Richik, 35 seconds. Walk through image verification and duplicate grouping, the fixed baseline, DE and equal-budget random search, higher-fidelity re-ranking, repeated final training, and locked evaluation. All three recipes share the same final data, epochs and stopping rule. Search uses validation only. The offline demo loads exact seed-zero weights.')
    chart(s,'fig_pipeline.png')
    s=new_slide(prs,'Implementation and Results (2/7)','A pre-registered baseline',
        'Richik, 35 seconds. EfficientNet-B0 starts with ImageNet weights. The baseline freezes every feature stage and trains a 256-unit head with dropout 0.2, Adam learning rate 0.001, no weight decay, no label smoothing and flips-only augmentation. This is fixed before search; we never weaken it after seeing results. Searched recipes use AdamW. Fine-tuning depth is one of the architecture decisions.')
    chart(s,'fig_architecture.png','The head and the number of unfrozen feature blocks are architecture genes.')
    s=new_slide(prs,'Implementation and Results (3/7)','Differential Evolution, from scratch',
        'Suraj, 50 seconds. Explain the three distinct donors and why the target is excluded. Mutation is x_r1 plus F times the difference of two others. Bounce-back repairs violations between the parent and the violated bound. Binomial crossover includes a forced mutant coordinate. Greedy selection accepts ties. Each evaluation saves population and RNG state so the run resumes exactly.')
    picture(s,Path('figures/fig_de_operators_diagram.png'),.65,2.35,7.4,4.25)
    text(s,'Seven evolved genes',8.45,2.6,4.1,.5,24)
    text(s,'Learning rate\nWeight decay\nDropout\nHead width\nUnfrozen blocks\nLabel smoothing\nAugmentation magnitude',8.48,3.3,3.9,3.1,19,ink=MUTED)
    s=new_slide(prs,'Implementation and Results (4/7)','Search under an equal budget',
        'Suraj, 40 seconds. The plot compares best validation macro-F1 against the same evaluation budget. Each candidate trains a fresh model using the same seed and fixed training subset. Search fitness is not a test score. The top three distinct configurations are re-ranked at full resolution before the final recipe is frozen. Discuss the actual relative curves, including ties or random-search advantages.')
    chart(s,'fig_de_vs_rs.png','Proxy validation fitness; final test performance is evaluated separately.')
    s=new_slide(prs,'Implementation and Results (5/7)','Before and after: measured performance',
        'Suraj, 45 seconds. Read the measured mean and standard deviation across all three seeds. Compare baseline, random-search and DE recipes honestly. Do not substitute the best seed or proxy-search score. State whether the gain is positive, neutral or negative, then discuss the per-class and statistical evidence that follows.')
    if metrics:
        base=metrics['models']['baseline']['aggregate'];de=metrics['models']['de']['aggregate']
        for i,(label,value,std) in enumerate([('BASELINE ACCURACY',base['accuracy']['mean'],base['accuracy']['std']),('DE ACCURACY',de['accuracy']['mean'],de['accuracy']['std']),('DE MACRO-F1',de['macro_f1']['mean'],de['macro_f1']['std'])]):
            x=.58+i*4.12;rect(s,x,2.43,3.9,1.6,PALE if i==1 else WHITE);text(s,label,x+.21,2.65,3.45,.3,11,ink=MUTED);text(s,pct(measured('final_metrics.json/'+label,value)),x+.2,3.1,3.45,.6,36);text(s,f'± {std*100:.2f} percentage points',x+.22,3.74,3.5,.2,10,ink=MUTED)
        picture(s,Path('figures/fig_before_after_bars.png'),.65,4.18,12.0,2.5)
    else:chart(s,'fig_before_after_bars.png','DRAFT DESIGN PREVIEW — final evaluation has not completed.')
    s=new_slide(prs,'Implementation and Results (6/7)','Which classes improve?',
        'Sushant, 35 seconds. Compare macro averages with the per-class view. Use identical normalized confusion-matrix scales. Mention the classes that actually remain difficult and do not assume a particular confusion pattern. Macro-F1 matters because the class distribution is imbalanced.')
    chart(s,'fig_confusion_side_by_side.png')
    s=new_slide(prs,'Implementation and Results (7/7)','Explanations and stress tests',
        'Sushant, 40 seconds. Grad-CAM is computed using gradients through the last convolution. Warm regions contribute more to the predicted class; attention is not proof of causality. The stress test applies deterministic blur, noise, illumination change, JPEG compression and low light to held-out images. It simulates some field conditions without claiming field validation.')
    picture(s,Path('figures/fig_gradcam_grid.png'),.6,2.3,4.4,4.35);picture(s,Path('figures/fig_robustness.png'),5.3,2.4,7.4,4.1)
    s=new_slide(prs,'Live Demo','Plantwise · intelligence, naturally',
        'Sushant, 60 seconds. Open http://127.0.0.1:8000 after starting python demo/serve.py. Press P for presentation mode. Choose a sample tomato leaf; analyze and compare top-three probabilities and Grad-CAM. Apply mild blur and repeat. Then open Evolution Lab, run Rastrigin, pause with Space, scrub the timeline and explain mutation, crossover and selection. If the server is unavailable, show demo/demo.gif. Do not upload audience personal images.')
    cards(s,[('Diagnose','Compare exact baseline and DE seed-zero checkpoints on a sample leaf.'),('Explain','Inspect top-three probabilities and genuine Grad-CAM attention.'),('Evolve','Change the landscape and watch the population converge.')])
    s=new_slide(prs,'Performance Metrics and Benchmarking (1/3)','Accuracy is only the beginning',
        'Sushant, 40 seconds. Accuracy measures all correct labels. Precision, recall and F1 are calculated per class and macro-averaged. ROC-AUC assesses ranking. We report bootstrap confidence intervals for seed-zero accuracy and macro-F1, an exact paired McNemar test, and calibration. These answer different questions; none alone proves deployment readiness.')
    cards(s,[('Quality','Accuracy · precision · recall\nMacro-F1 · ROC-AUC\nPer-class confusion'),('Uncertainty','All seeds reported\nBootstrap confidence intervals\nPaired McNemar comparison'),('Beyond clean images','Corruption robustness\nExpected calibration error\nInference latency')])
    s=new_slide(prs,'Performance Metrics and Benchmarking (2/3)','Five algorithms. Equal evaluations.',
        'Sushant, 40 seconds. This study validates the optimizer independently of the CNN. The benchmark suite includes Sphere, Rosenbrock, Rastrigin, Ackley and Griewank. DE, adaptive jDE, GA, PSO and random search each receive the same function-evaluation budget for every seed. Different functions favor different search behaviors; we report all runs.')
    chart(s,'fig_benchmark_convergence.png',f"{bench['settings']['runs']} runs per algorithm/function · {bench['settings']['budget']:,} evaluations per run · {bench['settings']['dimensions']} dimensions")
    s=new_slide(prs,'Performance Metrics and Benchmarking (3/3)','Variation matters',
        'Sushant, 35 seconds. The box plots show all independent seeds. Discuss median and spread rather than picking a favorable run. jDE adapts its control parameters and can behave differently on multimodal functions. Rank-sum p-values are exploratory and uncorrected for multiple comparisons. A logarithmic display floor is disclosed for very small objective values.')
    chart(s,'fig_benchmark_box.png')
    s=new_slide(prs,'Advantages and Limitations','What the evidence supports',
        'Sushant, 40 seconds. Strengths include an auditable optimizer, fixed baseline, leakage-aware splits, equal budgets and exact resumability. Limitations include lab-domain images, proxy-fidelity noise, one crop, one held-out split and computational cost. Repeated seeds quantify training variation but not every possible distribution shift. Negative and neutral outcomes remain visible.')
    cards(s,[('Advantages','Auditable operators\nReproducible data splits\nEqual-budget control\nOffline inference'),('Limitations','Controlled-condition images\nSingle crop and split\nProxy-fidelity noise\nTraining cost')])
    s=new_slide(prs,'Future Scope','From the lab toward the field',
        'Sushant, 30 seconds. Future work is prospective, not claimed as completed. Priorities are true field-domain evaluation, multi-objective search balancing quality and latency, adaptive DE variants such as SHADE, on-device export and farmer-assisted active learning. New datasets require careful class mapping and a fresh evaluation protocol.')
    cards(s,[('Field evidence','Validate on real field images with an explicit class mapping.'),('Efficient intelligence','Optimize quality alongside latency and model size.'),('Adaptive learning','Explore SHADE, multi-fidelity search and expert feedback.')])
    spec=importlib.util.spec_from_file_location('citation_tools',ROOT/'scripts/14_check_citations.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bib=module.parse_bibtex((ROOT/'report/references.bib').read_text(encoding='utf-8'))
    requested=['storn1997de','mohanty2016plant','tan2019efficientnet','das2011de','brest2006jde','selvaraju2017gradcam','dietterich1998approximate','guo2017calibration','hendrycks2019robustness','singh2020plantdoc']
    keys=[k for k in requested if k in bib]
    if len(keys)<8:
        patterns=['storn','mohanty','tan2019','das2011','brest','selvaraju','dietterich','guo','hendrycks','singh2020']
        keys=[]
        for pat in patterns:
            matches=[k for k in bib if pat in k]
            if matches and matches[0] not in keys:keys.append(matches[0])
    for start in range(0,len(keys),5):
        subset=keys[start:start+5];s=new_slide(prs,f'References ({start//5+1}/{(len(keys)+4)//5})','Verified DOI sources',
            'All listed references have verified DOI records. The bibliography checker compares normalized title, first author and year with Crossref or DataCite and checks DOI resolution. Refer to the full report for the complete bibliography and evidence cache. Source-specific factual claims are cited in the report, and the code links to the exact experiment outputs.')
        for i,key in enumerate(subset):
            entry=bib[key];title=re.sub(r'[{}]','',entry['title']);author=entry['author'].split(' and ')[0]
            text(s,f'{start+i+1:02d}',.6,2.55+i*.76,.45,.3,12,ink=GREEN)
            text(s,f"{author} et al. ({entry['year']}). {title}",1.12,2.5+i*.76,11.3,.44,13)
            text(s,'DOI: '+entry['doi'],1.12,2.96+i*.76,11.3,.25,10,ink=MUTED)
    s=new_slide(prs,'Thank You / Questions','Team 10 · 23DS7BSBIO',
        'All members. Thank the audience and invite questions. Use the QR code for reproducible code, report, figures and the offline demo. If asked for a result, refer to the measured files rather than recalling an unverified value. For algorithm questions, demonstrate the Evolution Lab or show the unit tests.')
    text(s,'A healthier future.\nOne leaf at a time.',.65,2.62,8.2,1.6,39)
    text(s,'Plantwise · Intelligence, naturally.',.68,4.65,8,.4,18,ink=GREEN)
    picture(s,Path('figures/repo_qr.png'),9.45,2.65,2.8,2.8)
    text(s,'github.com/Suraj-B12/\nBIO-INSPIRED-ALGO---AAT',8.9,5.65,3.8,.6,11,ink=MUTED,align=PP_ALIGN.CENTER)
    destination=ROOT/'slides'/('Plantwise_Design_Preview.pptx' if args.draft else 'BIA_AAT_Team10.pptx');destination.parent.mkdir(exist_ok=True)
    prs.save(destination);save_json('results/slides_manifest.json',{'draft':args.draft,'slides':MANIFEST,'sources':['results/final_metrics.json','results/data_summary.json','results/benchmark_summary.json','report/references.bib']})
    print(f'Saved {len(prs.slides)} slides: {destination}')

if __name__=='__main__':main()
