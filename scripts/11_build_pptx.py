"""Rebuild the botanical Team10 deck from measured JSON and figures."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import importlib.util
import json
import re
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls
from bioalgo.common import ROOT,load_config,read_json,save_json

INK='193B2B';GREEN='28734E';MUTED='5B6E55';PALE='EAF1E0';BG='F8FAF4';LINE='DDE6D3';WHITE='FFFFFF'
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
    transition=parse_xml(f'<p:transition {nsdecls("p")} spd="med"><p:fade/></p:transition>')
    s._element.insert_element_before(transition,'p:timing','p:extLst')
    picture(s,Path('assets/bmsce_logo.png'),.45,.24,.44,.44)
    text(s,'B.M.S. College of Engineering · Bengaluru',1.02,.32,7,.25,10,ink=MUTED)
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
    if caption:text(s,caption,.72,6.61,11.9,.31,14,ink=MUTED)

def pct(v):return f'{100*v:.2f}%'

def measured(path,value):
    MANIFEST[-1].setdefault('measured_values',[]).append({'source':path,'value':value});return value


def table(slide, headers, rows, widths, y=2.55, row_height=.56, size=16):
    x0=.58
    rect(slide,x0,y,12.16,row_height,PALE,False)
    for row_index,row in enumerate([headers]+rows):
        x=x0
        if row_index and row_index%2:rect(slide,x0,y+row_index*row_height,12.16,row_height,WHITE,False)
        for col,(value,width) in enumerate(zip(row,widths)):
            text(slide,value,x+.12,y+row_index*row_height+.12,width-.24,row_height-.12,
                 size if row_index else max(14,size-2),bold=row_index==0,ink=GREEN if row_index==0 else INK)
            x+=width


def search_seconds(method):
    path=ROOT/f'results/{method}_evaluations.csv'
    if not path.exists():return None
    with path.open(encoding='utf-8',newline='') as stream:
        return sum(float(row['train_time']) for row in csv.DictReader(stream) if row['cached'].lower()=='false')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');args=parser.parse_args()
    cfg=load_config();data=read_json('results/data_summary.json');metrics=read_json('results/final_metrics.json');bench=read_json('results/benchmark_summary.json')
    if not args.draft and metrics is None:raise SystemExit('Final evaluation required. Use --draft only for a clearly marked design preview.')
    prs=Presentation();prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
    prs.core_properties.title=TITLE
    prs.core_properties.subject='Bio Inspired Algorithms · 23DS7BSBIO · Academic Year 2026–2027'
    prs.core_properties.author='Team 10 · B.M.S. College of Engineering'
    prs.core_properties.keywords='Plantwise, EfficientNet-B0, Differential Evolution, PlantVillage, reproducible research'
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
    for i,(name,usn) in enumerate(TEAM):
        y=4.05+i*.57
        text(s,name,9.7,y,2.7,.24,13)
        text(s,usn,9.7,y+.25,2.7,.2,10,ink=MUTED)
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
    text(s,'Evolve seven training decisions. Maximize validation macro-F1.',.6,2.38,12.0,.48,23)
    text(s,f"{measured('data_summary.json/usable_total',data['usable_total']):,} images · {len(data['original_counts'])} classes · group-aware 70 / 15 / 15 split · {data['exact_duplicates_removed']} exact duplicates removed",.62,2.98,12.0,.35,15,ink=MUTED)
    picture(s,Path('figures/fig_dataset_slide.png'),.65,3.3,12.0,3.5)
    s=new_slide(prs,'Implementation and Results (1/10)','The full system',
        'Richik, 35 seconds. Walk through image verification and duplicate grouping, the fixed baseline, DE and equal-budget random search, higher-fidelity re-ranking, repeated final training, and locked evaluation. All three recipes share the same final data, epochs and stopping rule. Search uses validation only. The offline demo loads exact seed-zero weights.')
    chart(s,'fig_pipeline.png')
    s=new_slide(prs,'Implementation and Results (2/10)','A pre-registered baseline',
        'Richik, 35 seconds. EfficientNet-B0 starts with ImageNet weights. The baseline freezes every feature stage and trains a 256-unit head with dropout 0.2, Adam learning rate 0.001, no weight decay, no label smoothing and the common random-crop and flip pipeline without RandAugment. This is fixed before search; we never weaken it after seeing results. Searched recipes use AdamW. Fine-tuning depth is one of the architecture decisions.')
    chart(s,'fig_architecture.png','The head and the number of unfrozen feature blocks are architecture genes.')
    s=new_slide(prs,'Implementation and Results (3/10)','Differential Evolution, from scratch',
        'Suraj, 50 seconds. Explain the three distinct donors and why the target is excluded. Mutation is x_r1 plus F times the difference of two others. Bounce-back repairs violations between the parent and the violated bound. Binomial crossover includes a forced mutant coordinate. Greedy selection accepts ties. Each evaluation saves population and RNG state so the run resumes exactly.')
    chart(s,'fig_de_operators_diagram.png')
    s=new_slide(prs,'Implementation and Results (4/10)','The chromosome and its search budget',
        'Suraj, 30 seconds. All seven encoded genes lie in the unit interval. Learning rate and weight decay decode logarithmically; architecture and augmentation genes are quantized. Zero head units means a direct linear classifier. The exact ranges, population, F, CR and evaluation budget shown here are fixed before the search. The baseline is not changed to make a comparison favorable.')
    table(s,['GENE','DECODED CHOICE','SCALE'],[
        ['Learning rate','10⁻⁴·⁵ to 10⁻²','Logarithmic'],['Weight decay','10⁻⁶ to 10⁻²','Logarithmic'],
        ['Dropout','0.0 to 0.6','Linear'],['Head width','0, 128, 256, 512','Categorical'],
        ['Unfrozen blocks','0 to 9','Integer'],['Label smoothing','0.0 to 0.2','Linear'],
        ['Augmentation magnitude','0 to 15; 0 = no extra RandAugment','Integer']], [4.0,5.0,3.16],y=2.35,row_height=.47,size=15)
    tier=cfg['tiers'][cfg['tier']];budget=tier['population']*(tier['generations']+1)
    text(s,f"NP {tier['population']} · F {cfg['de']['F']} · CR {cfg['de']['CR']} · {tier['generations']} generations · {budget} evaluations per method",.72,6.3,12,.35,17,ink=GREEN)
    s=new_slide(prs,'Implementation and Results (5/10)','Search under an equal budget',
        'Suraj, 40 seconds. The plot compares best validation macro-F1 against the same evaluation budget. Each candidate trains a fresh model using the same seed and fixed training subset. Search fitness is not a test score. The top three distinct configurations are re-ranked at full resolution before the final recipe is frozen. Discuss the actual relative curves, including ties or random-search advantages.')
    chart(s,'fig_de_vs_rs.png','Proxy validation fitness; final test performance is evaluated separately.')
    s=new_slide(prs,'Implementation and Results (6/10)','Before and after: measured performance',
        'Suraj, 45 seconds. Read the measured mean and standard deviation across all three seeds. Compare baseline, random-search and DE recipes honestly. Do not substitute the best seed or proxy-search score. State whether the gain is positive, neutral or negative, then discuss the per-class and statistical evidence that follows.')
    if metrics:
        base=metrics['models']['baseline']['aggregate'];de=metrics['models']['de']['aggregate'];rs=metrics['models']['rs']['aggregate']
        for i,(label,value,std) in enumerate([('BASELINE ACCURACY',base['accuracy']['mean'],base['accuracy']['std']),('DE ACCURACY',de['accuracy']['mean'],de['accuracy']['std']),('BASELINE MACRO-F1',base['macro_f1']['mean'],base['macro_f1']['std']),('DE MACRO-F1',de['macro_f1']['mean'],de['macro_f1']['std'])]):
            x=.58+i*3.09;rect(s,x,2.43,2.9,1.42,PALE if i%2 else WHITE)
            text(s,label,x+.18,2.62,2.54,.3,14,ink=MUTED)
            text(s,pct(measured('final_metrics.json/models/'+('baseline' if label.startswith('BASELINE') else 'de')+'/aggregate/'+('accuracy' if 'ACCURACY' in label else 'macro_f1')+'/mean',value)),x+.18,2.98,2.54,.55,31)
            text(s,f'± {std*100:.2f} pp SD · {len(metrics["seeds"])} seeds',x+.18,3.55,2.54,.27,14,ink=MUTED)
        picture(s,Path('figures/fig_before_after_slide.png'),.65,3.96,12.0,2.45)
        reduction=metrics['relative_error_reduction']
        tied=abs(de['accuracy']['mean']-rs['accuracy']['mean'])<1e-12
        comparison=f"DE and RS tie at {pct(de['accuracy']['mean'])} mean accuracy" if tied else f"Mean accuracy: DE {pct(de['accuracy']['mean'])} · RS {pct(rs['accuracy']['mean'])}"
        reduction_text=f'DE error reduction vs baseline: {pct(reduction)}' if reduction is not None else 'Error reduction undefined: baseline error is zero'
        text(s,f'{comparison} · {reduction_text}',.72,6.56,12,.31,14,ink=GREEN)
        MANIFEST[-1]['evidence']='final_metrics.json/models/*/aggregate; final_metrics.json/relative_error_reduction'
    else:chart(s,'fig_before_after_bars.png','DRAFT DESIGN PREVIEW — final evaluation has not completed.')
    s=new_slide(prs,'Implementation and Results (7/10)','Where do the models disagree?',
        'Sushant, 35 seconds. Compare macro averages with the per-class view. Use identical normalized confusion-matrix scales. Mention the classes that actually remain difficult and do not assume a particular confusion pattern. Macro-F1 matters because the class distribution is imbalanced.')
    chart(s,'fig_confusion_side_by_side.png')
    s=new_slide(prs,'Implementation and Results (8/10)','Every class counts',
        'Sushant, 25 seconds. Read per-class F1 averaged across every seed. The classes are sorted by DE minus baseline gain so both improvements and regressions remain visible. Highlight the largest measured gain and the largest remaining weakness. Do not turn a class-specific gain into a claim of general field reliability.')
    chart(s,'fig_per_class_f1.png','Per-class F1 across all three seeds; classes ordered by measured gain.')
    s=new_slide(prs,'Implementation and Results (9/10)','Where the models look',
        'Sushant, 40 seconds. Grad-CAM is computed using gradients through the last convolution. Warm regions contribute more to the predicted class; attention is not proof of causality. The stress test applies deterministic blur, noise, illumination change, JPEG compression and low light to held-out images. It simulates some field conditions without claiming field validation.')
    chart(s,'fig_gradcam_slide.png','Within-image normalized activation; attention is not a lesion mask or causal proof.')
    s=new_slide(prs,'Implementation and Results (10/10)','Clean accuracy improves; robustness declines',
        'Sushant, 20 seconds. Both models are evaluated on the same five corruption types at three severities. Read the actual trend and the conditions that remain difficult. These controlled stress tests probe sensitivity, but they do not establish generalization to real farms or unseen cameras.')
    if metrics:
        robust=metrics['robustness']
        chart(s,'fig_robustness.png',f"Mean corruption accuracy · seed 0, 15 conditions: baseline {pct(robust['baseline']['mean_accuracy'])} · DE {pct(robust['de']['mean_accuracy'])} — DE is worse.")
        MANIFEST[-1]['evidence']='final_metrics.json/robustness/{baseline,de}/mean_accuracy'
    else:chart(s,'fig_robustness.png','Seed-zero models · identical held-out images and deterministic corruption settings.')
    s=new_slide(prs,'Live Demo','Plantwise · intelligence, naturally',
        'Sushant, 60 seconds. Open http://127.0.0.1:8000 after starting python demo/serve.py. Press P for presentation mode. Choose a sample tomato leaf; analyze and compare top-three probabilities and Grad-CAM. Apply mild blur and repeat. Then open Evolution Lab, run Rastrigin, pause with Space, scrub the timeline and explain mutation, crossover and selection. If the server is unavailable, show demo/demo.gif. Do not upload audience personal images.')
    cards(s,[('Diagnose','Compare exact baseline and DE seed-zero checkpoints on a sample leaf.'),('Explain','Inspect top-three probabilities and genuine Grad-CAM attention.'),('Evolve','Change the landscape and watch the population converge.')])
    s=new_slide(prs,'Performance Metrics and Benchmarking (1/5)','Accuracy is only the beginning',
        'Sushant, 40 seconds. Accuracy measures all correct labels. Precision, recall and F1 are calculated per class and macro-averaged. ROC-AUC assesses ranking. We report bootstrap confidence intervals for seed-zero accuracy and macro-F1, an exact paired McNemar test, and calibration. These answer different questions; none alone proves deployment readiness.')
    calibration=('Beyond accuracy','ROC-AUC: ranking quality\nECE: calibration gap\nLatency: time per forward pass')
    if metrics:
        ece={key:metrics['models'][key]['per_seed'][0]['calibration']['ece'] for key in ('baseline','de','rs')}
        calibration=('Calibration',f"ECE · seed 0 · lower is better\nBaseline {pct(ece['baseline'])} · DE {pct(ece['de'])}\nRS {pct(ece['rs'])}\nDE has a larger calibration gap.")
        MANIFEST[-1]['evidence']='final_metrics.json/models/*/per_seed/0/calibration/ece'
    cards(s,[('Classification','Accuracy = correct / total\nPrecision = TP / (TP + FP)\nRecall = TP / (TP + FN)'),('Macro-F1','F1 = 2PR / (P + R)\nMacro-F1 = mean class F1\nEvery class counts equally.'),calibration])
    s=new_slide(prs,'Performance Metrics and Benchmarking (2/5)','Uncertainty and paired evidence',
        'Sushant, 30 seconds. These intervals use 1,000 deterministic bootstrap resamples of seed-zero held-out predictions. They quantify resampling uncertainty on this split, not variation across farms. McNemar is an exact two-sided paired test of the seed-zero baseline and DE error pattern. A small p-value alone does not measure practical importance. The three-seed variation was shown separately.')
    if metrics:
        cirows=[]
        for key,label in [('baseline','Baseline'),('rs','Random search'),('de','DE-optimized')]:
            ci=metrics['models'][key]['per_seed'][0]['bootstrap_95_ci']
            cirows.append([label,*[' – '.join(pct(v) for v in ci[k]) for k in ('accuracy','macro_f1')]])
        table(s,['SEED-ZERO MODEL','ACCURACY · 95% CI','MACRO-F1 · 95% CI'],cirows,[3.5,4.33,4.33],row_height=.7,size=19)
        paired=metrics['mcnemar']
        text(s,f"Exact paired McNemar p = {paired['p_value']:.4g}",.7,5.85,11.9,.42,24,ink=GREEN)
        text(s,f"Baseline-only correct: {paired['baseline_only_correct']} · DE-only correct: {paired['de_only_correct']} · 1,000 bootstrap resamples",.72,6.43,11.9,.3,15,ink=MUTED)
        MANIFEST[-1]['evidence']='final_metrics.json/models/*/per_seed/0/bootstrap_95_ci; final_metrics.json/mcnemar'
    else:text(s,'Confidence intervals and paired statistics appear after the locked evaluation.',.7,3.3,11.9,1,23,ink=MUTED)
    s=new_slide(prs,'Performance Metrics and Benchmarking (3/5)','Five algorithms. Equal evaluations.',
        'Sushant, 40 seconds. This study validates the optimizer independently of the CNN. The benchmark suite includes Sphere, Rosenbrock, Rastrigin, Ackley and Griewank. DE, adaptive jDE, GA, PSO and random search each receive the same function-evaluation budget for every seed. Different functions favor different search behaviors; we report all runs.')
    chart(s,'fig_benchmark_convergence_slide.png',f"{bench['settings']['runs']} runs per algorithm/function · {bench['settings']['budget']:,} evaluations per run · {bench['settings']['dimensions']} dimensions")
    s=new_slide(prs,'Performance Metrics and Benchmarking (4/5)','No universal winner',
        'Sushant, 25 seconds. Compare median final objective values across all thirty seeds. Lower is better and the known optimum is zero. Different landscapes favor different methods. The full report also gives mean, standard deviation, best values and exploratory rank-sum tests. This is evidence about these functions and budgets, not a universal algorithm ranking.')
    algorithms=['DE','jDE','GA','PSO','Random Search']
    rows=[[function,*[f'{values["algorithms"][name]["median"]:.2e}' for name in algorithms]] for function,values in bench['functions'].items()]
    table(s,['FUNCTION',*algorithms],rows,[2.26,1.98,1.98,1.98,1.98,1.98],row_height=.61,size=16)
    text(s,'Median final objective · lower is better · 30 seeds per method and function',.72,6.43,11.9,.3,15,ink=MUTED)
    MANIFEST[-1]['evidence']='benchmark_summary.json/functions/*/algorithms/*/median'
    s=new_slide(prs,'Performance Metrics and Benchmarking (5/5)','The cost of a better recipe',
        'Sushant, 25 seconds. Report actual summed training time across all three seeds, and warmed model-forward latency at batch size one. Search time is the sum of uncached candidate-training durations, not a claim about exclusive wall-clock use. Evaluation latency uses the recorded device; the offline demo runs on CPU and also computes Grad-CAM, so its full response takes longer.')
    if metrics:
        rows=[]
        for key,label in [('baseline','Baseline'),('rs','Random search'),('de','DE-optimized')]:
            model=metrics['models'][key];a=model['aggregate']
            rows.append([label,f"{sum(r['training_seconds'] for r in model['per_seed'])/60:.1f} min",f"{a['latency_ms']['mean']:.2f} ± {a['latency_ms']['std']:.2f} ms",f"{model['per_seed'][0]['parameters']['total']/1e6:.2f} M"])
        table(s,['MODEL','ALL 3 SEEDS · TRAIN','INFERENCE · MEAN ± SD','PARAMETERS'],rows,[2.46,3.0,4.2,2.5],row_height=.7,size=18)
        text(s,f"Search training: DE {search_seconds('de')/60:.1f} min · RS {search_seconds('rs')/60:.1f} min",.72,5.83,11.9,.43,22,ink=GREEN)
        device=metrics['models']['baseline']['per_seed'][0]['latency_device']
        text(s,f"Batch-one latency on {device.upper()} · optimizer benchmark wall time: {bench['wall_seconds']:.1f} s",.72,6.43,11.9,.3,15,ink=MUTED)
        MANIFEST[-1]['evidence']='final_metrics.json/models/*; de_evaluations.csv/train_time; rs_evaluations.csv/train_time; benchmark_summary.json/wall_seconds'
    else:text(s,'Measured training and inference costs appear after all runs complete.',.7,3.3,11.9,1,23,ink=MUTED)
    s=new_slide(prs,'Advantages and Limitations','What the evidence supports',
        'Sushant, 40 seconds. Strengths include an auditable optimizer, fixed baseline, leakage-aware splits, equal budgets and exact resumability. Limitations include lab-domain images, proxy-fidelity noise, one crop, one held-out split and computational cost. Repeated seeds quantify training variation but not every possible distribution shift. Negative and neutral outcomes remain visible.')
    limitations='Controlled-condition images\nOne crop/split; one search per method\nProxy-fidelity noise\nTraining cost'
    if metrics:
        limitations=(f"DE ties RS: {pct(de['accuracy']['mean'])} mean accuracy\n"
                     'DE has lower corruption accuracy\nDE has a larger calibration gap\n'
                     'One crop/split; one search per method')
        MANIFEST[-1]['evidence']='final_metrics.json/models/*/aggregate/accuracy; final_metrics.json/robustness; final_metrics.json/models/*/per_seed/0/calibration/ece'
    cards(s,[('Advantages','Auditable operators\nReproducible data splits\nEqual-budget control\nOffline inference'),('Measured limitations',limitations)],y=2.6,h=3.9)
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
            text(s,f'{start+i+1:02d}',.6,2.55+i*.76,.45,.3,14,ink=GREEN)
            text(s,f"{author} et al. ({entry['year']}). {title}",1.12,2.5+i*.76,11.3,.44,14)
            text(s,'DOI: '+entry['doi'],1.12,2.96+i*.76,11.3,.25,14,ink=MUTED)
    s=new_slide(prs,'Thank You / Questions','Team 10 · 23DS7BSBIO',
        'All members. Thank the audience and invite questions. Use the QR code for reproducible code, report, figures and the offline demo. If asked for a result, refer to the measured files rather than recalling an unverified value. For algorithm questions, demonstrate the Evolution Lab or show the unit tests.')
    text(s,'A healthier future.\nOne leaf at a time.',.65,2.62,8.2,1.6,39)
    text(s,'Plantwise · Intelligence, naturally.',.68,4.65,8,.4,18,ink=GREEN)
    picture(s,Path('figures/repo_qr.png'),9.45,2.65,2.8,2.8)
    text(s,'github.com/Suraj-B12/\nBIO-INSPIRED-ALGO---AAT',8.9,5.65,3.8,.6,14,ink=MUTED,align=PP_ALIGN.CENTER)
    speaker_plan=read_json('docs/speaker_notes.json')
    if speaker_plan:
        entries=speaker_plan['slides']
        aligned=(len(entries)==len(prs.slides) and 600<=sum(e['seconds'] for e in entries)<=720
                 and all(e['slide']==m['slide'] and e['title']==m['title'] for e,m in zip(entries,MANIFEST)))
        if not aligned:
            if not args.draft:raise ValueError('Speaker plan must match every slide and cover a 10–12 minute talk.')
            print('Draft uses builder notes until the timed speaker plan matches the new slide order.')
        else:
            for slide,manifest,entry in zip(prs.slides,MANIFEST,entries):
                note=f"{entry['speaker']} — {entry['seconds']} seconds.\n\n{entry['script']}"
                slide.notes_slide.notes_text_frame.text=note;manifest['notes']=note
    elif not args.draft:
        raise ValueError('Final deck requires the complete timed docs/speaker_notes.json plan.')
    destination=ROOT/'slides'/('Plantwise_Design_Preview.pptx' if args.draft else 'BIA_AAT_Team10.pptx');destination.parent.mkdir(exist_ok=True)
    prs.save(destination);save_json('results/slides_manifest.json',{'draft':args.draft,'slides':MANIFEST,'sources':['results/final_metrics.json','results/data_summary.json','results/benchmark_summary.json','report/references.bib']})
    print(f'Saved {len(prs.slides)} slides: {destination}')

if __name__=='__main__':main()
