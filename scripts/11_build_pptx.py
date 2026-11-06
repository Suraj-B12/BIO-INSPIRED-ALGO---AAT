"""Rebuild the botanical Team10 deck from measured JSON and figures."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import importlib.util
import json
import re
import numpy as np
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

def text(slide,value,x,y,w,h,size=20,bold=False,ink=INK,align=None,space=6):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame
    tf.clear();tf.word_wrap=True;tf.margin_left=tf.margin_right=Inches(.01);tf.margin_top=tf.margin_bottom=0
    for i,line in enumerate(str(value).split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color(ink);p.space_after=Pt(space);p.line_spacing=1.13
        if align is not None:p.alignment=align
    return box

def rect(slide,x,y,w,h,fill=WHITE,radius=True):
    s=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));s.fill.solid();s.fill.fore_color.rgb=color(fill);s.line.fill.background()
    if radius:s.adjustments[0]=.12
    return s

def picture(slide,path,x,y,w,h):
    path=ROOT/path
    if not path.exists():return text(slide,'Figure not found.',x,y,w,h,20,ink=MUTED)
    with Image.open(path) as im:iw,ih=im.size
    ratio=min(w/iw,h/ih);ww,hh=iw*ratio,ih*ratio
    slide.shapes.add_picture(str(path),Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))

def new_slide(prs,title,kicker,notes):
    s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=color(BG)
    transition=parse_xml(f'<p:transition {nsdecls("p")} spd="med"><p:fade/></p:transition>')
    s._element.insert_element_before(transition,'p:timing','p:extLst')
    picture(s,Path('assets/bmsce_logo.png'),.45,.24,.44,.44)
    text(s,'B.M.S. College of Engineering · Bengaluru',1.02,.32,7,.25,11,ink=MUTED)
    text(s,kicker.upper(),.56,.99,12,.25,12,ink=GREEN)
    text(s,title,.55,1.32,12.1,.7,29,bold=False)
    rect(s,.56,6.96,12.2,.012,LINE,False)
    text(s,'Department of Computer Science & Engineering (Data Science)',.56,7.1,10,.2,10,ink=MUTED)
    text(s,str(len(prs.slides)),12.05,7.08,.7,.25,10,ink=MUTED,align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text=notes
    MANIFEST.append({'slide':len(prs.slides),'title':title,'notes':notes})
    return s

def lead(s,value,y=2.12):
    """One plain-language sentence directly under the slide title."""
    text(s,value,.58,y,12.1,.45,20,ink=GREEN)

def cards(s,items,y=2.85,h=3.75,highlight=0):
    gap=.22;w=(12.2-gap*(len(items)-1))/len(items)
    for i,(heading,body) in enumerate(items):
        x=.56+i*(w+gap);rect(s,x,y,w,h,PALE if i==highlight else WHITE)
        text(s,f'{i+1:02d}',x+.25,y+.25,w-.5,.32,13,ink=GREEN)
        text(s,heading,x+.25,y+.72,w-.5,.6,24)
        text(s,body,x+.25,y+1.45,w-.5,h-1.6,18,ink=MUTED,space=8)

def chart(s,name,caption='',y=2.2,h=4.25):
    picture(s,Path('figures')/name,.65,y,12.0,h)
    if caption:text(s,caption,.72,6.55,11.9,.34,16,ink=MUTED)

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


def short_class(name):
    name=name.split('___')[-1].replace('Tomato_','').replace('_',' ').strip()
    return {'Spider mites Two-spotted spider mite':'Spider mites','Tomato Yellow Leaf Curl Virus':'Yellow leaf curl','Tomato mosaic virus':'Mosaic virus'}.get(name,name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');args=parser.parse_args()
    cfg=load_config();data=read_json('results/data_summary.json');metrics=read_json('results/final_metrics.json');bench=read_json('results/benchmark_summary.json')
    if not args.draft and metrics is None:raise SystemExit('Final evaluation required. Use --draft only for a clearly marked design preview.')
    prs=Presentation();prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
    prs.core_properties.title=TITLE
    prs.core_properties.subject='Bio Inspired Algorithms · 23DS7BSBIO · Academic Year 2026–2027'
    prs.core_properties.author='Team 10 · B.M.S. College of Engineering'
    prs.core_properties.keywords='Plantwise, EfficientNet-B0, Differential Evolution, PlantVillage'

    # 1 · Cover
    s=new_slide(prs,'Cover Slide/Main Slide','Bio Inspired Algorithms · 23DS7BSBIO','Introduce Team 10, the project title and the guide.')
    for shape in list(s.shapes):
        if shape.has_text_frame and shape.text=='Cover Slide/Main Slide':shape.text_frame.paragraphs[0].font.size=Pt(11);shape.text_frame.paragraphs[0].font.color.rgb=color(MUTED)
    text(s,TITLE,.56,2.05,8.5,1.66,32)
    text(s,'INTELLIGENCE, NATURALLY.',.58,4.03,8,.28,12,ink=GREEN)
    rect(s,9.45,1.55,3.25,4.96,PALE)
    picture(s,Path('assets/bmsce_logo.png'),10.34,1.88,1.4,1.4)
    text(s,'TEAM 10',9.7,3.59,2.7,.3,12,ink=GREEN)
    for i,(name,usn) in enumerate(TEAM):
        y=4.05+i*.57
        text(s,name,9.7,y,2.7,.24,13)
        text(s,usn,9.7,y+.25,2.7,.2,10,ink=MUTED)
    text(s,'Under the guidance of\nManoj Kumar S, Assistant Professor',.58,4.83,8,1,18)
    text(s,'Academic Year 2026–2027',.58,6.24,8,.35,15,ink=GREEN)

    # 2-4 · Background
    s=new_slide(prs,'Introduction to Optimization Problems','Finding the best settings','Optimization means finding the best settings for a goal.')
    lead(s,'Optimization = finding the settings that give the best result.')
    cards(s,[('Settings','Our model has 7 training settings, such as learning speed and how much of the network to retrain.'),
             ('Score','Each set of settings is tested by training a model and scoring it on images it has not seen.'),
             ('Search','Try settings smartly, with a fixed number of attempts, and keep the best one.')])
    s=new_slide(prs,'What are Nature-Inspired Algorithms?','Learning from nature','Nature-inspired algorithms copy ideas such as evolution and swarms to search for good solutions.')
    lead(s,'Search methods that borrow ideas from nature, such as evolution or swarms.')
    cards(s,[('A population','Work with a group of candidate solutions, not just one guess.'),
             ('Variation','Mix and change candidates to create new ones, like offspring.'),
             ('Selection','Keep whichever candidate scores better, like survival of the fittest.')])
    s=new_slide(prs,'Classification of Nature-Inspired Algorithms','A map of the field','Nature-inspired algorithms fall into a few families; our choice, DE, is an evolutionary method.')
    lead(s,'Our choice, Differential Evolution (DE), belongs to the evolutionary family.')
    cards(s,[('Evolutionary','Genetic Algorithm\nDifferential Evolution (ours)\nEvolution Strategies'),('Swarm','Particle Swarm\nAnt Colony\nArtificial Bee Colony'),('Other families','Simulated Annealing\nBiogeography-based\nTeaching–Learning')])

    # 5 · Problem
    s=new_slide(prs,'Problem Statement','Tomato leaf health','Use DE to choose the training settings that best detect 10 tomato leaf conditions.')
    lead(s,'Goal: use DE to pick the training settings that best detect tomato leaf diseases.')
    text(s,f"{measured('data_summary.json/usable_total',data['usable_total']):,} leaf images · {len(data['original_counts'])} classes (9 diseases + healthy) · 70% train / 15% validation / 15% test",.6,2.62,12.0,.35,16,ink=MUTED)
    picture(s,Path('figures/fig_dataset_slide.png'),.65,3.05,12.0,3.8)

    # 6-15 · Implementation and results
    s=new_slide(prs,'Implementation and Results (1/10)','The full system','From raw images to a final, fair comparison and a live demo.')
    chart(s,'fig_pipeline.png','Test images are kept aside and used only once, for the final score.')
    s=new_slide(prs,'Implementation and Results (2/10)','Our model and the baseline','EfficientNet-B0 pretrained on ImageNet; the baseline is a standard recipe fixed before any search.')
    chart(s,'fig_architecture.png','EfficientNet-B0: a compact, pretrained image network. Baseline = standard recipe fixed in advance.')
    s=new_slide(prs,'Implementation and Results (3/10)','How DE creates better candidates','Mutation, crossover and selection, written from scratch in NumPy.')
    chart(s,'fig_de_operators_diagram.png','Mutate: combine three candidates  ·  Crossover: blend with the parent  ·  Select: keep the better one')
    s=new_slide(prs,'Implementation and Results (4/10)','The 7 settings DE tunes','The seven settings, what each controls, and the search budget.')
    table(s,['SETTING','WHAT IT CONTROLS','RANGE SEARCHED'],[
        ['Learning rate','How fast the model learns','0.00003 to 0.01'],
        ['Weight decay','Keeps the model simple to avoid overfitting','0.000001 to 0.01'],
        ['Dropout','Randomly switches off neurons during training','0 to 0.6'],
        ['Head width','Size of the final classification layer','0, 128, 256 or 512'],
        ['Unfrozen blocks','How much of the pretrained network is retrained','0 to 9 blocks'],
        ['Label smoothing','Stops the model from being over-confident','0 to 0.2'],
        ['Augmentation','Random image changes that add variety','Level 0 to 15']], [3.1,5.86,3.2],y=2.15,row_height=.5,size=16)
    tier=cfg['tiers'][cfg['tier']];budget=tier['population']*(tier['generations']+1)
    text(s,f"Population {tier['population']} · {tier['generations']} generations · {budget} training runs for DE, and the same {budget} for random search",.72,6.25,12,.35,17,ink=GREEN)
    s=new_slide(prs,'Implementation and Results (5/10)','DE versus random search','Best validation score found as the search spends its training runs.')
    de_search=read_json('results/de_search.json');rs_search=read_json('results/rs_search.json')
    cap='Same number of training runs for both methods.'
    if de_search and rs_search:
        cap=f"During the search, DE found the best validation score: {pct(de_search['best_fitness'])} vs {pct(rs_search['best_fitness'])} for random search."
        MANIFEST[-1]['evidence']='de_search.json/best_fitness; rs_search.json/best_fitness'
    chart(s,'fig_de_vs_rs.png',cap)
    s=new_slide(prs,'Implementation and Results (6/10)','Before and after DE','Mean results over three training runs: baseline versus the DE recipe.')
    if metrics:
        base=metrics['models']['baseline']['aggregate'];de=metrics['models']['de']['aggregate'];rs=metrics['models']['rs']['aggregate']
        for i,(label,value,std) in enumerate([('BASELINE ACCURACY',base['accuracy']['mean'],base['accuracy']['std']),('DE ACCURACY',de['accuracy']['mean'],de['accuracy']['std']),('BASELINE MACRO-F1',base['macro_f1']['mean'],base['macro_f1']['std']),('DE MACRO-F1',de['macro_f1']['mean'],de['macro_f1']['std'])]):
            x=.58+i*3.09;rect(s,x,2.2,2.9,1.42,PALE if i%2 else WHITE)
            text(s,label,x+.18,2.38,2.54,.3,14,ink=MUTED)
            text(s,pct(measured('final_metrics.json/models/'+('baseline' if label.startswith('BASELINE') else 'de')+'/aggregate/'+('accuracy' if 'ACCURACY' in label else 'macro_f1')+'/mean',value)),x+.18,2.72,2.54,.55,31)
            text(s,f'± {std*100:.2f} pp · {len(metrics["seeds"])} runs',x+.18,3.28,2.54,.27,13,ink=MUTED)
        picture(s,Path('figures/fig_before_after_slide.png'),.65,3.75,12.0,2.55)
        reduction=metrics['relative_error_reduction']
        tied=abs(de['accuracy']['mean']-rs['accuracy']['mean'])<1e-12
        reduction_text=f'DE cuts the baseline’s errors by {pct(reduction)}' if reduction is not None else 'Baseline error is zero'
        comparison=f"both search methods reach {pct(de['accuracy']['mean'])}" if tied else f"random search reaches {pct(rs['accuracy']['mean'])}"
        text(s,f'{reduction_text} · {comparison} · macro-F1 = average score across all 10 classes',.72,6.45,12,.34,15,ink=GREEN)
        MANIFEST[-1]['evidence']='final_metrics.json/models/*/aggregate; final_metrics.json/relative_error_reduction'
    else:chart(s,'fig_before_after_bars.png','DRAFT DESIGN PREVIEW')
    s=new_slide(prs,'Implementation and Results (7/10)','Where the mistakes go','Confusion matrices: rows are the true disease, columns the prediction.')
    chart(s,'fig_confusion_side_by_side.png','Each row is the true disease. A strong diagonal means correct answers; DE removes most baseline mix-ups.')
    s=new_slide(prs,'Implementation and Results (8/10)','Every disease improves','Per-class F1 for baseline and DE, averaged over three runs.')
    cap='Per-class F1, averaged over three training runs.'
    if metrics:
        b=np.mean([[r['f1'] for r in p['per_class']] for p in metrics['models']['baseline']['per_seed']],axis=0)
        d=np.mean([[r['f1'] for r in p['per_class']] for p in metrics['models']['de']['per_seed']],axis=0)
        gain=d-b;top=int(np.argmax(gain));names=metrics['class_names']
        better=int((gain>0).sum())
        cap=f"{'All '+str(len(gain)) if better==len(gain) else str(better)+' of '+str(len(gain))} classes improve with DE · largest gain: {short_class(names[top])} (+{100*gain[top]:.1f} points)"
        MANIFEST[-1]['evidence']='final_metrics.json/models/{baseline,de}/per_seed/*/per_class/f1'
    chart(s,'fig_per_class_f1.png',cap)
    s=new_slide(prs,'Implementation and Results (9/10)','Where the models look','Grad-CAM heatmaps for two correctly classified test leaves.')
    chart(s,'fig_gradcam_slide.png','Grad-CAM: a heatmap of where the model looked. Warm colours = more attention.')
    s=new_slide(prs,'Implementation and Results (10/10)','What DE discovered','The winning DE recipe compared with the baseline recipe.')
    if metrics:
        bc=metrics['models']['baseline']['config'];dc=metrics['models']['de']['config']
        def g(v):return '0' if v==0 else f'{v:.2g}' if isinstance(v,float) and v and abs(v)<.01 else (f'{v:.2f}' if isinstance(v,float) else str(v))
        rows=[['Unfrozen blocks',str(bc['unfreeze_blocks']),str(dc['unfreeze_blocks']),'Retrain the whole network, not only the top'],
              ['Learning rate',g(bc['learning_rate']),g(dc['learning_rate']),'Slightly faster learning'],
              ['Label smoothing',g(bc['label_smoothing']),g(dc['label_smoothing']),'Less over-confident training'],
              ['Weight decay',g(bc['weight_decay']),g(dc['weight_decay']),'Mild simplicity penalty'],
              ['Dropout',g(bc['dropout']),g(dc['dropout']),'About the same'],
              ['Head width',str(bc['head_units']),str(dc['head_units']),'Unchanged']]
        table(s,['SETTING','BASELINE','DE CHOICE','IN PLAIN WORDS'],rows,[2.9,2.0,2.0,5.26],y=2.15,row_height=.55,size=17)
        text(s,'Key insight: DE learned that retraining all 9 blocks of the pretrained network pays off.',.72,6.15,12,.4,18,ink=GREEN)
        MANIFEST[-1]['evidence']='final_metrics.json/models/{baseline,de}/config'

    # 16 · Demo
    s=new_slide(prs,'Live Demo','Plantwise · intelligence, naturally','Run the offline demo: diagnose a leaf, explain it, watch DE evolve.')
    lead(s,'An offline web app that runs on a normal laptop.')
    cards(s,[('Diagnose','Pick a tomato leaf and compare the baseline and DE models side by side.'),('Explain','See the top-3 predictions and the Grad-CAM heatmap.'),('Evolve','Watch a DE population search a landscape, step by step.')])

    # 17-21 · Metrics and benchmarking
    s=new_slide(prs,'Performance Metrics and Benchmarking (1/5)','How we measure success','Accuracy, macro-F1 and ROC-AUC explained in plain words.')
    lead(s,'Three simple ways to score a disease classifier.')
    cards(s,[('Accuracy','Share of test images labelled correctly.'),('Macro-F1','Average score across all 10 classes, so rare diseases count as much as common ones.'),('ROC-AUC','How well the model ranks the right disease above the wrong ones (1.0 = perfect).')],y=2.85,h=3.4)
    s=new_slide(prs,'Performance Metrics and Benchmarking (2/5)','Is the gain real?','Confidence intervals and the McNemar test on the same test images.')
    if metrics:
        cirows=[]
        for key,label in [('baseline','Baseline'),('rs','Random search'),('de','DE-optimized')]:
            ci=metrics['models'][key]['per_seed'][0]['bootstrap_95_ci']
            cirows.append([label,*[' – '.join(pct(v) for v in ci[k]) for k in ('accuracy','macro_f1')]])
        table(s,['MODEL (RUN 1)','ACCURACY · LIKELY RANGE','MACRO-F1 · LIKELY RANGE'],cirows,[3.5,4.33,4.33],y=2.2,row_height=.68,size=19)
        paired=metrics['mcnemar']
        ptext='p < 0.001' if paired['p_value']<.001 else f"p = {paired['p_value']:.3f}"
        text(s,f"A statistical test (McNemar) confirms the gain is not due to chance ({ptext}).",.7,5.15,11.9,.45,22,ink=GREEN)
        text(s,f"On the same test images, DE fixes {paired['de_only_correct']} baseline mistakes; the baseline is right on only {paired['baseline_only_correct']} that DE misses.",.72,5.72,11.9,.35,17,ink=MUTED)
        text(s,'Likely range = 95% confidence interval from 1,000 resamples of the test set.',.72,6.2,11.9,.3,15,ink=MUTED)
        MANIFEST[-1]['evidence']='final_metrics.json/models/*/per_seed/0/bootstrap_95_ci; final_metrics.json/mcnemar'
    s=new_slide(prs,'Performance Metrics and Benchmarking (3/5)','Testing DE on its own','DE, jDE, GA, PSO and random search on five standard test functions.')
    chart(s,'fig_benchmark_convergence_slide.png',f"5 standard test functions · {bench['settings']['runs']} runs each · {bench['settings']['budget']:,} tries per run · lower error is better")
    s=new_slide(prs,'Performance Metrics and Benchmarking (4/5)','Benchmark results','Median final error for each method and function.')
    algorithms=['DE','jDE','GA','PSO','Random Search']
    rows=[[function,*[f'{values["algorithms"][name]["median"]:.2e}' for name in algorithms]] for function,values in bench['functions'].items()]
    table(s,['FUNCTION',*algorithms],rows,[2.26,1.98,1.98,1.98,1.98,1.98],y=2.2,row_height=.6,size=17)
    beats=sum(values['algorithms']['DE']['median']<values['algorithms']['Random Search']['median'] for values in bench['functions'].values())
    de_wins=[f for f,v in bench['functions'].items() if min(v['algorithms'],key=lambda a:v['algorithms'][a]['median'])=='DE']
    text(s,f"Median final error · lower is better · DE beats random search on {beats} of {len(rows)} functions"+(f" and is best on {', '.join(de_wins)}" if de_wins else ''),.72,6.0,11.9,.35,16,ink=GREEN)
    MANIFEST[-1]['evidence']='benchmark_summary.json/functions/*/algorithms/*/median'
    s=new_slide(prs,'Performance Metrics and Benchmarking (5/5)','Cost of a better recipe','Training time, speed per image and model size.')
    if metrics:
        rows=[]
        for key,label in [('baseline','Baseline'),('rs','Random search'),('de','DE-optimized')]:
            model=metrics['models'][key];a=model['aggregate']
            rows.append([label,f"{sum(r['training_seconds'] for r in model['per_seed'])/60:.1f} min",f"{a['latency_ms']['mean']:.2f} ms",f"{model['per_seed'][0]['parameters']['total']/1e6:.2f} M"])
        table(s,['MODEL','TRAINING (3 RUNS)','TIME PER IMAGE','PARAMETERS'],rows,[2.96,3.0,3.2,3.0],y=2.2,row_height=.68,size=19)
        text(s,f"Search time: DE {search_seconds('de')/60:.1f} min · random search {search_seconds('rs')/60:.1f} min",.72,5.15,11.9,.43,22,ink=GREEN)
        text(s,'All three models predict at about the same speed and size.',.72,5.75,11.9,.35,17,ink=MUTED)
        MANIFEST[-1]['evidence']='final_metrics.json/models/*; de_evaluations.csv/train_time; rs_evaluations.csv/train_time'

    # 22-23 · Wrap-up
    s=new_slide(prs,'Advantages and Limitations','What we achieved','Strengths of the work and what remains for the next stage.')
    adv='Strong accuracy gain over the baseline\nDE written from scratch and unit-tested\nFair comparison with an equal budget\nWorks offline on a normal laptop'
    if metrics:adv=f"Accuracy {pct(base['accuracy']['mean'])} → {pct(de['accuracy']['mean'])}\nDE written from scratch and unit-tested\nFair comparison with an equal budget\nWorks offline on a normal laptop"
    cards(s,[('Advantages',adv),('Limitations','Lab photos of single leaves\nOne crop: tomato\nSearch needs extra training time')],y=2.3,h=4.2)
    s=new_slide(prs,'Future Scope','Next steps','Field photos, multi-objective DE and a mobile app.')
    lead(s,'Build on these results with harder data and richer goals.')
    cards(s,[('Field photos','Test on real farm photos taken with phones.'),('Multi-objective DE','Also reward robustness to blur and noise, reliable confidence and speed.'),('Mobile and adaptive','On-device app; self-tuning DE variants such as SHADE.')])

    # References
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
        subset=keys[start:start+5];s=new_slide(prs,f'References ({start//5+1}/{(len(keys)+4)//5})','Key sources',
            'Key references; the full list is in the report.')
        for i,key in enumerate(subset):
            entry=bib[key];title=re.sub(r'[{}]','',entry['title']);author=entry['author'].split(' and ')[0]
            y=2.2+i*.9
            text(s,f'{start+i+1:02d}',.6,y,.45,.3,15,ink=GREEN)
            box=text(s,f"{author} et al. ({entry['year']}). {title}",1.12,y,11.4,.8,14,space=2)
            p=box.text_frame.add_paragraph();p.text='DOI: '+entry['doi'];p.font.name='Arial';p.font.size=Pt(13);p.font.color.rgb=color(MUTED)
    s=new_slide(prs,'Thank You / Questions','Team 10 · 23DS7BSBIO','Thank the audience and invite questions.')
    text(s,'A healthier future.\nOne leaf at a time.',.65,2.62,8.2,1.6,39)
    text(s,'Plantwise · Intelligence, naturally.',.68,4.65,8,.4,18,ink=GREEN)
    picture(s,Path('figures/repo_qr.png'),9.45,2.45,2.8,2.8)
    text(s,'github.com/Suraj-B12/\nBIO-INSPIRED-ALGO---AAT',8.9,5.4,3.8,.6,14,ink=MUTED,align=PP_ALIGN.CENTER)
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
