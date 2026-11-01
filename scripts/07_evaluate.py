"""The only experiment script allowed to open held-out images.

Requires a frozen protocol and all nine checkpoints. Each completed component is
persisted so an interruption can resume without using test results for selection.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bioalgo.common import ROOT,load_config,read_json,save_json,pick_device,seed_everything,synchronize,timestamp,resolve_path
from bioalgo.data import load_rows,make_loader,image_path,evaluation_transform,CORRUPTIONS,corrupt_image,display_crop
from bioalgo.model import load_checkpoint,parameter_counts,gradcam
from bioalgo.training import evaluate_loader,fingerprint,split_fingerprint,training_controls,compatible_signature
from bioalgo.evaluation import classification_metrics,bootstrap_ci,roc_data
import hashlib
import time
import numpy as np
import torch
from PIL import Image,ImageOps
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar


def latency(model,tensor,device):
    with torch.no_grad():
        for _ in range(5):model(tensor)
        synchronize(device);start=time.perf_counter()
        for _ in range(30):model(tensor)
        synchronize(device)
    return (time.perf_counter()-start)/30*1000


def main():
    config=load_config();frozen=read_json('results/frozen_protocol.json');classes=read_json('results/class_names.json');device=pick_device();seed_everything(42)
    if frozen is None:raise FileNotFoundError('Final recipes must be frozen by scripts/06_train_final.py first')
    if config['baseline']!=frozen['recipes']['baseline']:raise RuntimeError('Baseline differs from frozen protocol')
    preregistered=read_json('results/baseline_preregistration.json')
    if not preregistered or preregistered['config']!=config['baseline']:raise RuntimeError('Baseline preregistration mismatch')
    if config['training']!=frozen['training'] or config['seeds']!=frozen['seeds']:raise RuntimeError('Final training protocol differs from frozen settings')
    checkpoints={f'{name}_s{seed}':ROOT/'models'/f'{name}_s{seed}.pt' for name in ('baseline','de','rs') for seed in config['seeds']}
    missing=[str(path) for path in checkpoints.values() if not path.exists()]
    if missing:raise FileNotFoundError('Missing final checkpoints: '+', '.join(missing))
    # Inspect checkpoint protocol before touching held-out data.
    for name in ('baseline','de','rs'):
        for seed in config['seeds']:
            result=read_json(f'results/{name}_s{seed}.json')
            if not result or result['config']!=frozen['recipes'][name] or result['protocol']['resolution']!=frozen['training']['resolution']:raise RuntimeError(f'{name}/seed{seed} was trained under a different protocol')
            expected_protocol={'recipe':frozen['recipes'][name],'seed':int(seed),'resolution':frozen['training']['resolution'],'fraction':1.,'epochs':frozen['training']['epochs'],'batch_size':frozen['training']['batch_size'],'patience':frozen['training'].get('patience',4),'split_fingerprint':split_fingerprint(config),'architecture':'efficientnet_b0','training_controls':training_controls(frozen['training'])}
            if fingerprint(result['protocol'])!=result['signature'] or not compatible_signature(result['signature'],expected_protocol,config,name):raise RuntimeError(f'{name}/seed{seed} training settings or split fingerprint mismatch')
            checkpoint=torch.load(checkpoints[f'{name}_s{seed}'],map_location='cpu',weights_only=True)
            if checkpoint['protocol']!=result['protocol']:raise RuntimeError(f'{name}/seed{seed} checkpoint protocol differs from its result')
            if checkpoint['config']!=frozen['recipes'][name] or checkpoint['signature']!=result['signature'] or checkpoint['seed']!=seed or checkpoint['resolution']!=frozen['training']['resolution'] or checkpoint['class_names']!=classes:raise RuntimeError(f'{name}/seed{seed} checkpoint provenance mismatch')
            if checkpoint['protocol']['fraction']!=1. or checkpoint['protocol']['epochs']!=frozen['training']['epochs']:raise RuntimeError('Final checkpoint must use the complete training split and fixed epoch budget')
            del checkpoint
    split_path=resolve_path(config['paths']['splits'])/'test.csv'
    identity={'frozen_signature':frozen['signature'],'heldout_manifest_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),'checkpoints':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in checkpoints.items()}}
    signature=fingerprint(identity);previous=read_json('results/evaluation_lock.json')
    if previous and previous['signature']!=signature:raise RuntimeError('The held-out set was already opened for different checkpoints; a new protocol requires a new untouched test set')
    if read_json('results/final_metrics.json'):
        print('Final evaluation already completed; committed results reused without reopening held-out images.');return
    if previous is None:save_json('results/evaluation_lock.json',{'signature':signature,**identity,'opened_at':timestamp()})
    rows=load_rows('test',config,allow_test=True);resolution=config['training']['resolution'];loader=make_loader(rows,config,resolution)
    output={'schema_version':1,'status':'complete','timestamp':timestamp(),'evaluation_signature':signature,'class_names':classes,'test_images':len(rows),'seeds':config['seeds'],'models':{},'robustness':{},'roc':{},'protocol':frozen}
    for name in ('baseline','de','rs'):
        per_seed=[]
        for seed in config['seeds']:
            identifier=f'{name}_s{seed}';metrics_path=f'results/evaluation_{identifier}.json';metrics=read_json(metrics_path)
            if metrics is None:
                model,metadata=load_checkpoint(checkpoints[identifier],device)
                prediction=evaluate_loader(model,loader,device,embeddings=(seed==0 and name in ('baseline','de')))
                y,p=prediction['labels'],prediction['probabilities'];metrics=classification_metrics(y,p,classes)
                tensor=next(iter(loader))[0][:1].to(device)
                metrics.update({'seed':seed,'latency_device':str(device),'latency_batch_size':1,'latency_repetitions':30,'latency_ms':latency(model,tensor,device),'parameters':parameter_counts(model),'model_size_mb':checkpoints[identifier].stat().st_size/1024**2,'training_seconds':read_json(f'results/{identifier}.json')['training_seconds']})
                if seed==0:metrics['bootstrap_95_ci']=bootstrap_ci(y,p,samples=1000)
                np.savez_compressed(ROOT/'results'/f'predictions_{identifier}.npz',labels=y,probabilities=p,embeddings=prediction['embeddings'])
                save_json(metrics_path,metrics);del model
                if device.type=='mps':torch.mps.empty_cache()
                print(f"Final evaluation {identifier}: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}",flush=True)
            per_seed.append(metrics)
        keys=['accuracy','macro_precision','macro_recall','macro_f1','weighted_precision','weighted_recall','weighted_f1','roc_auc','top3_accuracy','latency_ms','training_seconds']
        aggregate={key:{'mean':float(np.mean([row[key] for row in per_seed])),'std':float(np.std([row[key] for row in per_seed],ddof=1)) if len(per_seed)>1 else 0.} for key in keys}
        output['models'][name]={'per_seed':per_seed,'aggregate':aggregate,'config':frozen['recipes'][name]}
        saved=np.load(ROOT/'results'/f'predictions_{name}_s0.npz');output['roc'][name]=roc_data(saved['labels'],saved['probabilities'])
    base=np.load(ROOT/'results'/'predictions_baseline_s0.npz');de=np.load(ROOT/'results'/'predictions_de_s0.npz');y=base['labels'];base_correct=base['probabilities'].argmax(1)==y;de_correct=de['probabilities'].argmax(1)==y
    table=[[int((base_correct&de_correct).sum()),int((base_correct&~de_correct).sum())],[int((~base_correct&de_correct).sum()),int((~base_correct&~de_correct).sum())]]
    discordant=table[0][1]+table[1][0]
    test=mcnemar(table,exact=True) if discordant else None
    output['mcnemar']={'table':table,'baseline_only_correct':table[0][1],'de_only_correct':table[1][0],'p_value':float(test.pvalue) if test else 1.,'method':'exact two-sided binomial','seed':0}
    err_base=1-output['models']['baseline']['aggregate']['accuracy']['mean'];err_de=1-output['models']['de']['aggregate']['accuracy']['mean']
    output['relative_error_reduction']=(err_base-err_de)/err_base if err_base>0 else None
    for name in ('baseline','de'):
        model,_=load_checkpoint(checkpoints[f'{name}_s0'],device);robust=read_json(f'results/robustness_{name}.json',{})
        for kind in CORRUPTIONS:
            robust.setdefault(kind,{})
            for severity in (1,2,3):
                if str(severity) in robust[kind]:continue
                stress_loader=make_loader(rows,config,resolution,corruption=kind,severity=severity)
                prediction=evaluate_loader(model,stress_loader,device)
                robust[kind][str(severity)]={'accuracy':prediction['accuracy'],'macro_f1':prediction['macro_f1']}
                save_json(f'results/robustness_{name}.json',robust)
                print(f'Stress test {name} {kind}/{severity}: {prediction["accuracy"]:.4f}',flush=True)
        output['robustness'][name]={'results':robust,'mean_accuracy':float(np.mean([robust[k][str(s)]['accuracy'] for k in CORRUPTIONS for s in (1,2,3)]))}
        del model
        if device.type=='mps':torch.mps.empty_cache()
    qualitative(rows,config,classes,device)
    save_json('results/final_metrics.json',output)
    print('Final held-out evaluation complete. All output statistics were measured; test data remains excluded from selection.')


def qualitative(rows,config,classes,device):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples=ROOT/'demo'/'examples';examples.mkdir(parents=True,exist_ok=True)
    chosen=[];manifest=[]
    for label,name in enumerate(classes):
        row=next(row for row in rows if int(row['label'])==label);chosen.append(row)
        with Image.open(image_path(row,config)) as image:
            image=image.convert('RGB');image.thumbnail((320,320));filename=f'{label:02d}.jpg';image.save(examples/filename,quality=85)
        manifest.append({'id':label,'file':filename,'class_name':name,'label':label,'source_path':row['path'],'split':'test'})
    save_json(examples/'index.json',manifest)
    save_json('results/demo_examples.json',manifest)
    fig,axes=plt.subplots(6,3,figsize=(10,17));records=[];asset_images=[];asset_cams=[]
    models={name:load_checkpoint(ROOT/'models'/f'{name}_s0.pt',device)[0] for name in ('baseline','de')}
    for index,row in enumerate(chosen[:6]):
        with Image.open(image_path(row,config)) as opened:image=opened.convert('RGB')
        display=display_crop(image,224);asset_images.append(np.asarray(display));leaf_cams=[];axes[index,0].imshow(display);axes[index,0].set_title(row['class_name'].replace('Tomato___','').replace('_',' '),fontsize=9)
        record={'source_path':row['path'],'true_label':int(row['label'])}
        for col,(name,model) in enumerate(models.items(),start=1):
            tensor=evaluation_transform(224)(image).unsqueeze(0).to(device);cam,prob=gradcam(model,tensor);leaf_cams.append(cam.astype(np.float32))
            axes[index,col].imshow(display);axes[index,col].imshow(cam,cmap='jet',alpha=.42);axes[index,col].set_title(f"{name}: {classes[int(prob.argmax())].replace('Tomato___','').replace('_',' ')}\n{float(prob.max()):.1%}",fontsize=8)
            record[name]={'predicted_index':int(prob.argmax()),'probability':float(prob.max())}
        for axis in axes[index]:axis.axis('off')
        records.append(record);asset_cams.append(leaf_cams)
    fig.tight_layout();fig.savefig(ROOT/'figures'/'fig_gradcam_grid.png',dpi=300);fig.savefig(ROOT/'figures'/'fig_gradcam_grid.pdf');plt.close(fig)
    save_json('results/qualitative.json',records)
    saved=np.load(ROOT/'results'/'predictions_de_s0.npz');pred=saved['probabilities'].argmax(1);wrong=np.flatnonzero(pred!=saved['labels'])
    selected=list(wrong[:3])+[i for i in range(len(rows)) if i not in set(wrong[:3])][:12-min(3,len(wrong))]
    fig,axes=plt.subplots(3,4,figsize=(13,10));prediction_records=[];prediction_pixels=[]
    for axis,index in zip(axes.flat,selected):
        row=rows[index]
        with Image.open(image_path(row,config)) as image:
            pixels=np.asarray(display_crop(image,224));prediction_pixels.append(pixels);axis.imshow(pixels)
        correct=int(row['label'])==int(pred[index]);axis.set_title(f"True: {classes[int(row['label'])].replace('Tomato___','').replace('_',' ')}\nPred: {classes[int(pred[index])].replace('Tomato___','').replace('_',' ')}",fontsize=8,color='#147D55' if correct else '#B43232');axis.axis('off')
        prediction_records.append({'path':row['path'],'true_label':int(row['label']),'prediction':int(pred[index]),'correct':correct})
    fig.tight_layout();fig.savefig(ROOT/'figures'/'fig_predictions_grid.png',dpi=300);fig.savefig(ROOT/'figures'/'fig_predictions_grid.pdf');plt.close(fig);save_json('results/prediction_examples.json',prediction_records)
    with Image.open(image_path(chosen[0],config)) as opened:image=opened.convert('RGB')
    fig,axes=plt.subplots(5,3,figsize=(9,14));corruption_pixels=[]
    for i,kind in enumerate(CORRUPTIONS):
        for j in range(3):
            pixels=np.asarray(display_crop(corrupt_image(image,kind,j+1),224));corruption_pixels.append(pixels);axes[i,j].imshow(pixels);axes[i,j].set_title(f'{kind.replace("_"," ")} · severity {j+1}',fontsize=10);axes[i,j].axis('off')
    np.savez_compressed(ROOT/'results'/'qualitative_assets.npz',gradcam_images=np.asarray(asset_images),gradcam_maps=np.asarray(asset_cams),prediction_images=np.asarray(prediction_pixels),corruption_images=np.asarray(corruption_pixels).reshape(5,3,224,224,3))
    fig.tight_layout();fig.savefig(ROOT/'figures'/'fig_corruptions.png',dpi=300);fig.savefig(ROOT/'figures'/'fig_corruptions.pdf');plt.close(fig)

if __name__=='__main__':main()
