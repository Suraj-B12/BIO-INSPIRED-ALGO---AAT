"""One transparent training loop shared by baseline, search and final recipes."""
from .common import ROOT, load_config, read_json, save_json, seed_everything, pick_device, synchronize, timestamp
from .data import load_rows, subset_rows, make_loader
from .model import build_model, training_mode, parameter_counts
import copy
import gc
import hashlib
import json
import math
import os
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()



def training_controls(settings):
    return {'warmup_epochs':int(settings.get('warmup_epochs',1)),'grad_clip':float(settings.get('grad_clip',1.)),'num_workers':int(settings.get('num_workers',0))}


def compatible_signature(signature, identity, config, name):
    """Accept only the audited initial-baseline metadata format as a legacy case.

    Initial baseline training began before controls were added to the signature.
    Its immutable preregistration records those exact controls. Never extend this
    exception to searches, changed settings, or an unproved resume checkpoint.
    """
    if signature==fingerprint(identity):return True
    legacy={key:value for key,value in identity.items() if key!='training_controls'}
    preregistration=read_json('results/baseline_preregistration.json')
    return (name=='baseline' and signature==fingerprint(legacy) and preregistration is not None
            and preregistration.get('training')==config['training']
            and preregistration.get('config')==identity['recipe']
            and identity['seed'] in preregistration.get('seeds',[]))


def split_fingerprint(config):
    from .common import resolve_path
    directory = resolve_path(config.get('paths',{}).get('splits','data/splits'))
    # Selection fingerprints deliberately exclude the held-out manifest.
    return fingerprint({name:hashlib.sha256((directory/f'{name}.csv').read_bytes()).hexdigest() for name in ('train','val')})


def atomic_checkpoint(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value,temp)
    os.replace(temp,path)


def evaluate_loader(model, loader, device, criterion=None, embeddings=False):
    model.eval(); truths, probabilities, vectors = [], [], []; loss_sum=0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device); labels = labels.to(device)
            if embeddings:
                features = model.avgpool(model.features(images)).flatten(1)
                logits = model.classifier(features); vectors.extend(features.cpu().tolist())
            else: logits = model(images)
            if not torch.isfinite(logits).all(): raise RuntimeError('Non-finite model predictions')
            if criterion is not None: loss_sum += float(criterion(logits,labels).cpu()) * len(labels)
            probabilities.extend(logits.softmax(1).cpu().tolist()); truths.extend(labels.cpu().tolist())
    y = np.asarray(truths); p = np.asarray(probabilities); predicted = p.argmax(1)
    return {'loss':loss_sum/len(y),'accuracy':float(accuracy_score(y,predicted)),'macro_f1':float(f1_score(y,predicted,labels=range(10),average='macro',zero_division=0)),'labels':y,'probabilities':p,'embeddings':np.asarray(vectors)}


def train_recipe(config, recipe, name, seed, *, resolution=None, fraction=1., epochs=None, save_model=True, resume=True, quiet=False):
    settings = config['training']; resolution=int(resolution or settings['resolution']); epochs=int(epochs or settings['epochs'])
    if epochs<1 or resolution<32 or int(settings['batch_size'])<2 or int(settings.get('patience',4))<1:raise ValueError('Training requires positive epochs/patience, resolution>=32 and batch_size>=2')
    recipe = dict(recipe); device=pick_device(); seed_everything(seed)
    identity={'recipe':recipe,'seed':int(seed),'resolution':resolution,'fraction':fraction,'epochs':epochs,'batch_size':settings['batch_size'],'patience':settings.get('patience',4),'split_fingerprint':split_fingerprint(config),'architecture':'efficientnet_b0','training_controls':training_controls(settings)}
    signature=fingerprint(identity)
    result_path=ROOT/'results'/f'{name}_s{seed}.json'; curve_path=ROOT/'results'/f'curves_{name}_s{seed}.json'
    checkpoint_path=ROOT/'models'/f'{name}_s{seed}.pt'; resume_path=ROOT/'models'/f'.{name}_s{seed}.resume.pt'
    existing=read_json(result_path)
    if resume and existing:
        if not compatible_signature(existing.get('signature'),identity,config,name): raise RuntimeError(f'{result_path.name} belongs to a different protocol; archive previous results before rerunning')
        if not save_model or checkpoint_path.exists(): return existing
    train_rows=subset_rows(load_rows('train',config),fraction,seed=int(config.get('split_seed',42)))
    val_rows=load_rows('val',config)
    train_loader=make_loader(train_rows,config,resolution,train=True,augmentation=recipe.get('augmentation',0),seed=seed)
    val_loader=make_loader(val_rows,config,resolution,seed=seed)
    model=build_model(recipe,pretrained=True).to(device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer_name=recipe.get('optimizer','adamw').lower()
    optimizer=(torch.optim.Adam if optimizer_name=='adam' else torch.optim.AdamW)(parameters,lr=float(recipe['learning_rate']),weight_decay=float(recipe['weight_decay']))
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=float(recipe.get('label_smoothing',0)))
    curve=[]; best=-1.; best_epoch=0; waiting=0; start_epoch=0; elapsed_previous=0.; best_state=None
    if resume and resume_path.exists():
        state=torch.load(resume_path,map_location='cpu',weights_only=True)
        if not compatible_signature(state['signature'],identity,config,name): raise RuntimeError('Resume checkpoint protocol mismatch')
        if state['signature']!=signature:
            # Preserve the original signature/checkpoint metadata; controls are
            # verified against immutable preregistration by the compatibility gate.
            signature=state['signature'];identity.pop('training_controls')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        # load_state_dict migrates parameter buffers while retaining CPU step scalars.
        curve=state['curve'];best=state['best'];best_epoch=state['best_epoch'];waiting=state['waiting'];start_epoch=state['epoch']+1;elapsed_previous=state['elapsed'];best_state=state['best_state']
        torch.set_rng_state(state['torch_rng']);train_loader.generator.set_state(state['loader_rng'])
        if device.type=='cuda' and state.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        if device.type=='mps' and state.get('mps_rng') is not None: torch.mps.set_rng_state(state['mps_rng'])
    if waiting>=int(settings.get('patience',4)):start_epoch=epochs
    started=time.perf_counter(); steps_per_epoch=len(train_loader); total_steps=epochs*steps_per_epoch; warmup=max(1,steps_per_epoch*int(settings.get('warmup_epochs',1)))
    for epoch in range(start_epoch,epochs):
        training_mode(model); train_loss=0.; correct=0; count=0; epoch_start=time.perf_counter()
        for step,(images,labels) in enumerate(train_loader):
            global_step=epoch*steps_per_epoch+step
            if global_step<warmup: factor=(global_step+1)/warmup
            else: factor=.5*(1+math.cos(math.pi*(global_step-warmup)/max(1,total_steps-warmup)))
            for group in optimizer.param_groups: group['lr']=float(recipe['learning_rate'])*factor
            images=images.to(device);labels=labels.to(device);optimizer.zero_grad(set_to_none=True)
            logits=model(images);loss=criterion(logits,labels)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,float(settings.get('grad_clip',1.)));optimizer.step()
            train_loss+=float(loss.detach().cpu())*len(labels);correct+=int((logits.argmax(1)==labels).sum().cpu());count+=len(labels)
        validation=evaluate_loader(model,val_loader,device,criterion)
        row={'epoch':epoch+1,'train_loss':train_loss/count,'train_accuracy':correct/count,'val_loss':validation['loss'],'val_accuracy':validation['accuracy'],'val_macro_f1':validation['macro_f1'],'learning_rate':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-epoch_start}
        curve.append(row);save_json(curve_path,curve)
        if validation['macro_f1']>best:
            best=validation['macro_f1'];best_epoch=epoch+1;waiting=0
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            if save_model:
                atomic_checkpoint(checkpoint_path,{'architecture':'efficientnet_b0','state_dict':best_state,'config':recipe,'class_names':read_json('results/class_names.json'),'seed':int(seed),'resolution':resolution,'signature':signature,'trained_at':timestamp(),'best_validation_macro_f1':best,'epoch':best_epoch,'protocol':identity})
        else: waiting+=1
        if save_model:
            atomic_checkpoint(resume_path,{'signature':signature,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'curve':curve,'best':best,'best_epoch':best_epoch,'waiting':waiting,'epoch':epoch,'elapsed':elapsed_previous+time.perf_counter()-started,'best_state':best_state,'torch_rng':torch.get_rng_state(),'loader_rng':train_loader.generator.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None})
        if not quiet: print(f"{name}/seed{seed} epoch {epoch+1}/{epochs}: val F1={best:.4f}, loss={validation['loss']:.4f}, {row['seconds']:.1f}s",flush=True)
        if waiting>=int(settings.get('patience',4)): break
    result={'name':name,'seed':int(seed),'signature':signature,'config':recipe,'protocol':identity,'validation_macro_f1':best,'validation_accuracy':curve[best_epoch-1]['val_accuracy'],'validation_loss':curve[best_epoch-1]['val_loss'],'best_epoch':best_epoch,'epochs_trained':len(curve),'training_seconds':elapsed_previous+time.perf_counter()-started,'device':str(device),'parameter_counts':parameter_counts(model),'train_images':len(train_rows),'validation_images':len(val_rows),'checkpoint':str(checkpoint_path.relative_to(ROOT)) if save_model else None,'timestamp':timestamp()}
    save_json(result_path,result)
    if resume_path.exists(): resume_path.unlink()
    del model,optimizer,best_state,train_loader,val_loader
    gc.collect()
    if device.type=='mps': torch.mps.empty_cache()
    elif device.type=='cuda': torch.cuda.empty_cache()
    return result
