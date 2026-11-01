"""Validation-only search fitness, durable cache, complete evaluation provenance."""
from .common import ROOT,load_config,read_json,save_json,timestamp,pick_device
from .training import train_recipe,fingerprint,split_fingerprint
from .de import decode
import csv
import json
import math
import time



def cnn_de_controls(config):
    settings=config['de']
    if settings.get('adaptive',False):raise ValueError('jDE is restricted to benchmark experiments; CNN search uses DE/rand/1/bin')
    dither=settings.get('dither',False)
    if dither is True:dither=(.5,.9)
    elif dither is False or dither is None:dither=None
    elif not isinstance(dither,(list,tuple)) or len(dither)!=2:raise ValueError('Dither must be false, true, or a two-value range')
    return {'F':settings['F'],'CR':settings['CR'],'adaptive':False,'dither':dither}


def setup_search(config,method,fresh=False):
    tier=config['tiers'][config['tier']]
    protocol={'tier':tier,'split_fingerprint':split_fingerprint(config),'seed':int(config.get('split_seed',42)),'training':config['training'],'search_space_version':1}
    identity=fingerprint(protocol)
    protocol_path=ROOT/'results'/f'{method}_protocol.json'
    previous=read_json(protocol_path)
    if previous is None and (ROOT/'results'/f'{method}_state.json').exists() and not fresh:raise RuntimeError('Search state exists without its protocol fingerprint; archive it or explicitly restart fresh')
    if previous and previous['signature']!=identity and not fresh: raise ValueError('Search protocol changed; use --fresh after archiving results')
    save_json(protocol_path,{'signature':identity,**protocol})
    cache_path=ROOT/'results'/'search_cache.json'
    cache=read_json(cache_path,{})
    log_path=ROOT/'results'/f'{method}_evaluations.csv'
    if fresh and log_path.exists(): log_path.unlink()
    pending={}
    def objective(genes):
        recipe=decode(genes)
        # Rounding is only a cache identity; rounded values are also actually trained.
        recipe={k:round(v,10) if isinstance(v,float) else v for k,v in recipe.items()}
        key=fingerprint({'protocol':identity,'recipe':recipe})
        started=time.perf_counter()
        if key in cache: record=dict(cache[key]);record['cached']=True
        else:
            try:
                result=train_recipe(config,recipe,f'proxy_{key[:12]}',protocol['seed'],resolution=tier['resolution'],fraction=tier['fraction'],epochs=tier['epochs'],save_model=False,quiet=True)
                record={'recipe':recipe,'fitness':result['validation_macro_f1'],'val_accuracy':result['validation_accuracy'],'val_loss':result['validation_loss'],'train_time':result['training_seconds'],'cached':False,'error':None}
            except RuntimeError as exc:
                if not any(term in str(exc).lower() for term in ('out of memory','non-finite','nan')): raise
                record={'recipe':recipe,'fitness':0.,'val_accuracy':0.,'val_loss':None,'train_time':time.perf_counter()-started,'cached':False,'error':str(exc)}
            cache[key]=record;save_json(cache_path,cache)
        pending.clear();pending.update(record)
        return float(record['fitness'])
    records_path=ROOT/'results'/f'{method}_evaluation_records.json'
    records=read_json(records_path,{}) if not fresh else {}
    if fresh: save_json(records_path,{})
    def callback(event):
        recipe=decode(event['genes'])
        recipe={k:round(v,10) if isinstance(v,float) else v for k,v in recipe.items()}
        key=fingerprint({'protocol':identity,'recipe':recipe})
        cached_record=cache.get(key)
        if cached_record is None: raise RuntimeError('Evaluation cache missing during durable log reconciliation')
        record={'evaluation':event['evaluation'],'generation':event['generation'],'individual':event['individual'],**{f'x{i+1}':float(x) for i,x in enumerate(event['genes'])},**recipe,'fitness':event['fitness'],'val_accuracy':cached_record['val_accuracy'],'val_loss':cached_record['val_loss'],'train_time':cached_record['train_time'],'cached':pending.get('cached',True),'error':cached_record['error'],'timestamp':timestamp(),'device':str(pick_device())}
        index=str(event['evaluation'])
        is_new=index not in records
        if is_new:
            records[index]=record;save_json(records_path,records)
        # The JSON ledger is authoritative. Rebuild even for an already-seen event:
        # a crash may have occurred after its durable write but before CSV replace.
        ordered=[records[key] for key in sorted(records,key=int)]
        temp=log_path.with_suffix('.csv.tmp')
        with temp.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(ordered[0]));writer.writeheader();writer.writerows(ordered);stream.flush();__import__('os').fsync(stream.fileno())
        __import__('os').replace(temp,log_path)
        if is_new:print(f"{method.upper()} evaluation {event['evaluation']}: validation macro-F1 {event['fitness']:.4f}",flush=True)
    return objective,callback,tier


def rerank(config,method):
    path=ROOT/'results'/f'{method}_evaluations.csv'
    with path.open(encoding='utf-8',newline='') as stream: rows=list(csv.DictReader(stream))
    rows.sort(key=lambda row:float(row['fitness']),reverse=True)
    seen=set();ranking=[]
    for row in rows:
        recipe=decode([float(row[f'x{i+1}']) for i in range(7)])
        recipe={k:round(v,10) if isinstance(v,float) else v for k,v in recipe.items()}
        key=fingerprint(recipe)
        if key in seen: continue
        seen.add(key)
        result=train_recipe(config,recipe,f'{method}_rerank_{len(ranking)}',int(config.get('split_seed',42)),resolution=config['training']['resolution'],epochs=int(config.get('de',{}).get('rerank_epochs',3)),save_model=False)
        ranking.append({'config':recipe,'validation_macro_f1':result['validation_macro_f1'],'training_seconds':result['training_seconds'],'proxy_macro_f1':float(row['fitness'])})
        if len(ranking)>=int(config.get('de',{}).get('rerank_top',3)): break
    ranking.sort(key=lambda row:row['validation_macro_f1'],reverse=True)
    output={'method':method,'ranking':ranking,'best_config':ranking[0]['config'],'selection_split':'val','timestamp':timestamp()}
    save_json(f'results/{method}_rerank.json',output)
    return output
