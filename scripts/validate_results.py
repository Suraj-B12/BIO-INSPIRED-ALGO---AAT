"""Recompute final statistics from saved predictions without reopening test images."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bioalgo.common import ROOT,load_config,read_json,save_json,timestamp
from bioalgo.evaluation import classification_metrics
from bioalgo.data import CORRUPTIONS
import csv
import hashlib
import json
import math
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

METRICS=['accuracy','macro_precision','macro_recall','macro_f1','weighted_precision','weighted_recall','weighted_f1','roc_auc','top3_accuracy']


def compare_predictions(saved,labels,probabilities,classes):
    actual=classification_metrics(labels,probabilities,classes)
    for key in METRICS:
        if not math.isclose(saved[key],actual[key],rel_tol=1e-12,abs_tol=1e-12):raise AssertionError(f'{key}: saved value does not match predictions')
    if saved['confusion_matrix']!=actual['confusion_matrix']:raise AssertionError('Confusion matrix does not match predictions')
    if len(saved['per_class'])!=len(classes):raise AssertionError('Missing per-class metrics')
    for first,second in zip(saved['per_class'],actual['per_class']):
        if first['class_name']!=second['class_name'] or first['support']!=second['support']:raise AssertionError('Per-class label/support mismatch')
        for key in ('precision','recall','f1'):
            if not math.isclose(first[key],second[key],rel_tol=1e-12,abs_tol=1e-12):raise AssertionError(f'Per-class {key} mismatch')
    if saved['calibration']!=actual['calibration']:raise AssertionError('Calibration bins or ECE mismatch')
    return actual


def require_finite(value,path='root'):
    if isinstance(value,dict):
        for key,child in value.items():require_finite(child,f'{path}.{key}')
    elif isinstance(value,list):
        for index,child in enumerate(value):require_finite(child,f'{path}[{index}]')
    elif isinstance(value,float) and not math.isfinite(value):raise AssertionError(f'Non-finite value at {path}')


def main():
    config=load_config();final=read_json('results/final_metrics.json')
    if not final or final.get('status')!='complete':raise SystemExit('Final experiment results are not complete yet.')
    require_finite(final);classes=read_json('results/class_names.json');frozen=read_json('results/frozen_protocol.json');lock=read_json('results/evaluation_lock.json')
    if final['class_names']!=classes or len(classes)!=10:raise AssertionError('Class mapping mismatch')
    if final['seeds']!=[0,1,2] or final['protocol']!=frozen:raise AssertionError('Final protocol/seeds mismatch')
    if final['evaluation_signature']!=lock['signature']:raise AssertionError('Evaluation signature mismatch')
    hashes={};predictions={};checks=[]
    for method in ('baseline','de','rs'):
        model=final['models'][method]
        if [row['seed'] for row in model['per_seed']]!=[0,1,2]:raise AssertionError('Missing or reordered seed')
        if model['config']!=frozen['recipes'][method]:raise AssertionError('Final recipe mismatch')
        for row in model['per_seed']:
            identifier=f"{method}_s{row['seed']}";path=ROOT/'results'/f'predictions_{identifier}.npz'
            saved=np.load(path);labels=saved['labels'];probabilities=saved['probabilities']
            if len(labels)!=final['test_images']:raise AssertionError('Held-out count mismatch')
            compare_predictions(row,labels,probabilities,classes)
            if not np.array_equal(labels,predictions.get('labels',labels)):raise AssertionError('Prediction rows are not aligned across models')
            predictions['labels']=labels;predictions[identifier]=probabilities
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoint=ROOT/'models'/f'{identifier}.pt'
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=lock['checkpoints'][identifier]:raise AssertionError('Checkpoint changed after final evaluation')
            checks.append(f'{identifier}: metrics recomputed from saved probabilities; checkpoint unchanged')
        for key,summary in model['aggregate'].items():
            values=[row[key] for row in model['per_seed']]
            if not math.isclose(summary['mean'],float(np.mean(values)),abs_tol=1e-12) or not math.isclose(summary['std'],float(np.std(values,ddof=1)),abs_tol=1e-12):raise AssertionError(f'{method} aggregate {key} mismatch')
        interval=model['per_seed'][0]['bootstrap_95_ci']
        if interval['samples']!=1000:raise AssertionError('Incorrect bootstrap count')
        for key in ('accuracy','macro_f1'):
            if not 0<=interval[key][0]<=interval[key][1]<=1:raise AssertionError('Invalid confidence interval')
    labels=predictions['labels'];base=predictions['baseline_s0'].argmax(1)==labels;de=predictions['de_s0'].argmax(1)==labels
    table=[[int((base&de).sum()),int((base&~de).sum())],[int((~base&de).sum()),int((~base&~de).sum())]]
    p=float(mcnemar(table,exact=True).pvalue) if table[0][1]+table[1][0] else 1.
    if final['mcnemar']['table']!=table or not math.isclose(final['mcnemar']['p_value'],p,abs_tol=1e-15):raise AssertionError('McNemar result mismatch')
    for method in ('baseline','de'):
        robust=final['robustness'][method]['results']
        if sorted(robust)!=sorted(CORRUPTIONS):raise AssertionError('Missing corruption')
        values=[]
        for kind in CORRUPTIONS:
            if sorted(robust[kind])!=['1','2','3']:raise AssertionError('Missing corruption severity')
            for severity in ('1','2','3'):
                score=robust[kind][severity]['accuracy']
                if not 0<=score<=1:raise AssertionError('Invalid robustness accuracy')
                values.append(score)
        if not math.isclose(final['robustness'][method]['mean_accuracy'],float(np.mean(values)),abs_tol=1e-12):raise AssertionError('Robustness mean mismatch')
    error_base=1-final['models']['baseline']['aggregate']['accuracy']['mean'];error_de=1-final['models']['de']['aggregate']['accuracy']['mean']
    if error_base>0 and not math.isclose(final['relative_error_reduction'],(error_base-error_de)/error_base,abs_tol=1e-12):raise AssertionError('Error reduction mismatch')
    tier=config['tiers'][config['tier']];budget=tier['population']*(tier['generations']+1)
    for method in ('de','rs'):
        with (ROOT/'results'/f'{method}_evaluations.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
        if [int(row['evaluation']) for row in rows]!=list(range(1,budget+1)):raise AssertionError(f'{method}: incomplete or duplicated search budget')
        state=read_json(f'results/{method}_state.json');events=state['evaluation_events'] if method=='de' else state['evaluations']
        if len(events)!=budget:raise AssertionError(f'{method}: durable evaluation count mismatch')
        for row,event in zip(rows,events):
            if float(row['fitness'])!=event['fitness'] or not np.array_equal([float(row[f'x{i+1}']) for i in range(7)],event['genes']):raise AssertionError('CSV differs from durable search events')
        checks.append(f'{method}: exactly {budget} candidates, complete durable event and CSV agreement')
    audit={'status':'passed','timestamp':timestamp(),'test_images':final['test_images'],'models_checked':9,'checks':checks,'prediction_file_sha256':hashes,'heldout_images_reopened':False,'scope':'Metrics, aggregates, McNemar, confidence-interval bounds, robustness completeness, unchanged checkpoints and equal search budgets.'}
    save_json('results/result_validation.json',audit);print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
