"""Numerical evaluation helpers; these functions never load any data split."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, f1_score, roc_curve


def calibration(labels, probabilities, bins=15):
    confidence=probabilities.max(1);correct=probabilities.argmax(1)==labels
    edges=np.linspace(0,1,bins+1);rows=[];ece=0.
    for index in range(bins):
        mask=(confidence>=edges[index]) & (confidence < edges[index+1] if index<bins-1 else confidence<=edges[index+1])
        count=int(mask.sum());acc=float(correct[mask].mean()) if count else 0.;conf=float(confidence[mask].mean()) if count else 0.
        rows.append({'lower':float(edges[index]),'upper':float(edges[index+1]),'count':count,'accuracy':acc,'confidence':conf});ece+=count/len(labels)*abs(acc-conf)
    return {'ece':float(ece),'bins':rows}


def classification_metrics(labels, probabilities, class_names):
    labels=np.asarray(labels);probabilities=np.asarray(probabilities)
    if probabilities.shape != (len(labels),len(class_names)):raise ValueError('Probability shape does not match labels and classes')
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities')
    predicted=probabilities.argmax(1);result={'accuracy':float(accuracy_score(labels,predicted))}
    for average in ('macro','weighted'):
        p,r,f,_=precision_recall_fscore_support(labels,predicted,labels=range(len(class_names)),average=average,zero_division=0)
        result.update({f'{average}_precision':float(p),f'{average}_recall':float(r),f'{average}_f1':float(f)})
    p,r,f,support=precision_recall_fscore_support(labels,predicted,labels=range(len(class_names)),zero_division=0)
    result['per_class']=[{'class_name':name,'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i]),'support':int(support[i])} for i,name in enumerate(class_names)]
    result['roc_auc']=float(roc_auc_score(labels,probabilities,multi_class='ovr',average='macro',labels=range(len(class_names))))
    result['confusion_matrix']=confusion_matrix(labels,predicted,labels=range(len(class_names))).tolist()
    result['top3_accuracy']=float(np.mean(np.any(np.argsort(probabilities,axis=1)[:,-3:]==labels[:,None],axis=1)))
    result['calibration']=calibration(labels,probabilities)
    return result


def bootstrap_ci(labels, probabilities, samples=1000, seed=42):
    rng=np.random.default_rng(seed);labels=np.asarray(labels);predicted=np.asarray(probabilities).argmax(1);accuracies=[];scores=[]
    for _ in range(samples):
        chosen=rng.integers(0,len(labels),len(labels));y=labels[chosen];p=predicted[chosen]
        accuracies.append(float(np.mean(y==p)));scores.append(float(f1_score(y,p,labels=range(probabilities.shape[1]),average='macro',zero_division=0)))
    return {'samples':samples,'seed':seed,'accuracy':np.quantile(accuracies,[.025,.975]).tolist(),'macro_f1':np.quantile(scores,[.025,.975]).tolist()}


def roc_data(labels, probabilities):
    grid=np.linspace(0,1,201);rows=[]
    for i in range(probabilities.shape[1]):
        fpr,tpr,_=roc_curve(np.asarray(labels)==i,probabilities[:,i]);rows.append({'class_index':i,'fpr':fpr.tolist(),'tpr':tpr.tolist()})
    return {'per_class':rows,'macro_fpr':grid.tolist(),'macro_tpr':np.mean([np.interp(grid,row['fpr'],row['tpr']) for row in rows],axis=0).tolist()}
