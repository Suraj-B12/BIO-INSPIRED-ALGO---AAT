import csv
import numpy as np
import pytest
import torch
import bioalgo.search as search


def setup(tmp_path,monkeypatch,trainer):
    monkeypatch.setattr(search,'ROOT',tmp_path)
    monkeypatch.setattr(search,'split_fingerprint',lambda config:'train-val-fingerprint')
    monkeypatch.setattr(search,'train_recipe',trainer)
    monkeypatch.setattr(search,'pick_device',lambda:torch.device('cpu'))
    return {'tier':'C','tiers':{'C':{'resolution':112,'fraction':.1,'epochs':2}},'training':{'batch_size':32},'split_seed':42}


def test_fitness_cache_and_callback_crash_reconciliation(tmp_path,monkeypatch):
    calls=[]
    def trainer(*args,**kwargs):
        calls.append(args[1]);return {'validation_macro_f1':.75,'validation_accuracy':.8,'validation_loss':.3,'training_seconds':2.}
    config=setup(tmp_path,monkeypatch,trainer)
    objective,callback,_=search.setup_search(config,'de')
    genes=np.full(7,.5);event={'evaluation':1,'generation':0,'individual':0,'genes':genes.tolist(),'fitness':.75}
    assert objective(genes)==.75
    assert objective(genes)==.75
    assert len(calls)==1
    # Simulate the optimizer persisting its event before the process lost CSV output.
    _,recovered,_=search.setup_search(config,'de')
    recovered(event);recovered(event)
    with (tmp_path/'results'/'de_evaluations.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==1 and rows[0]['fitness']=='0.75'


def test_failed_training_is_explicit_finite_zero(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('MPS backend out of memory')
    config=setup(tmp_path,monkeypatch,fail);objective,callback,_=search.setup_search(config,'de')
    genes=np.zeros(7);assert objective(genes)==0
    callback({'evaluation':1,'generation':0,'individual':0,'genes':genes.tolist(),'fitness':0.})
    with (tmp_path/'results'/'de_evaluations.csv').open(encoding='utf-8',newline='') as stream:row=next(csv.DictReader(stream))
    assert 'out of memory' in row['error'] and row['fitness']=='0.0'


def test_unknown_training_failure_is_not_hidden(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('Unexpected tensor size')
    config=setup(tmp_path,monkeypatch,fail);objective,_,_=search.setup_search(config,'de')
    with pytest.raises(RuntimeError,match='Unexpected'):objective(np.zeros(7))


def test_ledger_recovers_missing_or_stale_csv(tmp_path,monkeypatch):
    def trainer(*args,**kwargs):return {'validation_macro_f1':.75,'validation_accuracy':.8,'validation_loss':.3,'training_seconds':2.}
    config=setup(tmp_path,monkeypatch,trainer);objective,callback,_=search.setup_search(config,'de')
    genes=np.full(7,.3);objective(genes)
    event={'evaluation':1,'generation':0,'individual':0,'genes':genes.tolist(),'fitness':.75}
    callback(event)
    csv_path=tmp_path/'results'/'de_evaluations.csv';csv_path.unlink()
    # Ledger survived, CSV replacement did not. Resume must rebuild it.
    _,replay,_=search.setup_search(config,'de');replay(event)
    assert csv_path.exists()
    csv_path.write_text('stale partial CSV',encoding='utf-8');replay(event)
    with csv_path.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==1 and rows[0]['evaluation']=='1' and rows[0]['fitness']=='0.75'


def test_cnn_controls_wire_dither_and_reject_jde():
    config={'de':{'F':.6,'CR':.9,'dither':True,'adaptive':False}}
    assert search.cnn_de_controls(config)['dither']==(.5,.9)
    config['de']['dither']=[.4,.8]
    assert search.cnn_de_controls(config)['dither']==[.4,.8]
    config['de']['dither']=False
    assert search.cnn_de_controls(config)['dither'] is None
    config['de']['adaptive']=True
    with pytest.raises(ValueError,match='restricted'):search.cnn_de_controls(config)
