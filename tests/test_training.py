"""Exercise resumable training state with a small real differentiable model."""
import torch
import pytest
from torch.utils.data import TensorDataset,DataLoader
import bioalgo.training as training


@pytest.mark.parametrize('device_name',['cpu','mps'])
def test_interrupted_training_matches_uninterrupted(tmp_path,monkeypatch,device_name):
    if device_name=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS hardware not available')
    monkeypatch.setattr(training,'ROOT',tmp_path)
    monkeypatch.setattr(training,'split_fingerprint',lambda config:'fixed-training-and-validation')
    monkeypatch.setattr(training,'pick_device',lambda:torch.device(device_name))
    monkeypatch.setattr(training,'load_rows',lambda split,config:[{'path':str(i),'label':i%10} for i in range(20)])
    monkeypatch.setattr(training,'build_model',lambda recipe,pretrained:torch.nn.Sequential(torch.nn.Linear(6,12),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(12,10)))
    monkeypatch.setattr(training,'training_mode',lambda model:model.train())
    original_read=training.read_json
    monkeypatch.setattr(training,'read_json',lambda path,default=None:[str(i) for i in range(10)] if str(path)=='results/class_names.json' else original_read(path,default))
    images=torch.linspace(0,1,120).reshape(20,6);labels=torch.arange(20)%10
    def loader(rows,config,resolution,train=False,augmentation=0,seed=42,**kwargs):
        return DataLoader(TensorDataset(images,labels),batch_size=5,shuffle=train,generator=torch.Generator().manual_seed(seed))
    monkeypatch.setattr(training,'make_loader',loader)
    config={'training':{'resolution':32,'epochs':3,'batch_size':5,'patience':4,'warmup_epochs':1},'split_seed':42}
    recipe={'learning_rate':.01,'weight_decay':0.,'optimizer':'adam'}
    reference=training.train_recipe(config,recipe,'reference',7,quiet=True)
    original_evaluate=training.evaluate_loader;counter=[0]
    def interrupted(*args,**kwargs):
        counter[0]+=1
        if counter[0]==2:raise RuntimeError('Simulated process interruption')
        return original_evaluate(*args,**kwargs)
    monkeypatch.setattr(training,'evaluate_loader',interrupted)
    with pytest.raises(RuntimeError,match='Simulated'):training.train_recipe(config,recipe,'resumed',7,quiet=True)
    assert (tmp_path/'models'/'.resumed_s7.resume.pt').exists()
    monkeypatch.setattr(training,'evaluate_loader',original_evaluate)
    resumed=training.train_recipe(config,recipe,'resumed',7,quiet=True)
    assert reference['validation_macro_f1']==resumed['validation_macro_f1']
    assert reference['best_epoch']==resumed['best_epoch']
    first=torch.load(tmp_path/'models'/'reference_s7.pt',weights_only=True)['state_dict']
    second=torch.load(tmp_path/'models'/'resumed_s7.pt',weights_only=True)['state_dict']
    assert all(torch.equal(first[key],second[key]) for key in first)
    assert not (tmp_path/'models'/'.resumed_s7.resume.pt').exists()
    changed={**recipe,'learning_rate':.005}
    with pytest.raises(RuntimeError,match='different protocol'):training.train_recipe(config,changed,'resumed',7,quiet=True)


def test_legacy_baseline_identity_requires_exact_preregistration(monkeypatch):
    settings={'epochs':15,'batch_size':32,'patience':4,'warmup_epochs':1,'grad_clip':1.,'num_workers':0}
    recipe={'learning_rate':.001};config={'training':settings}
    identity={'recipe':recipe,'seed':0,'resolution':224,'fraction':1.,'epochs':15,'batch_size':32,'patience':4,'split_fingerprint':'same','architecture':'efficientnet_b0','training_controls':training.training_controls(settings)}
    legacy={key:value for key,value in identity.items() if key!='training_controls'}
    signature=training.fingerprint(legacy)
    preregistration={'config':recipe,'training':dict(settings),'seeds':[0,1,2]}
    monkeypatch.setattr(training,'read_json',lambda path:preregistration)
    assert training.compatible_signature(signature,identity,config,'baseline')
    assert not training.compatible_signature(signature,identity,config,'de')
    for key,value in [('warmup_epochs',2),('grad_clip',.5),('num_workers',2),('batch_size',16),('patience',3)]:
        altered={'training':{**settings,key:value}}
        assert not training.compatible_signature(signature,identity,altered,'baseline')
    changed={**identity,'split_fingerprint':'different'}
    assert not training.compatible_signature(signature,changed,config,'baseline')
