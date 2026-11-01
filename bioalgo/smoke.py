"""Fast offline end-to-end plumbing check, intentionally not a scientific result."""
from .common import seed_everything,save_json,timestamp
from .model import build_model,training_mode,parameter_counts,gradcam,load_checkpoint
from .training import atomic_checkpoint
import tempfile
from pathlib import Path
import time
import numpy as np
import torch


def run_smoke():
    seed_everything(42);started=time.perf_counter()
    recipe={'learning_rate':.001,'weight_decay':0.,'dropout':.2,'head_units':128,'unfreeze_blocks':0,'label_smoothing':0.,'augmentation':0}
    model=build_model(recipe,pretrained=False);training_mode(model)
    images=torch.rand(2,3,64,64);labels=torch.tensor([0,1]);optimizer=torch.optim.Adam(model.classifier.parameters(),lr=.001)
    logits=model(images);loss=torch.nn.functional.cross_entropy(logits,labels);loss.backward();optimizer.step()
    model.eval()
    with torch.no_grad():before=model(images)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'smoke.pt';atomic_checkpoint(path,{'architecture':'efficientnet_b0','state_dict':model.state_dict(),'config':recipe,'class_names':[str(i) for i in range(10)]})
        restored,_=load_checkpoint(path)
        with torch.no_grad():after=restored(images)
        if not torch.equal(before,after):raise AssertionError('Checkpoint round trip changed predictions')
        cam,prob=gradcam(restored,images[:1])
        if cam.shape!=(64,64) or not np.isfinite(cam).all() or not np.isclose(prob.sum(),1):raise AssertionError('Invalid Grad-CAM or probabilities')
    result={'status':'passed','kind':'synthetic pipeline verification; not model accuracy','timestamp':timestamp(),'seconds':time.perf_counter()-started,'batch_shape':list(images.shape),'output_shape':list(logits.shape),'finite_loss':bool(torch.isfinite(loss)),'checkpoint_round_trip_exact':True,'gradcam_shape':list(cam.shape),'parameters':parameter_counts(model),'network_required':False}
    return result
