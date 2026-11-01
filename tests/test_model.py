import pytest
import torch
from bioalgo.common import seed_everything
from bioalgo.model import build_model,training_mode,parameter_counts,gradcam


@pytest.mark.parametrize('unfreeze',[0,3,9])
def test_freezing_and_batchnorm(unfreeze):
    seed_everything(4);model=build_model({'unfreeze_blocks':unfreeze},pretrained=False);training_mode(model)
    expected=sum(p.numel() for p in model.classifier.parameters())
    for index,block in enumerate(model.features):
        trainable=index>=9-unfreeze
        assert all(p.requires_grad==trainable for p in block.parameters())
        if trainable:expected+=sum(p.numel() for p in block.parameters())
        for layer in block.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):assert layer.training==trainable
    assert parameter_counts(model)['trainable']==expected


@pytest.mark.parametrize('units',[0,128,256,512])
def test_forward_output_shape(units):
    seed_everything(1);model=build_model({'head_units':units},pretrained=False).eval()
    with torch.no_grad():output=model(torch.rand(2,3,64,64))
    assert output.shape==(2,10) and torch.isfinite(output).all()


def test_gradcam_frozen_backbone_no_hook_leak():
    seed_everything(2);model=build_model(pretrained=False).eval();tensor=torch.rand(1,3,64,64)
    for _ in range(2):
        cam,prob=gradcam(model,tensor)
        assert cam.shape==(64,64) and prob.shape==(10,)
        assert 0<=cam.min()<=cam.max()<=1
    assert len(model.features[-1]._forward_hooks)==0


@pytest.mark.parametrize('unfreeze',[0,3,9])
def test_package_gradcam_preserves_predictions_weights_and_input(unfreeze):
    import numpy as np
    seed_everything(12);model=build_model({'unfreeze_blocks':unfreeze},pretrained=False).eval()
    tensor=torch.rand(1,3,64,64);before={key:value.clone() for key,value in model.state_dict().items()}
    with torch.no_grad():expected=model(tensor).softmax(1)[0].numpy()
    with torch.inference_mode():cam,probabilities=gradcam(model,tensor)
    np.testing.assert_allclose(probabilities,expected,rtol=1e-6,atol=1e-7)
    assert np.isfinite(cam).all() and cam.shape==(64,64) and 0<=cam.min()<=cam.max()<=1
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not tensor.requires_grad and tensor.grad is None
    assert not model.features[-1]._forward_hooks


def test_package_gradcam_releases_hooks_on_failure(monkeypatch):
    from pytorch_grad_cam import GradCAM
    seed_everything(3);model=build_model(pretrained=False).eval()
    def fail(*args,**kwargs):raise RuntimeError('Simulated explanation failure')
    monkeypatch.setattr(GradCAM,'__call__',fail)
    with pytest.raises(RuntimeError,match='Simulated'):gradcam(model,torch.rand(1,3,64,64))
    assert not model.features[-1]._forward_hooks
    assert all(parameter.grad is None for parameter in model.parameters())


def test_package_gradcam_rejects_invalid_target_without_hooks():
    model=build_model(pretrained=False).eval()
    with pytest.raises(ValueError,match='class range'):gradcam(model,torch.rand(1,3,64,64),target=10)
    assert not model.features[-1]._forward_hooks


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS hardware not available')
def test_package_gradcam_mps_batch_one():
    import numpy as np
    seed_everything(31);model=build_model(pretrained=False).to('mps').eval();tensor=torch.rand(1,3,64,64,device='mps')
    with torch.no_grad():expected=model(tensor).softmax(1)[0].cpu().numpy()
    cam,probabilities=gradcam(model,tensor)
    np.testing.assert_allclose(probabilities,expected,rtol=1e-6,atol=1e-7)
    assert cam.shape==(64,64) and np.isfinite(cam).all() and 0<=cam.min()<=cam.max()<=1
    assert not model.features[-1]._forward_hooks
    assert all(parameter.grad is None for parameter in model.parameters())
    del model,tensor
    torch.mps.empty_cache()
