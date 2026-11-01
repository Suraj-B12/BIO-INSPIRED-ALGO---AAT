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
