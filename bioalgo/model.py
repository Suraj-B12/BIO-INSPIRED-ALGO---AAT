"""EfficientNet-B0 transfer learning and genuine gradient-based explanations."""
from .common import ROOT
import time
import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(config=None, pretrained=True, num_classes=10):
    config = config or {}
    units = int(config.get('head_units', config.get('units', 256)))
    dropout = float(config.get('dropout', .2))
    unfreeze = int(config.get('unfreeze_blocks', config.get('unfreeze', 0)))
    if units not in (0, 128, 256, 512): raise ValueError('Invalid hidden units')
    if not 0 <= dropout <= 1: raise ValueError('Invalid dropout')
    if not 0 <= unfreeze <= 9: raise ValueError('Unfreeze must be between 0 and 9')
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    if units:
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, units), nn.SiLU(), nn.Dropout(dropout), nn.Linear(units, num_classes))
    else:
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
    for index, block in enumerate(model.features):
        for parameter in block.parameters(): parameter.requires_grad = index >= 9 - unfreeze
    model._unfreeze = unfreeze
    model._recipe = dict(config)
    return model


def training_mode(model):
    """Frozen blocks must keep BatchNorm statistics and stochastic depth fixed."""
    model.train()
    for block in list(model.features)[:9 - model._unfreeze]: block.eval()
    return model


def parameter_counts(model):
    return {'total': sum(p.numel() for p in model.parameters()), 'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def model_summary(model, resolution=224):
    rows = []
    model.eval()
    with torch.no_grad():
        value = torch.zeros(1, 3, resolution, resolution, device=next(model.parameters()).device)
        for index, block in enumerate(model.features):
            value = block(value)
            rows.append({'block': index, 'output_shape': list(value.shape), **parameter_counts(block)})
    rows.append({'block': 'classifier', 'output_shape': [1, 10], **parameter_counts(model.classifier)})
    return rows


def load_checkpoint(path, device='cpu'):
    """Load only a project-generated weights checkpoint; no network needed."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'efficientnet_b0': raise ValueError('Unsupported checkpoint architecture')
    model = build_model(checkpoint['config'], pretrained=False, num_classes=len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval(), checkpoint


def gradcam(model, tensor, target=None):
    """Use pytorch-grad-cam on the final convolution, including frozen backbones."""
    import os
    cache_directory = ROOT / 'work' / 'matplotlib'
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache_directory))
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError('Grad-CAM expects one image tensor with shape (1, C, H, W)')
    if target is not None:
        target = int(target)
        if not 0 <= target < model.classifier[-1].out_features:
            raise ValueError('Grad-CAM target is outside the model class range')
    model.eval()
    try:
        # The input needs gradients even when every backbone parameter is frozen.
        # This changes explanation graph construction only, never model weights.
        with torch.inference_mode(False), torch.enable_grad():
            image_tensor = tensor.detach().clone().requires_grad_(True)
            targets = None if target is None else [ClassifierOutputTarget(target)]
            with GradCAM(model=model, target_layers=[model.features[-1]]) as explanation:
                heatmap = explanation(input_tensor=image_tensor, targets=targets,
                                      aug_smooth=False, eigen_smooth=False)[0]
                probabilities = explanation.outputs.detach().softmax(1)[0].cpu().numpy()
            return np.asarray(heatmap, dtype=np.float32), probabilities
    finally:
        model.zero_grad(set_to_none=True)


def predict_image(model, image, resolution=224, explain=True):
    from .data import evaluation_transform
    from .common import synchronize
    device = next(model.parameters()).device
    tensor = evaluation_transform(resolution)(image.convert('RGB')).unsqueeze(0).to(device)
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(): probabilities = model(tensor).softmax(1)[0].cpu().numpy()
    synchronize(device)
    latency = (time.perf_counter() - started) * 1000
    heatmap = gradcam(model, tensor)[0] if explain else None
    return {'probabilities': probabilities.tolist(), 'predicted_index': int(np.argmax(probabilities)), 'latency_ms': latency, 'heatmap': heatmap}
