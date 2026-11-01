"""Offline Plantwise demo. Start with python demo/serve.py; no external assets."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import base64
from io import BytesIO
import json
import threading
import time
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title='Plantwise · Team 10', docs_url='/api/docs')
STATIC = ROOT / 'demo' / 'static'
EXAMPLES = ROOT / 'demo' / 'examples'
EXAMPLES.mkdir(parents=True, exist_ok=True)
app.mount('/static', StaticFiles(directory=STATIC), name='static')
app.mount('/examples', StaticFiles(directory=EXAMPLES), name='examples')
LOCK = threading.Lock()
MODELS = {}
MAX_UPLOAD = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 24_000_000


def read_result(name, default=None):
    path = ROOT / 'results' / name
    if not path.exists(): return default
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError): return default


def checkpoint(name):
    return ROOT / 'models' / f'{name}_s0.pt'


def model_ready(name):
    # Best-epoch checkpoints exist before a run finishes. The training summary
    # is written only after completion, so its presence is required for serving.
    record = read_result(f'{name}_s0.json', {})
    if not isinstance(record, dict):
        return False
    epochs = record.get('epochs_trained')
    return (checkpoint(name).is_file() and record.get('seed') == 0
            and isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0
            and str(record.get('checkpoint', '')).replace('\\', '/') == f'models/{name}_s0.pt')


def label_name(name):
    return name.replace('Tomato___', '').replace('_', ' ').replace('  ', ' ').strip()


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/api/status')
def status():
    classes = read_result('class_names.json', [])
    if isinstance(classes, dict): classes = classes.get('class_names', list(classes))
    final = read_result('final_metrics.json')
    return {'app': 'Plantwise', 'course': '23DS7BSBIO', 'year': '2026–2027',
            'classes': classes, 'class_count': len(classes),
            'models': {n: model_ready(n) for n in ['baseline', 'de', 'rs']},
            'evaluation_ready': isinstance(final, dict) and final.get('status') == 'complete',
            'data': read_result('data_summary.json'),
            'system': read_result('system_info.json'),
            'search_ready': (ROOT / 'results/de_history.json').exists(),
            'benchmarks_ready': (ROOT / 'results/benchmarks.json').exists()}


@app.get('/api/results')
def results():
    return {'metrics': read_result('final_metrics.json'), 'frozen': read_result('frozen_protocol.json'),
            'benchmarks': read_result('benchmark_summary.json'), 'verification': read_result('test_report.json')}


@app.get('/api/search')
def search():
    return {'history': read_result('de_history.json', (read_result('de_state.json') or {}).get('history')), 'rerank': read_result('de_rerank.json')}


@app.get('/api/examples')
def examples():
    manifest = EXAMPLES / 'index.json'
    if manifest.exists(): return json.loads(manifest.read_text(encoding='utf-8'))
    return []


class EvolutionRequest(BaseModel):
    function: str = 'Rastrigin'
    population: int = Field(default=28, ge=4, le=80)
    generations: int = Field(default=70, ge=1, le=160)
    F: float = Field(default=.6, gt=0, le=2)
    CR: float = Field(default=.9, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**32-1)


@app.post('/api/evolve')
def evolve(request: EvolutionRequest):
    from bioalgo.de import optimize
    from bioalgo.benchmarks import FUNCTIONS, unit_objective
    if request.function not in FUNCTIONS: raise HTTPException(422, 'Unknown benchmark function')
    result = optimize(unit_objective(request.function), dimensions=2,
        population_size=request.population, generations=request.generations,
        F=request.F, CR=request.CR, seed=request.seed, record_positions=True)
    spec = FUNCTIONS[request.function]
    lo, hi = spec['bounds']
    grid = np.linspace(lo, hi, 64)
    xx, yy = np.meshgrid(grid, grid)
    z = spec['function'](np.stack([xx, yy], axis=-1))
    return {'function': request.function, 'bounds': [lo, hi], 'landscape': z.tolist(),
            'optimum': spec['optimum_coordinate'], 'history': result['history'],
            'evaluations': result['evaluations'], 'best_value': -result['best_fitness']}


def parse_image(raw):
    if not raw or len(raw) > MAX_UPLOAD: raise HTTPException(413, 'Choose an image smaller than 12 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            source = Image.open(BytesIO(raw))
            if source.format not in ('JPEG', 'PNG', 'WEBP'): raise HTTPException(415, 'Please use a JPEG, PNG, or WebP image.')
            if source.width < 32 or source.height < 32: raise HTTPException(422, 'The image must be at least 32 × 32 pixels.')
            source.load()
            return ImageOps.exif_transpose(source).convert('RGB')
    except HTTPException: raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(422, 'This image could not be opened. Please choose a valid image under 24 megapixels.')


def encode_image(image):
    stream = BytesIO(); image.save(stream, format='JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()


@app.post('/api/predict')
def predict(file: UploadFile = File(...), corruption: str = Form('none'), severity: int = Form(1)):
    if corruption not in ('none', 'blur', 'noise', 'brightness', 'jpeg', 'low_light') or severity not in (1, 2, 3):
        raise HTTPException(422, 'Invalid stress-test setting.')
    raw = file.file.read(MAX_UPLOAD + 1)
    image = parse_image(raw)
    if not all(model_ready(n) for n in ['baseline', 'de']):
        raise HTTPException(503, 'The comparison models are still being trained. The Evolution Lab is ready to explore.')
    with LOCK:
        from bioalgo.model import load_checkpoint, predict_image
        from bioalgo.data import display_crop
        # CPU inference keeps the live showcase available alongside GPU training.
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        if corruption != 'none':
            image = corrupt_image(image, corruption, severity)
        response = {'models': [], 'input': encode_image(ImageOps.contain(image, (800, 800))),
                    'corruption': corruption, 'severity': severity}
        for name in ('baseline', 'de'):
            signature = checkpoint(name).stat().st_mtime_ns
            if name not in MODELS or MODELS[name][0] != signature:
                model, metadata = load_checkpoint(checkpoint(name), 'cpu')
                MODELS[name] = (signature, model, metadata)
            _, model, metadata = MODELS[name]
            predicted = predict_image(model, image, resolution=224, explain=True)
            names = metadata['class_names']
            probabilities = predicted['probabilities']
            top = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]
            cam = predicted.pop('heatmap')
            crop = display_crop(image, 224)
            from matplotlib import colormaps
            heat = colormaps['inferno'](cam)[..., :3]
            overlay_pixels = np.clip(np.asarray(crop) * .60 + heat * 255 * .40, 0, 255)
            # A contour complements hue; CAM values themselves remain untouched.
            above = cam >= .5
            boundary = np.zeros_like(above)
            boundary[1:] |= above[1:] != above[:-1]
            boundary[:, 1:] |= above[:, 1:] != above[:, :-1]
            overlay_pixels[boundary] = 255
            overlay = Image.fromarray(np.uint8(overlay_pixels))
            response['models'].append({'name': name, 'label': label_name(names[top[0]]),
                'confidence': probabilities[top[0]], 'top3': [{'label': label_name(names[i]), 'probability': probabilities[i]} for i in top],
                'latency_ms': predicted['latency_ms'], 'gradcam': encode_image(overlay),
                'checkpoint': checkpoint(name).name})
        return response


def corrupt_image(image, name, severity):
    # Same deterministic corruption function as final evaluation, mapped for the UI.
    from bioalgo.data import corrupt_image as apply_corruption
    return apply_corruption(image, name, severity, seed=42)


DOWNLOADS = {'report': ROOT / 'report/BIA_AAT_Team10_Report.pdf',
             'slides': ROOT / 'slides/BIA_AAT_Team10.pptx',
             'slides-pdf': ROOT / 'slides/BIA_AAT_Team10.pdf'}


@app.get('/api/download/{kind}')
def download(kind: str):
    path = DOWNLOADS.get(kind)
    if path is None or not path.exists(): raise HTTPException(404, 'This document is not available yet.')
    return FileResponse(path, filename=path.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the offline Plantwise demo')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    import uvicorn
    print(f'Plantwise is ready at http://{args.host}:{args.port}', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
