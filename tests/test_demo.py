"""API behavior, actual optimizer output, input boundaries and offline assets."""
from io import BytesIO
import base64
from pathlib import Path
import importlib.util
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

spec=importlib.util.spec_from_file_location('plantwise_server',Path(__file__).resolve().parents[1]/'demo/serve.py')
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
client=TestClient(server.app)


def image_bytes(size=(64,64),format='PNG'):
    stream=BytesIO();Image.new('RGB',size,(50,140,60)).save(stream,format=format);return stream.getvalue()


@pytest.mark.parametrize('route',['/','/static/style.css','/static/app.js','/static/mark.svg','/static/leaf.svg','/api/status','/api/results','/api/search','/api/examples'])
def test_local_routes(route):
    r=client.get(route);assert r.status_code==200
    assert r.content


@pytest.mark.parametrize('function',['Sphere','Rastrigin','Ackley','Rosenbrock','Griewank'])
@pytest.mark.parametrize('seed',range(12))
def test_live_optimizer_is_real_and_monotone(function,seed):
    payload={'function':function,'population':8,'generations':9,'seed':seed}
    r=client.post('/api/evolve',json=payload);assert r.status_code==200
    result=r.json();history=result['history']
    assert len(history)==10 and result['evaluations']==80
    assert len(result['landscape'])==64
    assert np.all(np.diff([h['best'] for h in history])>=0)
    assert all(len(h['population'])==8 for h in history)
    from bioalgo.benchmarks import unit_objective
    objective=unit_objective(function)
    for h in history:
        assert np.allclose([objective(x) for x in h['population']],h['fitness'])
    repeat=client.post('/api/evolve',json=payload).json()
    assert [h['population'] for h in repeat['history']]==[h['population'] for h in history]


@pytest.mark.parametrize('payload',[{'population':3},{'population':81},{'generations':0},{'generations':161},{'seed':-1},{'seed':2**32},{'F':0},{'F':2.1},{'CR':-.1},{'CR':1.1},{'function':'unknown'},{'population':'bad'}])
def test_reject_invalid_experiment(payload):
    assert client.post('/api/evolve',json=payload).status_code==422


@pytest.mark.parametrize('raw',[b'',b'not an image',b'<script>bad()</script>',b'\x89PNG\r\n',b'%PDF-1.5'])
def test_invalid_upload_never_crashes(raw):
    assert client.post('/api/predict',files={'file':('leaf.png',raw,'image/png')}).status_code in (413,422)


@pytest.mark.parametrize('dimensions',[(1,1),(31,31),(31,80),(80,31)])
def test_tiny_image_rejected(dimensions):
    assert client.post('/api/predict',files={'file':('leaf.png',image_bytes(dimensions),'image/png')}).status_code==422


@pytest.mark.parametrize('format',['PNG','JPEG','WEBP'])
def test_valid_image_decoder(format):
    result=server.parse_image(image_bytes(format=format));assert result.size==(64,64) and result.mode=='RGB'


def test_gif_not_silently_accepted():
    r=client.post('/api/predict',files={'file':('leaf.gif',image_bytes(format='GIF'),'image/gif')});assert r.status_code==415


def test_large_upload_rejected():
    r=client.post('/api/predict',files={'file':('leaf.png',b'0'*(server.MAX_UPLOAD+1),'image/png')});assert r.status_code==413


def test_unready_models_are_not_fabricated(tmp_path,monkeypatch):
    monkeypatch.setattr(server,'checkpoint',lambda n:tmp_path/f'{n}.pt')
    r=client.post('/api/predict',files={'file':('leaf.png',image_bytes(),'image/png')});assert r.status_code==503
    assert 'models' not in r.json()


def test_intermediate_epoch_checkpoints_are_not_served_as_completed_models(tmp_path, monkeypatch):
    records = {}
    for name in ('baseline', 'de'):
        (tmp_path / f'{name}.pt').write_bytes(b'isolated incomplete checkpoint fixture')
    monkeypatch.setattr(server, 'checkpoint', lambda name: tmp_path / f'{name}.pt')
    monkeypatch.setattr(server, 'read_result', lambda name, default=None: records.get(name, default))
    assert client.get('/api/status').json()['models'] == {'baseline': False, 'de': False, 'rs': False}
    assert client.post('/api/predict', files={'file': ('leaf.png', image_bytes(), 'image/png')}).status_code == 503
    for name in ('baseline', 'de'):
        records[f'{name}_s0.json'] = {'seed': 0, 'epochs_trained': 4, 'checkpoint': f'models\\{name}_s0.pt'}
    assert client.get('/api/status').json()['models'] == {'baseline': True, 'de': True, 'rs': False}


@pytest.mark.parametrize('kind',['../config.yaml','config','weights','unknown'])
def test_downloads_are_allowlisted(kind):
    assert client.get('/api/download/'+kind).status_code==404


@pytest.mark.parametrize('kind',['blur','noise','brightness','jpeg','low_light'])
@pytest.mark.parametrize('severity',[1,2,3])
def test_stress_matches_evaluation(kind,severity):
    from bioalgo.data import corrupt_image
    image=Image.open(BytesIO(image_bytes()))
    assert np.array_equal(server.corrupt_image(image,kind,severity),corrupt_image(image,kind,severity,seed=42))


def test_no_network_asset_dependency():
    root=Path(__file__).resolve().parents[1]/'demo/static'
    html=(root/'index.html').read_text(encoding='utf-8')
    assert 'src="http' not in html
    assert '@import' not in (root/'style.css').read_text(encoding='utf-8')
    assert 'fonts.googleapis' not in html


@pytest.mark.parametrize('corruption', ['none', 'blur'])
def test_prediction_endpoint_runs_models_and_encodes_explanations(tmp_path, monkeypatch, corruption):
    """Exercise real forward/backward computation with an untrained test model.

    No dataset or scientific checkpoint is read or written by this test.
    """
    import torch
    import bioalgo.model as modeling
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = modeling.build_model({'head_units': 128, 'unfreeze_blocks': 0}, pretrained=False).eval()
    classes = [f'Tomato___Test_class_{i}' for i in range(10)]
    for name in ('baseline', 'de'):
        (tmp_path / f'{name}.pt').write_bytes(b'test checkpoint identity')
    monkeypatch.setattr(server, 'checkpoint', lambda name: tmp_path / f'{name}.pt')
    monkeypatch.setattr(server, 'read_result', lambda name, default=None: {'seed': 0, 'epochs_trained': 1,
        'checkpoint': f'models/{name.removesuffix(".json")}.pt'} if name in ('baseline_s0.json', 'de_s0.json') else default)
    monkeypatch.setattr(server, 'MODELS', {})
    monkeypatch.setattr(modeling, 'load_checkpoint', lambda path, device: (model, {'class_names': classes}))
    response = client.post('/api/predict', files={'file': ('test.png', image_bytes(), 'image/png')},
                           data={'corruption': corruption, 'severity': '1'})
    assert response.status_code == 200
    data = response.json()
    assert [entry['name'] for entry in data['models']] == ['baseline', 'de']
    for entry in data['models']:
        probabilities = [row['probability'] for row in entry['top3']]
        assert len(probabilities) == 3 and probabilities == sorted(probabilities, reverse=True)
        assert 0 <= entry['confidence'] <= 1 and entry['latency_ms'] > 0
        assert entry['confidence'] == probabilities[0]
        encoded = base64.b64decode(entry['gradcam'].split(',', 1)[1])
        with Image.open(BytesIO(encoded)) as overlay:
            assert overlay.size == (224, 224) and overlay.format == 'JPEG'
    assert data['corruption'] == corruption
