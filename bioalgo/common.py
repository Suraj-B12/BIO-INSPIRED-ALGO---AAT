"""Small cross-platform helpers shared by the numbered experiment scripts."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import json
import random
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None):
    with open(path or ROOT / 'config.yaml', encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path

def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(f'Cannot encode {type(value).__name__}')

def save_json(path, value):
    """Atomic replacement keeps interrupted runs readable."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, default=json_default, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def read_json(path, default=None):
    path = resolve_path(path)
    if not path.exists(): return default
    return json.loads(path.read_text(encoding='utf-8'))

def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Avoid extreme CPU oversubscription on laptops and CI runners.
    torch.set_num_threads(min(4, os.cpu_count() or 1))

def pick_device():
    import torch
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def synchronize(device):
    import torch
    if device.type == 'cuda': torch.cuda.synchronize()
    elif device.type == 'mps': torch.mps.synchronize()

def timestamp():
    return datetime.now(timezone.utc).isoformat()


def revision_provenance(root=None):
    """Identify this project's revision without borrowing an enclosing repository.

    Delivery ZIPs intentionally contain no Git metadata. Their manifest can
    preserve the recorded source revision, but that is labeled inherited evidence
    rather than a newly verified Git HEAD. Exact artifact hashes remain separate.
    """
    project = Path(root or ROOT).resolve()

    def valid_revision(value):
        return (isinstance(value, str) and len(value) in (40, 64)
                and all(character in '0123456789abcdef' for character in value))

    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=project,
                             capture_output=True, text=True, timeout=10)
        if top.returncode == 0 and Path(top.stdout.strip()).resolve() == project:
            head = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=project,
                                  capture_output=True, text=True, timeout=10)
            revision = head.stdout.strip()
            if head.returncode == 0 and valid_revision(revision):
                return {'revision': revision, 'source': 'project_git_head',
                        'description': 'Git HEAD verified for this project at generation time.'}
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        manifest = json.loads((project / 'results/delivery_manifest.json').read_text(encoding='utf-8'))
        revision = manifest.get('revision') if isinstance(manifest, dict) else None
        if valid_revision(revision):
            return {'revision': revision, 'source': 'delivery_manifest',
                    'description': 'Source revision inherited from the delivery manifest; project Git history is unavailable and this is not a fresh Git verification.'}
    except (OSError, ValueError):
        pass
    return {'revision': None, 'source': 'unavailable',
            'description': 'Source revision is unavailable: no matching project Git root or valid delivery manifest.'}
