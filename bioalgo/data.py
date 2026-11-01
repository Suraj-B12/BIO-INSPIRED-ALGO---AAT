"""Deterministic image transforms, safe split loading and leakage-aware grouping."""
from .common import ROOT, resolve_path
import csv
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]
CORRUPTIONS = ['blur', 'noise', 'brightness', 'jpeg', 'low_light']


def display_crop(image, resolution=224):
    return transforms.CenterCrop(resolution)(transforms.Resize(round(resolution * 1.14))(image.convert('RGB')))


def evaluation_transform(resolution=224):
    return transforms.Compose([transforms.Resize(round(resolution * 1.14)), transforms.CenterCrop(resolution), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def training_transform(resolution=224, augmentation=0):
    operations = [transforms.RandomResizedCrop(resolution, scale=(.7, 1.)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    if augmentation: operations.append(transforms.RandAugment(num_ops=2, magnitude=int(augmentation)))
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_rows(split, config, *, allow_test=False):
    if split == 'test' and not allow_test: raise ValueError('The held-out split is accessible only during final evaluation')
    if split not in ('train', 'val', 'test'): raise ValueError('Unknown split')
    path = resolve_path(config.get('paths', {}).get('splits', 'data/splits')) / f'{split}.csv'
    if not path.exists(): raise FileNotFoundError(f'Missing {path}; run scripts/01_prepare_data.py first')
    with path.open(encoding='utf-8', newline='') as stream: rows = list(csv.DictReader(stream))
    for row in rows: row['label'] = int(row['label'])
    return rows


def subset_rows(rows, fraction, seed=42):
    if not 0 < fraction <= 1: raise ValueError('fraction must be in (0, 1]')
    if fraction == 1: return list(rows)
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted({r['label'] for r in rows}):
        group = [r for r in rows if r['label'] == label]
        for index in rng.permutation(len(group))[:max(1, round(len(group) * fraction))]: selected.append(group[index])
    return sorted(selected, key=lambda r: r['path'])


def image_path(row, config, cached=True):
    raw = resolve_path(config.get('paths', {}).get('raw', 'data/raw/PlantVillage-Dataset/raw/color'))
    if cached:
        path = resolve_path(config.get('paths', {}).get('cache', 'data/cache_256')) / row['path']
        if path.exists(): return path
    return raw / row['path']


class LeafDataset(Dataset):
    def __init__(self, rows, config, transform=None, corruption=None, severity=1, seed=42):
        self.rows, self.config, self.transform = rows, config, transform
        self.corruption, self.severity, self.seed = corruption, severity, seed
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(image_path(row, self.config)) as opened: image = opened.convert('RGB')
        if self.corruption: image = corrupt_image(image, self.corruption, self.severity, self.seed + index)
        return self.transform(image) if self.transform else image, int(row['label'])


def make_loader(rows, config, resolution, train=False, augmentation=0, seed=42, corruption=None, severity=1):
    dataset = LeafDataset(rows, config, training_transform(resolution, augmentation) if train else evaluation_transform(resolution), corruption, severity, seed)
    settings = config.get('training', {})
    return DataLoader(dataset, batch_size=int(settings.get('batch_size', 32)), shuffle=train, num_workers=int(settings.get('num_workers', 0)), pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(seed), drop_last=False)


def corrupt_image(image, kind, severity=1, seed=42):
    if kind not in CORRUPTIONS: raise ValueError('Unknown corruption')
    if severity not in (1, 2, 3): raise ValueError('Severity must be 1, 2 or 3')
    image = image.convert('RGB')
    if kind == 'blur': return image.filter(ImageFilter.GaussianBlur([1, 2, 3][severity - 1]))
    if kind == 'brightness':
        return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance([.8, .6, .4][severity - 1])).enhance([.8, .65, .5][severity - 1])
    if kind == 'jpeg':
        buffer = io.BytesIO(); image.save(buffer, format='JPEG', quality=[55, 30, 10][severity - 1]); buffer.seek(0)
        with Image.open(buffer) as decoded: return decoded.convert('RGB')
    array = np.asarray(image, dtype=np.float32) / 255
    if kind == 'noise': array += np.random.default_rng(seed).normal(0, [.03, .06, .1][severity - 1], array.shape)
    if kind == 'low_light': array = array ** [1.5, 2, 3][severity - 1]
    return Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8))


def group_hashes(hashes, radius=4):
    """BK-tree neighbour lookup plus union-find, including transitive groups."""
    n = len(hashes); parents = list(range(n)); sizes = [1] * n
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    def union(i, j):
        a, b = root(i), root(j)
        if a == b: return
        if sizes[a] < sizes[b]: a, b = b, a
        parents[b] = a; sizes[a] += sizes[b]
    tree = None
    for i, value in enumerate(hashes):
        value = int(value, 16) if isinstance(value, str) else int(value)
        if tree is None: tree = [value, i, {}]; continue
        pending = [tree]
        while pending:
            node = pending.pop(); distance = (value ^ node[0]).bit_count()
            if distance <= radius: union(i, node[1])
            pending.extend(child for d, child in node[2].items() if distance - radius <= d <= distance + radius)
        node = tree
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0: break
            if distance not in node[2]: node[2][distance] = [value, i, {}]; break
            node = node[2][distance]
    return [root(i) for i in range(n)]


def stratified_group_split(rows, seed=42, ratios=(.7, .15, .15)):
    """Place whole near-duplicate groups with a class-weighted deficit objective."""
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1) or min(ratios) <= 0: raise ValueError('Invalid split ratios')
    classes = sorted({int(row['label']) for row in rows}); position = {label:i for i,label in enumerate(classes)}
    groups = {}
    for row in rows: groups.setdefault(str(row['group']), []).append(row)
    rng = np.random.default_rng(seed)
    ordered = list(groups.values()); rng.shuffle(ordered); ordered.sort(key=len, reverse=True)
    counts = np.zeros((3, len(classes)), dtype=float)
    total = np.array([sum(int(r['label']) == c for r in rows) for c in classes], dtype=float)
    target = np.outer(ratios, total)
    answer = {'train': [], 'val': [], 'test': []}
    for group in ordered:
        contribution = np.bincount([position[int(r['label'])] for r in group], minlength=len(classes))
        costs = []
        for split in range(3):
            new = counts.copy(); new[split] += contribution
            costs.append(float(np.sum((new - target) ** 2 / np.maximum(target, 1))))
        choice = int(np.argmin(costs)); counts[choice] += contribution
        answer[list(answer)[choice]].extend(group)
    for values in answer.values(): values.sort(key=lambda r: r['path'])
    return answer
