"""Download, validate, deduplicate, group and stratify the real tomato dataset."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bioalgo.common import ROOT, load_config, resolve_path, save_json, timestamp
from bioalgo.data import group_hashes, stratified_group_split
import argparse
import csv
import hashlib
import subprocess
import time
from collections import Counter
from PIL import Image, ImageOps
import imagehash
from tqdm import tqdm


DATASET_REVISION = '7f7ecc7e1eaca78107e3affe7cb5abd9427e139a'


def download(raw, revision=DATASET_REVISION):
    repository = raw.parents[1]
    if not (repository / '.git').exists():
        repository.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse','https://github.com/spMohanty/PlantVillage-Dataset.git',str(repository)], check=True)
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
    if current!=revision:
        # Reproduce the recorded dataset snapshot even when upstream HEAD advances.
        subprocess.run(['git','fetch','--depth','1','origin',revision],cwd=repository,check=True)
        subprocess.run(['git','checkout','--detach',revision],cwd=repository,check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
    if actual!=revision:raise RuntimeError('Dataset checkout does not match its pinned source revision')
    folders = subprocess.check_output(['git','ls-tree','--name-only','HEAD:raw/color'], cwd=repository, text=True).splitlines()
    folders = [f'raw/color/{name}' for name in folders if name.startswith('Tomato___')]
    if len(folders) != 10: raise RuntimeError(f'Expected ten tomato classes; found {len(folders)}')
    subprocess.run(['git','sparse-checkout','set',*folders], cwd=repository, check=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--download',action='store_true'); parser.add_argument('--fresh', action='store_true'); args = parser.parse_args()
    config = load_config(); raw = resolve_path(config.get('paths',{}).get('raw','data/raw/PlantVillage-Dataset/raw/color'))
    splits_dir = resolve_path(config.get('paths',{}).get('splits','data/splits')); cache = resolve_path(config.get('paths',{}).get('cache','data/cache_256'))
    complete_manifests=all((splits_dir/f'{name}.csv').exists() for name in ('train','val','test')) and (ROOT/'results'/'class_names.json').exists() and (ROOT/'results'/'data_summary.json').exists()
    if complete_manifests and not args.fresh:
        print('Committed splits already exist; validating raw paths and filling the image cache.')
        rows = []
        for path in sorted(splits_dir.glob('*.csv')):
            with path.open(encoding='utf-8',newline='') as stream: rows.extend(csv.DictReader(stream))
        missing = [row['path'] for row in rows if not (raw / row['path']).exists()]
        if missing and args.download: download(raw); missing = [row['path'] for row in rows if not (raw / row['path']).exists()]
        if missing: raise FileNotFoundError(f'{len(missing)} images absent; rerun with --download')
        for row in tqdm(rows,desc='Verify committed image digests'):
            with Image.open(raw / row['path']) as opened:image=ImageOps.exif_transpose(opened).convert('RGB')
            digest=hashlib.sha256(image.size.__repr__().encode()+image.tobytes()).hexdigest()
            if digest!=row['sha256']:raise RuntimeError(f"Dataset image changed from committed manifest: {row['path']}")
        write_group_audit(rows)
        write_cache(rows, raw, cache)
        return
    if args.download or not raw.exists(): download(raw)
    folders = sorted(p for p in raw.iterdir() if p.is_dir() and p.name.startswith('Tomato___'))
    if len(folders) != 10: raise RuntimeError(f'Expected ten tomato folders, got {len(folders)}')
    rows, corrupt, exact_duplicates, seen = [], [], [], {}
    original = {}
    started = time.perf_counter()
    for label, folder in enumerate(folders):
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
        original[folder.name] = len(files)
        for path in tqdm(files, desc=folder.name):
            relative = path.relative_to(raw).as_posix()
            try:
                with Image.open(path) as opened: opened.verify()
                with Image.open(path) as opened: image = ImageOps.exif_transpose(opened).convert('RGB')
                digest = hashlib.sha256(image.size.__repr__().encode() + image.tobytes()).hexdigest()
                if digest in seen:
                    exact_duplicates.append({'removed':relative,'kept':seen[digest]}); continue
                seen[digest] = relative
                rows.append({'path':relative,'label':label,'class_name':folder.name,'phash':str(imagehash.phash(image)),'sha256':digest})
            except (OSError, ValueError) as exc: corrupt.append({'path':relative,'error':str(exc)})
    if not 16000 <= sum(original.values()) <= 20000: raise RuntimeError(f'Unexpected dataset size: {sum(original.values())}')
    print('Grouping perceptually similar images across all classes...', flush=True)
    groups = group_hashes([row['phash'] for row in rows], radius=4)
    for row, group in zip(rows, groups): row['group'] = str(group)
    splits = stratified_group_split(rows, seed=int(config.get('split_seed',42)))
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        with (splits_dir / f'{name}.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['path','label','class_name','phash','sha256','group']); writer.writeheader(); writer.writerows(values)
    summary = {'timestamp':timestamp(),'source':'https://github.com/spMohanty/PlantVillage-Dataset','variant':'raw/color','original_counts':original,'original_total':sum(original.values()),'usable_total':len(rows),'exact_duplicates_removed':len(exact_duplicates),'corrupt_removed':len(corrupt),'near_duplicate_groups':len(set(groups)),'near_duplicate_threshold':4,'split_seed':int(config.get('split_seed',42)),'counts':{name:dict(Counter(r['class_name'] for r in values)) for name,values in splits.items()},'sizes':{name:len(values) for name,values in splits.items()},'preprocessing_seconds':time.perf_counter()-started}
    try: summary['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=raw.parents[1],text=True).strip()
    except subprocess.CalledProcessError: pass
    save_json('results/class_names.json',[p.name for p in folders]); save_json('results/data_summary.json',summary)
    save_json('results/data_cleaning.json',{'corrupt':corrupt,'exact_duplicates':exact_duplicates})
    write_group_audit(rows)
    write_cache(rows, raw, cache)
    print(summary)



def write_group_audit(rows):
    sizes=Counter(row['group'] for row in rows)
    save_json('results/data_group_audit.json',{'retained_images':len(rows),'all_groups':len(sizes),'groups_with_multiple_images':sum(size>1 for size in sizes.values()),'images_in_multiple_image_groups':sum(size for size in sizes.values() if size>1),'largest_group':max(sizes.values()),'excess_members_over_one_per_group':sum(size-1 for size in sizes.values()),'note':'Near-duplicate images were retained within one split per group; only exact pixel duplicates were removed.'})


def write_cache(rows, raw, cache):
    for row in tqdm(rows, desc='256px cache'):
        path = cache / row['path']
        if path.exists(): continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(raw / row['path']) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
            ratio = 256 / min(image.size)
            image = image.resize((round(image.width*ratio),round(image.height*ratio)),Image.Resampling.LANCZOS)
            image.save(path,format='JPEG',quality=95)

if __name__ == '__main__': main()
