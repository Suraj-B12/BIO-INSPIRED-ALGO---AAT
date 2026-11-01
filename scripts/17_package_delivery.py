"""Package only verified project artifacts, excluding raw data and local caches."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ('.github', 'assets', 'bioalgo', 'data/splits', 'demo', 'docs',
           'figures', 'models', 'report', 'results', 'scripts', 'slides', 'tests', 'references')
ROOT_FILES = ('.gitignore', 'AGENTS.md', 'LAB_NOTEBOOK.md', 'LICENSE', 'README.md',
              'config.yaml', 'requirements.txt', 'requirements-lock.txt', 'run_all.py')
EXCLUDED_SUFFIXES = {'.pyc', '.aux', '.log', '.out', '.toc', '.bbl', '.blg', '.gz', '.tmp'}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def deliverable_files():
    selected = {ROOT / name for name in ROOT_FILES}
    for folder in FOLDERS:
        for path in (ROOT / folder).rglob('*'):
            relative = path.relative_to(ROOT)
            if not path.is_file() or path.is_symlink():
                continue
            if any(part in ('__pycache__', '.pytest_cache', 'benchmark_runs') for part in relative.parts):
                continue
            if path.name.startswith('.') or path.suffix in EXCLUDED_SUFFIXES:
                continue
            if 'Design_Preview' in path.name or path.name.startswith('template_page_'):
                continue
            if str(relative) in ('report/main.pdf', 'results/delivery_manifest.json'):
                continue
            selected.add(path)
    return sorted(path for path in selected if path.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='User-facing output directory')
    parser.add_argument('--require-ci', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / 'scripts/16_verify_delivery.py')]
    if args.require_ci:
        command.append('--require-ci')
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/validate_results.py')], cwd=ROOT, check=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    files = deliverable_files()
    manifest = {
        'revision': revision,
        'scope': 'Complete source, measured results, nine model checkpoints, offline demo and academic deliverables',
        'excluded': 'Raw dataset, virtual environment, caches, credentials, temporary previews and Git internals',
        'file_count': len(files),
        'total_uncompressed_bytes': sum(path.stat().st_size for path in files),
        'files': {str(path.relative_to(ROOT)): {'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in files},
        'manifest_note': 'This manifest excludes its own hash to avoid circularity; archive checksum is saved alongside the ZIP.',
    }
    manifest_path = ROOT / 'results/delivery_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    destination = args.output.resolve()
    destination.mkdir(exist_ok=True, parents=True)
    archive = destination / 'Plantwise-complete-project.zip'
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files + [manifest_path]:
            bundle.write(path, Path('Plantwise') / path.relative_to(ROOT))
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f'Archive integrity failure: {bad}')
    for relative, name in (
        ('report/BIA_AAT_Team10_Report.pdf', 'Plantwise-Report.pdf'),
        ('slides/BIA_AAT_Team10.pptx', 'Plantwise-Presentation.pptx'),
        ('slides/BIA_AAT_Team10.pdf', 'Plantwise-Presentation.pdf'),
        ('docs/verification_report.md', 'Plantwise-Verification.md'),
        ('demo/demo.gif', 'Plantwise-Demo.gif'),
    ):
        shutil.copy2(ROOT / relative, destination / name)
    checksum = sha256(archive)
    (destination / 'Plantwise-checksums.txt').write_text(f'{checksum}  {archive.name}\n', encoding='utf-8')
    print(f'Packaged {len(files)} files: {archive.stat().st_size / 1048576:.1f} MiB')
    print(archive)


if __name__ == '__main__':
    main()
