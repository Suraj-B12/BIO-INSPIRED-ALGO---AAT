"""Delivery evidence must identify the same files on POSIX and Windows."""
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
from types import SimpleNamespace
import zipfile
from bioalgo.common import revision_provenance

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location('delivery_' + name, ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def windows_tree(native_root):
    """Use Windows path semantics against an isolated temporary file tree."""
    class WindowsTreePath(PureWindowsPath):
        def native(self):
            return native_root.joinpath(*self.relative_to('C:/fixture').parts)

        def is_file(self):
            return self.native().is_file()

        def is_symlink(self):
            return self.native().is_symlink()

        def open(self, *args, **kwargs):
            return self.native().open(*args, **kwargs)

        def rglob(self, pattern):
            return (self / path.relative_to(self.native()).as_posix()
                    for path in self.native().rglob(pattern))

    return WindowsTreePath('C:/fixture')


def write_tree(root, names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'Fixture for {name}\n', encoding='utf-8')


def test_identical_source_bytes_have_platform_independent_fingerprint(tmp_path):
    module = load_script('16_verify_delivery.py')
    write_tree(tmp_path, ['run_all.py', 'config.yaml', '.gitattributes',
                         'requirements.txt', 'requirements-lock.txt',
                         '.github/workflows/ci.yml', 'bioalgo/Z.py', 'bioalgo/a.py',
                         'demo/static/index.html', 'demo/static/app.js'])
    module.ROOT = tmp_path
    expected = module.source_fingerprint()
    module.ROOT = windows_tree(tmp_path)
    assert module.source_fingerprint() == expected
    (tmp_path / 'demo/static/app.js').write_text('Changed application\n', encoding='utf-8')
    assert module.source_fingerprint() != expected


def test_repeat_packaging_excludes_previous_manifest_and_previews_on_both_platforms(tmp_path):
    module = load_script('17_package_delivery.py')
    kept = {'.gitattributes', 'README.md', 'report/BIA_AAT_Team10_Report.pdf',
            'results/final_metrics.json', 'models/de_s0.pt', 'demo/static/leaf.svg'}
    omitted = {'results/delivery_manifest.json', 'report/main.pdf',
               'slides/Plantwise_Design_Preview.pptx', 'models/.de_s0.resume.pt',
               'results/benchmark_runs/run.json', 'data/raw/private.jpg'}
    write_tree(tmp_path, kept | omitted)
    for root in (tmp_path, windows_tree(tmp_path)):
        module.ROOT = root
        selected = [path.relative_to(root).as_posix() for path in module.deliverable_files()]
        assert selected == sorted(kept)
        assert len(selected) == len(set(selected))


def test_git_checkout_preserves_locked_bytes_even_with_windows_autocrlf(tmp_path):
    """Exercise Git's real checkout conversion, without touching project data."""
    source = tmp_path / 'source'
    clone = tmp_path / 'clone'
    source.mkdir()
    shutil.copy2(ROOT / '.gitattributes', source / '.gitattributes')
    original = {'source.py': b'print("unchanged")\n',
                'split.csv': b'path,label\r\nleaf.jpg,0\r\n'}
    for name, contents in original.items():
        (source / name).write_bytes(contents)
    git = ['git', '-c', 'core.autocrlf=true', '-c', 'user.name=Delivery Test',
           '-c', 'user.email=delivery-test@example.invalid', '-c', 'commit.gpgsign=false',
           '-c', f'core.hooksPath={tmp_path / "unused-hooks"}']
    subprocess.run(git + ['init', '--quiet', str(source)], check=True, capture_output=True)
    subprocess.run(git + ['add', '.'], cwd=source, check=True, capture_output=True)
    subprocess.run(git + ['commit', '--quiet', '-m', 'Byte preservation fixture'],
                   cwd=source, check=True, capture_output=True)
    subprocess.run(git + ['clone', '--quiet', str(source), str(clone)],
                   check=True, capture_output=True)
    for name, contents in original.items():
        assert hashlib.sha256((clone / name).read_bytes()).digest() == hashlib.sha256(contents).digest()


def make_git_project(root):
    root.mkdir()
    (root / 'README.md').write_text('Temporary provenance fixture\n', encoding='utf-8')
    git = ['git', '-c', 'user.name=Delivery Test',
           '-c', 'user.email=delivery-test@example.invalid', '-c', 'commit.gpgsign=false',
           '-c', f'core.hooksPath={root / "unused-hooks"}']
    subprocess.run(git + ['init', '--quiet', str(root)], check=True, capture_output=True)
    subprocess.run(git + ['add', 'README.md'], cwd=root, check=True, capture_output=True)
    subprocess.run(git + ['commit', '--quiet', '-m', 'Provenance fixture'],
                   cwd=root, check=True, capture_output=True)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def test_revision_provenance_rejects_parent_repository_and_labels_archive(tmp_path):
    parent = tmp_path / 'parent'
    parent_revision = make_git_project(parent)
    assert revision_provenance(parent)['revision'] == parent_revision
    assert revision_provenance(parent)['source'] == 'project_git_head'
    extracted = parent / 'Plantwise'
    (extracted / 'results').mkdir(parents=True)
    assert revision_provenance(extracted)['revision'] is None
    assert revision_provenance(extracted)['source'] == 'unavailable'
    inherited_revision = 'a' * 40
    (extracted / 'results/delivery_manifest.json').write_text(
        json.dumps({'revision': inherited_revision}), encoding='utf-8')
    inherited = revision_provenance(extracted)
    assert inherited['revision'] == inherited_revision
    assert inherited['revision'] != parent_revision
    assert inherited['source'] == 'delivery_manifest'
    assert 'not a fresh Git verification' in inherited['description']


def test_revision_provenance_works_without_git_or_valid_manifest(tmp_path, monkeypatch):
    import bioalgo.common as common

    def no_git(*args, **kwargs):
        raise FileNotFoundError('Git is not installed in this isolated fixture')

    monkeypatch.setattr(common.subprocess, 'run', no_git)
    (tmp_path / 'results').mkdir()
    manifest = tmp_path / 'results/delivery_manifest.json'
    manifest.write_text(json.dumps({'revision': 'b' * 40}), encoding='utf-8')
    assert revision_provenance(tmp_path)['source'] == 'delivery_manifest'
    manifest.write_text('{broken JSON', encoding='utf-8')
    assert revision_provenance(tmp_path)['revision'] is None
    assert revision_provenance(tmp_path)['source'] == 'unavailable'


def test_extracted_archive_repackages_without_git_or_duplicate_manifest(tmp_path, monkeypatch):
    """Exercise the real archive writer; scientific gates are outside this fixture."""
    module = load_script('17_package_delivery.py')
    extracted = tmp_path / 'Plantwise'
    names = {'README.md', 'report/BIA_AAT_Team10_Report.pdf',
             'slides/BIA_AAT_Team10.pptx', 'slides/BIA_AAT_Team10.pdf',
             'docs/verification_report.md', 'demo/demo.gif'}
    write_tree(extracted, names)
    (extracted / 'results').mkdir()
    (extracted / 'results/delivery_manifest.json').write_text(
        json.dumps({'revision': 'c' * 40}), encoding='utf-8')
    module.ROOT = extracted
    gate_calls = []
    # Replace this script's subprocess binding only; provenance still calls real Git.
    module.subprocess = SimpleNamespace(run=lambda command, **kwargs: gate_calls.append(command))
    output = tmp_path / 'output'
    monkeypatch.setattr(module.sys, 'argv', ['17_package_delivery.py', '--output', str(output)])
    for _ in range(2):
        module.main()
        with zipfile.ZipFile(output / 'Plantwise-complete-project.zip') as archive:
            entries = archive.namelist()
            manifest = json.loads(archive.read('Plantwise/results/delivery_manifest.json'))
            assert entries.count('Plantwise/results/delivery_manifest.json') == 1
            assert len(entries) == manifest['file_count'] + 1
            assert 'results/delivery_manifest.json' not in manifest['files']
            assert manifest['revision'] == 'c' * 40
            assert manifest['revision_provenance']['source'] == 'delivery_manifest'
            for name, evidence in manifest['files'].items():
                assert hashlib.sha256(archive.read('Plantwise/' + name)).hexdigest() == evidence['sha256']
    assert len(gate_calls) == 4
