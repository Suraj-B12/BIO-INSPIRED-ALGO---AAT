"""Verify the completed delivery against its measured evidence, without training.

This is an artifact gate, not a substitute for the independent review. Local
completion and remote CI/publication are deliberately reported separately.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import hashlib
import math
import subprocess
import numpy as np
from bioalgo.common import ROOT, load_config, read_json, save_json, timestamp

FIGURES = """de_convergence de_vs_rs de_diversity search_landscape parallel_coords
gene_importance learning_curves before_after_bars per_class_f1
confusion_side_by_side roc robustness reliability gradcam_grid predictions_grid
embeddings benchmark_convergence benchmark_box de_operators_diagram architecture
pipeline benchmark_surfaces class_distribution dataset_samples augmentation_examples
corruptions de_rastrigin_2d""".split()
FILES = [
    'README.md', 'LICENSE', 'requirements-lock.txt', 'config.yaml',
    'report/BIA_AAT_Team10_Report.pdf', 'report/main.tex', 'report/references.bib',
    'slides/BIA_AAT_Team10.pptx', 'slides/BIA_AAT_Team10.pdf',
    'demo/demo.gif', 'demo/screenshot.jpg', 'demo/examples/index.json',
    'docs/speaker_script.md', 'docs/viva_qa.md',
    'docs/research/literature_review.md', 'docs/verification_report.md',
    'results/system_info.json', 'results/final_metrics.json',
    'results/report_numbers.json', 'results/manual_citation_spotcheck.json',
    'figures/fig_parallel_coords.html', 'figures/fig_de_rastrigin_2d.gif',
] + [f'figures/fig_{name}.{ext}' for name in FIGURES for ext in ('png', 'pdf')]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_fingerprint():
    """Bind executed tests to the code, configuration and dependencies checked."""
    paths = {ROOT / name for name in ('run_all.py', 'config.yaml', 'requirements.txt', 'requirements-lock.txt', '.github/workflows/ci.yml')}
    for folder in ('bioalgo', 'scripts', 'tests', 'demo'):
        paths.update(path for path in (ROOT / folder).rglob('*') if path.suffix in ('.py', '.js', '.cjs', '.html', '.css'))
    payload = '\n'.join(f'{path.relative_to(ROOT)}:{digest(path)}' for path in sorted(paths) if path.is_file())
    return hashlib.sha256(payload.encode()).hexdigest()


def audit_artifact_hashes():
    """Exclude the audit itself to avoid circular hashes."""
    paths = [name for name in FILES if name != 'docs/verification_report.md']
    paths += [f'results/{name}' for name in (
        'data_summary.json', 'benchmark_summary.json', 'evaluation_lock.json',
        'de_evaluations.csv', 'rs_evaluations.csv', 'de_rerank.json', 'rs_rerank.json',
        'qualitative.json', 'prediction_examples.json', 'class_names.json',
    )]
    return {name: digest(ROOT / name) for name in sorted(set(paths)) if (ROOT / name).is_file()}


def validate_metrics(metrics, config):
    """Return integrity errors using persisted per-seed evidence only."""
    errors = []
    if not metrics or metrics.get('status') != 'complete':
        return ['Final held-out evaluation has not completed.']
    if metrics.get('seeds') != config['seeds']:
        errors.append('Evaluation seeds differ from the configured seeds.')
    if metrics['models']['baseline']['config'] != config['baseline']:
        errors.append('Reported baseline differs from its fixed recipe.')
    for name in ('baseline', 'de', 'rs'):
        model = metrics['models'][name]
        rows = model['per_seed']
        if [r['seed'] for r in rows] != config['seeds']:
            errors.append(f'{name}: missing or reordered final seed.')
        for key, summary in model['aggregate'].items():
            values = [r[key] for r in rows]
            expected = (float(np.mean(values)), float(np.std(values, ddof=1)))
            if not all(math.isclose(summary[k], v, rel_tol=1e-10, abs_tol=1e-12)
                       for k, v in zip(('mean', 'std'), expected)):
                errors.append(f'{name}: {key} aggregate does not match all seeds.')
        for row in rows:
            confusion = np.asarray(row['confusion_matrix'])
            if confusion.sum() != metrics['test_images']:
                errors.append(f'{name}/seed{row["seed"]}: confusion support mismatch.')
            if not math.isclose(float(np.trace(confusion) / confusion.sum()), row['accuracy']):
                errors.append(f'{name}/seed{row["seed"]}: accuracy mismatch.')
        intervals = rows[0].get('bootstrap_95_ci', {})
        if intervals.get('samples') != 1000:
            errors.append(f'{name}: missing 1,000-resample confidence intervals.')
        for key in ('accuracy', 'macro_f1'):
            ci = intervals.get(key, [])
            if len(ci) != 2 or not 0 <= ci[0] <= ci[1] <= 1:
                errors.append(f'{name}: invalid {key} confidence interval.')
    for name in ('baseline', 'de'):
        conditions = metrics['robustness'][name]['results']
        if set(conditions) != {'blur', 'noise', 'brightness', 'jpeg', 'low_light'}:
            errors.append(f'{name}: incomplete stress-test conditions.')
        for kind, values in conditions.items():
            if set(values) != {'1', '2', '3'}:
                errors.append(f'{name}/{kind}: incomplete stress-test severities.')
    if not 0 <= metrics['mcnemar']['p_value'] <= 1:
        errors.append('Invalid McNemar p-value.')
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-ci', action='store_true', help='Also require successful remote CI evidence')
    args = parser.parse_args()
    errors = []
    config = load_config()
    for name in FILES:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f'Missing or empty delivery file: {name}')
    metrics = read_json('results/final_metrics.json')
    errors.extend(validate_metrics(metrics, config))
    for name, predicate in (
        ('facts_check.json', lambda r: r.get('status') == 'passed' and r.get('scope') == 'full'),
        ('citation_check.json', lambda r: r.get('final_gate_passed') is True),
        ('test_report.json', lambda r: r.get('status') == 'passed' and r.get('python', {}).get('passed', 0) >= 2000 and r.get('frontend', {}).get('passed', 0) >= 6),
        ('smoke.json', lambda r: r.get('status') == 'passed' and r.get('seconds', 601) < 600),
        ('verification_signoff.json', lambda r: r.get('status') == 'passed'),
    ):
        if not predicate(read_json('results/' + name, {})):
            errors.append(f'Gate has not passed: {name}')
    facts = read_json('results/facts_check.json', {})
    for name in facts.get('checked', []):
        path = ROOT / name
        if not path.is_file() or facts.get('sha256', {}).get(name) != digest(path):
            errors.append(f'Locked-fact evidence is stale: {name}')
    citation = read_json('results/citation_check.json', {})
    bibliography = ROOT / 'report/references.bib'
    if not bibliography.exists() or citation.get('bibliography_sha256') != digest(bibliography):
        errors.append('Citation check is stale for the current bibliography.')
    fingerprint = source_fingerprint()
    if read_json('results/test_report.json', {}).get('source_fingerprint') != fingerprint:
        errors.append('Test results do not identify the current source files.')
    signoff = read_json('results/verification_signoff.json', {})
    if signoff.get('source_fingerprint') != fingerprint or signoff.get('artifact_hashes') != audit_artifact_hashes():
        errors.append('Independent sign-off does not identify the current code and artifacts.')
    tier = config['tiers'][config['tier']]
    budget = tier['population'] * (tier['generations'] + 1)
    for method in ('de', 'rs'):
        path = ROOT / f'results/{method}_evaluations.csv'
        rows = list(csv.DictReader(path.open(encoding='utf-8', newline=''))) if path.exists() else []
        if len(rows) != budget or [int(r['evaluation']) for r in rows] != list(range(1, budget + 1)):
            errors.append(f'{method}: incomplete or nonconsecutive {budget}-evaluation search ledger.')
        ranked = read_json(f'results/{method}_rerank.json', {})
        if len(ranked.get('ranking', [])) != config['de']['rerank_top'] or ranked.get('selection_split') != 'val':
            errors.append(f'{method}: validation-only reranking incomplete.')
    lock = read_json('results/evaluation_lock.json', {})
    for name, expected in lock.get('checkpoints', {}).items():
        path = ROOT / 'models' / f'{name}.pt'
        if not path.exists() or digest(path) != expected:
            errors.append(f'{name}: checkpoint differs from locked evaluation.')
    if len(lock.get('checkpoints', {})) != 9:
        errors.append('Evaluation must identify all nine final checkpoints.')
    test_manifest = ROOT / config['paths']['splits'] / 'test.csv'
    if not test_manifest.exists() or digest(test_manifest) != lock.get('heldout_manifest_sha256'):
        errors.append('Held-out manifest differs from the evaluated manifest.')
    if metrics:
        if metrics['evaluation_signature'] != lock.get('signature'):
            errors.append('Metrics do not identify the locked evaluation.')
        import pymupdf
        for name in ('report/BIA_AAT_Team10_Report.pdf', 'slides/BIA_AAT_Team10.pdf'):
            path = ROOT / name
            if path.exists():
                with pymupdf.open(path) as document:
                    content = '\n'.join(p.get_text() for p in document).lower()
                for token in ('pending', 'draft design preview', 'becomes available after', '??', '[?]'):
                    if token in content:
                        errors.append(f'{name}: unresolved draft content {token!r}.')
    ci = read_json('results/ci_evidence.json', {})
    ci_passed = ci.get('conclusion') == 'success' and ci.get('platforms') == ['ubuntu', 'macos', 'windows']
    if args.require_ci and not ci_passed:
        errors.append('Remote three-platform CI has not passed.')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    save_json('results/delivery_gate.json', {
        'status': 'passed' if not errors else 'failed', 'checked_at': timestamp(),
        'revision': revision, 'scope': 'local_and_ci' if args.require_ci else 'local',
        'remote_ci': 'passed' if ci_passed else 'not_verified', 'errors': errors,
        'required_artifact_count': len(FILES),
    })
    for error in errors:
        print('FAIL:', error)
    print(f'Delivery gate: {"passed" if not errors else "failed"}; {len(errors)} issues')
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
