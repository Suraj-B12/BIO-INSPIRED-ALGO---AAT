"""Run engineering checks and retain auditable counts, logs and source identity.

This does not train experimental models or create scientific performance results.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bioalgo.common import save_json, timestamp


def parse_junit(path):
    """Count actual test cases and cross-check the declared leaf-suite totals."""
    root = ET.parse(path).getroot()
    suites = [element for element in root.iter('testsuite') if not element.findall('testsuite')]
    if not suites:
        raise ValueError('JUnit report contains no test suites.')
    result = dict(total=0, passed=0, failed=0, errors=0, skipped=0)
    for suite in suites:
        cases = suite.findall('testcase')
        counts = dict(total=len(cases), passed=0, failed=0, errors=0, skipped=0)
        for case in cases:
            outcomes = [key for tag, key in (('failure', 'failed'), ('error', 'errors'), ('skipped', 'skipped'))
                        if case.find(tag) is not None]
            if len(outcomes) > 1:
                raise ValueError('JUnit case has conflicting outcomes.')
            counts[outcomes[0] if outcomes else 'passed'] += 1
        for attribute, key in (('tests', 'total'), ('failures', 'failed'), ('errors', 'errors'), ('skipped', 'skipped')):
            if int(suite.get(attribute, '-1')) != counts[key]:
                raise ValueError(f'JUnit {attribute} summary does not match its actual cases.')
        for key, value in counts.items():
            result[key] += value
    if not result['total']:
        raise ValueError('JUnit report contains no tests.')
    return result


def parse_tap(path):
    """Use Node's final TAP summary, keeping skipped/todo/cancelled distinct."""
    content = path.read_text(encoding='utf-8', errors='replace')
    counts = {}
    for source, target in (('tests', 'total'), ('pass', 'passed'), ('fail', 'failed'),
                           ('cancelled', 'cancelled'), ('skipped', 'skipped'), ('todo', 'todo')):
        matches = re.findall(rf'^# {source} (\d+)\s*$', content, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f'TAP report must contain exactly one final {source} count.')
        counts[target] = int(matches[0])
    if not counts['total'] or counts['total'] != sum(counts[key] for key in ('passed', 'failed', 'cancelled', 'skipped', 'todo')):
        raise ValueError('TAP outcome counts do not add up to its test count.')
    return counts


def fingerprint():
    spec = importlib.util.spec_from_file_location('delivery_gate_checks', ROOT / 'scripts/16_verify_delivery.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_fingerprint()


def outcome_status(phases, initial_source, final_source):
    return ('passed' if initial_source == final_source and len(phases) == 3
            and all(phase.get('status') == 'passed' for phase in phases.values()) else 'failed')


def run_phase(command, log):
    started = time.perf_counter()
    result = {'command': command, 'log': log.relative_to(ROOT).as_posix()}
    with log.open('wb') as stream:
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            result['returncode'] = completed.returncode
        except OSError as error:
            stream.write((str(error) + '\n').encode('utf-8'))
            result.update(returncode=None, error=str(error))
    result['elapsed_seconds'] = round(time.perf_counter() - started, 4)
    return result


def environment():
    import torch
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'pytest', 'fastapi', 'httpx', 'grad-cam'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        node = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=10)
        node_version = node.stdout.strip() if node.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        node_version = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'architecture': platform.machine(), 'packages': packages, 'node': node_version,
            'cuda_available': torch.cuda.is_available(), 'mps_available': torch.backends.mps.is_available()}


def checked_counts(phase, parser, evidence):
    try:
        phase.update(parser(evidence))
        phase['counts_available'] = True
    except (OSError, ValueError, ET.ParseError) as error:
        phase.update(counts_available=False, parse_error=str(error))
    phase['status'] = ('passed' if phase['returncode'] == 0 and phase['counts_available']
                       and not any(phase.get(key, 0) for key in ('failed', 'errors', 'cancelled')) else 'failed')
    return phase


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    evidence = ROOT / 'results/checks' / run_id
    evidence.mkdir(parents=True)
    initial = fingerprint()
    report = {'schema_version': 1, 'status': 'running', 'run_id': run_id,
              'started_at': timestamp(), 'scope': 'Engineering tests and synthetic smoke verification; not scientific model accuracy.',
              'source_fingerprint': initial, 'environment': environment()}
    phases = {}

    def persist():
        save_json(ROOT / 'results/test_report.json', report)
        save_json(evidence / 'report.json', report)

    persist()
    try:
        junit = evidence / 'pytest.xml'
        print('Running Python tests…', flush=True)
        python = run_phase([sys.executable, '-m', 'pytest', '-q', f'--junitxml={junit}'], evidence / 'pytest-output.txt')
        python = checked_counts(python, parse_junit, junit)
        python['junit'] = junit.relative_to(ROOT).as_posix()
        if junit.exists():
            shutil.copy2(junit, ROOT / 'results/test_suite.xml')
        else:
            (ROOT / 'results/test_suite.xml').unlink(missing_ok=True)
        phases['python'] = report['python'] = python
        persist()

        print('Running frontend tests…', flush=True)
        tap = evidence / 'frontend.tap'
        frontend = run_phase(['node', '--test', '--test-reporter=tap', 'tests/frontend.test.cjs'], tap)
        frontend = checked_counts(frontend, parse_tap, tap)
        frontend['tap'] = tap.relative_to(ROOT).as_posix()
        shutil.copy2(tap, ROOT / 'results/frontend.tap')
        phases['frontend'] = report['frontend'] = frontend
        persist()

        print('Running the offline pipeline smoke check…', flush=True)
        smoke_path = ROOT / 'results/smoke.json'
        previous = smoke_path.read_bytes() if smoke_path.exists() else None
        smoke = run_phase([sys.executable, 'run_all.py', '--smoke'], evidence / 'smoke-output.txt')
        try:
            current = smoke_path.read_bytes()
            value = json.loads(current)
            if current == previous:
                raise ValueError('Smoke evidence was not refreshed by this invocation.')
            if value.get('status') != 'passed' or not 0 <= value.get('seconds', -1) < 600:
                raise ValueError('Smoke result did not pass within the required ten minutes.')
            smoke.update(status='passed' if smoke['returncode'] == 0 else 'failed',
                         evidence='results/smoke.json', evidence_sha256=hashlib.sha256(current).hexdigest(),
                         smoke_seconds=value['seconds'])
            shutil.copy2(smoke_path, evidence / 'smoke.json')
        except (OSError, ValueError, TypeError) as error:
            smoke.update(status='failed', parse_error=str(error))
        phases['smoke'] = report['smoke'] = smoke
    except KeyboardInterrupt:
        report['interrupted'] = True
    finally:
        try:
            final = fingerprint()
        except Exception as error:
            final = None
            report['source_validation_error'] = str(error)
        report.update(status='failed' if report.get('interrupted') else outcome_status(phases, initial, final),
                      source_fingerprint_end=final, source_changed=initial != final,
                      completed_at=timestamp(), elapsed_seconds=round(time.perf_counter() - started, 4))
        persist()
    for name, phase in phases.items():
        counts = f"; {phase['passed']} passed, {phase.get('skipped', 0)} skipped" if phase.get('counts_available') else ''
        print(f"{name}: {phase['status']}{counts}")
    if report['source_changed']:
        print('FAIL: source files changed during verification; rerun checks on stable source.')
    print(f"Engineering checks: {report['status']}. Evidence: results/test_report.json")
    raise SystemExit(report['status'] != 'passed')


if __name__ == '__main__':
    main()
