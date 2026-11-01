"""Verification counters are tested with fixtures, never experimental results."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('check_runner', ROOT / 'scripts/run_checks.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

JUNIT = '''<testsuites><testsuite name="fixture" tests="4" failures="1" errors="1" skipped="1">
<testcase name="pass"/><testcase name="fail"><failure>assertion</failure></testcase>
<testcase name="error"><error>setup error</error></testcase>
<testcase name="skip"><skipped>device unavailable</skipped></testcase>
</testsuite></testsuites>'''
TAP = '''TAP version 13
# tests 5
# suites 0
# pass 1
# fail 1
# cancelled 1
# skipped 1
# todo 1
# duration_ms 12.3
'''


def test_junit_counts_real_outcomes_without_treating_skips_or_errors_as_passes(tmp_path):
    path = tmp_path / 'fixture.xml'
    path.write_text(JUNIT, encoding='utf-8')
    assert runner.parse_junit(path) == dict(total=4, passed=1, failed=1, errors=1, skipped=1)
    phase = runner.checked_counts({'returncode': 1}, runner.parse_junit, path)
    assert phase['status'] == 'failed' and phase['passed'] == 1


def test_junit_rejects_inflated_summary_counts(tmp_path):
    path = tmp_path / 'fixture.xml'
    path.write_text(JUNIT.replace('tests="4"', 'tests="4000"'), encoding='utf-8')
    with pytest.raises(ValueError, match='summary does not match'):
        runner.parse_junit(path)


def test_tap_keeps_cancelled_skipped_and_todo_separate(tmp_path):
    path = tmp_path / 'fixture.tap'
    path.write_text(TAP, encoding='utf-8')
    assert runner.parse_tap(path) == dict(total=5, passed=1, failed=1, cancelled=1, skipped=1, todo=1)
    assert runner.checked_counts({'returncode': 1}, runner.parse_tap, path)['status'] == 'failed'


@pytest.mark.parametrize('content', [TAP.replace('# tests 5', '# tests 5000'), TAP.replace('# pass 1\n', ''), TAP + '# pass 1\n'])
def test_tap_rejects_incomplete_or_inconsistent_summaries(tmp_path, content):
    path = tmp_path / 'fixture.tap'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        runner.parse_tap(path)


def test_exit_failure_cannot_be_hidden_by_passing_tap_counts(tmp_path):
    path = tmp_path / 'fixture.tap'
    path.write_text(TAP.replace('# fail 1', '# fail 0').replace('# cancelled 1', '# cancelled 0').replace('# tests 5', '# tests 3'), encoding='utf-8')
    phase = runner.checked_counts({'returncode': 2}, runner.parse_tap, path)
    assert phase['passed'] == 1 and phase['failed'] == 0 and phase['status'] == 'failed'


def test_success_requires_all_three_phases_and_unchanged_source():
    phases = {name: {'status': 'passed'} for name in ('python', 'frontend', 'smoke')}
    assert runner.outcome_status(phases, 'before', 'before') == 'passed'
    assert runner.outcome_status(phases, 'before', 'changed') == 'failed'
    assert runner.outcome_status({'python': phases['python']}, 'before', 'before') == 'failed'
    phases['frontend']['status'] = 'failed'
    assert runner.outcome_status(phases, 'before', 'before') == 'failed'


def test_failed_python_phase_retains_evidence_and_still_runs_frontend_and_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'fingerprint', lambda: 'fixture-source')
    monkeypatch.setattr(runner, 'environment', lambda: {'kind': 'isolated test fixture'})
    monkeypatch.setattr(runner.sys, 'argv', ['run_checks.py'])
    called = []

    def isolated_phase(command, log):
        called.append(command)
        if '-m' in command:
            Path(command[-1].split('=', 1)[1]).write_text(JUNIT, encoding='utf-8')
            log.write_text('Expected fixture failure evidence\n', encoding='utf-8')
            code = 1
        elif command[0] == 'node':
            log.write_text(TAP.replace('# fail 1', '# fail 0').replace('# cancelled 1', '# cancelled 0').replace('# tests 5', '# tests 3'), encoding='utf-8')
            code = 0
        else:
            (tmp_path / 'results/smoke.json').write_text(json.dumps({'status': 'passed', 'seconds': 0.1}), encoding='utf-8')
            log.write_text('Isolated smoke fixture\n', encoding='utf-8')
            code = 0
        return {'returncode': code, 'elapsed_seconds': 0.1, 'log': log.relative_to(tmp_path).as_posix()}

    monkeypatch.setattr(runner, 'run_phase', isolated_phase)
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code is True
    result = json.loads((tmp_path / 'results/test_report.json').read_text(encoding='utf-8'))
    assert len(called) == 3 and result['status'] == 'failed'
    assert result['python']['passed'] == 1 and result['python']['errors'] == 1
    assert result['frontend']['status'] == 'passed' and result['smoke']['status'] == 'passed'
    assert (tmp_path / result['python']['log']).read_text().startswith('Expected fixture failure')
    assert (tmp_path / 'results/test_suite.xml').read_text() == JUNIT
    assert (tmp_path / 'results/frontend.tap').exists()
    assert list((tmp_path / 'results/checks').glob('*/report.json'))
