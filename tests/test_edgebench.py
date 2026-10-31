"""EdgeBench contracts without Docker or model calls; SForge-specific checks are optional."""
import argparse
import csv
import json
import subprocess
from types import SimpleNamespace

import pytest

from evals import edgebench


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def rollout(path, totals):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps({
        'timestamp': f'2026-10-31T00:00:{n:02d}Z',
        'payload': {'type': 'token_count', 'info': {'total_token_usage': total}},
    }) for n, total in enumerate(totals)) + '\n{partial')


def test_usage_deduplicates_and_restarts_per_session(tmp_path):
    first = dict(input_tokens=100, cached_input_tokens=60, output_tokens=10)
    second = dict(input_tokens=150, cached_input_tokens=80, output_tokens=15)
    rollout(tmp_path / 'rollout-a.jsonl', [first, first, {}, second])
    rollout(tmp_path / 'nested/rollout-b.jsonl', [first, first])
    events = edgebench.usage(tmp_path)
    assert tuple(sum(e[i] for e in events) for i in (1, 2, 3)) == (250, 140, 25)
    assert len(events) == 3


def test_catalog_preserves_instructions(tmp_path):
    prompt = tmp_path / 'prompt.md'
    prompt.write_text('Exact base instructions.\n')
    entry = json.loads(edgebench.catalog(tmp_path, 'custom-model', prompt).read_text())['models'][0]
    assert entry['base_instructions'] == prompt.read_text()
    assert entry['context_window'] == edgebench.CATALOG_WINDOW


def test_report_prefers_final_snapshot_and_lists_missing_tasks(tmp_path, capsys):
    pytest.importorskip('sforge')
    bundle, out = tmp_path / 'bundle', tmp_path / 'out'
    write_json(out / 'run.json', {'run_id': 'check', 'tasks': ['task', 'missing']})
    for task in ('task', 'missing'):
        write_json(bundle / f'tasks/{task}.json', {'judge': {'rescale': {
            'kind': 'linear', 'lower': 0, 'upper': 2}}})
    write_json(bundle / 'logs/runs/check-clm/task/submissions/1/report.json',
               {'score': 1, 'submitted_at': 1793404810})
    first = dict(input_tokens=100, output_tokens=10)
    rollout(out / 'clm/task/sessions/rollout-old.jsonl', [first])
    rollout(out / 'clm/task/final/sessions/rollout-final.jsonl',
            [first, dict(input_tokens=200, output_tokens=20)])
    csv_path = tmp_path / 'report.csv'
    edgebench.report(out, bundle, csv_path, 0.01, 0.02)
    output = capsys.readouterr().out
    assert '50.00 ($2.40' in output
    assert '| missing | — |' in output
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]['input_tokens'] == '200'
    assert rows[-1]['best_score'] == '50.0'


@pytest.mark.parametrize('strategy,catalog', [('off', 'off'), ('off', 'catalog.json'), ('clm', 'catalog.json')])
def test_agent_setup_and_resume(strategy, catalog):
    pytest.importorskip('sforge')
    from sforge.harness.config import SForgeConfig
    from evals.edgebench_agent import CodexRelayAgent, CATALOG
    config = SForgeConfig()
    config.agent_extra_env = {'RELAY_STRATEGY': strategy, 'RELAY_CATALOG': catalog}
    agent = CodexRelayAgent(config)
    env = {'OPENAI_BASE_URL': 'https://example.test/v1', **config.agent_extra_env}
    agent.augment_env(env, 'test-model')
    assert env['RELAY_UPSTREAM'] == 'https://example.test/v1'
    assert env['OPENAI_BASE_URL'] == ('http://127.0.0.1:8787/v1' if strategy == 'clm' else env['RELAY_UPSTREAM'])
    for resume in (False, True):
        cmd = agent.format_run_cmd('/tmp/prompt', resume=resume)
        assert (CATALOG in cmd) == (catalog != 'off')
        assert ('resume --last' in cmd) == resume
    for cmd in agent.install_cmds:
        subprocess.run(['bash', '-n'], input=cmd, text=True, check=True)
    if strategy == 'off':
        assert all('python3' not in cmd for cmd in agent._setup())


def test_cleanup_collects_before_removal_even_if_copy_fails(tmp_path, monkeypatch):
    pytest.importorskip('sforge')
    from sforge.harness.backend.docker_backend import DockerBackend
    calls = []
    original = DockerBackend.cleanup_container
    monkeypatch.setattr(DockerBackend, 'cleanup_container', lambda *a: calls.append('cleanup'))
    monkeypatch.setattr(edgebench, 'collect_container', lambda name, path: calls.append((name, path)))
    edgebench.collect_before_cleanup('check-clm', tmp_path)
    wrapped = DockerBackend.cleanup_container
    wrapped(None, SimpleNamespace(name='sforge.run.task.check-clm'))
    assert calls == [('sforge.run.task.check-clm', tmp_path / 'task/final'), 'cleanup']
    calls.clear()
    def fail(*args):
        raise OSError('copy failed')
    monkeypatch.setattr(edgebench, 'collect_container', fail)
    wrapped(None, SimpleNamespace(name='sforge.run.task.check-clm'))
    assert calls == ['cleanup']
    monkeypatch.setattr(DockerBackend, 'cleanup_container', original)


def test_arm_launcher_registers_with_real_sforge(tmp_path, monkeypatch):
    pytest.importorskip('sforge')
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'relay-0.1.0-py3-none-any.whl').touch()
    (assets / 'pip-25.0-py3-none-any.whl').touch()
    out = tmp_path / 'clm'
    out.mkdir()
    options = argparse.Namespace(asset_port=8899, asset_host='host.docker.internal', window=None,
        pip_index=None, model='test-model', effort='medium', timeout=60, eval_interval=30,
        stagger=0, run_id='check', tasks=['task'], submission_cooldown=None,
        enable_internet=False, bundle=tmp_path)
    popen = subprocess.Popen
    captured = {}
    monkeypatch.setenv('SFORGE_AGENT_EXTRA_ENV', 'EXISTING=value')
    def launch(argv, **kwargs):
        captured.update(kwargs)
        code = argv[2].replace('sys.exit(main())', "assert factory.get_agent_class('codex-relay') is CodexRelayAgent;print('codex-relay');sys.argv=['sforge','run','--help'];sys.exit(main())")
        return popen([argv[0], '-c', code], **kwargs)
    monkeypatch.setattr(edgebench.subprocess, 'Popen', launch)
    job = edgebench.arm_run('clm', out, options, assets)
    assert job.wait(timeout=30) == 0, (out / 'sforge.log').read_text()
    assert 'codex-relay' in (out / 'sforge.log').read_text()
    assert 'EXISTING=value,RELAY_STRATEGY=clm' in captured['env']['SFORGE_AGENT_EXTRA_ENV']


def test_driver_resolves_paths_and_cleans_up_after_launch_failure(tmp_path, monkeypatch):
    import sys
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'bundle').mkdir()
    calls = []
    class Server:
        def shutdown(self):
            calls.append('shutdown')
        def server_close(self):
            calls.append('close')
    class Job:
        def poll(self):
            return None
        def terminate(self):
            calls.append('terminate')
        def wait(self, timeout=None):
            calls.append('wait')
            return 0
    def launch(arm, out, options, assets_dir):
        assert options.bundle == tmp_path / 'bundle'
        assert out.is_absolute() and assets_dir.is_absolute()
        if arm == 'native':
            raise RuntimeError('launch failed')
        return Job()
    prompt = tmp_path / 'prompt.md'
    prompt.write_text('base')
    monkeypatch.setattr(sys, 'argv', ['edgebench', '--bundle', 'bundle', '--out', 'runs',
        '--tasks', 'task', '--arms', 'codex', 'native', '--codex-prompt', str(prompt), '--arm-delay', '0'])
    monkeypatch.setattr(edgebench, 'assets', lambda *args: Server())
    monkeypatch.setattr(edgebench, 'arm_run', launch)
    with pytest.raises(RuntimeError, match='launch failed'):
        edgebench.main()
    assert calls == ['terminate', 'wait', 'shutdown', 'close']
