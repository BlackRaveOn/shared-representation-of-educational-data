"""Verify complete schedules, paired observations, model lineage and file integrity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .runtime import save

ROOT = Path(__file__).resolve().parent
EXPECTED = {'selection': (60, 18), 'common': (6480, 324),
            'scaling': (6480, 972), 'maintenance': (3600, 540)}
VARIANTS = {'selection': {'joint_catalog', 'independent_union'},
            'common': {'separate_selected', 'shared_selected'},
            'scaling': {'separate_daily', 'separate_selected', 'shared_daily', 'shared_selected'},
            'maintenance': {'B', 'BB', 'S', 'A'}}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_detail(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'max_abs_diff':
                assert item <= 1e-12
            if key in ('exact', 'ordered_rows_exact'):
                assert item is True
            if key == 'mismatches':
                assert item == 0
            inspect_detail(item)
    elif isinstance(value, list):
        for item in value:
            inspect_detail(item)


def verify(results):
    results = Path(results)
    campaign = json.loads((results / 'campaign.json').read_text(encoding='utf-8'))
    assert campaign['status'] == 'completed'
    assert campaign['completed_modules'] == campaign['requested_modules']
    freeze = json.loads((results / 'code_freeze.json').read_text(encoding='utf-8'))
    for name, expected in freeze.items():
        assert digest(ROOT / name) == expected, ('Code differs from measured source', name)
    input_path = results / 'input_freeze.json'
    if input_path.exists():
        inputs = json.loads(input_path.read_text(encoding='utf-8'))
        for name, expected in inputs.items():
            assert digest(ROOT.parent / name) == expected, ('Input differs', name)
    else:
        inputs = {}
    integrity = {'scope': 'current campaign source and inputs',
                 'code_files_verified': len(freeze), 'input_files_verified': len(inputs)}
    base_environment = json.loads((results / 'environment.json').read_text(encoding='utf-8'))
    assert base_environment['server_version'] == '25.3.2.39'
    assert base_environment['query_settings']['max_threads'] == '2'
    assert base_environment['query_settings']['max_memory_usage'] == str(4 * 1024**3)
    report = {'campaign_complete': True, 'smoke': campaign['smoke'],
              'integrity': integrity, 'modules': {}}
    for module in campaign['completed_modules']:
        folder = results / module
        complete = json.loads((folder / 'run_complete.json').read_text(encoding='utf-8'))
        assert complete['complete']
        assert digest(folder / 'timings.jsonl') == complete['timings_sha256']
        rows = [json.loads(x) for x in (folder / 'timings.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        frame = pd.DataFrame(rows)
        assert len(frame) == complete['timing_rows']
        assert np.isfinite(frame.seconds).all() and (frame.seconds > 0).all()
        assert set(frame.variant) == VARIANTS[module]
        assert frame.warmup.map(type).eq(bool).all()
        assert (frame.warmup == (frame['round'] < 0)).all()
        # Freeze the complete raw row set: no filtered primary analysis file exists.
        measured, warmup = frame.loc[~frame.warmup], frame.loc[frame.warmup]
        if not campaign['smoke']:
            assert (len(measured), len(warmup)) == EXPECTED[module]
            assert len(measured.session.unique()) == 3
        context = [x for x in ('dataset', 'scenario', 'snapshot', 'factor', 'fraction', 'contract',
                               'phase', 'session', 'round') if x in frame]
        assert not frame.duplicated(context + ['variant']).any()
        for _, group in frame.groupby(context, dropna=False):
            assert set(group.variant) == VARIANTS[module]
            assert sorted(group['order']) == list(range(len(VARIANTS[module])))
        if not campaign['smoke']:
            strata_keys = [x for x in context if x != 'round']
            for _, group in measured.groupby(strata_keys, dropna=False):
                counts = group.groupby(['variant', 'order']).size()
                assert len(counts) == len(VARIANTS[module]) ** 2
                assert counts.nunique() == 1, 'Unbalanced variant position'
                order = group.pivot(index='round', columns='variant', values='order')
                variants = sorted(VARIANTS[module])
                for i, left in enumerate(variants):
                    for right in variants[i+1:]:
                        assert int((order[left] < order[right]).sum()) * 2 == len(order)
        events = [json.loads(x) for x in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        assert not any(e['kind'] in ('technical_failure', 'incomplete_group') for e in events)
        env_path = results / ('environment_before_' + module + '.json')
        if env_path.exists():
            env = json.loads(env_path.read_text(encoding='utf-8'))
            assert env['server_identity'] == base_environment['server_identity']
            assert env['packages'] == base_environment['packages']
            assert env['query_settings'] == base_environment['query_settings']
            assert env['server_settings'] == base_environment['server_settings']
            assert env['server_uptime_seconds'] >= base_environment['server_uptime_seconds']
            after = json.loads((results / ('environment_after_' + module + '.json')).read_text(encoding='utf-8'))
            assert after['server_identity'] == env['server_identity']
            assert after['server_uptime_seconds'] >= env['server_uptime_seconds']
            assert after['query_settings'] == env['query_settings']
            assert after['server_settings'] == env['server_settings']
        for name in ('checks.json', 'raw_checks.json', 'replica_checks.json', 'context.json'):
            p = folder / name
            if p.exists():
                inspect_detail(json.loads(p.read_text(encoding='utf-8')))
        if module == 'maintenance':
            assert (frame.mismatches == 0).all()
            assert (frame.g_read_queries == 3).all()
        report['modules'][module] = {'measured_rows': len(measured), 'warmup_rows': len(warmup),
                                     'paired_group_count': frame.groupby(context, dropna=False).ngroups,
                                     'timings_sha256': complete['timings_sha256']}
    if 'selection' in campaign['completed_modules']:
        overlay = results / 'selection/overlay.json'
        complete = json.loads((results / 'selection/run_complete.json').read_text(encoding='utf-8'))
        assert digest(overlay) == complete['overlay_sha256']
        report['fresh_model_overlay_sha256'] = digest(overlay)
        models = json.loads(overlay.read_text(encoding='utf-8'))
        for module in campaign['completed_modules']:
            if module != 'selection':
                lineage = json.loads((results / ('selection_overlay_before_' + module + '.json')).read_text(encoding='utf-8'))
                assert lineage['sha256'] == digest(overlay)
            if module == 'common':
                schedule = json.loads((results / 'common/schedule.json').read_text(encoding='utf-8'))
                for snapshot in schedule:
                    for variant in VARIANTS['common']:
                        path = results / f"common/serialized_models/c{snapshot['index']}_{variant}.json"
                        assert json.loads(path.read_text(encoding='utf-8')) == models[snapshot['dataset']]['models']
            if module == 'scaling':
                for path in (results / 'scaling/serialized_models').glob('*.json'):
                    assert json.loads(path.read_text(encoding='utf-8')) == models['MoocRadar']['models']
            if module == 'maintenance':
                path = results / 'maintenance/models.json'
                assert json.loads(path.read_text(encoding='utf-8')) == models['MoocRadar']['fixed_maintenance_models']
                context = json.loads((results / 'maintenance/context.json').read_text(encoding='utf-8'))
                assert context['model_source']['overlay_sha256'] == digest(overlay)
    save(results / 'verification.json', report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    verify(args.results)
