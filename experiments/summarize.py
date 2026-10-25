"""Uniform paired means and moving-block uncertainty for the local campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .runtime import save

ROOT = Path(__file__).resolve().parent
CONTEXT = ['dataset', 'scenario', 'phase', 'factor', 'fraction', 'contract']


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def clean_context(frame):
    defaults = {'dataset': 'MoocRadar', 'scenario': 'native', 'phase': 'answer',
                'factor': 1, 'fraction': 1.0, 'contract': 'none'}
    for key, value in defaults.items():
        if key not in frame:
            frame[key] = value
        else:
            frame[key] = frame[key].fillna(value)
    return frame


def contrasts(module, variants):
    variants = set(variants)
    if module == 'common':
        assert variants == {'separate_selected', 'shared_selected'}, variants
        return [('separate_selected', 'shared_selected')]
    if module == 'scaling':
        assert variants == {'separate_daily', 'separate_selected', 'shared_daily', 'shared_selected'}, variants
        return [('separate_selected', x) for x in ('separate_daily', 'shared_daily', 'shared_selected')] + [
            ('shared_daily', 'shared_selected'), ('separate_daily', 'separate_selected'),
            ('separate_daily', 'shared_daily')]
    if module == 'maintenance':
        assert variants == {'B', 'BB', 'S', 'A'}, variants
        return [('B', x) for x in ('BB', 'S', 'A')] + [('BB', x) for x in ('S', 'A')]
    if module == 'selection':
        assert variants == {'independent_union', 'joint_catalog'}, variants
        return [('independent_union', 'joint_catalog')]
    raise ValueError(module)


def paired_summary(frame, baseline, variant, uncertainty, module):
    selected = frame[frame.variant.isin([baseline, variant])]
    keys = ['snapshot', 'session', 'round']
    assert not selected.duplicated(keys + ['variant']).any()
    pairs = selected.pivot(index=keys, columns='variant', values='seconds').sort_index()
    assert set(pairs.columns) == {baseline, variant} and not pairs.isna().any().any()
    arrays = [group[[baseline, variant]].to_numpy(dtype=float)
              for _, group in pairs.groupby(level=['snapshot', 'session'], sort=True)]
    assert arrays and len({len(x) for x in arrays}) == 1, 'Unequal or empty strata'
    values = np.stack(arrays)
    assert np.isfinite(values).all() and (values > 0).all()
    means = values.mean(axis=1).mean(axis=0)
    intervals = {}
    blocks = uncertainty['selection_sensitivity_block_lengths'] if module == 'selection' else uncertainty['sensitivity_block_lengths']
    if values.shape[1] < max(blocks):
        # Smoke summaries are descriptive only and cannot replace reported analysis.
        blocks = [min(b, values.shape[1]) for b in blocks]
    for block in sorted(set(blocks)):
        rng = np.random.default_rng(uncertainty['seed'] + block)
        draws = uncertainty['resamples']
        strata, rounds, _ = values.shape
        starts = rng.integers(0, rounds, size=(draws, strata, (rounds + block - 1) // block))
        indices = ((starts[..., None] + np.arange(block)) % rounds).reshape(draws, strata, -1)[..., :rounds]
        sampled = values[np.arange(strata)[None, :, None], indices]
        sample_means = sampled.mean(axis=2).mean(axis=1)
        reduction = 100 * (1 - sample_means[:, 1] / sample_means[:, 0])
        intervals[str(block)] = {
            'ci95_reduction_percent': np.quantile(reduction, [.025, .975]).tolist(),
            'baseline_ci95_ms': (1000 * np.quantile(sample_means[:, 0], [.025, .975])).tolist(),
            'variant_ci95_ms': (1000 * np.quantile(sample_means[:, 1], [.025, .975])).tolist(),
            'difference_ci95_ms': (1000 * np.quantile(sample_means[:, 1] - sample_means[:, 0], [.025, .975])).tolist()}
    primary = str(min(uncertainty['primary_block_length'], values.shape[1]))
    return {'baseline': baseline, 'variant': variant,
            'baseline_mean_ms': float(1000 * means[0]), 'variant_mean_ms': float(1000 * means[1]),
            'baseline_median_ms_secondary': float(1000 * np.median(values[:, :, 0])),
            'variant_median_ms_secondary': float(1000 * np.median(values[:, :, 1])),
            'reduction_percent': float(100 * (1 - means[1] / means[0])),
            **intervals[primary], 'intervals_by_block_length': intervals,
            'n_pairs': int(len(pairs)), 'n_strata': len(arrays),
            'rounds_per_stratum': values.shape[1], 'bootstrap_resamples': uncertainty['resamples']}


def summarize_module(path, protocol):
    module = path.name
    complete = json.loads((path / 'run_complete.json').read_text(encoding='utf-8'))
    assert complete['complete']
    raw = load_rows(path / 'timings.jsonl')
    assert len(raw) == complete['timing_rows']
    assert hashlib.sha256((path / 'timings.jsonl').read_bytes()).hexdigest() == complete['timings_sha256']
    frame = clean_context(pd.DataFrame(raw))
    assert frame.warmup.map(type).eq(bool).all()
    frame = frame.loc[~frame.warmup].copy()
    if module == 'maintenance' and 'service_plus_completion_seconds' in frame:
        exact_total = frame.copy()
        exact_total['phase'] = 'exact_state_total'
        exact_total['seconds'] = exact_total['service_plus_completion_seconds']
        frame = pd.concat([frame, exact_total], ignore_index=True)
    comparisons = []
    for context, group in frame.groupby(CONTEXT, dropna=False, sort=True):
        fields = dict(zip(CONTEXT, context))
        for baseline, variant in contrasts(module, group.variant.unique()):
            comparisons.append({'module': module, **fields,
                                **paired_summary(group, baseline, variant, protocol['uncertainty'], module)})
    storage = []
    storage_path = path / 'storage.json'
    if storage_path.exists():
        sizes = clean_context(pd.DataFrame(json.loads(storage_path.read_text(encoding='utf-8'))))
        size_keys = [k for k in CONTEXT if k != 'phase']
        for context, group in sizes.groupby(size_keys, dropna=False, sort=True):
            fields = dict(zip(size_keys, context))
            for baseline, variant in contrasts(module, group.variant.unique()):
                pair = group[group.variant.isin([baseline, variant])].pivot(
                    index='snapshot', columns='variant', values='total_bytes')
                assert not pair.isna().any().any()
                a, b = float(pair[baseline].mean()), float(pair[variant].mean())
                storage.append({'module': module, **fields, 'baseline': baseline, 'variant': variant,
                                'baseline_mean_bytes': a, 'variant_mean_bytes': b,
                                'reduction_percent': 100 * (1 - b / a), 'fixed_snapshots': len(pair)})
    result = {'module': module, 'comparisons': comparisons, 'storage': storage,
              'counts': {k: complete[k] for k in ('timing_rows', 'measured_rows', 'warmup_rows')},
              'timings_sha256': complete['timings_sha256']}
    result = json.loads(json.dumps(result, allow_nan=False,
                                   default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()))
    save(path / 'summary.json', result)
    pd.DataFrame(comparisons).to_csv(path / 'comparisons.csv', index=False)
    pd.DataFrame(storage).to_csv(path / 'storage_comparisons.csv', index=False)
    return result


def summarize(results):
    results = Path(results)
    protocol = json.loads((results / 'protocol_frozen.json').read_text(encoding='utf-8'))
    state = json.loads((results / 'campaign.json').read_text(encoding='utf-8'))
    assert state['status'] == 'completed', 'Incomplete campaigns cannot generate publication summaries'
    modules = [summarize_module(results / name, protocol) for name in state['completed_modules']]
    aggregate = {'protocol_version': protocol['version'], 'smoke': state['smoke'],
                 'comparisons': [x for m in modules for x in m['comparisons']],
                 'storage': [x for m in modules for x in m['storage']],
                 'modules': {m['module']: m['counts'] for m in modules},
                 'primary_statistic': protocol['primary_statistic'], 'uncertainty': protocol['uncertainty']}
    save(results / 'summary.json', aggregate)
    print(json.dumps({'modules': aggregate['modules'], 'comparisons': len(aggregate['comparisons'])}), flush=True)
    return aggregate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT.parent / 'results')
    summarize(parser.parse_args().results)
