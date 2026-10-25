"""Four-plan history scaling and served-population ablation on the same server."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .common import benchmark_snapshot, load_snapshots, plan_snapshot
from .native import Native, compare
from .runtime import Journal, connect, environment, numeric_context, sha

VARIANTS = {'separate_daily': (False, True), 'separate_selected': (False, False),
            'shared_daily': (True, True), 'shared_selected': (True, False)}
FACTORS = [1, 2, 5, 10, 50]
FRACTIONS = [0.25, 0.5, 1.0]


def replicate(snapshot, factor):
    """Replicate participants and event windows; keep each course catalog fixed."""
    factor = int(factor)
    assert factor >= 1
    result = copy.copy(snapshot)
    stride = (int(snapshot.raw.eid.max()) // 100 + 1) * 100
    def copies(frame):
        pieces = []
        for replica in range(factor):
            part = frame.copy()
            part['eid'] = part.eid.to_numpy(dtype=np.uint64) + np.uint64(replica * stride)
            part['qid'] = part.qid.to_numpy(dtype=np.uint32) + np.uint32(replica * snapshot.population)
            pieces.append(part)
        return pd.concat(pieces, ignore_index=True)
    result.raw = copies(snapshot.raw)
    result.raw_changed = copies(snapshot.raw_changed)
    people = []
    for replica in range(factor):
        part = snapshot.people.copy()
        part['qid'] = part.qid.to_numpy(dtype=np.uint32) + np.uint32(replica * snapshot.population)
        part['replica_id'] = replica
        people.append(part)
    result.people = pd.concat(people, ignore_index=True)
    result.npeople = result.population = factor * snapshot.population
    result.sizes = np.tile(snapshot.sizes, factor)
    result.offsets = np.r_[0, np.cumsum(result.sizes)[:-1]].astype(np.int64)
    result.target = np.tile(snapshot.target, factor)
    result.static = np.tile(snapshot.static, (factor, 1))
    result.static_changed = np.tile(snapshot.static_changed, (factor, 1))
    result.total_items = int(result.sizes.sum())
    result.replica_factor = factor
    result.replica_eid_stride = stride
    assert result.raw.eid.is_unique and result.raw_changed.eid.is_unique
    assert np.array_equal(result.raw.loc[result.raw.eid % 100 != 0, 'eid'], result.raw_changed.eid)
    assert result.raw.eid.is_monotonic_increasing
    assert int(result.raw.qid.max()) < 2**32
    return result


def subset(snapshot, fraction, seed):
    """A fixed seeded sample of eligible participants, with compact local ids."""
    count = max(1, int(np.floor(snapshot.population * fraction)))
    chosen = np.sort(np.random.default_rng(seed).permutation(snapshot.population)[:count])
    result = copy.copy(snapshot)
    remap = np.full(snapshot.population, -1, dtype=np.int64)
    remap[chosen] = np.arange(count)
    def select(frame):
        keep = remap[frame.qid.to_numpy()] >= 0
        selected = frame.loc[keep].copy()
        selected['qid'] = remap[selected.qid.to_numpy()].astype(np.uint32)
        return selected.reset_index(drop=True)
    result.raw, result.raw_changed = select(snapshot.raw), select(snapshot.raw_changed)
    result.people = snapshot.people.iloc[chosen].copy().reset_index(drop=True)
    result.people['source_qid'] = chosen
    result.people['qid'] = np.arange(count, dtype=np.uint32)
    result.npeople = result.population = count
    result.sizes = snapshot.sizes[chosen].copy()
    result.offsets = np.r_[0, np.cumsum(result.sizes)[:-1]].astype(np.int64)
    result.target = snapshot.target[chosen].copy()
    pieces = [np.arange(snapshot.offsets[q], snapshot.offsets[q] + snapshot.sizes[q]) for q in chosen]
    item_indices = np.concatenate(pieces)
    result.static = snapshot.static[item_indices].copy()
    result.static_changed = snapshot.static_changed[item_indices].copy()
    result.total_items = int(result.sizes.sum())
    result.source_qids = chosen
    return result


def metadata(snapshot, out):
    """Materialize common metadata once, outside timers, for measured byte counts."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot.people.to_parquet(out / 'participants.parquet', index=False)
    snapshot.catalog.to_parquet(out / 'catalog.parquet', index=False)
    snapshot.catalog_changed.to_parquet(out / 'catalog_changed.parquet', index=False)
    np.savez_compressed(out / 'context.npz', offsets=snapshot.offsets, sizes=snapshot.sizes,
                        target=snapshot.target, static=snapshot.static)
    np.savez_compressed(out / 'context_changed.npz', static=snapshot.static_changed)
    before = ['participants.parquet', 'catalog.parquet', 'context.npz']
    after = ['participants.parquet', 'catalog_changed.parquet', 'context_changed.npz']
    snapshot.metadata_bytes = sum((out / name).stat().st_size for name in before)
    snapshot.changed_metadata_bytes = sum((out / name).stat().st_size for name in after)
    return {name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
            for name in sorted(set(before + after)) for path in [out / name]}


def verify_replication(base, expanded, factor):
    checks = []
    for changed in (False, True):
        expected, actual = base.oracle(changed), expanded.oracle(changed)
        details = {}
        for key, value in expected.items():
            tiled = np.tile(value, (factor,) + (1,) * (value.ndim - 1))
            details[key] = compare({key: actual[key]}, {key: tiled})[key]
        checks.append({'factor': factor, 'changed': changed, 'details': details})
    return checks


def preflight_large(journal, client, snapshot, factor):
    """Untimed correctness and query-limit gate at the largest declared scale."""
    native = Native(client)
    checks, raw_checks = [], []
    try:
        for changed in (False, True):
            name = 'large_changed' if changed else 'large_raw'
            data = snapshot.raw_changed if changed else snapshot.raw
            native.load(name, data)
            raw_checks.append({'factor': factor, 'changed': changed, **native.verify_raw(name, data)})
            expected = snapshot.oracle(changed)
            for index, (variant, (shared, daily)) in enumerate(VARIANTS.items()):
                adapter = plan_snapshot(snapshot, daily)
                plan = native.build(f'large_{int(changed)}_{index}', name, adapter.specs(shared))
                _, actual = native.answer(adapter, plan, shared, changed)
                checks.append({'factor': factor, 'variant': variant, 'changed': changed,
                               'details': compare(actual, expected)})
                native.retire(plan)
        journal.save('largest_scale_correctness.json', {'factor': factor, 'checks': checks,
                                                       'raw_checks': raw_checks,
                                                       'reported_timings': False})
        print('SCALING LARGE PREFLIGHT PASS factor', factor, flush=True)
    finally:
        native.close()


def run(out: Path, smoke=False):
    journal = Journal(out, 'scaling')
    client = None
    checks, raw_checks, storage, inputs, replica_checks, ownership = [], [], [], [], [], []
    try:
        with numeric_context():
            snapshots = [snapshot for snapshot in load_snapshots()
                         if snapshot.dataset == 'MoocRadar']
            base = next(snapshot for snapshot in snapshots if snapshot.cut == '2020-11-30')
            client = connect()
            journal.save('environment.json', environment(client))
            factors = [1] if smoke else FACTORS
            tasks = [('history', base, factor, 1.0, index)
                     for index, factor in enumerate(factors)]
            if not smoke:
                tasks += [('population', snapshot, 1, fraction, 100 + index * 10 + fi)
                          for index, snapshot in enumerate(snapshots)
                          for fi, fraction in enumerate(FRACTIONS)]
            journal.save('schedule.json', [{'scenario': scenario, 'snapshot': snapshot.name,
                                            'factor': factor, 'fraction': fraction, 'index': index}
                                           for scenario, snapshot, factor, fraction, index in tasks])
            for scenario, source, factor, fraction, index in tasks:
                snapshot = (replicate(source, factor) if scenario == 'history'
                            else subset(source, fraction, 790153 + index // 10))
                prefix = f's{index}'
                meta = metadata(snapshot, journal.out / 'common_metadata' / prefix)
                journal.save('common_metadata/' + prefix + '/manifest.json', meta)
                if scenario == 'history':
                    replica_checks += verify_replication(source, snapshot, factor)
                    journal.save('replica_checks.json', replica_checks)
                print('SCALING START', scenario, snapshot.name, factor, fraction,
                      'rows', len(snapshot.raw), 'people', snapshot.population, flush=True)
                native = Native(client)
                ownership.append({'scenario': scenario, 'snapshot': snapshot.name,
                                  'factor': factor, 'fraction': fraction, 'database': native.db})
                journal.save('ownership.json', ownership)
                try:
                    result = benchmark_snapshot(
                        journal, native, snapshot, variants=VARIANTS,
                        phases=['answer', 'build', 'update'] if scenario == 'history' else ['answer'],
                        sessions=1 if smoke else 3, rounds=4 if smoke else 20,
                        warmups=1 if smoke else 3, seed=790153 + index * 1000,
                        factor=factor, fraction=fraction, scenario=scenario, prefix=prefix)
                finally:
                    native.close()
                checks += result['checks']; raw_checks += result['raw_checks']
                storage += result['storage']; inputs.append(result['common_inputs'])
                journal.save('checks.json', checks); journal.save('raw_checks.json', raw_checks)
                journal.save('storage.json', storage); journal.save('common_inputs.json', inputs)
                del snapshot
            if smoke:
                large = replicate(base, 50)
                replica_checks += verify_replication(base, large, 50)
                journal.save('replica_checks.json', replica_checks)
                preflight_large(journal, client, large, 50)
                del large
            expected = 48 if smoke else 6480
            report = journal.finish({'smoke': smoke, 'correctness_checks': len(checks),
                                     'native_history_checks': len(raw_checks),
                                     'replica_checks': len(replica_checks),
                                     'expected_measured_rows': expected})
            assert report['measured_rows'] == expected
            return report
    except BaseException as exc:
        journal.event('technical_failure', exception_type=type(exc).__name__)
        raise
    finally:
        if client is not None:
            client.close()
