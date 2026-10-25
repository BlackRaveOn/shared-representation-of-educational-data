"""Compare shared and separate representations on the three datasets."""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path

from .runtime import Journal, connect, dumps, environment, now, numeric_context, timed, variant_orders
from .native import Native, compare
import numpy as np
from scipy.special import expit
from adapters.moocradar import get_snapshots as load_moocradar, ranking, linear_score
from adapters.transfers import get_snapshots as load_transfers


def config(text):
    days, style = text.split('_')
    days = int(days[1:])
    assert days in (1, 7, 14, 28) and style in ('N', 'NS')
    return days, style


class MoocSnapshot:
    """Selected feature resolution is independent of the physical plan resolution."""
    def __init__(self, snapshot, choice):
        self.__dict__.update(snapshot.__dict__)
        self.dataset = 'MoocRadar'
        self.window_days = 28
        self.class_config = choice['class_config']
        self.rank_config = choice['rank_config']
        self.class_days, self.class_style = config(self.class_config)
        self.rank_days, self.rank_style = config(self.rank_config)
        self.models = self.model_payload = choice['models']
        self.daily = False

    def specs(self, shared):
        if shared:
            days = 1 if self.daily else min(self.class_days, self.rank_days)
            return [{'slot': 0, 'name': 'shared', 'days': days,
                     'by_item': True, 'with_sum': True}]
        return [
            {'slot': 0, 'name': 'classifier', 'days': 1 if self.daily else self.class_days,
             'by_item': True, 'with_sum': self.class_style == 'NS'},
            {'slot': 1, 'name': 'ranker', 'days': 1 if self.daily else self.rank_days,
             'by_item': True, 'with_sum': self.rank_style == 'NS'},
            {'slot': 2, 'name': 'monitor', 'days': 1 if self.daily else 28,
             'by_item': False, 'with_sum': True}]

    def _reduce(self, matrix, native_days, target_days):
        assert target_days % native_days == 0
        return matrix.reshape(self.total_items, 28 // target_days,
                              target_days // native_days).sum(axis=2)

    def _outputs(self, cn, cs, rn, rs, class_native_days, rank_native_days, monitor, changed):
        cn = self._reduce(cn, class_native_days, self.class_days)
        rn = self._reduce(rn, rank_native_days, self.rank_days)
        if self.class_style == 'NS':
            cs = self._reduce(cs, class_native_days, self.class_days)
        if self.rank_style == 'NS':
            rs = self._reduce(rs, rank_native_days, self.rank_days)
        static = self.static_changed if changed else self.static
        target = self.offsets + self.target
        class_parts = [np.add.reduceat(cn, self.offsets), cn[target]]
        if self.class_style == 'NS':
            class_parts = [np.add.reduceat(cn, self.offsets), np.add.reduceat(cs, self.offsets),
                           cn[target], cs[target]]
        rank_parts = [rn] + ([rs] if self.rank_style == 'NS' else [])
        class_features = np.column_stack(class_parts + [static[target]])
        rank_features = np.column_stack(rank_parts + [static])
        return {
            'prediction': expit(linear_score(class_features, self.models['classifier'])),
            'ranking': ranking(linear_score(rank_features, self.models['ranker']), self.offsets, self.sizes),
            'monitor': np.asarray(monitor, dtype=np.int64)}

    def evaluate(self, rows, shared, changed=False):
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 6)
        specs = self.specs(shared)
        def block(slot):
            part = rows[rows[:, 0] == slot]
            days = specs[slot]['days']
            n = np.zeros((self.total_items, 28 // days), dtype=np.int64)
            s = np.zeros_like(n) if specs[slot]['with_sum'] else None
            index = self.offsets[part[:, 1]] + part[:, 2]
            assert (part[:, 2] >= 0).all()
            n[index, part[:, 3]] = part[:, 4]
            if s is not None:
                s[index, part[:, 3]] = part[:, 5]
            return n, s, days
        cn, cs, cd = block(0)
        if shared:
            rn, rs, rd = cn, cs, cd
            monitor = np.column_stack([np.add.reduceat(cn.sum(axis=1), self.offsets),
                                       np.add.reduceat(cs.sum(axis=1), self.offsets)])
        else:
            rn, rs, rd = block(1)
            monitor = np.zeros((self.npeople, 2), dtype=np.int64)
            part = rows[rows[:, 0] == 2]
            np.add.at(monitor, part[:, 1], part[:, 4:6])
        return self._outputs(cn, cs, rn, rs, cd, rd, monitor, changed)

    def oracle(self, changed=False):
        raw = self.raw_changed if changed else self.raw
        days_width = min(self.class_days, self.rank_days)
        grouped = raw.assign(bin=raw.day // days_width).groupby(['qid', 'item', 'bin'], sort=False).value.agg(n='size', s='sum')
        n = np.zeros((self.total_items, 28 // days_width), dtype=np.int64)
        s = np.zeros_like(n)
        qids, items, days = [grouped.index.get_level_values(key).to_numpy() for key in ('qid', 'item', 'bin')]
        index = self.offsets[qids] + items
        n[index, days], s[index, days] = grouped.n.to_numpy(), grouped.s.to_numpy()
        person = raw.groupby('qid', sort=False).value.agg(n='size', s='sum')
        monitor = np.zeros((self.npeople, 2), dtype=np.int64)
        monitor[person.index.to_numpy()] = person[['n', 's']].to_numpy()
        return self._outputs(n, s, n, s, days_width, days_width, monitor, changed)


def load_snapshots():
    path = os.environ.get('GFORM_SELECTION_OVERLAY')
    if not path:
        raise RuntimeError('Run selection first; GFORM_SELECTION_OVERLAY is required.')
    overlay = json.loads(Path(path).read_text(encoding='utf-8'))
    choice = overlay['MoocRadar']
    mooc = [MoocSnapshot(snapshot, choice)
            for snapshot in load_moocradar(choice['models'])]
    return mooc + load_transfers(overlay)


def plan_snapshot(snapshot, daily=False):
    result = copy.copy(snapshot)
    result.daily = daily
    return result


def benchmark_snapshot(journal, native, snapshot, *, variants, phases, sessions, rounds,
                       warmups, seed, factor=1, scenario='common', fraction=1.0, prefix='s'):
    """All comparisons share loaded inputs, one server, durable paired rounds."""
    raw_name, changed_name = prefix + '_raw', prefix + '_changed'
    native.load(raw_name, snapshot.raw)
    native.load(changed_name, snapshot.raw_changed)
    raw_checks = []
    for changed, name, data in ((False, raw_name, snapshot.raw), (True, changed_name, snapshot.raw_changed)):
        raw_checks.append({'dataset': snapshot.dataset, 'snapshot': snapshot.name,
                           'factor': factor, 'fraction': fraction, 'scenario': scenario,
                           'changed': changed, **native.verify_raw(name, data)})
    oracles = {changed: snapshot.oracle(changed) for changed in (False, True)}
    context = {'dataset': snapshot.dataset, 'snapshot': snapshot.name,
               'factor': factor, 'fraction': fraction, 'scenario': scenario}
    checks, storage = [], []
    adapters, base, build, update, views = {}, {}, {}, {}, {}
    for variant, (shared, daily) in variants.items():
        adapter = adapters[variant] = plan_snapshot(snapshot, daily)
        base[variant] = native.build(prefix + '_base_' + variant, raw_name, adapter.specs(shared))
        _, actual = native.answer(adapter, base[variant], shared)
        checks.append({**context, 'variant': variant, 'phase': 'base',
                       'changed': False, 'details': compare(actual, oracles[False])})
        storage.append({**context, 'variant': variant, **native.storage(adapter, base[variant])})
        if 'build' in phases:
            build[variant] = native.build(prefix + '_build_' + variant, raw_name, adapter.specs(shared))
        if 'update' in phases:
            update[variant] = native.build(prefix + '_update_' + variant, changed_name, adapter.specs(shared))
            views[variant] = native.publish(prefix + '_current_' + variant, update[variant])
        for folder, payload in (('serialized_models', adapter.model_payload),
                                ('serialized_plans', adapter.specs(shared))):
            artifact = journal.out / folder / (prefix + '_' + variant + '.json')
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(dumps(payload), encoding='utf-8')
    inputs = {**context, 'population': snapshot.population, 'window_days': snapshot.window_days,
              'event_rows': len(snapshot.raw), 'changed_event_rows': len(snapshot.raw_changed),
              'deleted_rows': len(snapshot.raw) - len(snapshot.raw_changed),
              'task_set': ['classification', 'top5_ranking', 'monitoring'] if snapshot.dataset == 'MoocRadar' else ['classification', 'monitoring'],
              'canonical_native_bytes': native.table_bytes([raw_name]),
              'changed_canonical_native_bytes': native.table_bytes([changed_name]),
              'common_metadata_bytes': snapshot.metadata_bytes,
              'changed_common_metadata_bytes': getattr(snapshot, 'changed_metadata_bytes', snapshot.metadata_bytes)}
    if snapshot.dataset == 'MoocRadar':
        inputs.update({'class_config': snapshot.class_config, 'rank_config': snapshot.rank_config})
    else:
        inputs['selected_bin_days'] = snapshot.days
    schedule = {}
    for session in range(sessions):
        for phase_index, phase in enumerate(phases):
            schedule[(session, phase)] = variant_orders(list(variants), rounds, warmups,
                                                       seed + session * 100 + phase_index * 10000)
    journal.save('schedules/' + prefix + '.json', [
        {'session': session, 'phase': phase, 'round': round_, 'order': order}
        for (session, phase), orders in schedule.items() for round_, order in orders.items()])
    for session in range(sessions):
        for round_ in range(-warmups, rounds):
            for phase in phases:
                rows, completed = [], []
                gc.collect()
                try:
                    for order, variant in enumerate(schedule[(session, phase)][round_]):
                        shared = variants[variant][0]
                        adapter = adapters[variant]
                        stamp = now()
                        if phase == 'answer':
                            elapsed, actual = native.answer(adapter, base[variant], shared)
                            completed.append((variant, actual, None, None))
                        elif phase == 'build':
                            elapsed, (plan, _) = timed(lambda: native.replace(
                                prefix + '_build_' + variant, raw_name, adapter.specs(shared),
                                previous=build[variant]))
                            build[variant] = plan
                            completed.append((variant, None, plan, None))
                        elif phase == 'update':
                            elapsed, (plan, view) = timed(lambda: native.replace(
                                prefix + '_update_' + variant, changed_name, adapter.specs(shared),
                                previous=update[variant], view=views[variant]))
                            update[variant] = plan
                            completed.append((variant, None, plan, view))
                        else:
                            raise ValueError('Unknown phase')
                        rows.append({**context, 'session': session, 'round': round_,
                                     'warmup': round_ < 0, 'variant': variant, 'phase': phase,
                                     'seconds': elapsed, 'order': order, 'started_utc': stamp})
                    journal.pair(rows)
                except BaseException as exc:
                    journal.event('incomplete_group', **context, session=session, round=round_, phase=phase,
                                  exception_type=type(exc).__name__, completed_rows=rows)
                    raise
                if round_ in (-1, rounds - 1):
                    for variant, actual, plan, view in completed:
                        changed = phase == 'update'
                        if actual is None:
                            _, actual = native.answer(adapters[variant], plan, variants[variant][0], changed, view)
                        checks.append({**context, 'variant': variant, 'phase': phase,
                                       'session': session, 'round': round_, 'changed': changed,
                                       'details': compare(actual, oracles[changed])})
            if round_ >= 0 and (round_ + 1) % 10 == 0:
                print(journal.module.upper(), snapshot.name, 'factor', factor, 'fraction', fraction,
                      'session', session + 1, 'round', round_ + 1, flush=True)
    return {'checks': checks, 'raw_checks': raw_checks, 'storage': storage, 'common_inputs': inputs}


def run(out: Path, smoke=False):
    journal = Journal(out, 'common')
    client = native = None
    checks, raw_checks, storage, inputs = [], [], [], []
    try:
        with numeric_context():
            snapshots = load_snapshots()
            client = connect()
            native = Native(client)
            journal.save('environment.json', environment(client))
            journal.save('ownership.json', {'database': native.db, 'scope': 'new owned database only'})
            order = np.random.default_rng(790021).permutation(len(snapshots)).tolist()
            journal.save('schedule.json', [{'index': i, 'dataset': snapshots[i].dataset,
                                            'snapshot': snapshots[i].name} for i in order])
            for sequence, index in enumerate(order):
                snapshot = snapshots[index]
                print('COMMON START', snapshot.dataset, snapshot.name, flush=True)
                result = benchmark_snapshot(
                    journal, native, snapshot, variants={'separate_selected': (False, False),
                                                       'shared_selected': (True, False)},
                    phases=['answer', 'build', 'update'], sessions=1 if smoke else 3,
                    rounds=2 if smoke else 60, warmups=1 if smoke else 3,
                    seed=790021 + index * 1000, prefix=f'c{index}')
                checks += result['checks']; raw_checks += result['raw_checks']
                storage += result['storage']; inputs.append(result['common_inputs'])
                journal.save('checks.json', checks); journal.save('raw_checks.json', raw_checks)
                journal.save('storage.json', storage); journal.save('common_inputs.json', inputs)
            expected = 6 * (1 if smoke else 3) * (2 if smoke else 60) * 3 * 2
            report = journal.finish({'smoke': smoke, 'correctness_checks': len(checks),
                                     'native_history_checks': len(raw_checks),
                                     'expected_measured_rows': expected})
            assert report['measured_rows'] == expected
            return report
    except BaseException as exc:
        journal.event('technical_failure', exception_type=type(exc).__name__)
        raise
    finally:
        if native is not None:
            native.close()
            journal.event('owned_database_removed')
        if client is not None:
            client.close()
