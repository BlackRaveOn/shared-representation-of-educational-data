"""Local correction-response experiment with a frozen four-policy schedule.

The response includes three reads of the same prepared state, reconstruction,
historical lookups, inference and the requested output contract. Publication is
not part of this response experiment. Any work needed to finish an exact state
after a decision-only response is timed and checked separately.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import uuid
import gc

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from . import maintenance_algorithms as alg
from .runtime import connect, Journal, numeric_context, variant_orders

ROOT = Path(__file__).resolve().parent
REPRO = ROOT.parent
CUT = '2020-11-30'
CALIBRATION_CUT = '2020-11-02'
FACTORS = [1, 2, 5, 10, 50]
METHODS = {'B': 'full_update', 'BB': 'bounds_batch', 'S': 'selective', 'A': 'adaptive'}
CONTRACTS = {'D': 'decisions', 'DM': 'decisions_monitor', 'PM': 'numeric_monitor'}
SEED = 790481


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_models(smoke=False):
    path = os.getenv('GFORM_SELECTION_OVERLAY')
    if not path:
        raise RuntimeError('Run selection first; GFORM_SELECTION_OVERLAY is required.')
    overlay = json.loads(Path(path).read_text(encoding='utf-8'))
    record = overlay['MoocRadar']
    models = record['fixed_maintenance_models']
    source = {'overlay_sha256': digest(path), 'fresh_local_training': True,
              'classifier': 'd28_NS', 'ranker': 'd28_N',
              'training_seed': record.get('training_seed'),
              'calibration_cut': record.get('calibration_cut')}
    assert len(models['classifier']['coef']) == 6
    assert len(models['ranker']['coef']) == 3
    return models, source


def prepare_case(history, models, cut=CUT, calibration=False):
    """Rebuild the fixed request population and an independent corrected oracle."""
    when = pd.Timestamp(cut)
    past = history.loc[history.ts < when]
    window = past.loc[past.ts >= when - pd.Timedelta(days=28)]
    allcat = past.groupby(['course_id', 'exercise_id'], sort=True).is_correct.agg(n='size', s='sum').reset_index()
    allcat['bid'] = allcat.groupby('course_id', sort=False).cumcount().astype('int32')
    if calibration:
        catsizes = allcat.groupby('course_id').size()
        counts = window.groupby(['user_id', 'course_id'], sort=True).size()
        future = history.loc[(history.ts >= when) & (history.ts < when + pd.Timedelta(days=7))]
        targets = future.sort_values(['ts', 'eid']).groupby(['user_id', 'course_id'], sort=False).head(1)
        eligible = pd.DataFrame([k for k, n in counts.items() if n >= 10 and catsizes[k[1]] >= 6],
                                columns=['user_id', 'course_id'])
        requests = eligible.merge(targets[['user_id', 'course_id', 'exercise_id']],
                                  on=['user_id', 'course_id'], validate='one_to_one')
        requests = requests.merge(allcat[['course_id', 'exercise_id', 'bid']],
                                  on=['course_id', 'exercise_id'], validate='many_to_one')
        requests = requests.rename(columns={'bid': 'target_index', 'exercise_id': 'target_block'})
        requests = requests.sort_values(['user_id', 'course_id']).reset_index(drop=True)
        requests.insert(0, 'qid', np.arange(len(requests), dtype=np.int64))
    else:
        requests = pd.read_parquet(REPRO / 'data/moocradar/workloads' / f'requests_{cut}.parquet')
    assert np.array_equal(requests.qid.to_numpy(), np.arange(len(requests)))
    meta = requests[['qid', 'user_id', 'course_id']].copy()
    catalog = allcat.loc[allcat.course_id.isin(meta.course_id)].reset_index(drop=True)
    raw = window.merge(meta, on=['user_id', 'course_id'], validate='many_to_one')
    raw = raw.merge(catalog[['course_id', 'exercise_id', 'bid']],
                    on=['course_id', 'exercise_id'], validate='many_to_one')
    raw = raw.sort_values('eid').reset_index(drop=True)
    cats = {k: group.sort_values('bid') for k, group in catalog.groupby('course_id', sort=False)}
    cases = []
    for row in requests.itertuples(index=False):
        ids = cats[row.course_id].exercise_id.to_numpy()
        assert ids[int(row.target_index)] == row.target_block
        cases.append({'key': (row.user_id, row.course_id), 'ids': ids, 'target': int(row.target_index)})
    if calibration:
        rng = np.random.default_rng(790482)
        selected = rng.random(len(raw)) < 0.05
        journal = raw.loc[selected, ['eid', 'qid', 'course_id', 'bid']].copy()
        journal['new'] = rng.integers(0, 2, len(journal))
    else:
        journal = pd.read_parquet(REPRO / 'data/moocradar/workloads' / f'journal_{cut}.parquet')
    assert journal.eid.is_unique and journal.eid.isin(raw.eid).all()
    joined = journal.merge(raw[['eid', 'qid', 'bid']], on='eid', suffixes=('', '_raw'), validate='one_to_one')
    assert np.array_equal(joined.qid, joined.qid_raw) and np.array_equal(joined.bid, joined.bid_raw)
    q = alg.structure(cases, catalog[['course_id', 'bid', 'n', 's']], 1)
    q['N'] = np.zeros(q['basecells'], dtype=np.int64)
    q['S'] = np.zeros(q['basecells'], dtype=np.int64)
    index = q['baseoff'][raw.qid.to_numpy()] + raw.bid.to_numpy()
    np.add.at(q['N'], index, 1)
    np.add.at(q['S'], index, raw.is_correct.to_numpy())
    q.update(catN=catalog.n.to_numpy(dtype=np.int64), catS=catalog.s.to_numpy(dtype=np.int64), scale=1)

    # Oracle uses actual corrected raw event values and a complete new catalog
    # groupby. No correction bound or selective-retrieval state is used here.
    corrected = past.copy()
    replacement = pd.Series(journal['new'].to_numpy(), index=journal.eid)
    mask = corrected.eid.isin(replacement.index)
    corrected.loc[mask, 'is_correct'] = corrected.loc[mask, 'eid'].map(replacement).to_numpy()
    corrected_cat = corrected.groupby(['course_id', 'exercise_id'], sort=True).is_correct.agg(n='size', s='sum')
    corrected_cat = corrected_cat.reindex(pd.MultiIndex.from_frame(catalog[['course_id', 'exercise_id']]))
    corrected_raw = raw[['eid', 'qid', 'bid']].merge(
        corrected[['eid', 'is_correct']], on='eid', validate='one_to_one')
    corrected_sums = np.zeros(q['basecells'], dtype=np.int64)
    np.add.at(corrected_sums, q['baseoff'][corrected_raw.qid.to_numpy()] + corrected_raw.bid.to_numpy(),
              corrected_raw.is_correct.to_numpy())
    oracle_q = dict(q, catN=corrected_cat.n.to_numpy(dtype=np.int64),
                    catS=corrected_cat.s.to_numpy(dtype=np.int64), S=corrected_sums)
    oracle = alg.answer(oracle_q, corrected_sums, models)
    case = {'cut': cut, 'cases': cases, 'journal': journal, 'meta': meta, 'catalog': catalog,
            'raw': raw, 'q': q, 'tables': {},
            'updated': {'d28_NS': {'p': oracle['p']}, 'd28_N': {'r': oracle['r']}},
            'updated_monitor': oracle['mon']}
    expanded = alg.expanded_journal(case, q)
    exact = q['S'].copy()
    old = history.set_index('eid').is_correct.reindex(expanded['eid']).to_numpy(dtype=np.int64)
    np.add.at(exact, expanded['gid'], expanded['new'] - old)
    assert np.array_equal(exact, corrected_sums)
    result = alg.answer(q, exact, models)
    assert np.max(np.abs(result['p'] - oracle['p'])) <= 1e-12
    assert np.array_equal(result['r'], oracle['r']) and np.array_equal(result['mon'], oracle['mon'])
    if not calibration:
        assert len(requests) == 235 and len(journal) == 541 and len(raw) == 10864
    return case


class Store:
    """Manage the temporary database owned by one maintenance run."""
    def __init__(self, client, history):
        self.client = client
        self.database = 'gform_maintenance_' + uuid.uuid4().hex[:12]
        assert not client.query(f'EXISTS DATABASE {self.database}').first_row[0]
        self.owned = []
        self.history_rows = len(history)
        assert self.history_rows == alg.HROWS
        assert history.eid.is_unique and int(history.eid.min()) == 0 and int(history.eid.max()) == alg.HROWS - 1
        client.command(f'CREATE DATABASE {self.database} ENGINE=Atomic')
        try:
            self.create('history_base', "eid UInt64 CODEC(ZSTD(3)),user_id String CODEC(ZSTD(3)),"
                        "course_id String CODEC(ZSTD(3)),exercise_id String CODEC(ZSTD(3)),"
                        "ts DateTime64(3, 'UTC') CODEC(ZSTD(3)),is_correct UInt8 CODEC(ZSTD(3))", 'eid')
            cols = ['eid', 'user_id', 'course_id', 'exercise_id', 'ts', 'is_correct']
            for start in range(0, len(history), 100000):
                client.insert_df(self.full('history_base'), history.iloc[start:start + 100000][cols])
            self.compact('history_base')
        except BaseException:
            client.command(f'DROP DATABASE {self.database} SYNC')
            raise

    def full(self, name):
        assert name.replace('_', '').isalnum()
        return self.database + '.' + name

    def create(self, name, schema, order):
        assert name not in self.owned
        self.client.command(f'CREATE TABLE {self.full(name)} ({schema}) ENGINE=MergeTree '
                            f'ORDER BY {order} SETTINGS index_granularity=512,index_granularity_bytes=0,'
                            'min_rows_for_wide_part=0,min_bytes_for_wide_part=0')
        self.owned.append(name)

    def compact(self, name):
        self.client.command(f'OPTIMIZE TABLE {self.full(name)} FINAL')

    def prepare(self, case, factor, prefix):
        hist = 'history_base'
        if factor != 1:
            hist = prefix + '_history'
            assert hist not in self.owned
            self.client.command(f'CREATE TABLE {self.full(hist)} AS {self.full("history_base")}')
            self.owned.append(hist)
            self.client.command(f'INSERT INTO {self.full(hist)} SELECT toUInt64(eid+r.number*{alg.HROWS}),'
                                'user_id,course_id,exercise_id,ts,is_correct '
                                f'FROM {self.full("history_base")} CROSS JOIN numbers({factor}) r')
            self.compact(hist)
        tables = {'history': self.full(hist)}
        definitions = {
            'mapping': ('replica UInt16,qid UInt32,user_id String,course_id String', '(replica,qid)', case['meta']),
            'catalog': ('replica UInt16,course_id String,bid UInt32,n Int64,s Int64', '(replica,course_id,bid)',
                        case['catalog'][['course_id', 'bid', 'n', 's']]),
            'aggregate': ('replica UInt16,qid UInt32,bid UInt32,n Int64,s Int64', '(replica,qid,bid)',
                          case['raw'].groupby(['qid', 'bid']).is_correct.agg(n='size', s='sum').reset_index())}
        for kind, (schema, order, frame) in definitions.items():
            name = prefix + '_' + kind
            schema = ','.join(column + ' CODEC(ZSTD(3))' for column in schema.split(','))
            self.create(name, schema, order)
            for replica in range(factor):
                rows = frame.copy()
                rows.insert(0, 'replica', replica)
                self.client.insert_df(self.full(name), rows)
            self.compact(name)
            tables[kind] = self.full(name)
        case['tables'][factor] = tables
        got = self.client.query(f'SELECT count(),min(eid),max(eid),sum(is_correct) FROM {tables["history"]}').first_row
        expected_sum = self.client.query(f'SELECT sum(is_correct) FROM {self.full("history_base")}').first_row[0]
        assert got[0] == alg.HROWS * factor and got[1] == 0 and got[2] == alg.HROWS * factor - 1
        assert got[3] == expected_sum * factor
        return {'factor': factor, 'history_rows': got[0], 'history_values_checksum': got[3],
                'population': len(case['cases']) * factor, 'window_rows': len(case['raw']) * factor,
                'journal_records': len(case['journal']) * factor,
                'replica_count_sum_and_eid_range_verified': True}

    def drop_prepared(self, case, factor):
        tables = case['tables'].pop(factor)
        for full_name in tables.values():
            name = full_name.split('.')[-1]
            if name == 'history_base':
                continue
            assert full_name == self.full(name) and name in self.owned
            self.client.command(f'DROP TABLE {full_name} SYNC')
            self.owned.remove(name)

    def close(self):
        self.client.command(f'DROP DATABASE {self.database} SYNC')


def calibrate(client, case, models):
    q = alg.load_snapshot(client, case, 1)
    journal = alg.expanded_journal(case, q)
    state = alg.state(q, journal)
    low, high = alg.bounds(q, state)
    alg.certify(q, low, high, models)
    check_times = []
    for _ in range(7):
        start = time.perf_counter()
        alg.certify(q, low, high, models)
        check_times.append(time.perf_counter() - start)
    rng = np.random.default_rng(790483)
    rows = []
    ids = case['raw'].eid.to_numpy(dtype=np.int64)
    for size in (128, 512, 1024, 4096, 8192):
        if size > len(ids):
            continue
        chosen = rng.choice(ids, size, replace=False)
        for repeat in range(6):
            reader = alg.Reader(client, q['table'])
            reader.read(chosen)
            rows.append({'size': size, 'repeat': repeat, 'warmup': repeat == 0,
                         'seconds': reader.seconds, 'batches': int(np.ceil(size / 4096))})
    frame = pd.DataFrame([row for row in rows if not row['warmup']])
    medians = frame.groupby('size', as_index=False).median(numeric_only=True)
    coef, _ = nnls(np.column_stack([medians.batches, medians['size']]), medians.seconds)
    return {'read_fixed': float(coef[0]), 'read_per_value': float(coef[1]),
            'check_per_case': float(np.quantile(check_times, 0.75) / q['n']),
            'calibration_cut': CALIBRATION_CUT, 'calibration_population': q['n'],
            'check_seconds': check_times, 'historical_reads': rows,
            'selection_rule': 'nonnegative read-cost fit and 75th-percentile per-query check cost; calibration data only'}


def run(out: Path, smoke=False):
    out = Path(out)
    journal = Journal(out, 'maintenance')
    client = store = None
    total_rows = 0
    try:
        with numeric_context():
            models, model_source = load_models(smoke)
            hpath = REPRO / 'data/moocradar/history.parquet'
            history = pd.read_parquet(hpath).sort_values('eid').reset_index(drop=True)
            case = prepare_case(history, models)
            calibration = prepare_case(history, models, CALIBRATION_CUT, calibration=True)
            client = connect()
            store = Store(client, history)
            preparation = store.prepare(calibration, 1, 'calibration')
            profile = calibrate(client, calibration, models)
            journal.save('profile.json', profile)
            journal.save('models.json', models)
            store.drop_prepared(calibration, 1)
            factors, sessions, rounds, warmups = ([1], 1, 1, 1) if smoke else (FACTORS, 3, 20, 3)
            source_files = [hpath,
                REPRO / 'data/moocradar/workloads/requests_2020-11-30.parquet',
                REPRO / 'data/moocradar/workloads/journal_2020-11-30.parquet']
            context = {'model_source': model_source, 'models': ['d28_NS', 'd28_N'],
                       'cut': CUT, 'calibration_cut': CALIBRATION_CUT, 'factors': factors,
                       'sessions': sessions, 'rounds': rounds, 'warmups': warmups,
                       'base_population': len(case['cases']), 'base_history_rows': len(history),
                       'base_window_rows': len(case['raw']), 'base_journal_records': len(case['journal']),
                       'journal_rate': len(case['journal']) / len(case['raw']),
                       'source_hashes': {str(p.relative_to(REPRO)).replace('\\', '/'): digest(p) for p in source_files},
                       'response_boundary': 'three prepared-state HTTP reads, array reconstruction, historical lookups, all requested answers; no publication',
                       'completion_boundary': 'remaining historical lookups and exact numerical state recomputation after the response; no publication',
                       'calibration_preparation': preparation, 'native_preparations': [], 'checks': []}
            for factor in factors:
                preparation = store.prepare(case, factor, f'f{factor}')
                context['native_preparations'].append(preparation)
                for contract, contract_name in CONTRACTS.items():
                    for variant, method in METHODS.items():
                        result = alg.service(client, case, factor, models, contract_name, method, profile)
                        context['checks'].append({'factor': factor, 'contract': contract, 'variant': variant,
                                                   'mismatches': result['mismatches'],
                                                   'max_probability_error': result['max_probability_error']})
                journal.save('context.json', context)
                for session in range(sessions):
                    schedules = {contract: variant_orders(list(METHODS), rounds, warmups,
                                  SEED + factor * 1000 + session * 100 + i)
                                 for i, contract in enumerate(CONTRACTS)}
                    rng = np.random.default_rng(SEED + factor * 100 + session)
                    for round_index in range(-warmups, rounds):
                        for contract in rng.permutation(list(CONTRACTS)):
                            rows = []
                            gc.collect()
                            for order, variant in enumerate(schedules[contract][round_index]):
                                try:
                                    result = alg.service(client, case, factor, models,
                                                         CONTRACTS[contract], METHODS[variant], profile)
                                except Exception as exc:
                                    journal.event('incomplete_group', factor=factor, session=session,
                                                  round=round_index, contract=str(contract),
                                                  failed_variant=variant,
                                                  exception_type=type(exc).__name__,
                                                  completed_observations=rows)
                                    raise
                                rows.append({**result, 'module': 'maintenance', 'dataset': 'MoocRadar',
                                             'snapshot': CUT, 'factor': factor, 'session': session,
                                             'round': round_index, 'warmup': round_index < 0,
                                             'variant': variant, 'phase': 'response', 'order': order,
                                             'contract': str(contract), 'seconds': result['service_seconds']})
                            journal.pair(rows)
                            total_rows += len(rows)
                        if round_index in (-1, rounds - 1) or (round_index + 1) % 5 == 0:
                            print(json.dumps({'module': 'maintenance', 'factor': factor,
                                              'session': session, 'round': round_index,
                                              'rows': total_rows}), flush=True)
                store.drop_prepared(case, factor)
            journal.save('context.json', context)
            store.close()
            store = None
            client.close()
            client = None
            return journal.finish({'rows': total_rows, 'smoke': smoke, 'correctness_mismatches': 0,
                                   'all_native_sources_cleaned': True})
    except Exception as exc:
        journal.event('technical_failure', exception_type=type(exc).__name__)
        raise
    finally:
        if store is not None:
            try:
                store.close()
                journal.event('cleanup', completed=True)
            except Exception as exc:
                journal.event('cleanup', completed=False, exception_type=type(exc).__name__,
                              remaining_own_database=store.database)
        if client is not None:
            client.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    arguments = parser.parse_args()
    run(arguments.out, arguments.smoke)
