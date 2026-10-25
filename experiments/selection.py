"""Fit task models, validate temporal folds, and select aggregate representations."""
from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import itertools
import json
import random
import time
import uuid

from . import runtime
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

ROOT = Path(__file__).resolve().parent
REPRO = ROOT.parent
from adapters import mooc_features as mc

CONFIGS = list(mc.CFG)
LIMITS = {'auc_drop': .01, 'logloss_increase': .02,
          'ndcg5_drop': .02, 'hit5_drop': .02, 'time_ratio_max': 1.25}
CATALOG_DAYS = [1, 7, 14, 28, 56]
TEST_CUTS = ['2020-11-16', '2020-11-23', '2020-11-30', '2020-12-07']


def clean(value):
    return json.loads(json.dumps(value, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()))


def save(path, value):
    runtime.save(path, clean(value))


def export_model(model):
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    assert np.array_equal(classifier.classes_, [0, 1])
    return {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coef': classifier.coef_[0].tolist(),
            'intercept': float(np.asarray(classifier.intercept_).reshape(-1)[0])}


class Portable:
    def __init__(self, payload):
        self.payload = payload
        self.mean = np.asarray(payload['mean'])
        self.scale = np.asarray(payload['scale'])
        self.coef = np.asarray(payload['coef'])
        self.intercept = float(payload['intercept'])

    def decision_function(self, features):
        return ((features - self.mean) / self.scale) @ self.coef + self.intercept

    def predict_proba(self, features):
        p = expit(self.decision_function(features))
        return np.column_stack((1-p, p))


def train_mooc(cases, seed):
    n, s, static, offsets, target = mc.matrices(cases)
    y = np.asarray([case['y'] for case in cases])
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for i, case in enumerate(cases):
        for j in rng.choice(np.delete(np.arange(len(case['ids'])), case['target']), 5, replace=False):
            pos.append(offsets[i] + case['target'])
            neg.append(offsets[i] + j)
    pos, neg = np.asarray(pos), np.asarray(neg)
    models, payload = {'class': {}, 'rank': {}}, {'class': {}, 'rank': {}}
    for config in CONFIGS:
        xc, xr = mc.features(n, s, static, offsets, target, config)
        diff = xr[pos] - xr[neg]
        fitted = {
            'class': make_pipeline(StandardScaler(), LogisticRegression(
                C=1, max_iter=1000, random_state=seed)).fit(xc, y),
            'rank': make_pipeline(StandardScaler(), LogisticRegression(
                C=1, max_iter=1000, fit_intercept=False, random_state=seed)).fit(
                    np.r_[diff, -diff], np.r_[np.ones(len(diff)), np.zeros(len(diff))])}
        for task, original in fitted.items():
            payload[task][config] = export_model(original)
            models[task][config] = Portable(payload[task][config])
            features = xc if task == 'class' else xr
            assert np.max(np.abs(original.decision_function(features) -
                                 models[task][config].decision_function(features))) <= 1e-12
    return models, payload


def raw_window(history, cases, cut):
    cut = pd.Timestamp(cut)
    mapping = pd.DataFrame([(i, *case['key']) for i, case in enumerate(cases)],
                           columns=['qid', 'user_id', 'course_id'])
    catalogs = []
    seen = set()
    for case in cases:
        course = case['key'][1]
        if course in seen:
            continue
        seen.add(course)
        catalogs.extend((course, ex, i) for i, ex in enumerate(case['ids']))
    catalog = pd.DataFrame(catalogs, columns=['course_id', 'exercise_id', 'item'])
    raw = history.loc[(history.ts < cut) & (history.ts >= cut-pd.Timedelta(days=28))]
    raw = raw.merge(mapping, on=['user_id', 'course_id'], validate='many_to_one')
    raw = raw.merge(catalog, on=['course_id', 'exercise_id'], validate='many_to_one')
    raw['day'] = ((raw.ts-(cut-pd.Timedelta(days=28))).dt.total_seconds()//86400).astype('uint8')
    raw = raw.rename(columns={'is_correct': 'value'})
    return raw[['eid', 'qid', 'item', 'day', 'value']].sort_values('eid').reset_index(drop=True)


class CatalogNative:
    """Owned immutable generations; one HTTP bundle read per candidate/task."""
    def __init__(self, client, proof_path=None):
        self.c = client
        self.db = 'gform_selection_' + uuid.uuid4().hex[:12]
        assert not self.c.query(f'EXISTS DATABASE {self.db}').first_row[0]
        self.c.command(f'CREATE DATABASE {self.db} ENGINE=Atomic')
        self.owned = set()
        self.serial = 0
        self.proof_path = Path(proof_path) if proof_path is not None else None
        self.history_checks = []

    def full(self, name):
        assert name.replace('_', '').isalnum()
        return self.db + '.' + name

    def create(self, name, fields, order):
        assert name not in self.owned
        self.c.command(f'CREATE TABLE {self.full(name)} ({fields}) ENGINE=MergeTree '
                       f'ORDER BY {order} SETTINGS index_granularity=512,index_granularity_bytes=0,'
                       'min_rows_for_wide_part=0,min_bytes_for_wide_part=0')
        self.owned.add(name)

    def load(self, name, raw):
        self.create(name, 'eid UInt64 CODEC(ZSTD(3)),qid UInt32 CODEC(ZSTD(3)),'
                    'item UInt32 CODEC(ZSTD(3)),day UInt8 CODEC(ZSTD(3)),value UInt8 CODEC(ZSTD(3))', 'eid')
        for offset in range(0, len(raw), 250000):
            self.c.insert_df(self.full(name), raw.iloc[offset:offset+250000])
        self.c.command(f'OPTIMIZE TABLE {self.full(name)} FINAL')
        got = np.asarray(self.c.query(f'SELECT eid,qid,item,day,value FROM {self.full(name)} ORDER BY eid').result_rows,
                         dtype=np.uint64).reshape(-1, 5)
        expected = raw.sort_values('eid').to_numpy(dtype=np.uint64)
        assert np.array_equal(got, expected)
        digest = lambda array: hashlib.sha256(np.asarray(array, dtype='<u8').tobytes()).hexdigest()
        self.history_checks.append({'table': name, 'rows': len(raw),
                                    'columns': ['eid', 'qid', 'item', 'day', 'value'],
                                    'ordering': 'eid', 'serialization': 'little-endian uint64, C row-major',
                                    'expected_sha256': digest(expected), 'native_sha256': digest(got),
                                    'ordered_rows_exact': True})
        if self.proof_path is not None:
            save(self.proof_path, self.history_checks)

    def materialize(self, raw_name, mooc=True):
        self.serial += 1
        prefix = 'g' + str(self.serial)
        tables = {}
        configs = CONFIGS if mooc else [f'd{d}_N' for d in CATALOG_DAYS]
        for config in configs:
            days, style = mc.cfg(config)
            name = prefix + '_' + config
            fields = 'qid UInt32 CODEC(ZSTD(3)),item UInt32 CODEC(ZSTD(3)),bin UInt8 CODEC(ZSTD(3)),n UInt64 CODEC(ZSTD(3))'
            if style == 'NS':
                fields += ',s UInt64 CODEC(ZSTD(3))'
            self.create(name, fields, '(qid,item,bin)')
            suffix = ',sum(value)' if style == 'NS' else ''
            self.c.command(f'INSERT INTO {self.full(name)} SELECT qid,item,toUInt8(intDiv(day,{days})),count(){suffix} '
                           f'FROM {self.full(raw_name)} GROUP BY qid,item,intDiv(day,{days})')
            self.c.command(f'OPTIMIZE TABLE {self.full(name)} FINAL')
            tables[config] = name
        # Native exact-monitor component is also available to independent tasks.
        name = prefix + '_monitor'
        self.create(name, 'qid UInt32 CODEC(ZSTD(3)),n UInt64 CODEC(ZSTD(3)),s UInt64 CODEC(ZSTD(3))', 'qid')
        self.c.command(f'INSERT INTO {self.full(name)} SELECT qid,count(),sum(value) '
                       f'FROM {self.full(raw_name)} GROUP BY qid')
        self.c.command(f'OPTIMIZE TABLE {self.full(name)} FINAL')
        tables['monitor'] = name
        quoted = ','.join("'"+x+"'" for x in tables.values())
        sizes = dict(self.c.query(f"SELECT table,sum(bytes_on_disk) FROM system.parts WHERE active "
                                 f"AND database='{self.db}' AND table IN ({quoted}) GROUP BY table").result_rows)
        return tables, {config: int(sizes[name]) for config, name in tables.items()}

    def retire(self, tables):
        for name in tables.values():
            assert name in self.owned
            self.c.command(f'DROP TABLE {self.full(name)} SYNC')
            self.owned.remove(name)

    def mooc_answer(self, tables, cases, models, ck, rk, task=None):
        cd, ct = mc.cfg(ck)
        rd, rt = mc.cfg(rk)
        if task == 'class':
            days, style, key = cd, ct, ck
        elif task == 'rank':
            days, style, key = rd, rt, rk
        else:
            days, style = min(cd, rd), 'NS'
            key = f'd{days}_NS'
        table = tables[key]
        rows = np.asarray(self.c.query(f'SELECT qid,item,bin,n,' + ('s' if style == 'NS' else 'toUInt64(0)') +
                          f' FROM {self.full(table)}').result_rows, dtype=np.int64).reshape(-1, 5)
        sizes = np.asarray([len(c['ids']) for c in cases])
        offsets = np.r_[0, np.cumsum(sizes)[:-1]]
        n = np.zeros((int(sizes.sum()), 28//days), dtype=np.int64)
        s = np.zeros_like(n)
        n[offsets[rows[:, 0]]+rows[:, 1], rows[:, 2]] = rows[:, 3]
        s[offsets[rows[:, 0]]+rows[:, 1], rows[:, 2]] = rows[:, 4]
        static = np.concatenate([c['static'] for c in cases])
        target = np.asarray([c['target'] for c in cases])
        result = {}
        if task in (None, 'class'):
            xc, _ = mc.features(n, s, static, offsets, target, ck, (days, days))
            result['p'] = models['class'][ck].predict_proba(xc)[:, 1]
        if task in (None, 'rank'):
            _, xr = mc.features(n, s, static, offsets, target, rk, (days, days))
            result['r'] = mc.rankings(models['rank'][rk].decision_function(xr), offsets)
        if task is None:
            result['monitor'] = np.c_[np.add.reduceat(n.sum(1), offsets), np.add.reduceat(s.sum(1), offsets)]
        return result

    def transfer_answer(self, tables, people, model, days, channels, course_flag):
        rows = np.asarray(self.c.query(f'SELECT qid,bin,item,n FROM {self.full(tables[f"d{days}_N"])}').result_rows,
                          dtype=np.int64).reshape(-1, 4)
        a = np.zeros((len(people), 56//days, channels), dtype=np.int64)
        a[rows[:, 0], rows[:, 1], rows[:, 2]] = rows[:, 3]
        return {'p': model.predict_proba(transfer_features(a, people, course_flag))[:, 1],
                'monitor': a.sum(1)}

    def close(self):
        assert self.db.startswith('gform_selection_')
        self.c.command(f'DROP DATABASE {self.db} SYNC')


def model_bytes(payload):
    return len(runtime.dumps(payload).encode('utf-8'))


def mooc_refs(pred):
    q = {k: pred[k]['quality'] for k in CONFIGS}
    cr = min(CONFIGS, key=lambda k: (-q[k]['auc'], q[k]['logloss'], k))
    rr = min(CONFIGS, key=lambda k: (-q[k]['ndcg5'], -q[k]['hit5'], k))
    return {'class': cr, 'rank': rr, 'auc': q[cr]['auc'], 'logloss': q[cr]['logloss'],
            'hit5': q[rr]['hit5'], 'ndcg5': q[rr]['ndcg5']}


def feasible(row, refs, limits=LIMITS):
    return (row['auc'] >= refs['auc']-limits['auc_drop']-1e-12 and
            row['logloss'] <= refs['logloss']+limits['logloss_increase']+1e-12 and
            row['hit5'] >= refs['hit5']-limits['hit5_drop']-1e-12 and
            row['ndcg5'] >= refs['ndcg5']-limits['ndcg5_drop']-1e-12 and
            row['seconds'] <= refs['seconds']*limits['time_ratio_max'])


def catalog_rows(payload, pred, sizes, latencies):
    rows = []
    for ck, rk in itertools.product(CONFIGS, repeat=2):
        physical_days = min(mc.cfg(ck)[0], mc.cfg(rk)[0])
        models = {'classifier': payload['class'][ck], 'ranker': payload['rank'][rk]}
        plan = [{'slot': 0, 'name': 'shared', 'days': physical_days,
                 'by_item': True, 'with_sum': True}]
        table_bytes = sizes[f'd{physical_days}_NS']
        mb, pb = model_bytes(models), model_bytes(plan)
        rows.append({'id': ck+'__'+rk, 'class': ck, 'rank': rk,
                     'physical_days': physical_days, 'table_bytes': table_bytes,
                     'model_bytes': mb, 'metadata_bytes': pb, 'total_bytes': table_bytes+mb+pb,
                     'auc': pred[ck]['quality']['auc'], 'logloss': pred[ck]['quality']['logloss'],
                     'hit5': pred[rk]['quality']['hit5'], 'ndcg5': pred[rk]['quality']['ndcg5'],
                     'seconds': float(latencies[ck+'__'+rk])})
    return rows


def choose(rows, payload, pred, sizes, refs, limits=LIMITS):
    good = [row for row in rows if feasible(row, refs, limits)]
    chosen = min(good, key=lambda row: (row['total_bytes'], row['seconds'], row['id'])) if good else None
    q = {k: pred[k]['quality'] for k in CONFIGS}
    eligible_c = [k for k in CONFIGS if q[k]['auc'] >= refs['auc']-limits['auc_drop']-1e-12
                  and q[k]['logloss'] <= refs['logloss']+limits['logloss_increase']+1e-12]
    eligible_r = [k for k in CONFIGS if q[k]['hit5'] >= refs['hit5']-limits['hit5_drop']-1e-12
                  and q[k]['ndcg5'] >= refs['ndcg5']-limits['ndcg5_drop']-1e-12]
    ck = min(eligible_c, key=lambda k: (sizes[k]+model_bytes(payload['class'][k]), k)) if eligible_c else None
    rk = min(eligible_r, key=lambda k: (sizes[k]+model_bytes(payload['rank'][k]), k)) if eligible_r else None
    union = next((row for row in rows if row['class'] == ck and row['rank'] == rk), None)
    return {'chosen': chosen, 'union': union, 'union_feasible': bool(union and feasible(union, refs, limits)),
            'feasible_pairs': len(good), 'reference': refs}


def check_mooc(actual, pred, monitor, ck, rk):
    if 'p' in actual:
        assert np.max(np.abs(actual['p']-pred[ck]['p'])) <= 1e-12
    if 'r' in actual:
        assert np.array_equal(actual['r'], pred[rk]['r'])
    if 'monitor' in actual:
        assert np.array_equal(actual['monitor'], monitor)


def calibration_orders(variants, rounds, warmups, seed):
    """Random permutation/reversal pairs balance every candidate-pair precedence.

    Ten rounds cannot balance 64 candidates over all 64 absolute positions.
    Five seeded permutations and their reversals nevertheless give each pair
    exactly five precedences in each direction, including the daily reference.
    """
    variants = list(variants)
    rng = random.Random(seed)
    measured = []
    if rounds == 1:
        order = variants.copy()
        rng.shuffle(order)
        measured.append(order)
    else:
        assert rounds % 2 == 0
        for _ in range(rounds//2):
            order = variants.copy()
            rng.shuffle(order)
            measured.extend((order, list(reversed(order))))
        rng.shuffle(measured)
        positions = [{v: order.index(v) for v in variants} for order in measured]
        for a, b in itertools.combinations(variants, 2):
            assert sum(pos[a] < pos[b] for pos in positions) == rounds//2
    result = {}
    for rep in range(-warmups, 0):
        order = variants.copy()
        rng.shuffle(order)
        result[rep] = order
    result.update(enumerate(measured))
    return result


def calibrate_mooc(native, name, raw, cases, models, payload, out, smoke=False):
    native.load(name, raw)
    tables, sizes = native.materialize(name)
    pred, monitor = mc.raw_predictions(cases, models)
    refs = mooc_refs(pred)
    ids = [a+'__'+b for a, b in itertools.product(CONFIGS, repeat=2)]
    journal = runtime.Journal(out/'calibration', 'selection_calibration')
    timings = {key: [] for key in ids}
    checks = []
    try:
        rounds, warmups = (1, 0) if smoke else (10, 3)
        orders = calibration_orders(ids, rounds, warmups, 790041)
        journal.save('candidate_order.json', {
            'seed': 790041, 'orders': orders, 'measured_rounds': rounds,
            'method': ('One seeded random permutation' if smoke else
                       'Five seeded random permutations plus each reversal; random sequence of the ten rounds'),
            'pairwise_precedence_per_direction': None if smoke else rounds//2,
            'pairwise_precedence_check_passed': not smoke,
            'absolute_position_uniformity': False,
            'position_note': ('Pairwise precedence balance is not assessed in a one-round smoke check.' if smoke else
                              'Ten rounds cannot cover all 64 positions; precedence for every pair, including the daily reference, is balanced.')})
        for rep, order in orders.items():
            gc.collect()
            rows = []
            for position, key in enumerate(order):
                ck, rk = key.split('__')
                seconds, actual = runtime.timed(lambda: native.mooc_answer(tables, cases, models, ck, rk))
                check_mooc(actual, pred, monitor, ck, rk)
                rows.append({'dataset': 'MoocRadar', 'snapshot': name, 'factor': 1, 'session': 0,
                             'round': rep, 'warmup': rep < 0, 'variant': key, 'phase': 'calibration_answer',
                             'order': position, 'seconds': seconds})
                if rep >= 0:
                    timings[key].append(seconds)
            journal.pair(rows)
        latencies = {key: float(np.mean(values)) for key, values in timings.items()}
        refs['seconds'] = latencies['d1_NS__d1_NS']
        candidates = catalog_rows(payload, pred, sizes, latencies)
        result = choose(candidates, payload, pred, sizes, refs)
        if result['chosen'] is None:
            raise RuntimeError('No calibration-feasible MoocRadar representation under frozen constraints')
        result.update({'population': len(cases), 'cut': name, 'test_labels_used': False,
                       'physical_catalog': 'One combined count/success table at finest requested day interval',
                       'candidate_count': len(candidates), 'oracle_checks': rounds*64+warmups*64})
        save(out/'selection.json', result)
        save(out/'candidates.json', candidates)
        save(out/'models.json', payload)
        pd.DataFrame(candidates).to_csv(out/'candidates.csv', index=False)
        journal.finish({'native_oracle_checks': result['oracle_checks']})
        return result, candidates, pred, sizes
    finally:
        native.retire(tables)


def predictions(cases, models, ck, rk, cr, rr, fold, cut):
    pred, _ = mc.raw_predictions(cases, models)
    rows = []
    for i, case in enumerate(cases):
        row = {'fold': fold, 'cut': cut, 'user_id': case['key'][0], 'course_id': case['key'][1],
               'y': case['y'], 'catalog_size': len(case['ids'])}
        popular = np.lexsort((np.arange(len(case['ids'])), -case['static'][:, 0]))[:5]
        for name, cl, ra in [('selected', ck, rk), ('reference', cr, rr), ('full', 'd1_NS', 'd1_NS')]:
            rank = pred[ra]['r'][i]
            where = np.flatnonzero(rank == case['target'])
            row[name+'_p'] = float(pred[cl]['p'][i])
            row[name+'_hit'] = int(len(where) > 0)
            row[name+'_ndcg'] = float(1/np.log2(where[0]+2)) if len(where) else 0.
        where = np.flatnonzero(popular == case['target'])
        row['popularity_hit'] = int(len(where) > 0)
        row['popularity_ndcg'] = float(1/np.log2(where[0]+2)) if len(where) else 0.
        row['random_hit'] = 5/len(case['ids'])
        row['random_ndcg'] = float(np.sum(1/np.log2(np.arange(5)+2))/len(case['ids']))
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(frame, name):
    result = {'hit5': float(frame[name+'_hit'].mean()), 'ndcg5': float(frame[name+'_ndcg'].mean())}
    if name in ('selected', 'reference', 'full'):
        result.update(auc=float(roc_auc_score(frame.y, frame[name+'_p'])) if frame.y.nunique() == 2 else None,
                      logloss=float(log_loss(frame.y, frame[name+'_p'], labels=[0, 1])))
    return result


def describe(frame, bootstrap=1000):
    groups = []
    for (fold, cut), group in frame.groupby(['fold', 'cut']):
        a, b = metrics(group, 'selected'), metrics(group, 'reference')
        groups.append({'fold': int(fold), 'cut': cut, 'n': len(group), 'learners': int(group.user_id.nunique()),
                       'selected': a, 'reference': b,
                       'class_pass': a['auc'] >= b['auc']-.01 and a['logloss'] <= b['logloss']+.02,
                       'rank_pass': a['hit5'] >= b['hit5']-.02 and a['ndcg5'] >= b['ndcg5']-.02})
    result = {'observations': len(frame), 'learners': int(frame.user_id.nunique()),
              'catalog': {'minimum': int(frame.catalog_size.min()), 'median': float(frame.catalog_size.median()),
                          'maximum': int(frame.catalog_size.max())},
              'pooled': {name: metrics(frame, name) for name in ('selected', 'reference', 'full', 'popularity', 'random')},
              'per_cut': groups, 'class_pass_cuts': sum(g['class_pass'] for g in groups),
              'rank_pass_cuts': sum(g['rank_pass'] for g in groups), 'independent': False,
              'bootstrap': {'unit': 'learner', 'replicates': bootstrap, 'seed': 530017}}
    rng = np.random.default_rng(530017)
    users = frame.user_id.unique()
    indices = {u: np.flatnonzero(frame.user_id.to_numpy() == u) for u in users}
    stats = []
    for _ in range(bootstrap):
        group = frame.iloc[np.concatenate([indices[u] for u in rng.choice(users, len(users), replace=True)])]
        if group.y.nunique() < 2:
            continue
        a, b = metrics(group, 'selected'), metrics(group, 'reference')
        stats.append([a[k]-b[k] for k in ('auc', 'logloss', 'hit5', 'ndcg5')])
    arr = np.asarray(stats)
    a, b = result['pooled']['selected'], result['pooled']['reference']
    result['paired_differences'] = {k: {'delta': a[k]-b[k], 'ci95': np.quantile(arr[:, j], [.025, .975]).tolist()}
                                    for j, k in enumerate(('auc', 'logloss', 'hit5', 'ndcg5'))}
    return result


def sensitivity(candidates, payload, pred, sizes, refs, out):
    rows = []
    for au, ll, nd, hi, tr in itertools.product([0, .005, .01, .02, .05], [0, .01, .02, .05],
                                               [0, .005, .01, .02, .05], [0, .01, .02, .05], [1, 1.25, 1.5]):
        limits = dict(auc_drop=au, logloss_increase=ll, ndcg5_drop=nd, hit5_drop=hi, time_ratio_max=tr)
        result = choose(candidates, payload, pred, sizes, refs, limits)
        chosen, union = result['chosen'], result['union']
        rows.append({**limits, 'chosen': chosen['id'] if chosen else None, 'union': union['id'] if union else None,
                     'union_feasible': result['union_feasible'], 'same_models': bool(chosen and union and chosen['id'] == union['id']),
                     'same_layout': bool(chosen and union and chosen['physical_days'] == union['physical_days'])})
    pd.DataFrame(rows).to_csv(out/'sensitivity.csv', index=False)
    save(out/'sensitivity_summary.json', {'combinations': len(rows),
         'joint_feasible': sum(row['chosen'] is not None for row in rows),
         'union_feasible': sum(row['union_feasible'] for row in rows),
         'same_models': sum(row['same_models'] for row in rows), 'same_layout': sum(row['same_layout'] for row in rows)})


def transfer_cube(events, npeople, days, channels):
    result = np.zeros((npeople, 56//days, channels), dtype=np.int64)
    np.add.at(result, (events.qid.to_numpy(dtype=int), events.day.to_numpy(dtype=int)//days,
                      events.channel.to_numpy(dtype=int)), 1)
    return result


def transfer_features(cube, people, course_flag):
    result = np.log1p(cube.reshape(len(cube), -1))
    return np.column_stack((result, people.course_flag.to_numpy())) if course_flag else result


def transfer_score(y, probability):
    return {'n': len(y), 'auc': float(roc_auc_score(y, probability)),
            'logloss': float(log_loss(y, probability, labels=[0, 1])),
            'brier': float(brier_score_loss(y, probability)),
            'accuracy': float(accuracy_score(y, probability >= .5))}


def transfer_bootstrap(people, reference, selected, course_flag, replicates):
    y = people.y.to_numpy()
    if course_flag:
        cluster_values = people.USER_ID.to_numpy()
        clusters = sorted(np.unique(cluster_values))
        indices = {cluster: np.flatnonzero(cluster_values == cluster) for cluster in clusters}
    else:
        indices = [np.flatnonzero(y == value) for value in sorted(np.unique(y))]
    rng = np.random.default_rng(700022)
    rows = []
    for _ in range(replicates):
        if course_flag:
            draw = rng.choice(clusters, len(clusters), replace=True)
            ix = np.concatenate([indices[cluster] for cluster in draw])
        else:
            ix = np.concatenate([rng.choice(index, len(index), replace=True) for index in indices])
        if len(np.unique(y[ix])) < 2:
            continue
        a, b = transfer_score(y[ix], reference[ix]), transfer_score(y[ix], selected[ix])
        rows.append([b['auc']-a['auc'], b['logloss']-a['logloss']])
    a, b = transfer_score(y, reference), transfer_score(y, selected)
    return {'resamples': replicates, 'valid_resamples': len(rows), 'seed': 700022,
            'unit': 'published USER_ID cluster, retaining all course participations' if course_flag else 'participation, stratified by target',
            'cluster_count': len(clusters) if course_flag else len(people),
            'participation_count': len(people),
            'identity_note': 'Repeated published identifiers are kept together without asserting independently verified person identity.' if course_flag else 'One unique published participant identifier per test-period row.',
            **{key+'_selected_minus_reference': {'estimate': b[key]-a[key],
               'ci95': np.quantile(np.asarray(rows)[:, i], [.025, .975]).tolist()} for i, key in enumerate(['auc', 'logloss'])}}


def run_transfer(native, dataset, out, smoke):
    is_alpacas = dataset == 'ALPACAS'
    source = REPRO / 'data' / ('alpacas' if is_alpacas else 'moodle') / 'training'
    names = ['1819', '1920', '2021'] if is_alpacas else ['train', 'calibration', 'test']
    train, cal = [pd.read_parquet(source/f'participants_{name}.parquet') for name in names[:2]]
    ht, hc = [pd.read_parquet(source/f'events_{name}.parquet') for name in names[:2]]
    channels = 2 if is_alpacas else len(json.loads((source/'channel_vocabulary.json').read_text())['channels'])
    models, payload, predictions_cal = {}, {}, {}
    for days in CATALOG_DAYS:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=700021)).fit(
            transfer_features(transfer_cube(ht, len(train), days, channels), train, is_alpacas), train.y)
        payload[days] = export_model(model)
        models[days] = Portable(payload[days])
        features = transfer_features(transfer_cube(hc, len(cal), days, channels), cal, is_alpacas)
        predictions_cal[days] = models[days].predict_proba(features)[:, 1]
        assert np.max(np.abs(model.predict_proba(features)[:, 1]-predictions_cal[days])) <= 1e-12
    raw = hc.rename(columns={'channel': 'item'}).copy()
    raw['value'] = np.uint8(0)
    raw = raw[['eid', 'qid', 'item', 'day', 'value']]
    name = dataset.lower() + '_cal'
    native.load(name, raw)
    tables, sizes = native.materialize(name, mooc=False)
    journal = runtime.Journal(out/'calibration', 'selection_calibration')
    latency = {days: [] for days in CATALOG_DAYS}
    try:
        rounds, warmups = (1, 0) if smoke else (10, 3)
        monitor = transfer_cube(hc, len(cal), 56, channels)[:, 0, :]
        for rep, order in runtime.variant_orders([str(d) for d in CATALOG_DAYS], rounds, warmups, 790042).items():
            gc.collect()
            rows = []
            for position, text in enumerate(order):
                days = int(text)
                seconds, actual = runtime.timed(lambda: native.transfer_answer(tables, cal, models[days], days, channels, is_alpacas))
                assert np.max(np.abs(actual['p']-predictions_cal[days])) <= 1e-12
                assert np.array_equal(actual['monitor'], monitor)
                rows.append({'dataset': dataset, 'snapshot': names[1], 'factor': 1, 'session': 0,
                             'round': rep, 'warmup': rep < 0, 'variant': 'd'+text, 'phase': 'calibration_answer',
                             'order': position, 'seconds': seconds})
                if rep >= 0:
                    latency[days].append(seconds)
            journal.pair(rows)
        candidates = []
        for days in CATALOG_DAYS:
            models_payload = {'predictor': payload[days]}
            plan_payload = [{'slot': 0, 'name': 'prediction', 'days': days,
                             'by_item': True, 'with_sum': False}]
            mb, pb = model_bytes(models_payload), model_bytes(plan_payload)
            candidates.append({'days': days, **transfer_score(cal.y, predictions_cal[days]),
                               'table_bytes': sizes[f'd{days}_N'], 'model_bytes': mb, 'metadata_bytes': pb,
                               'total_bytes': sizes[f'd{days}_N']+mb+pb, 'seconds': float(np.mean(latency[days]))})
        ref = min(candidates, key=lambda r: (-r['auc'], r['logloss'], r['days']))
        for row in candidates:
            row['feasible'] = row['auc'] >= ref['auc']-.01 and row['logloss'] <= ref['logloss']+.02 and row['seconds'] <= 1.25*float(np.mean(latency[1]))
        valid = [r for r in candidates if r['feasible']]
        if not valid:
            raise RuntimeError('No calibration-feasible transfer representation under frozen constraints')
        chosen = min(valid, key=lambda r: (r['total_bytes'], r['seconds'], r['days']))
        selection = {'dataset': dataset, 'selected_days': chosen['days'], 'reference_days': ref['days'],
                     'chosen': chosen, 'reference': ref, 'test_labels_used': False,
                     'train_n': len(train), 'calibration_n': len(cal), 'training_seed': 700021}
        save(out/'selection.json', selection)
        save(out/'models.json', payload)
        save(out/'candidates.json', candidates)
        journal.finish({'native_oracle_checks': (rounds+warmups)*5})
    finally:
        native.retire(tables)
    people = pd.read_parquet(source/f'participants_{names[2]}.parquet')
    events = pd.read_parquet(source/f'events_{names[2]}.parquet')
    ds, dr = chosen['days'], ref['days']
    predictions_test = {d: models[d].predict_proba(transfer_features(
        transfer_cube(events, len(people), d, channels), people, is_alpacas))[:, 1] for d in set([1, ds, dr])}
    quality = {'dataset': dataset, 'selected_days': ds, 'reference_days': dr,
               'selected': transfer_score(people.y, predictions_test[ds]),
               'reference': transfer_score(people.y, predictions_test[dr]),
               'daily': transfer_score(people.y, predictions_test[1]),
               'paired_difference': transfer_bootstrap(people, predictions_test[dr], predictions_test[ds], is_alpacas, 25 if smoke else 2000),
               'train_n': len(train), 'calibration_n': len(cal), 'test_n': len(people), 'fresh_local_fitting': True}
    if is_alpacas:
        quality['per_course'] = {str(course): {
            'selected': transfer_score(group.y, predictions_test[ds][group.index]),
            'reference': transfer_score(group.y, predictions_test[dr][group.index])} for course, group in people.groupby('course')}
        prior = train.groupby('course').y.mean().to_dict()
        quality['course_prior_baseline'] = transfer_score(people.y, people.course.map(prior).to_numpy())
    elif 'seen_in_earlier_period' in people:
        ix = np.flatnonzero(~people.seen_in_earlier_period.to_numpy(dtype=bool))
        quality['unseen_id_sensitivity'] = {'n': len(ix), 'selected': transfer_score(people.y.iloc[ix], predictions_test[ds][ix]),
                                          'reference': transfer_score(people.y.iloc[ix], predictions_test[dr][ix])}
    save(out/'quality.json', quality)
    pd.DataFrame({'qid': people.qid, 'y': people.y,
                  'bootstrap_cluster_id': pd.factorize(people.USER_ID, sort=True)[0] if is_alpacas else people.qid,
                  'selected_probability': predictions_test[ds],
                  'reference_probability': predictions_test[dr], 'daily_probability': predictions_test[1]}).to_csv(out/'test_predictions.csv', index=False)
    return {'selected_days': ds, 'models': {'predictor': payload[ds]}, 'training_seed': 700021,
            'selection_file': dataset.lower()+'/selection.json'}, quality


def pipeline_once(native, raw_name, train_cases, calibration_cases, variant, seed):
    """Full fitting/materialization/evaluation/selection cost, not just search CPU."""
    stages = {}
    started = time.perf_counter()
    models, payload = train_mooc(train_cases, seed)
    stages['training_seconds'] = time.perf_counter()-started
    started = time.perf_counter()
    tables, sizes = native.materialize(raw_name)
    stages['materialization_seconds'] = time.perf_counter()-started
    try:
        started = time.perf_counter()
        pred, monitor = mc.raw_predictions(calibration_cases, models)
        refs = mooc_refs(pred)
        outputs, latency = {}, {}
        if variant == 'joint_catalog':
            pairs = list(itertools.product(CONFIGS, repeat=2))
            for ck, rk in pairs:
                seconds, actual = runtime.timed(lambda: native.mooc_answer(tables, calibration_cases, models, ck, rk))
                outputs[ck+'__'+rk] = actual
                latency[ck+'__'+rk] = seconds
            refs['seconds'] = latency['d1_NS__d1_NS']
            rows = catalog_rows(payload, pred, sizes, latency)
            choice = choose(rows, payload, pred, sizes, refs)
            evaluations = 64
        else:
            # Independently evaluate all eight task models, then construct/test their union.
            for task in ('class', 'rank'):
                for config in CONFIGS:
                    seconds, actual = runtime.timed(lambda: native.mooc_answer(
                        tables, calibration_cases, models, config, config, task))
                    outputs[task+'_'+config] = actual
            q = {k: pred[k]['quality'] for k in CONFIGS}
            cls = [k for k in CONFIGS if q[k]['auc'] >= refs['auc']-.01-1e-12 and q[k]['logloss'] <= refs['logloss']+.02+1e-12]
            ranks = [k for k in CONFIGS if q[k]['hit5'] >= refs['hit5']-.02-1e-12 and q[k]['ndcg5'] >= refs['ndcg5']-.02-1e-12]
            ck = min(cls, key=lambda k: (sizes[k]+model_bytes(payload['class'][k]), k))
            rk = min(ranks, key=lambda k: (sizes[k]+model_bytes(payload['rank'][k]), k))
            # The daily reference and union are both measured for the same latency constraint.
            seconds, baseline = runtime.timed(lambda: native.mooc_answer(tables, calibration_cases, models, 'd1_NS', 'd1_NS'))
            refs['seconds'] = seconds
            outputs['d1_NS__d1_NS'] = baseline
            seconds, actual = runtime.timed(lambda: native.mooc_answer(tables, calibration_cases, models, ck, rk))
            outputs[ck+'__'+rk] = actual
            latency = {a+'__'+b: 0.0 for a, b in itertools.product(CONFIGS, repeat=2)}
            latency[ck+'__'+rk] = seconds
            rows = catalog_rows(payload, pred, sizes, latency)
            union = next(row for row in rows if row['class'] == ck and row['rank'] == rk)
            choice = {'chosen': union if feasible(union, refs) else None,
                      'union': union, 'union_feasible': feasible(union, refs), 'reference': refs}
            evaluations = 18
        stages['evaluation_selection_seconds'] = time.perf_counter()-started
        # Keep output checks outside the enclosing timer.
        result = {'choice': choice, 'evaluations': evaluations, 'stages': stages,
                  'outputs': outputs, 'pred': pred, 'monitor': monitor}
    finally:
        started = time.perf_counter()
        native.retire(tables)
        stages['retirement_seconds'] = time.perf_counter()-started
    return result


def check_pipeline(result):
    for key, actual in result['outputs'].items():
        if '__' in key:
            ck, rk = key.split('__')
        else:
            task, config = key.split('_', 1)
            ck = rk = config
        check_mooc(actual, result['pred'], result['monitor'], ck, rk)


def run(out: Path, smoke: bool = False):
    out = Path(out)
    journal = runtime.Journal(out, 'selection')
    client = native = None
    all_quality, overlay = {}, {}
    try:
        with runtime.failure_record(journal), runtime.numeric_context():
            history = pd.read_parquet(REPRO/'data/moocradar/history.parquet')
            history['ts'] = pd.to_datetime(history.ts)
            assert len(history) == 898933
            client = runtime.connect()
            native = CatalogNative(client, out/'native_history_checks.json')
            journal.save('environment.json', runtime.environment(client))
            cache = {}
            def snapshot(cut):
                key = str(pd.Timestamp(cut))
                if key not in cache:
                    cache[key] = mc.snapshot(history, cut)
                return cache[key]
            def training(cut):
                return [case for t in pd.date_range('2020-05-25', pd.Timestamp(cut)-pd.Timedelta(days=7), freq='7D')
                        for case in snapshot(t)]
            calibration_cut = '2020-11-02'
            train_cases, cal_cases = training(calibration_cut), snapshot(calibration_cut)
            raw = raw_window(history, cal_cases, calibration_cut)
            models, payload = train_mooc(train_cases, 400016)
            primary = out/'moocradar_primary'
            selection, candidates, pred, sizes = calibrate_mooc(native, 'primary_cal', raw, cal_cases, models, payload, primary, smoke)
            chosen, refs = selection['chosen'], selection['reference']
            frames = [predictions(snapshot(cut), models, chosen['class'], chosen['rank'], refs['class'], refs['rank'], -1, cut) for cut in TEST_CUTS]
            frame = pd.concat(frames, ignore_index=True)
            frame.to_parquet(primary/'test_predictions.parquet', index=False)
            quality = describe(frame, 25 if smoke else 1000)
            quality.update({'training_seed': 400016, 'train_n': len(train_cases), 'calibration_n': len(cal_cases),
                            'class_config': chosen['class'], 'rank_config': chosen['rank'], 'reference': refs, 'fresh_local_fitting': True})
            save(primary/'quality.json', quality)
            sensitivity(candidates, payload, pred, sizes, refs, primary)
            all_quality['MoocRadar_primary'] = quality
            overlay['MoocRadar'] = {'class_config': chosen['class'], 'rank_config': chosen['rank'],
                                    'models': {'classifier': payload['class'][chosen['class']], 'ranker': payload['rank'][chosen['rank']]},
                                    'fixed_maintenance_models': {'classifier': payload['class']['d28_NS'], 'ranker': payload['rank']['d28_N']},
                                    'training_seed': 400016, 'calibration_cut': calibration_cut}
            print('SELECTION primary MoocRadar quality complete', flush=True)
            temporal_frames, fold_records = [], []
            folds = mc.protocol()['temporal_folds']
            if smoke:
                folds = [folds[-1]]
            for fi, fold in enumerate(folds):
                cal = fold['calibration']
                tr, cases = training(cal), snapshot(cal)
                fresh, fresh_payload = train_mooc(tr, 530017)
                folder = out/'moocradar_temporal'/f'fold{fi}'
                selected, _, _, _ = calibrate_mooc(native, f'fold{fi}_cal', raw_window(history, cases, cal), cases,
                                                    fresh, fresh_payload, folder, smoke)
                choice, reference = selected['chosen'], selected['reference']
                frames = []
                for offset in fold['test_offsets_days']:
                    cut = str((pd.Timestamp(cal)+pd.Timedelta(days=offset)).date())
                    frames.append(predictions(snapshot(cut), fresh, choice['class'], choice['rank'], reference['class'], reference['rank'], fi, cut))
                df = pd.concat(frames, ignore_index=True)
                df.to_parquet(folder/'predictions.parquet', index=False)
                temporal_frames.append(df)
                fold_records.append({'fold': fi, 'calibration': cal, 'train_n': len(tr), 'calibration_n': len(cases),
                                     'validation_n': len(df), 'chosen': choice['id'], 'reference': [reference['class'], reference['rank']],
                                     'training_seed': 530017})
                print('SELECTION temporal fold ' + str(fi) + ' complete', flush=True)
            temporal = pd.concat(temporal_frames, ignore_index=True)
            temporal.to_parquet(out/'moocradar_temporal/predictions.parquet', index=False)
            temporal_quality = describe(temporal, 25 if smoke else 1000)
            temporal_quality['folds'] = fold_records
            temporal_quality['fresh_local_fitting'] = True
            save(out/'moocradar_temporal/quality.json', temporal_quality)
            all_quality['MoocRadar_temporal'] = temporal_quality
            for dataset in ('ALPACAS', 'Moodle'):
                overlay[dataset], all_quality[dataset] = run_transfer(native, dataset, out/dataset.lower(), smoke)
                print('SELECTION ' + dataset + ' quality complete', flush=True)
            save(out/'overlay.json', overlay)
            save(out/'quality_summary.json', all_quality)
            journal.event('selection_frozen', overlay_sha256=runtime.sha(out/'overlay.json'), test_labels_used=False)
            # Runtime comparison uses the declared fourth rolling-origin training workload.
            # It is separate from the primary model seed, while sharing this local server.
            sessions, rounds, warmups = (1, 1, 0) if smoke else (3, 10, 3)
            choices = []
            for session in range(sessions):
                orders = runtime.variant_orders(['joint_catalog', 'independent_union'], rounds, warmups, 790043+session)
                for rep, order in orders.items():
                    gc.collect()
                    rows = []
                    for position, variant in enumerate(order):
                        seconds, result = runtime.timed(lambda: pipeline_once(native, 'primary_cal', train_cases, cal_cases, variant, 530017))
                        check_pipeline(result)
                        choice = result['choice']['chosen']
                        rows.append({'dataset': 'MoocRadar', 'snapshot': '2020-11-02', 'factor': 1,
                                     'session': session, 'round': rep, 'warmup': rep < 0, 'variant': variant,
                                     'phase': 'selection_pipeline', 'order': position, 'seconds': seconds,
                                     'native_evaluations': result['evaluations'], **result['stages']})
                        choices.append({'session': session, 'round': rep, 'warmup': rep < 0,
                                        'variant': variant, 'chosen': choice['id'] if choice else None,
                                        'union_feasible': result['choice']['union_feasible']})
                    journal.pair(rows)
                    save(out/'pipeline_choices.json', choices)
                    print(f'SELECTION pipeline session={session+1} round={rep+1}', flush=True)
            journal.finish({'smoke': smoke, 'fresh_local_fitting': True,
                            'primary_training_seed': 400016, 'temporal_and_pipeline_seed': 530017,
                            'overlay_sha256': runtime.sha(out/'overlay.json'),
                            'native_history_checks': len(native.history_checks),
                            'native_history_checks_sha256': runtime.sha(out/'native_history_checks.json'),
                            'physical_catalog': 'Combined N/S table at finest requested day interval; exact success monitoring retained',
                            'pipeline_evaluations': {'joint_catalog': 64, 'independent_union': 18},
                            'pipeline_scope': 'Fit 16 models; materialize 8 candidate aggregates and monitor; evaluate native candidates; choose; retire owned aggregate generation. Common event window/cases supplied.'})
    finally:
        if native is not None:
            native.close()
        if client is not None:
            client.close()
    return clean(all_quality)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.out, args.smoke)
