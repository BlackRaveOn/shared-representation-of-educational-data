"""Contract tests for source identity, time windows, aggregates, and corrections."""
from __future__ import annotations

import importlib
import itertools
import json
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.common import MoocSnapshot
from experiments.native import compare
from experiments import maintenance
from experiments import maintenance_algorithms as alg

prep = importlib.import_module('scripts.prepare_moocradar')


def model(width, coef=None, intercept=0.):
    return {'mean': [0.] * width, 'scale': [1.] * width,
            'coef': list(coef if coef is not None else np.linspace(-.03, .04, width)),
            'intercept': intercept}


def history_fixture():
    """Same learner and item names in two course namespaces, plus window edges."""
    lower = pd.Timestamp('2020-01-04')
    rows = []
    for course in ['course_A', 'course_B']:
        times = [lower + pd.Timedelta(days=d) for d in range(20)]
        times += [lower - pd.Timedelta(nanoseconds=1), pd.Timestamp('2020-02-01') - pd.Timedelta(nanoseconds=1),
                  pd.Timestamp('2020-02-01'), pd.Timestamp('2020-02-02')]
        for i, when in enumerate(times):
            rows.append([len(rows), 'same_learner', course, 'item_' + str(i % 6), when, i % 2])
    return pd.DataFrame(rows, columns=prep.HISTORY_COLUMNS)


def assert_rejection_at(case, callback, fragment, error=AssertionError):
    """Require the intended existing guard, not an unrelated downstream error."""
    with case.assertRaises(error):
        callback()
    # guarded_action records the failing guard before unittest clears the traceback.
    case.assertIn(fragment, callback.rejected_line)


def guarded_action(function):
    def action():
        try:
            return function()
        except Exception as error:
            action.rejected_line = traceback.extract_tb(error.__traceback__)[-1].line or ''
            raise
    action.rejected_line = ''
    return action


class DomainPreparationTests(unittest.TestCase):
    def normalize_fixture(self, mutation=None):
        events = [
            {'log_id': 'e0', 'user_id': 'u', 'problem_id': 'pA', 'exercise_id': 'same_item', 'submit_time': '2020-01-10 00:00:00', 'is_correct': 1},
            {'log_id': 'e1', 'user_id': 'u', 'problem_id': 'pA', 'exercise_id': 'same_item', 'submit_time': '2020-01-10 00:00:00', 'is_correct': 1},
            {'log_id': 'e2', 'user_id': 'u', 'problem_id': 'pB', 'exercise_id': 'same_item', 'submit_time': '2020-01-11 00:00:00', 'is_correct': 0},
        ]
        problems = [{'problem_id': 'pA', 'course_id': 'A'}, {'problem_id': 'pB', 'course_id': 'B'}]
        if mutation:
            mutation(events, problems)
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / 'student-problem-middle.json').write_text(json.dumps([{'seq': events}]))
            (base / 'problem.json').write_text('\n'.join(json.dumps(p) for p in problems) + '\n')
            timestamps = pd.to_datetime([e['submit_time'] for e in events])
            manifest = {'source_files': [], 'expected_source_counts': {
                'sequences': 1, 'attempt_rows': len(events), 'problem_catalog_rows': len(problems),
                'users': len({e['user_id'] for e in events}),
                'observed_problem_ids': len({e['problem_id'] for e in events}),
                'observed_exercise_ids': len({e['exercise_id'] for e in events}),
                'catalog_course_ids': len({p['course_id'] for p in problems}),
                'minimum_timestamp': str(timestamps.min()), 'maximum_timestamp': str(timestamps.max())}}
            # Include fixture fingerprints to exercise source validation.
            for name in ('student-problem-middle.json', 'problem.json'):
                p = base / name
                manifest['source_files'].append({'name': name, 'bytes': p.stat().st_size, 'sha256': prep.sha(p)})
            return prep.normalize(base, base / 'out', manifest)

    def test_distinct_event_ids_preserve_equal_observations_and_course_context(self):
        h, report = self.normalize_fixture()
        self.assertEqual(len(h), 3)
        self.assertEqual(h.course_id.tolist(), ['A', 'A', 'B'])
        self.assertTrue(h.eid.is_unique)
        self.assertEqual(report['duplicate_rows_removed'], 0)

    def test_duplicate_event_identity_rejected(self):
        action = guarded_action(lambda: self.normalize_fixture(lambda e,p: e[1].update(log_id=e[0]['log_id'])))
        assert_rejection_at(self, action, 'events.log_id.is_unique')

    def test_ambiguous_problem_mapping_rejected(self):
        action = guarded_action(lambda: self.normalize_fixture(lambda e,p: p.append({'problem_id': 'pA', 'course_id': 'B'})))
        assert_rejection_at(self, action, 'problems.problem_id.is_unique')

    def test_unknown_problem_reference_rejected(self):
        action = guarded_action(lambda: self.normalize_fixture(lambda e,p: e[0].update(problem_id='absent')))
        assert_rejection_at(self, action, 'events.problem_id.isin')

    def test_nonbinary_result_rejected(self):
        action = guarded_action(lambda: self.normalize_fixture(lambda e,p: e[0].update(is_correct=2)))
        assert_rejection_at(self, action, 'events.is_correct.isin')


class WindowTests(unittest.TestCase):
    def test_half_open_window_and_no_future_catalog(self):
        h = history_fixture()
        when, past, window, catalog, keys = prep.common_at_cut(h, '2020-02-01')
        lower = when - pd.Timedelta(days=28)
        self.assertTrue(window.ts.eq(lower).any())
        self.assertFalse(window.ts.eq(when).any())
        self.assertTrue((window.ts >= lower).all() and (window.ts < when).all())
        self.assertEqual(len(window), 42)
        self.assertEqual(len(past), 44)
        future = pd.DataFrame([[1000, 'u_future', 'course_future', 'new_item', when, 1]], columns=prep.HISTORY_COLUMNS)
        _, _, _, catalog2, _ = prep.common_at_cut(pd.concat([h, future]), '2020-02-01')
        pd.testing.assert_frame_equal(catalog, catalog2)
        self.assertEqual(keys, [('same_learner', 'course_A'), ('same_learner', 'course_B')])

    def test_window_shift_rebuilds_membership_bins_and_separate_participations(self):
        h = history_fixture()
        with tempfile.TemporaryDirectory() as td, patch.object(prep, 'CUTS', ['2020-02-01', '2020-02-02']):
            out = Path(td)
            prep.serving_inputs(h, out)
            a = pd.read_parquet(out / '2020-02-01/raw.parquet')
            b = pd.read_parquet(out / '2020-02-02/raw.parquet')
            people = pd.read_parquet(out / '2020-02-01/participants.parquet')
            self.assertEqual(len(people), 2)
            self.assertEqual(people.user_id.nunique(), 1)
            self.assertEqual(people.course_id.nunique(), 2)
            leaving = set(h.loc[h.ts.eq(pd.Timestamp('2020-01-04')), 'eid'])
            entering = set(h.loc[h.ts.eq(pd.Timestamp('2020-02-01')), 'eid'])
            self.assertTrue(leaving <= set(a.eid) and not leaving & set(b.eid))
            self.assertTrue(entering <= set(b.eid) and not entering & set(a.eid))
            overlap = a[['eid', 'day']].merge(b[['eid', 'day']], on='eid', suffixes=('_a', '_b'))
            self.assertTrue((overlap.day_b.astype(int) == overlap.day_a.astype(int) - 1).all())
            self.assertTrue(a.day.between(0, 27).all() and b.day.between(0, 27).all())


def adapter_fixture(class_days=7, rank_days=14):
    raw = pd.DataFrame([(i, q, item, day, (i + q) % 2)
                        for i, (q, item, day) in enumerate(itertools.product(range(2), range(6), [0,6,7,13,14,27]))],
                       columns=['eid', 'qid', 'item', 'day', 'value'])
    context = SimpleNamespace(npeople=3, population=3, total_items=18,
                              sizes=np.array([6,6,6]), offsets=np.array([0,6,12]),
                              target=np.array([0,1,0]), raw=raw,
                              raw_changed=raw.loc[raw.eid % 3 != 0].copy(),
                              static=np.column_stack([np.ones(18), np.linspace(.2,.8,18)]))
    context.static_changed = context.static.copy()
    choice = {'class_config': f'd{class_days}_NS', 'rank_config': f'd{rank_days}_N',
              'models': {'classifier': model(4 * (28 // class_days) + 2),
                         'ranker': model(28 // rank_days + 2)}}
    return MoocSnapshot(context, choice)


def fixture_rows(raw, specs):
    """Independent tiny-fixture materialization; SQL is not executed here."""
    result = []
    for spec in specs:
        grouped = {}
        for row in raw.itertuples(index=False):
            key = (row.qid, row.item if spec['by_item'] else -1, row.day // spec['days'])
            count, value = grouped.get(key, (0,0))
            grouped[key] = count + 1, value + row.value
        for (qid, item, bin_), (count, value) in grouped.items():
            result.append([spec['slot'], qid, item, bin_, count, value if spec['with_sum'] else 0])
    return result


class AggregateTests(unittest.TestCase):
    def test_shared_and_separate_adapters_match_direct_outputs_with_empty_bins(self):
        for cd, rd in [(1,7), (7,14), (14,28), (28,28)]:
            snap = adapter_fixture(cd,rd)
            for changed in [False, True]:
                raw = snap.raw_changed if changed else snap.raw
                expected = snap.oracle(changed)
                for shared in [False, True]:
                    actual = snap.evaluate(fixture_rows(raw, snap.specs(shared)), shared, changed)
                    compare(actual, expected)
                    np.testing.assert_array_equal(actual['monitor'][2], [0,0])

    def test_non_nested_rollup_rejected(self):
        snap = adapter_fixture()
        with self.assertRaises(AssertionError):
            snap._reduce(np.zeros((18,4),dtype=np.int64), 7, 10)

    def test_same_width_different_anchor_is_not_equivalent(self):
        snap = adapter_fixture()
        aligned = np.zeros((18,28),dtype=np.int64)
        aligned[:,6], aligned[:,7] = 1,2
        shifted = np.roll(aligned, 1, axis=1)
        a = snap._reduce(aligned, 1, 7)
        b = snap._reduce(shifted, 1, 7)
        self.assertFalse(np.array_equal(a,b))
        np.testing.assert_array_equal(a.sum(axis=1), b.sum(axis=1))
        # The adapter requires aligned anchors but does not validate anchor metadata.


class JournalTests(unittest.TestCase):
    def invalid_case(self, mutation):
        history = history_fixture()
        requests = pd.DataFrame([[0, 'same_learner', 'course_A', 0, 'item_0']],
                                columns=['qid','user_id','course_id','target_index','target_block'])
        journal = pd.DataFrame([[0,0,'course_A',0,1]], columns=['eid','qid','course_id','bid','new'])
        journal = mutation(journal)
        def read(path, *args, **kwargs):
            return requests.copy() if Path(path).name.startswith('requests_') else journal.copy()
        models = {'classifier': model(6), 'ranker': model(3)}
        with patch.object(maintenance.pd, 'read_parquet', side_effect=read):
            return maintenance.prepare_case(history, models, cut='2020-02-01')

    def test_duplicate_journal_event_rejected(self):
        action = guarded_action(lambda: self.invalid_case(lambda j: pd.concat([j,j],ignore_index=True)))
        assert_rejection_at(self, action, 'journal.eid.is_unique')

    def test_unknown_journal_event_rejected(self):
        action = guarded_action(lambda: self.invalid_case(lambda j: j.assign(eid=9999)))
        assert_rejection_at(self, action, 'journal.eid.isin')

    def test_misrouted_journal_participation_rejected(self):
        action = guarded_action(lambda: self.invalid_case(lambda j: j.assign(qid=1)))
        assert_rejection_at(self, action, 'joined.qid')

    def test_exhaustive_binary_bounds_and_refinement(self):
        checked = 0
        for values in itertools.product([0,1],repeat=5):
            q = {'N': np.array([3,2]), 'S': np.array([sum(values[:3]),sum(values[3:])])}
            for new in itertools.product([0,1],repeat=3):
                j = {'eid': np.array([0,1,3]), 'gid': np.array([0,0,1]), 'new': np.array(new)}
                corrected = list(values)
                for eid, value in zip(j['eid'], new): corrected[eid] = value
                exact = np.array([sum(corrected[:3]),sum(corrected[3:])])
                st = alg.state(q,j)
                low, high = alg.bounds(q,st)
                self.assertTrue(np.all(low <= exact) and np.all(exact <= high))
                reader = SimpleNamespace(read=lambda ids: {int(i):values[int(i)] for i in ids})
                alg.refine(q,j,st,np.array([0]),reader)
                low2, high2 = alg.bounds(q,st)
                self.assertTrue(np.all(low2 >= low) and np.all(high2 <= high))
                self.assertTrue(np.all(low2 <= exact) and np.all(exact <= high2))
                alg.refine(q,j,st,np.array([1,2]),reader)
                low3, high3 = alg.bounds(q,st)
                np.testing.assert_array_equal(low3,exact)
                np.testing.assert_array_equal(high3,exact)
                checked += 1
        self.assertEqual(checked,256)

    def test_too_many_changed_records_rejected(self):
        q={'N':np.array([1]),'S':np.array([1])}
        j={'eid':np.array([0,1]),'gid':np.array([0,0]),'new':np.array([0,1])}
        with self.assertRaises(AssertionError): alg.state(q,j)

    def test_nonbinary_journal_value_is_an_unchecked_precondition(self):
        # The kernel assumes binary inputs. An invalid value can still yield a feasible sum.
        q={'N':np.array([3]),'S':np.array([0])}
        j={'eid':np.array([0]),'gid':np.array([0]),'new':np.array([2])}
        low,high=alg.bounds(q,alg.state(q,j))
        np.testing.assert_array_equal(low,[2])
        np.testing.assert_array_equal(high,[2])

    def test_class_boundary_requires_refinement_and_rank_ties_are_stable(self):
        cases=[{'key':('u','c'),'ids':np.arange(6),'target':0}]
        cat=pd.DataFrame({'course_id':['c']*6,'bid':np.arange(6),'n':[2]*6,'s':[1]*6})
        q=alg.structure(cases,cat,1)
        q.update(N=np.array([2]*6),S=np.array([1]*6),catN=np.array([2]*6),catS=np.array([1]*6))
        models={'classifier':model(6,[0.]*6),'ranker':model(3,[0.]*3)}
        out, class_ok, rank_ok=alg.certify(q,np.zeros(6),np.ones(6)*2,models)
        self.assertFalse(bool(class_ok[0]))
        self.assertTrue(bool(rank_ok[0]))
        np.testing.assert_array_equal(out['r'],[[0,1,2,3,4]])
        exact=alg.answer(q,q['S'],models)
        np.testing.assert_array_equal(exact['p'],[.5])
        np.testing.assert_array_equal(exact['cl'],[True])
        _, class_ok, rank_ok=alg.certify(q,q['S'],q['S'],models)
        self.assertTrue(bool(class_ok[0]) and bool(rank_ok[0]))


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Run tests without -O to enable assertion guards.')
    unittest.main(verbosity=2)
