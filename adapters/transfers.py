"""ALPACAS and Moodle adapters using the current campaign's selected models."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import expit

P = Path(__file__).resolve().parent


class TransferSnapshot:
    def __init__(self, dataset, directory, choice):
        root = P.parent / 'data' / directory / 'serving'
        self.dataset = dataset
        self.name = 'alpacas_2020-21' if dataset == 'ALPACAS' else 'moodle_2025-P1'
        self.window_days = 56
        self.days = int(choice['selected_days'])
        assert self.days in (1, 7, 14, 28, 56)
        self.people = pd.read_parquet(root / 'participants.parquet')
        self.npeople = self.population = len(self.people)
        self.channels = json.loads((root / 'channel_vocabulary.json').read_text())['channels']
        self.nchannels = len(self.channels)
        self.raw = pd.read_parquet(root / 'raw.parquet')
        self.raw_changed = self.raw.loc[self.raw.eid % 100 != 0].copy()
        self.model_payload = choice['models']
        self.models = self.model_payload
        self.metadata_bytes = sum((root / name).stat().st_size
                                 for name in ['serving_population.json', 'channel_vocabulary.json'])
        self.include_course_flag = dataset == 'ALPACAS'
        self._oracle_cache = {}
        assert list(self.raw.columns) == ['eid', 'qid', 'item', 'day', 'value']
        assert self.raw.eid.is_unique and np.array_equal(self.people.qid, np.arange(self.population))
        assert self.raw.qid.between(0, self.population-1).all()
        assert self.raw.item.between(0, self.nchannels-1).all()
        assert self.raw.day.between(0, 55).all() and self.raw.value.eq(0).all()

    def specs(self, shared):
        specs = [{'slot': 0, 'name': 'prediction', 'days': self.days,
                  'by_item': True, 'with_sum': False}]
        if not shared:
            specs.append({'slot': 1, 'name': 'monitoring', 'days': self.window_days,
                          'by_item': True, 'with_sum': False})
        return specs

    def _features(self, counts):
        result = np.log1p(counts.reshape(self.population, -1))
        if self.include_course_flag:
            result = np.column_stack((result, self.people.course_flag.to_numpy(dtype=np.float64)))
        return result

    def _predict(self, counts):
        parameters = self.model_payload['predictor']
        features = self._features(counts)
        mean, scale, coef = (np.asarray(parameters[key], dtype=np.float64)
                             for key in ['mean', 'scale', 'coef'])
        assert features.shape[1] == len(coef) == len(mean) == len(scale)
        return expit(((features - mean) / scale) @ coef + float(parameters['intercept']))

    def evaluate(self, rows, shared, changed=False):
        """Consume common SQL rows: slot, qid, item, bin, n, s."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 6)
        count = np.zeros((self.population, self.window_days // self.days, self.nchannels), dtype=np.int64)
        pred = rows[rows[:, 0] == 0]
        count[pred[:, 1], pred[:, 3], pred[:, 2]] = pred[:, 4]
        if shared:
            monitor = count.sum(axis=1)
        else:
            monitor = np.zeros((self.population, self.nchannels), dtype=np.int64)
            observed = rows[rows[:, 0] == 1]
            monitor[observed[:, 1], observed[:, 2]] = observed[:, 4]
        return {'probability': self._predict(count), 'monitor': monitor}

    def oracle(self, changed=False):
        """Reconstruct expectations independently by grouping canonical raw rows."""
        key = bool(changed)
        if key not in self._oracle_cache:
            raw = self.raw_changed if key else self.raw
            count = np.zeros((self.population, self.window_days // self.days, self.nchannels), dtype=np.int64)
            frame = raw.assign(bin=raw.day // self.days)
            grouped = frame.groupby(['qid', 'bin', 'item'], sort=False).size()
            for (qid, bin_, item), value in grouped.items():
                count[int(qid), int(bin_), int(item)] = int(value)
            monitor = np.zeros((self.population, self.nchannels), dtype=np.int64)
            for (qid, item), value in raw.groupby(['qid', 'item'], sort=False).size().items():
                monitor[int(qid), int(item)] = int(value)
            self._oracle_cache[key] = {'probability': self._predict(count), 'monitor': monitor}
        return self._oracle_cache[key]


def get_snapshots(choices):
    return [TransferSnapshot(dataset, directory, choices[dataset])
            for dataset, directory in [('ALPACAS', 'alpacas'), ('Moodle', 'moodle')]]
