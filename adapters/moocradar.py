"""MoocRadar adapter preserving classifier, top-5 ranking and exact monitoring.

Canonical inputs are loaded from prepared snapshots. Model parameters are
supplied by the current campaign's training and calibration stage.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
INPUT = P.parent / 'data/moocradar/serving'
CUTS = ['2020-11-16', '2020-11-23', '2020-11-30', '2020-12-07']


def linear_score(features, model):
    return ((features - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(model['coef']) + float(model['intercept'])


def ranking(scores, offsets, sizes):
    return np.asarray([np.lexsort((np.arange(int(size)), -scores[int(offset):int(offset + size)]))[:5]
                       for offset, size in zip(offsets, sizes)], dtype=np.int64)


class Snapshot:
    dataset = 'MoocRadar'
    window_days = 28

    def __init__(self, folder, models):
        self.folder = Path(folder)
        self.name = 'moocradar_' + self.folder.name
        self.cut = self.folder.name
        self.raw = pd.read_parquet(self.folder / 'raw.parquet')
        self.raw_changed = self.raw.loc[self.raw.eid % 100 != 0].copy()
        self.people = pd.read_parquet(self.folder / 'participants.parquet')
        self.catalog = pd.read_parquet(self.folder / 'catalog.parquet')
        self.catalog_changed = pd.read_parquet(self.folder / 'catalog_changed.parquet')
        with np.load(self.folder / 'context.npz') as context:
            self.offsets = context['offsets'].copy()
            self.sizes = context['sizes'].copy()
            self.target = context['target'].copy()
            self.static = context['static'].copy()
        with np.load(self.folder / 'context_changed.npz') as context:
            self.static_changed = context['static'].copy()
        self.npeople = self.population = len(self.people)
        self.total_items = int(self.sizes.sum())
        self.models = self.model_payload = models
        base_files = ['participants.parquet', 'catalog.parquet', 'context.npz']
        changed_files = ['participants.parquet', 'catalog_changed.parquet', 'context_changed.npz']
        self.metadata_bytes = sum((self.folder / f).stat().st_size for f in base_files)
        self.changed_metadata_bytes = sum((self.folder / f).stat().st_size for f in changed_files)
        assert np.array_equal(self.people.qid.to_numpy(), np.arange(self.npeople))
        assert self.raw.eid.is_unique


def get_snapshots(models):
    return [Snapshot(INPUT / cut, models) for cut in CUTS]

