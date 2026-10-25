"""Prepare MoocRadar history, observation windows, and correction workloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/moocradar_source.json'
CUTS = ['2020-11-16', '2020-11-23', '2020-11-30', '2020-12-07']
HISTORY_COLUMNS = ['eid', 'user_id', 'course_id', 'exercise_id', 'ts', 'is_correct']
SERVING_PARQUETS = ['raw.parquet', 'participants.parquet', 'catalog.parquet', 'catalog_changed.parquet']
SERVING_ARRAYS = ['context.npz', 'context_changed.npz']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_frame(frame, path):
    # Parquet 2.6 preserves the UInt32 columns in the serving schema.
    frame.to_parquet(path, index=False, engine='pyarrow', version='2.6')


def history_semantic_sha256(history):
    """Storage-independent SHA256 of explicit UTF-8 JSON rows in eid order."""
    digest = hashlib.sha256()
    timestamps = history.ts.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    for row, timestamp in zip(history.itertuples(index=False, name=None), timestamps):
        eid, user, course, exercise, _, value = row
        payload = [int(eid), str(user), str(course), str(exercise), int(timestamp), int(value)]
        digest.update((json.dumps(payload, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8'))
    return digest.hexdigest()


def check_empty_outputs(paths):
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError('Use fresh output paths; existing files are preserved: '+', '.join(existing[:5]))


def normalize(raw_dir, out_dir, manifest):
    """Normalize the pinned source records without filtering attempts."""
    source_checks = []
    for source in manifest['source_files']:
        path = raw_dir / source['name']
        actual = {'name': source['name'], 'bytes': path.stat().st_size, 'sha256': sha(path)}
        if actual['bytes'] != source['bytes'] or actual['sha256'] != source['sha256']:
            raise ValueError('Source fingerprint differs from the declared dataset: '+source['name'])
        source_checks.append(actual)
    sequences = json.loads((raw_dir/'student-problem-middle.json').read_text(encoding='utf-8'))
    if not isinstance(sequences, list):
        raise TypeError('The pinned middle source must be a JSON array of learner sequences')
    # Flatten in the published sequence/record order. No rows are filtered or deduplicated.
    rows = [record for sequence in sequences for record in sequence['seq']]
    events = pd.DataFrame(rows)
    problems = pd.read_json(raw_dir/'problem.json', lines=True)
    expected = manifest['expected_source_counts']
    assert len(sequences) == expected['sequences']
    assert len(events) == expected['attempt_rows']
    assert len(problems) == expected['problem_catalog_rows']
    assert events.user_id.nunique() == expected['users']
    assert events.problem_id.nunique() == expected['observed_problem_ids']
    assert events.exercise_id.nunique() == expected['observed_exercise_ids']
    assert problems.course_id.nunique() == expected['catalog_course_ids']
    assert problems.problem_id.is_unique and events.log_id.is_unique
    assert events.problem_id.isin(problems.problem_id).all()
    assert events.is_correct.isin([0, 1]).all()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_frame(events, out_dir/'events.parquet')
    write_frame(problems, out_dir/'problems.parquet')
    # The event's exercise_id is retained; only course_id comes from problem metadata.
    merged = events.merge(problems[['problem_id', 'course_id']], on='problem_id', validate='many_to_one')
    assert len(merged) == len(events)
    merged['ts'] = pd.to_datetime(merged.submit_time)
    assert merged.ts.notna().all()
    merged = merged.sort_values(['ts', 'log_id']).reset_index(drop=True)
    merged['eid'] = np.arange(len(merged), dtype=np.int64)
    history = merged[HISTORY_COLUMNS].copy()
    assert history.ts.dt.year.eq(2020).all()
    assert str(history.ts.min()) == expected['minimum_timestamp']
    assert str(history.ts.max()) == expected['maximum_timestamp']
    write_frame(history, out_dir/'history.parquet')
    semantic_hash = history_semantic_sha256(history)
    if manifest.get('expected_history_semantic_sha256') and semantic_hash != manifest['expected_history_semantic_sha256']:
        raise ValueError('Canonical history values differ from the validated semantic fingerprint')
    return history, {
        'sources': source_checks, 'rows': len(history), 'users': int(history.user_id.nunique()),
        'courses': int(history.course_id.nunique()), 'exercise_ids': int(history.exercise_id.nunique()),
        'history_semantic_sha256': semantic_hash,
        'history_semantic_serialization': 'UTF-8 compact JSON arrays [eid,user_id,course_id,exercise_id,unix_timestamp_ns,is_correct], LF after each row, sorted by eid',
        'columns': HISTORY_COLUMNS, 'dtypes': {name: str(dtype) for name, dtype in history.dtypes.items()},
        'minimum_timestamp': str(history.ts.min()), 'maximum_timestamp': str(history.ts.max()),
        'filters_applied': [], 'duplicate_rows_removed': 0,
        'normalization_logic': 'Source sequence flattening, course lookup, stable timestamp/log ordering and six-column history projection'}


def common_at_cut(history, cut):
    when = pd.Timestamp(cut)
    past = history.loc[history.ts < when]
    window = past.loc[past.ts >= when-pd.Timedelta(days=28)]
    catalog = past.groupby(['course_id', 'exercise_id'], sort=True).is_correct.agg(n='size', s='sum').reset_index()
    catalog['item'] = catalog.groupby('course_id', sort=False).cumcount().astype('uint32')
    sizes = catalog.groupby('course_id', sort=False).size().to_dict()
    counts = window.groupby(['user_id', 'course_id'], sort=True).size()
    keys = [key for key, count in counts.items() if count >= 10 and sizes[key[1]] >= 6]
    return when, past, window, catalog, keys


def serving_inputs(history, output):
    """Rebuild past-only serving populations and changed common context."""
    reports = []
    for cut in CUTS:
        when, past, window, catalog, keys = common_at_cut(history, cut)
        people = pd.DataFrame(keys, columns=['user_id', 'course_id'])
        people.insert(0, 'qid', np.arange(len(people), dtype='uint32'))
        groups = {key: group for key, group in catalog.groupby('course_id', sort=False)}
        sizes = people.course_id.map({key: len(group) for key, group in groups.items()}).to_numpy(dtype=np.int64)
        offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
        joined = window.merge(people, on=['user_id', 'course_id'], how='inner', validate='many_to_one').merge(
            catalog[['course_id', 'exercise_id', 'item']], on=['course_id', 'exercise_id'], validate='many_to_one')
        joined['day'] = ((joined.ts-(when-pd.Timedelta(days=28))).dt.total_seconds()//86400).astype('uint8')
        raw = joined.rename(columns={'is_correct': 'value'})[['eid', 'qid', 'item', 'day', 'value']].astype(
            {'eid': 'uint64', 'qid': 'uint32', 'item': 'uint32', 'day': 'uint8', 'value': 'uint8'}).sort_values('eid').reset_index(drop=True)
        assert raw.eid.is_unique and raw.day.between(0, 27).all()
        blocks = raw.groupby(['qid', 'item'], sort=True).value.agg(n='size', s='sum')
        target = blocks.reset_index().sort_values(['qid', 'n', 'item'], ascending=[True, False, True]).drop_duplicates('qid').sort_values('qid')['item'].to_numpy(dtype=np.int64)

        def static_for(frame):
            by_course = {key: group.sort_values('item') for key, group in frame.groupby('course_id', sort=False)}
            return np.concatenate([np.column_stack((np.log1p(by_course[key].n.to_numpy()),
                (by_course[key].s.to_numpy()+1)/(by_course[key].n.to_numpy()+2))) for key in people.course_id])

        deleted = joined.loc[joined.eid % 100 == 0]
        differences = deleted.groupby(['course_id', 'exercise_id']).is_correct.agg(n='size', s='sum')
        changed = catalog.copy()
        index = pd.MultiIndex.from_frame(catalog[['course_id', 'exercise_id']])
        changed[['n', 's']] -= differences.reindex(index, fill_value=0)[['n', 's']].to_numpy()
        direct = past.loc[~past.eid.isin(set(deleted.eid))].groupby(['course_id', 'exercise_id']).is_correct.agg(n='size', s='sum')
        direct = direct.reindex(index, fill_value=0)
        assert np.array_equal(changed[['n', 's']].to_numpy(), direct[['n', 's']].to_numpy())
        folder = output/cut
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in [('raw', raw), ('participants', people), ('catalog', catalog), ('catalog_changed', changed)]:
            write_frame(frame, folder/(name+'.parquet'))
        np.savez_compressed(folder/'context.npz', offsets=offsets, sizes=sizes, target=target, static=static_for(catalog))
        np.savez_compressed(folder/'context_changed.npz', static=static_for(changed))
        reports.append({'cut': cut, 'population': len(people), 'window_events': len(raw),
                        'deleted_events': len(deleted), 'candidate_items': int(sizes.sum()),
                        'selection_uses_future_labels': False, 'changed_context_matches_full_history': True})
    return reports


def quality_workloads(history, output):
    """Construct held-out queries and synthetic correction journals."""
    reports = []
    output.mkdir(parents=True, exist_ok=True)
    for cut_index, cut in enumerate(CUTS):
        when, _, window, catalog, eligible = common_at_cut(history, cut)
        future = history.loc[(history.ts >= when) & (history.ts < when+pd.Timedelta(days=7))]
        # The target is the first future record in chronological history.
        targets = future.groupby(['user_id', 'course_id'], sort=False).first()
        ids_by_course = {course: group.sort_values('item').exercise_id.to_numpy()
                         for course, group in catalog.groupby('course_id', sort=False)}
        rows = []
        for user, course in eligible:
            key = (user, course)
            if key not in targets.index:
                continue
            target = targets.loc[key]
            ids = ids_by_course[course]
            if target.exercise_id not in ids:
                continue
            rows.append({'qid': len(rows), 'user_id': user, 'course_id': course,
                         'target_block': str(target.exercise_id),
                         'target_index': int(np.flatnonzero(ids == target.exercise_id)[0]),
                         'y': int(target.is_correct)})
        requests = pd.DataFrame(rows)
        mapping = requests[['qid', 'user_id', 'course_id']]
        cat = catalog.loc[catalog.course_id.isin(mapping.course_id)].copy()
        cat['bid'] = cat.item.astype('int32')
        raw = window.merge(mapping, on=['user_id', 'course_id'], validate='many_to_one')
        raw = raw.merge(cat[['course_id', 'exercise_id', 'bid']], on=['course_id', 'exercise_id'], validate='many_to_one')
        raw['day'] = ((raw.ts-(when-pd.Timedelta(days=28))).dt.total_seconds()//86400).astype('int16')
        raw = raw.sort_values('eid')
        rng = np.random.default_rng(400020+cut_index)
        journal = raw.loc[rng.random(len(raw)) < .05, ['eid', 'qid', 'course_id', 'bid', 'day']].copy()
        journal['new'] = rng.integers(0, 2, len(journal))
        write_frame(requests, output/f'requests_{cut}.parquet')
        write_frame(journal, output/f'journal_{cut}.parquet')
        reports.append({'cut': cut, 'requests': len(requests), 'window_events': len(raw),
                        'journal_records': len(journal), 'journal_seed': 400020+cut_index,
                        'journal_selection_probability': .05, 'target_definition': 'First observed attempt in following seven days',
                        'journal_definition': 'Synthetic independent binary replacement; no-op replacements retained'})
    return reports



def prepare_moocradar(
    raw_dir: Path | str,
    out_dir: Path | str,
    *,
    serving_out_dir: Path | str | None = None,
    workloads_out_dir: Path | str | None = None,
) -> dict:
    """Normalize source files and optionally build serving and correction inputs."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    serving_out_dir = Path(serving_out_dir) if serving_out_dir is not None else None
    workloads_out_dir = Path(workloads_out_dir) if workloads_out_dir is not None else None
    outputs = [out_dir/name for name in ['events.parquet', 'problems.parquet', 'history.parquet', 'preparation_manifest.json']]
    if serving_out_dir is not None:
        outputs += [serving_out_dir/cut/name for cut in CUTS for name in SERVING_PARQUETS+SERVING_ARRAYS]
    if workloads_out_dir is not None:
        outputs += [workloads_out_dir/f'{kind}_{cut}.parquet' for cut in CUTS for kind in ['requests', 'journal']]
    check_empty_outputs(outputs)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    history, report = normalize(raw_dir, out_dir, manifest)
    report['software'] = {'python': platform.python_version(), 'numpy': np.__version__,
                          'pandas': pd.__version__, 'pyarrow': pyarrow.__version__}
    report['parquet_format_version'] = '2.6'
    if serving_out_dir is not None:
        report['serving_snapshots'] = serving_inputs(history, serving_out_dir)
    if workloads_out_dir is not None:
        report['quality_workloads'] = quality_workloads(history, workloads_out_dir)
    report['generated_files'] = [{'name': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
                                  for path in outputs if path.exists()]
    write_json(out_dir/'preparation_manifest.json', report)
    print(json.dumps({'status': 'complete', 'rows': len(history),
                      'serving_snapshots': len(report.get('serving_snapshots', [])),
                      'quality_workloads': len(report.get('quality_workloads', [])),
                      'manifest': str(out_dir/'preparation_manifest.json')}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True, help='Directory containing the pinned middle JSON and problem JSONL')
    parser.add_argument('--out-dir', type=Path, required=True, help='Fresh normalized-output files: events, problems and history Parquet')
    parser.add_argument('--serving-out-dir', type=Path, help='Optional destination for four serving snapshot folders')
    parser.add_argument('--workloads-out-dir', type=Path, help='Optional destination for four query/correction workload pairs')
    args = parser.parse_args()
    prepare_moocradar(args.raw_dir, args.out_dir,
                     serving_out_dir=args.serving_out_dir,
                     workloads_out_dir=args.workloads_out_dir)


if __name__ == '__main__':
    main()
