"""Export ALPACAS or Moodle serving inputs from the canonical evaluation split.

Model fitting and representation selection are performed by the experiment runner.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def prepare(dataset, training, output):
    if dataset not in {'alpacas', 'moodle'}:
        raise ValueError("dataset must be 'alpacas' or 'moodle'")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory.')
    split = '2021' if dataset == 'alpacas' else 'test'
    events_path = training / f'events_{split}.parquet'
    people_path = training / f'participants_{split}.parquet'
    events, people = pd.read_parquet(events_path), pd.read_parquet(people_path)
    vocabulary = ({'channels': ['lms_records', 'forum_contributions'],
                   'source': 'Original ALPACAS two-channel canonical event schema'}
                  if dataset == 'alpacas'
                  else json.loads((training / 'channel_vocabulary.json').read_text(encoding='utf-8')))
    assert events.eid.is_unique
    assert np.array_equal(people.qid.to_numpy(), np.arange(len(people)))
    assert events.qid.isin(people.qid).all() and events.day.between(0, 55).all()
    assert events.channel.between(0, len(vocabulary['channels']) - 1).all()
    raw = events.rename(columns={'channel': 'item'}).copy()
    raw['item'] = raw['item'].astype('uint32')
    raw['value'] = np.zeros(len(raw), dtype='uint8')
    raw = raw[['eid', 'qid', 'item', 'day', 'value']]
    population = {'qid': people.qid.tolist()}
    if dataset == 'alpacas':
        population['course_flag'] = people.course_flag.tolist()
    output.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output / 'raw.parquet', index=False, version='2.6')
    people.to_parquet(output / 'participants.parquet', index=False, version='2.6')
    save(output / 'serving_population.json', population)
    (output / 'channel_vocabulary.json').write_text(
        json.dumps(vocabulary, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8', newline='\n')
    report = {
        'dataset': dataset, 'events': len(raw), 'participants': len(people),
        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [events_path, people_path]},
        'model_fitting': False, 'representation_selection': False}
    save(output / 'preparation.json', report)
    print(json.dumps({'dataset': dataset, 'events': len(raw), 'participants': len(people)}))
    return report


def prepare_serving(dataset: str, training_dir: Path | str, out_dir: Path | str) -> dict:
    """Build serving inputs from a prepared ALPACAS or Moodle evaluation split."""
    return prepare(dataset, Path(training_dir), Path(out_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['alpacas', 'moodle'], required=True)
    parser.add_argument('--training-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare_serving(args.dataset, args.training_dir or ROOT / 'data' / args.dataset / 'training', args.out_dir)
