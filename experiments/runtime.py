"""Common local execution settings and durable, append-only measurement journals."""
from __future__ import annotations

import contextlib
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time

ROOT = Path(__file__).resolve().parent
QUERY_SETTINGS = {'max_threads': 2, 'max_insert_threads': 1,
                  'max_memory_usage': 4294967296, 'use_query_cache': 0,
                  'max_execution_time': 600}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def connect():
    """Credentials come from the process environment, never command-line values."""
    import clickhouse_connect
    host = os.environ.get('CLICKHOUSE_HOST', '127.0.0.1')
    assert host in ('127.0.0.1', 'localhost'), 'This campaign is local only.'
    required = ('CLICKHOUSE_USER', 'CLICKHOUSE_PASSWORD')
    if any(key not in os.environ for key in required):
        raise RuntimeError('Set local ClickHouse connection environment first.')
    client = clickhouse_connect.get_client(
        host=host, port=int(os.environ.get('CLICKHOUSE_PORT', '8123')),
        username=os.environ['CLICKHOUSE_USER'], password=os.environ['CLICKHOUSE_PASSWORD'],
        connect_timeout=10, send_receive_timeout=660, settings=QUERY_SETTINGS)
    assert client.query('SELECT version()').first_row[0] == '25.3.2.39'
    return client


def numeric_context():
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1)


def timed(function):
    start = time.perf_counter()
    result = function()
    return time.perf_counter() - start, result


def variant_orders(variants, rounds, warmups, seed):
    """Randomized Williams schedules balance position and pairwise precedence."""
    variants = list(variants)
    assert len(variants) == len(set(variants)) and len(variants) >= 2
    rng = random.Random(int(seed))
    basis = variants.copy()
    rng.shuffle(basis)
    n = len(basis)
    indices = [0]
    for i in range(1, n):
        indices.append((i + 1) // 2 if i % 2 else n - i // 2)
    design = [[basis[(index + offset) % n] for index in indices] for offset in range(n)]
    if n % 2:
        design += [list(reversed(order)) for order in design]
    measured = [design[i % len(design)] for i in range(rounds)]
    rng.shuffle(measured)
    result = {}
    for round_ in range(-warmups, 0):
        order = variants.copy()
        rng.shuffle(order)
        result[round_] = order
    result.update({i: order for i, order in enumerate(measured)})
    return result


def environment(client=None):
    packages = {}
    for name in ('clickhouse-connect', 'numpy', 'pandas', 'scipy', 'scikit-learn',
                 'pyarrow', 'threadpoolctl', 'psutil'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {'utc': now(), 'python': platform.python_version(),
              'platform': platform.platform(), 'processor': platform.processor(),
              'logical_cpus': os.cpu_count(), 'packages': packages,
              'requested_query_settings': QUERY_SETTINGS,
              'python_numeric_threads': 1, 'garbage_collection': 'enabled; full collection before each pair outside timers'}
    if client is not None:
        result['server_version'] = client.query('SELECT version()').first_row[0]
        result['server_host'] = client.query('SELECT hostName()').first_row[0]
        result['query_settings'] = dict(client.query(
            "SELECT name,value FROM system.settings WHERE name IN "
            "('max_threads','max_insert_threads','max_memory_usage','use_query_cache','max_execution_time')"
        ).result_rows)
        result['server_resources'] = dict(client.query(
            "SELECT metric,value FROM system.asynchronous_metrics WHERE metric IN "
            "('OSMemoryTotal','OSMemoryAvailable','LoadAverage1')"
        ).result_rows)
    return result


class Journal:
    """One output directory per module/attempt; incomplete attempts stay visible."""

    def __init__(self, out, module):
        self.out = Path(out)
        self.module = module
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / 'timings.jsonl'
        assert not self.path.exists(), 'Use a fresh output directory.'
        self.path.touch(exist_ok=False)
        self.started = now()
        self.rows = 0
        self.pairs = 0
        self.event('started', reported_timings=False)

    def _append(self, path, entries):
        payload = ''.join(dumps(entry) + '\n' for entry in entries)
        with Path(path).open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def pair(self, rows):
        rows = [dict(row) for row in rows]
        assert len(rows) >= 2
        assert len({row['variant'] for row in rows}) == len(rows)
        keys = ('dataset', 'snapshot', 'factor', 'session', 'round', 'warmup', 'phase')
        for key in keys:
            assert len({row.get(key) for row in rows}) == 1, key
        for row in rows:
            assert math.isfinite(float(row['seconds'])) and float(row['seconds']) > 0
            row.setdefault('module', self.module)
            row.setdefault('committed_utc', now())
        self._append(self.path, rows)
        self.rows += len(rows)
        self.pairs += 1
        save(self.out / 'checkpoint.json', {'utc': now(), 'rows': self.rows,
                                          'paired_groups': self.pairs, 'complete': False})

    def event(self, kind, **fields):
        self._append(self.out / 'events.jsonl', [{'utc': now(), 'kind': kind, **fields}])

    def save(self, name, value):
        save(self.out / name, value)

    def finish(self, extra=None):
        rows = [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if extra and 'expected_measured_rows' in extra:
            assert sum(not row['warmup'] for row in rows) == extra['expected_measured_rows']
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (self.out / 'timings.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report = {'module': self.module, 'started_utc': self.started,
                  'completed_utc': now(), 'timing_rows': len(rows),
                  'measured_rows': sum(not row['warmup'] for row in rows),
                  'warmup_rows': sum(row['warmup'] for row in rows),
                  'timings_sha256': sha(self.path), 'complete': True, **(extra or {})}
        self.save('run_complete.json', report)
        self.save('checkpoint.json', {'utc': now(), 'rows': len(rows),
                                    'paired_groups': self.pairs, 'complete': True})
        self.event('completed', timing_rows=len(rows))
        return report


@contextlib.contextmanager
def failure_record(journal, **context):
    try:
        yield
    except BaseException as exc:
        # No raw exception messages: database errors can include credentials.
        journal.event('technical_failure', exception_type=type(exc).__name__, **context)
        raise
