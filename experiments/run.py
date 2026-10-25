"""Execute one complete local campaign in isolated, sequential module processes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
MODULES = ('selection', 'common', 'scaling', 'maintenance')
ENVIRONMENT_KEYS = (
    'CLICKHOUSE_HOST', 'CLICKHOUSE_PORT', 'CLICKHOUSE_USER', 'CLICKHOUSE_PASSWORD',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'PYTHONHASHSEED', 'GFORM_SELECTION_DIR', 'GFORM_SELECTION_OVERLAY',
)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def credentials(path):
    if path:
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                if key.strip() in ('CLICKHOUSE_HOST', 'CLICKHOUSE_PORT', 'CLICKHOUSE_USER', 'CLICKHOUSE_PASSWORD'):
                    os.environ[key.strip()] = value
    if os.getenv('CLICKHOUSE_HOST', '127.0.0.1') not in ('localhost', '127.0.0.1'):
        raise ValueError('This campaign is restricted to the declared local ClickHouse endpoint')
    if not os.environ.get('CLICKHOUSE_USER') or not os.environ.get('CLICKHOUSE_PASSWORD'):
        raise ValueError('Provide ClickHouse authentication through environment or an external connection file')
    os.environ.setdefault('CLICKHOUSE_HOST', '127.0.0.1')
    os.environ.setdefault('CLICKHOUSE_PORT', '8123')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONHASHSEED'] = '790021'


@contextmanager
def execution_environment(connection_file=None):
    """Temporarily configure database access and reproducible worker settings."""
    previous = {name: os.environ.get(name) for name in ENVIRONMENT_KEYS}
    connection_file = connection_file or (
        ROOT.parent / '.env' if (ROOT.parent / '.env').is_file() else None)
    try:
        credentials(connection_file)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(list(ROOT.glob('*.py')) + [ROOT / 'protocol.json'])}


def input_hashes():
    """Hash prepared inputs, configuration, adapters, and preparation code."""
    suffixes = {'.parquet', '.json', '.csv', '.pkl', '.py', '.npz', '.txt'}
    paths = [path for folder in ('data', 'adapters', 'scripts', 'configs')
             for path in (ROOT.parent / folder).rglob('*')
             if path.is_file() and '__pycache__' not in path.parts
             and path.suffix.lower() in suffixes]
    return {path.relative_to(ROOT.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def environment():
    import psutil
    from .runtime import connect
    client = connect()
    try:
        server = client.query('SELECT version(),hostName(),uptime()').first_row
        settings = dict(client.query(
            "SELECT name,value FROM system.settings WHERE name IN "
            "('max_threads','max_memory_usage','max_insert_threads','use_query_cache')").result_rows)
        memory = dict(client.query(
            "SELECT metric,value FROM system.asynchronous_metrics WHERE metric IN "
            "('OSMemoryTotal','OSMemoryAvailable','LoadAverage1')").result_rows)
        server_settings = dict(client.query(
            "SELECT name,value FROM system.server_settings WHERE name IN "
            "('max_server_memory_usage','max_server_memory_usage_to_ram_ratio','background_pool_size')").result_rows)
    finally:
        client.close()
    packages = {n: importlib.metadata.version(n) for n in (
        'clickhouse-connect', 'numpy', 'pandas', 'scipy', 'scikit-learn',
        'pyarrow', 'threadpoolctl', 'psutil')}
    processor = platform.processor()
    if sys.platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                processor = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return {'recorded_at_utc': stamp(), 'python': platform.python_version(),
            'platform': platform.platform(), 'processor': processor,
            'cpu_physical': psutil.cpu_count(logical=False), 'cpu_logical': psutil.cpu_count(),
            'host_memory_bytes': psutil.virtual_memory().total,
            'host_disk_free_bytes': psutil.disk_usage(str(ROOT)).free, 'packages': packages,
            'server_version': server[0], 'server_identity': server[1], 'server_uptime_seconds': server[2],
            'query_settings': settings, 'server_metrics': memory, 'server_settings': server_settings,
            'endpoint': os.environ['CLICKHOUSE_HOST'] + ':' + os.environ['CLICKHOUSE_PORT'],
            'numeric_threads': 1}


def _worker(name, out, smoke, log_path):
    """Call a measurement module in a fresh interpreter with its own log."""
    out, log_path = Path(out), Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                importlib.import_module(f'experiments.{name}').run(out, smoke=smoke)
            except Exception as exc:
                # Driver exception messages can contain connection credentials.
                import traceback
                frames = [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name}
                          for f in traceback.extract_tb(exc.__traceback__)]
                save(out / 'worker_failure.json', {
                    'time': stamp(), 'exception_type': type(exc).__name__,
                    'frames': frames, 'reported_performance': False})
                print('WORKER FAILED ' + type(exc).__name__, flush=True)
                raise SystemExit(1) from None


def _run_stage(name, out, smoke, log_path):
    process = multiprocessing.get_context('spawn').Process(
        target=_worker, args=(name, out, smoke, log_path), name=f'analytics-{name}')
    started = False
    begin = last_progress = time.monotonic()
    try:
        process.start()
        started = True
        while process.is_alive():
            process.join(timeout=1)
            if time.monotonic() - last_progress > 30:
                print(f'RUNNING {name} {time.monotonic() - begin:.0f}s', flush=True)
                last_progress = time.monotonic()
        process.join()
        return process.exitcode
    except BaseException:
        if started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        raise
    finally:
        process.close()


def run_experiment(mode='measure', out=None, connection_file=None, modules=None,
                   selection_dir=None) -> Path:
    """Run, verify, and summarize an experiment; return its output directory.

    Each measurement stage calls its Python function in a fresh worker process.
    The caller's environment is restored on completion, failure, or interruption.
    """
    if mode not in ('smoke', 'measure'):
        raise ValueError('mode must be smoke or measure')
    if isinstance(modules, str):
        modules = tuple(x.strip() for x in modules.split(',') if x.strip())
    requested = MODULES if modules is None else tuple(modules)
    if not requested or any(x not in MODULES for x in requested) or len(set(requested)) != len(requested):
        raise ValueError('Unknown or duplicate module')
    if requested != tuple(x for x in MODULES if x in requested):
        raise ValueError('Modules must follow selection,common,scaling,maintenance order')
    timestamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = Path(out or ROOT.parent / 'outputs' / f'{mode}_{timestamp}').resolve()
    if out.exists():
        raise FileExistsError('Choose a new output directory; completed observations are never overwritten')
    selection_dir = Path(selection_dir or out / 'selection').resolve()
    if any(name != 'selection' for name in requested) and 'selection' not in requested:
        if not (selection_dir / 'overlay.json').is_file():
            raise FileNotFoundError('Run selection first or supply a completed selection_dir')
    smoke = mode == 'smoke'
    with execution_environment(connection_file):
        os.environ['GFORM_SELECTION_DIR'] = str(selection_dir)
        os.environ['GFORM_SELECTION_OVERLAY'] = str(selection_dir / 'overlay.json')
        out.mkdir(parents=True)
        state = {'started_at_utc': stamp(), 'smoke': smoke, 'requested_modules': requested,
                 'status': 'running', 'completed_modules': [], 'module_status': {}}
        save(out / 'campaign.json', state)
        active_module = None
        try:
            protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
            save(out / 'protocol_frozen.json', protocol)
            frozen_code = code_hashes()
            save(out / 'code_freeze.json', frozen_code)
            save(out / 'input_freeze.json', input_hashes())
            env = environment()
            save(out / 'environment.json', env)
            (out / 'logs').mkdir()
            for name in requested:
                active_module = name
                if code_hashes() != frozen_code:
                    raise RuntimeError('Campaign source changed after freeze')
                current = environment()
                if (current['server_identity'] != env['server_identity'] or current['packages'] != env['packages']
                        or current['server_uptime_seconds'] < env['server_uptime_seconds']
                        or current['server_settings'] != env['server_settings']
                        or current['query_settings'] != env['query_settings']):
                    raise RuntimeError('Server lifecycle, resource settings or Python dependency versions changed')
                save(out / ('environment_before_' + name + '.json'), current)
                if name != 'selection':
                    overlay_path = selection_dir / 'overlay.json'
                    save(out / ('selection_overlay_before_' + name + '.json'), {
                        'sha256': hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
                        'source': 'selection/overlay.json', 'recorded_at_utc': stamp()})
                state['module_status'][name] = {'status': 'running', 'started_at_utc': stamp()}
                save(out / 'campaign.json', state)
                print('START ' + name, flush=True)
                begin = time.monotonic()
                exit_code = _run_stage(name, out / name, smoke, out / 'logs' / (name + '.log'))
                state['module_status'][name].update(
                    finished_at_utc=stamp(), elapsed_seconds=time.monotonic() - begin,
                    exit_code=exit_code, status='completed' if exit_code == 0 else 'failed')
                if exit_code:
                    raise RuntimeError(f'{name} failed with exit code {exit_code}; see {out / "logs" / (name + ".log")}')
                state['completed_modules'].append(name)
                after = environment()
                save(out / ('environment_after_' + name + '.json'), after)
                if (after['server_identity'] != current['server_identity']
                        or after['server_uptime_seconds'] < current['server_uptime_seconds']
                        or after['server_settings'] != current['server_settings']
                        or after['query_settings'] != current['query_settings']):
                    raise RuntimeError('Server lifecycle or resource settings changed during the module')
                save(out / 'campaign.json', state)
                print('DONE ' + name, flush=True)
                active_module = None
            if code_hashes() != frozen_code:
                raise RuntimeError('Campaign source changed after freeze')
            state.update(status='completed', finished_at_utc=stamp())
            save(out / 'campaign.json', state)
            from .verify import verify
            from .summarize import summarize
            verify(out)
            summarize(out)
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                         finished_at_utc=stamp(), failure_type=type(exc).__name__)
            if active_module is not None:
                state['failed_module'] = active_module
                if active_module in state['module_status']:
                    state['module_status'][active_module].update(
                        status=state['status'], finished_at_utc=stamp())
            save(out / 'campaign.json', state)
            raise
        print('CAMPAIGN COMPLETED', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--connection-file', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--modules', default=','.join(MODULES))
    parser.add_argument('--selection-dir', type=Path)
    args = parser.parse_args()
    run_experiment(mode='smoke' if args.smoke else 'measure', out=args.out,
                   connection_file=args.connection_file, modules=args.modules,
                   selection_dir=args.selection_dir)


if __name__ == '__main__':
    main()
