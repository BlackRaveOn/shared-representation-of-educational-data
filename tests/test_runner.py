"""Checks for imported experiment execution and isolated worker failures."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import run as runner


ENVIRONMENT_KEYS = (
    'CLICKHOUSE_HOST', 'CLICKHOUSE_PORT', 'CLICKHOUSE_USER', 'CLICKHOUSE_PASSWORD',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'PYTHONHASHSEED', 'GFORM_SELECTION_DIR', 'GFORM_SELECTION_OVERLAY',
)
CONNECTION = {
    'CLICKHOUSE_HOST': '127.0.0.1', 'CLICKHOUSE_PORT': '18123',
    'CLICKHOUSE_USER': 'test', 'CLICKHOUSE_PASSWORD': 'test-only-secret',
}


def environment_snapshot():
    return {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}


def fake_server():
    return {
        'server_identity': 'test-server', 'server_uptime_seconds': 100,
        'packages': {}, 'server_settings': {}, 'query_settings': {},
    }


class EnvironmentTests(unittest.TestCase):
    def test_environment_restored_after_success_and_error(self):
        with tempfile.TemporaryDirectory() as folder:
            connection = Path(folder) / '.env'
            connection.write_text('\n'.join(f'{key}={value}' for key, value in CONNECTION.items()),
                                  encoding='utf-8')
            with patch.dict(os.environ, {'OMP_NUM_THREADS': '7', 'CLICKHOUSE_USER': 'original'}):
                before = environment_snapshot()
                with runner.execution_environment(connection):
                    self.assertEqual(os.environ['CLICKHOUSE_USER'], 'test')
                    self.assertEqual(os.environ['OMP_NUM_THREADS'], '1')
                    os.environ['GFORM_SELECTION_DIR'] = 'temporary-selection'
                self.assertEqual(environment_snapshot(), before)
                with self.assertRaisesRegex(RuntimeError, 'test failure'):
                    with runner.execution_environment(connection):
                        os.environ['GFORM_SELECTION_OVERLAY'] = 'temporary-overlay'
                        raise RuntimeError('test failure')
                self.assertEqual(environment_snapshot(), before)

    def test_invalid_connection_does_not_leak_partial_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            connection = Path(folder) / '.env'
            connection.write_text('CLICKHOUSE_USER=changed\nCLICKHOUSE_HOST=example.invalid\n',
                                  encoding='utf-8')
            before = environment_snapshot()
            with self.assertRaises(ValueError):
                with runner.execution_environment(connection):
                    self.fail('An external endpoint was accepted.')
            self.assertEqual(environment_snapshot(), before)


class ExperimentRunnerTests(unittest.TestCase):
    def runner_context(self):
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, CONNECTION))
        stack.enter_context(patch.object(runner, 'environment', side_effect=fake_server))
        stack.enter_context(patch.object(runner, 'code_hashes', return_value={}))
        stack.enter_context(patch.object(runner, 'input_hashes', return_value={}))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    @staticmethod
    def stage(name, out, smoke, log_path):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'timings.jsonl').write_text('{"fixture":true}\n', encoding='utf-8')
        if name == 'selection':
            (out / 'overlay.json').write_text('{}\n', encoding='utf-8')
        return 0

    def test_invalid_requests_never_contact_database(self):
        requests = [
            {'mode': 'invalid'},
            {'modules': []},
            {'modules': ['selection', 'selection']},
            {'modules': ['common', 'selection']},
            {'modules': ['unknown']},
        ]
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, CONNECTION), \
                patch.object(runner, 'environment') as server:
            for index, request in enumerate(requests):
                output = Path(folder) / str(index)
                with self.subTest(request=request), self.assertRaises(ValueError):
                    runner.run_experiment(out=output, **request)
                self.assertFalse(output.exists())
            server.assert_not_called()

    def test_existing_results_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, CONNECTION):
            output = Path(folder) / 'experiment'
            output.mkdir()
            existing = output / 'timings.jsonl'
            existing.write_text('original measurement', encoding='utf-8')
            with patch.object(runner, 'environment') as server, self.assertRaises(FileExistsError):
                runner.run_experiment(out=output)
            self.assertEqual(existing.read_text(encoding='utf-8'), 'original measurement')
            server.assert_not_called()

    def test_stages_run_in_order_and_analysis_is_called_directly(self):
        with tempfile.TemporaryDirectory() as folder, self.runner_context(), \
                patch.object(runner, '_run_stage', side_effect=self.stage) as stage, \
                patch('experiments.verify.verify') as verify, \
                patch('experiments.summarize.summarize') as summarize:
            output = Path(folder) / 'experiment'
            before = environment_snapshot()
            result = runner.run_experiment(mode='smoke', out=output)
            self.assertEqual(result, output.resolve())
            self.assertEqual([call.args[0] for call in stage.call_args_list], list(runner.MODULES))
            verify.assert_called_once_with(output.resolve())
            summarize.assert_called_once_with(output.resolve())
            state = json.loads((output / 'campaign.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['completed_modules'], list(runner.MODULES))
            self.assertTrue(state['smoke'])
            self.assertEqual(environment_snapshot(), before)

    def test_failed_worker_keeps_partial_observations(self):
        def fail(name, out, smoke, log_path):
            self.stage(name, out, smoke, log_path)
            return 1

        with tempfile.TemporaryDirectory() as folder, self.runner_context(), \
                patch.object(runner, '_run_stage', side_effect=fail) as stage, \
                patch('experiments.verify.verify') as verify, \
                patch('experiments.summarize.summarize') as summarize:
            output = Path(folder) / 'experiment'
            before = environment_snapshot()
            with self.assertRaises(RuntimeError):
                runner.run_experiment(out=output)
            state = json.loads((output / 'campaign.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['completed_modules'], [])
            self.assertTrue((output / 'selection/timings.jsonl').is_file())
            self.assertEqual(stage.call_count, 1)
            verify.assert_not_called()
            summarize.assert_not_called()
            self.assertEqual(environment_snapshot(), before)

    def test_interruption_preserves_partial_observations(self):
        def interrupt(name, out, smoke, log_path):
            self.stage(name, out, smoke, log_path)
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as folder, self.runner_context(), \
                patch.object(runner, '_run_stage', side_effect=interrupt):
            output = Path(folder) / 'experiment'
            before = environment_snapshot()
            with self.assertRaises(KeyboardInterrupt):
                runner.run_experiment(out=output)
            state = json.loads((output / 'campaign.json').read_text(encoding='utf-8'))
            self.assertIn(state['status'], {'interrupted', 'failed'})
            self.assertEqual(state['completed_modules'], [])
            self.assertTrue((output / 'selection/timings.jsonl').is_file())
            self.assertEqual(environment_snapshot(), before)


class SpawnedWorkerTests(unittest.TestCase):
    def test_interrupted_worker_is_terminated_and_joined(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(runner.multiprocessing, 'get_context') as get_context:
            process = get_context.return_value.Process.return_value
            process.is_alive.return_value = True
            process.join.side_effect = [KeyboardInterrupt, None, None]
            with self.assertRaises(KeyboardInterrupt):
                runner._run_stage('selection', Path(folder) / 'worker', True,
                                  Path(folder) / 'worker.log')
            get_context.assert_called_once_with('spawn')
            process.start.assert_called_once()
            process.terminate.assert_called_once()
            process.kill.assert_called_once()
            process.join.assert_called()
            process.close.assert_called_once()

    def test_spawned_import_failure_is_recorded_without_credentials(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, CONNECTION):
            output = Path(folder) / 'worker'
            log = Path(folder) / 'worker.log'
            with redirect_stdout(io.StringIO()):
                exit_code = runner._run_stage('missing_module', output, True, log)
            self.assertNotEqual(exit_code, 0)
            report = json.loads((output / 'worker_failure.json').read_text(encoding='utf-8'))
            self.assertEqual(report['exception_type'], 'ModuleNotFoundError')
            self.assertTrue(report['frames'])
            self.assertFalse(report['reported_performance'])
            self.assertNotIn(CONNECTION['CLICKHOUSE_PASSWORD'], log.read_text(encoding='utf-8'))
            self.assertNotIn(CONNECTION['CLICKHOUSE_PASSWORD'], json.dumps(report))


if __name__ == '__main__':
    unittest.main(verbosity=2)
