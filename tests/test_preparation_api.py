"""Preparation API checks using small local inputs."""

import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from scripts import prepare_alpacas, prepare_moocradar, prepare_moodle, prepare_serving


class PreparationApiTests(unittest.TestCase):
    def test_existing_alpacas_output_rejected_before_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'prepared'
            output.mkdir()
            original = output / 'events_1819.parquet'
            original.write_bytes(b'existing data')
            module = importlib.import_module('scripts.prepare_alpacas')
            with patch.object(module, 'download') as download:
                with self.assertRaises(FileExistsError):
                    prepare_alpacas(temporary, str(output))
                download.assert_not_called()
            self.assertEqual(original.read_bytes(), b'existing data')

    def test_existing_moocradar_output_rejected_before_reading_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'prepared'
            output.mkdir()
            history = output / 'history.parquet'
            history.write_bytes(b'existing history')
            with self.assertRaises(FileExistsError):
                prepare_moocradar('missing-source', str(output))
            self.assertEqual(history.read_bytes(), b'existing history')

    def test_existing_moodle_provenance_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provenance = root / 'provenance'
            provenance.mkdir()
            (provenance / 'source_manifest.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                prepare_moodle('missing.zip', root / 'raw', root / 'prepared',
                               provenance_dir=provenance)
            self.assertFalse((root / 'raw').exists())

    def test_unknown_serving_dataset_rejected_before_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'serving'
            with self.assertRaises(ValueError):
                prepare_serving('unknown', 'missing-input', output)
            self.assertFalse(output.exists())

    def test_moodle_archive_to_serving_with_string_paths(self):
        module = importlib.import_module('scripts.prepare_moodle')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / 'moodle.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('Readme.txt', 'Synthetic fixture for preparation tests.')
                for period in module.PERIODS:
                    rows = []
                    for index in range(2):
                        rows.append({
                            module.ID: f'learner_{index}',
                            'Fecha': '01/01/2024' if period == '2023 P2' else '2024-01-01',
                            'Hora': f'0{index}:00:00', module.COMPONENT: 'Page',
                            'Nombre evento': 'Viewed', module.TARGET: index,
                            'Status': 'Accredited' if index else 'Not accredited',
                            'Year': period,
                            **{f'grade_{column}': 0 for column in range(13)},
                        })
                    archive.writestr(f'Records_{period}.csv', pd.DataFrame(rows).to_csv(index=False))
            training, serving = root / 'training', root / 'serving'
            with contextlib.redirect_stdout(io.StringIO()):
                audit = prepare_moodle(str(archive_path), str(root / 'raw'), str(training))
                report = prepare_serving('moodle', str(training), str(serving))
            self.assertEqual(audit['source_rows'], 8)
            self.assertEqual(report['events'], 2)
            self.assertEqual(report['participants'], 2)
            self.assertEqual(json.loads((serving / 'preparation.json').read_text()), report)
            self.assertEqual(json.loads((training / 'input_audit.json').read_text()), audit)
            self.assertTrue((training / 'provenance/source_manifest.json').is_file())
            raw = pd.read_parquet(serving / 'raw.parquet')
            self.assertEqual(raw.columns.tolist(), ['eid', 'qid', 'item', 'day', 'value'])
            self.assertEqual(raw.qid.tolist(), [0, 1])
            self.assertTrue(raw.value.eq(0).all())
            self.assertTrue(raw.day.eq(0).all())


if __name__ == '__main__':
    unittest.main()
