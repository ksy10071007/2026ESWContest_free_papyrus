import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from database.app import load_project_environment


class DatabaseAppEnvironmentTest(unittest.TestCase):
    def test_root_dotenv_precedes_optional_json_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text(
                'MEDIFLOW_TEST_SOURCE=dotenv\nMEDIFLOW_TEST_DOTENV_ONLY=loaded\n',
                encoding='utf-8',
            )
            config_path = root / 'config.local.json'
            config_path.write_text(
                json.dumps({
                    'MEDIFLOW_TEST_SOURCE': 'json',
                    'MEDIFLOW_TEST_JSON_ONLY': 'loaded',
                }),
                encoding='utf-8',
            )

            keys = (
                'MEDIFLOW_TEST_SOURCE',
                'MEDIFLOW_TEST_DOTENV_ONLY',
                'MEDIFLOW_TEST_JSON_ONLY',
            )
            with mock.patch.dict(os.environ, {}, clear=False):
                for key in keys:
                    os.environ.pop(key, None)
                load_project_environment(str(root), str(config_path))
                self.assertEqual(os.environ['MEDIFLOW_TEST_SOURCE'], 'dotenv')
                self.assertEqual(os.environ['MEDIFLOW_TEST_DOTENV_ONLY'], 'loaded')
                self.assertEqual(os.environ['MEDIFLOW_TEST_JSON_ONLY'], 'loaded')
                for key in keys:
                    os.environ.pop(key, None)

    def test_process_environment_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text('MEDIFLOW_TEST_PRIORITY=dotenv\n', encoding='utf-8')
            config_path = root / 'config.local.json'
            config_path.write_text(
                json.dumps({'MEDIFLOW_TEST_PRIORITY': 'json'}),
                encoding='utf-8',
            )

            with mock.patch.dict(
                os.environ,
                {'MEDIFLOW_TEST_PRIORITY': 'process'},
                clear=False,
            ):
                load_project_environment(str(root), str(config_path))
                self.assertEqual(os.environ['MEDIFLOW_TEST_PRIORITY'], 'process')


if __name__ == '__main__':
    unittest.main()
