import tempfile
import unittest
from pathlib import Path

from utils.service_control import service_manager_argv


class ServiceControlTest(unittest.TestCase):
    def test_builds_fixed_manager_argv_without_shell(self):
        with tempfile.TemporaryDirectory(prefix='project with spaces ') as directory:
            project_root = Path(directory)
            manager = project_root / 'scripts' / 'mediflow-kiosk'
            manager.parent.mkdir()
            manager.write_text('#!/usr/bin/env python3\n', encoding='utf-8')

            self.assertEqual(
                service_manager_argv(project_root, 'restart'),
                [str(manager.resolve()), 'restart'],
            )

    def test_rejects_unapproved_action(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            service_manager_argv('/tmp/project', 'start;id')

    def test_server_has_no_missing_platform_manager_references(self):
        source = (Path(__file__).resolve().parents[1] / 'eye_server.py').read_text(
            encoding='utf-8'
        )
        self.assertNotIn('start_services_jetson.sh', source)
        self.assertNotIn('stop_services_jetson.sh', source)
        self.assertNotIn("['/bin/bash', '-lc'", source)


if __name__ == '__main__':
    unittest.main()
