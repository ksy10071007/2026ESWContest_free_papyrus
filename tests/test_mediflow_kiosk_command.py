import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


COMMAND_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mediflow-kiosk"
LOADER = importlib.machinery.SourceFileLoader("mediflow_kiosk_command", str(COMMAND_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
mediflow_kiosk = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mediflow_kiosk
LOADER.exec_module(mediflow_kiosk)


class MediflowKioskCommandTest(unittest.TestCase):
    def test_parse_action_accepts_only_one_known_action(self):
        self.assertEqual(
            mediflow_kiosk.parse_action(["mediflow-kiosk", "status"]),
            "status",
        )
        for argv in (
            ["mediflow-kiosk"],
            ["mediflow-kiosk", "status", "extra"],
            ["mediflow-kiosk", "start;id"],
            ["mediflow-kiosk", "unknown"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(mediflow_kiosk.UsageError):
                    mediflow_kiosk.parse_action(argv)

    def test_tail_lines_returns_only_requested_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            path.write_text("one\ntwo\nthree\n", encoding="utf-8")
            self.assertEqual(mediflow_kiosk.tail_lines(path, 2), ["two\n", "three\n"])

    def test_dashboard_url_uses_external_override(self):
        with mock.patch.dict(
            os.environ,
            {"EXTERNAL_BASE_URL": "https://kiosk.example.test/base/"},
            clear=False,
        ):
            self.assertEqual(
                mediflow_kiosk.resolve_dashboard_url(),
                "https://kiosk.example.test/base",
            )

    def test_dashboard_url_resolves_wildcard_host_to_lan(self):
        environment = {
            "EXTERNAL_BASE_URL": "",
            "SERVER_HOST": "0.0.0.0",
            "SERVER_IP": "",
            "SERVER_PORT": "5050",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            with mock.patch.object(mediflow_kiosk, "get_lan_ip", return_value="10.0.0.25"):
                self.assertEqual(
                    mediflow_kiosk.resolve_dashboard_url(),
                    "http://10.0.0.25:5050",
                )

    def test_manager_source_has_no_private_ip_literal(self):
        source = COMMAND_PATH.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"192\.168\.\d+\.\d+")

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "requires Linux /proc")
    def test_process_record_detects_start_time_tampering(self):
        snapshot = mediflow_kiosk.read_process_snapshot(os.getpid())
        self.assertIsNotNone(snapshot)
        spec = mediflow_kiosk.ServiceSpec(
            key="test",
            label="test",
            launch_argv=snapshot.argv,
            executable=snapshot.executable,
            cwd=snapshot.cwd,
            port=0,
            health_url="http://127.0.0.1/",
        )
        record = mediflow_kiosk.build_process_record(spec, os.getpid())
        valid, _, _ = mediflow_kiosk.validate_record(spec, record)
        self.assertTrue(valid)

        record["start_ticks"] = int(record["start_ticks"]) + 1
        valid, reason, _ = mediflow_kiosk.validate_record(spec, record)
        self.assertFalse(valid)
        self.assertEqual(reason, "PID was reused")


if __name__ == "__main__":
    unittest.main()
