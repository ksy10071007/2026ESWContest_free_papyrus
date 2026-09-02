from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "llm-cluster"
LEGACY_START = Path(__file__).resolve().parents[1] / "dashboard" / "start.sh"
LEGACY_STOP = Path(__file__).resolve().parents[1] / "dashboard" / "stop.sh"


def load_launcher():
    loader = importlib.machinery.SourceFileLoader("llm_cluster_launcher", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError("could not create launcher module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = load_launcher()

    def test_action_allowlist_rejects_missing_extra_and_injected_arguments(self) -> None:
        for arguments in ([], ["start", "extra"], ["start;id"], ["--help"]):
            completed = subprocess.run(
                ["python3", str(SCRIPT), *arguments],
                cwd="/tmp",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 64, arguments)
            self.assertIn("start|stop|restart|status|logs", completed.stderr)

    def test_legacy_dashboard_scripts_delegate_to_the_single_manager(self) -> None:
        start_text = LEGACY_START.read_text(encoding="utf-8")
        stop_text = LEGACY_STOP.read_text(encoding="utf-8")
        self.assertIn('exec "$PROJECT_ROOT/scripts/llm-cluster" start', start_text)
        self.assertIn('exec "$PROJECT_ROOT/scripts/llm-cluster" stop', stop_text)
        for text in (start_text, stop_text):
            self.assertNotIn("kill ", text)
            self.assertNotIn("nohup ", text)

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux /proc is required")
    def test_process_start_ticks_are_read_from_proc_without_shelling_out(self) -> None:
        state, start_ticks = self.launcher.read_proc_stat(os.getpid())
        self.assertNotEqual(state, "Z")
        self.assertGreater(start_ticks, 0)

    def test_atomic_record_is_private_and_identity_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            identity = self.launcher.ProcessIdentity(
                component="dashboard",
                pid=1234,
                uid=1000,
                start_ticks=5678,
                exe="/usr/bin/python3.10",
                cwd="/project",
                argv=("/project/.venv/bin/python", "-m", "uvicorn"),
            )
            self.launcher.atomic_write(
                path,
                json.dumps(identity.as_record(), sort_keys=True) + "\n",
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            record = self.launcher.read_identity_record(path)
            self.assertIsNotNone(record)
            self.assertTrue(self.launcher.identity_record_matches(record, identity))
            record["start_ticks"] = 9999
            self.assertFalse(self.launcher.identity_record_matches(record, identity))

    def test_malformed_pid_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.pid"
            for value in ("", "1", "-10", "123 extra", "$(id)"):
                path.write_text(value, encoding="utf-8")
                with self.assertRaises(self.launcher.LauncherError):
                    self.launcher.read_pid_file(path)

    def test_logs_are_strictly_limited_to_requested_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            path.write_text("".join(f"line-{index}\n" for index in range(250)), encoding="utf-8")
            self.assertEqual(
                list(self.launcher.tail_lines(path, 3)),
                ["line-247", "line-248", "line-249"],
            )

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux /proc is required")
    def test_tampered_pid_file_does_not_signal_unrelated_process(self) -> None:
        process = subprocess.Popen(["sleep", "30"])
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pid_file = root / "foreign.pid"
                pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
                spec = self.launcher.ProcessSpec(
                    name="foreign",
                    module="cluster.dashboard.app",
                    host="0.0.0.0",
                    port=18080,
                    health_path="/dashboard/health",
                    pid_file=pid_file,
                    log_file=root / "foreign.log",
                    identity_file=root / "foreign.identity.json",
                    access_log=False,
                )
                with self.assertRaises(self.launcher.LauncherError):
                    self.launcher.stop_component(spec, os.getuid())
                self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=5)

    @unittest.skipUnless(
        hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
        "Linux pidfd support is required",
    )
    def test_failed_child_rollback_escalates_without_leaving_process(self) -> None:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(30)",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(process.stdout.readline().strip(), "ready")
        pidfd = os.pidfd_open(process.pid, 0)
        old_term_timeout = self.launcher.TERM_TIMEOUT_SECONDS
        old_kill_timeout = self.launcher.KILL_TIMEOUT_SECONDS
        self.launcher.TERM_TIMEOUT_SECONDS = 0.1
        self.launcher.KILL_TIMEOUT_SECONDS = 2.0
        try:
            self.launcher.terminate_failed_child(process, pidfd)
            self.assertIsNotNone(process.poll())
        finally:
            self.launcher.TERM_TIMEOUT_SECONDS = old_term_timeout
            self.launcher.KILL_TIMEOUT_SECONDS = old_kill_timeout
            os.close(pidfd)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()


if __name__ == "__main__":
    unittest.main()
