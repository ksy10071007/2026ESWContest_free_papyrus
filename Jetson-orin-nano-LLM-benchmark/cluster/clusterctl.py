#!/usr/bin/env python3
"""Control a small head/worker Jetson LLM benchmark cluster over SSH."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import secrets
import shlex
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CLUSTER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLUSTER_DIR.parent
DEFAULT_INVENTORY = PROJECT_ROOT / ".run" / "cluster" / "nodes.local.csv"
DEFAULT_IDENTITY = Path.home() / ".ssh" / "id_ed25519_llm_cluster"
DEFAULT_WORKER_TOKEN = PROJECT_ROOT / ".run" / "cluster" / "worker.token"
DEFAULT_SETTINGS = PROJECT_ROOT / ".run" / "cluster" / "settings.json"
_worker_token_lock = threading.Lock()
SYSTEM_PACKAGE_ALLOWLIST = {
    "ca-certificates",
    "curl",
    "git",
    "rsync",
    "openssh-client",
    "iproute2",
    "build-essential",
    "cmake",
    "ninja-build",
    "pkg-config",
    "python3",
    "python3-dev",
    "python3-venv",
    "libopenblas-dev",
    "util-linux",
}
WORKER_READINESS_MARKER = "CLUSTER_READINESS_JSON="
ENVIRONMENT_MARKER = "CLUSTER_ENVIRONMENT_JSON="
READINESS_STATUSES = {
    "ready",
    "needs_setup",
    "manual",
    "unavailable",
    "failed",
    "not_checked",
    "repairable",
    "blocked",
}


def validate_project_dir(project_dir: str, user: str = "") -> str:
    """Reject broad or ambiguous sync targets before any remote write."""
    if (
        not re.fullmatch(r"/(?:home|opt|srv)/[a-zA-Z0-9._/-]+", project_dir)
        or ".." in Path(project_dir).parts
    ):
        raise ValueError("project_dir must be a safe path below /home, /opt or /srv")
    normalized = str(Path(project_dir))
    broad = {"/", "/home", "/opt", "/srv"}
    if user:
        broad.add(f"/home/{user}")
    parts = Path(normalized).parts
    if normalized in broad or (len(parts) >= 2 and parts[1] == "home" and len(parts) < 4):
        raise ValueError(f"project_dir is too broad for code synchronization: {project_dir}")
    return normalized

DISCOVERY_SCRIPT = r"""
set -eu
project_dir=$1
board=$(if [ -r /proc/device-tree/model ]; then tr -d '\000' </proc/device-tree/model; else uname -m; fi)
if [ -f /etc/nv_tegra_release ] || [ -d /etc/nv_tegra_release.d ] || command -v nvpmodel >/dev/null 2>&1; then
  platform_kind=jetson
elif printf '%s' "$board" | grep -qi 'raspberry pi'; then
  platform_kind=raspberry-pi
else
  platform_kind=unsupported
fi
pretty_os=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release 2>/dev/null | head -n 1 | sed 's/^"//;s/"$//')
packages='ca-certificates curl git rsync openssh-client iproute2 build-essential cmake ninja-build pkg-config python3 python3-dev python3-venv util-linux'
[ "$platform_kind" = raspberry-pi ] && packages="$packages libopenblas-dev"
missing=''
venv_works=false
venv_probe=$(mktemp -d 2>/dev/null || true)
if [ -n "$venv_probe" ] && python3 -m venv "$venv_probe/check" >/dev/null 2>&1; then
  venv_works=true
fi
[ -n "$venv_probe" ] && rm -rf "$venv_probe"
if command -v dpkg-query >/dev/null 2>&1; then
  for package_name in $packages; do
    if [ "$package_name" = python3-venv ] && [ "$venv_works" = true ]; then
      continue
    fi
    if ! dpkg-query -W -f='${db:Status-Abbrev}' "$package_name" 2>/dev/null | grep -q '^ii'; then
      missing="$missing $package_name"
    fi
  done
fi
sudo_nopasswd=false
if [ "$(id -u)" -eq 0 ]; then
  sudo_nopasswd=true
elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  sudo_nopasswd=true
fi
project_exists=false
[ -d "$project_dir" ] && project_exists=true
python_version=$(python3 --version 2>&1 || true)
disk_free_kb=$(df -Pk "${project_dir%/*}" 2>/dev/null | awk 'NR==2 {print $4}' || true)
ntp_sync=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || true)
printf 'platform_kind=%s\n' "$platform_kind"
printf 'board_model=%s\n' "$board"
printf 'architecture=%s\n' "$(uname -m)"
printf 'os=%s\n' "$pretty_os"
printf 'kernel=%s\n' "$(uname -r)"
printf 'python=%s\n' "$python_version"
printf 'sudo_nopasswd=%s\n' "$sudo_nopasswd"
printf 'project_exists=%s\n' "$project_exists"
printf 'missing_packages=%s\n' "$(printf '%s' "$missing" | xargs)"
printf 'disk_free_kb=%s\n' "$disk_free_kb"
printf 'ntp_synchronized=%s\n' "$ntp_sync"
""".strip()


@dataclass(frozen=True)
class Node:
    name: str
    role: str
    host: str
    user: str
    ssh_port: int
    api_port: int
    project_dir: str
    enabled: bool
    identity_file: str = ""
    platform: str = "auto"

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.api_port}"

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    @property
    def is_local(self) -> bool:
        if self.role != "head":
            return False
        local_names = {
            "127.0.0.1",
            "localhost",
            "::1",
            socket.gethostname(),
            socket.getfqdn(),
        }
        return self.host in local_names


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def load_nodes(path: Path, include_disabled: bool = False) -> List[Node]:
    if not path.exists():
        raise FileNotFoundError(
            f"Inventory not found: {path}. Copy cluster/config/nodes.example.csv to "
            ".run/cluster/nodes.local.csv and edit it."
        )

    nodes: List[Node] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "name",
            "role",
            "host",
            "user",
            "ssh_port",
            "api_port",
            "project_dir",
            "enabled",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Inventory is missing columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            if not row.get("name", "").strip():
                continue
            try:
                node = Node(
                    name=row["name"].strip(),
                    role=row["role"].strip().lower(),
                    host=row["host"].strip(),
                    user=row["user"].strip(),
                    ssh_port=int(row["ssh_port"]),
                    api_port=int(row["api_port"]),
                    project_dir=row["project_dir"].strip(),
                    enabled=_as_bool(row["enabled"]),
                    identity_file=row.get("identity_file", "").strip(),
                    platform=(row.get("platform", "auto") or "auto").strip().lower(),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid inventory row {line_number}: {exc}") from exc

            if node.role not in {"head", "worker"}:
                raise ValueError(f"Invalid role for {node.name}: {node.role}")
            if node.platform not in {"auto", "jetson", "raspberry-pi"}:
                raise ValueError(f"Invalid platform for {node.name}: {node.platform}")
            if not node.host:
                raise ValueError(f"Host is empty for {node.name}")
            try:
                validate_project_dir(node.project_dir, node.user)
            except ValueError as exc:
                raise ValueError(f"Invalid project_dir for {node.name}: {exc}") from exc
            if node.ssh_port < 1 or node.api_port < 1:
                raise ValueError(f"Ports must be positive for {node.name}")
            nodes.append(node)

    names = [node.name for node in nodes]
    if len(names) != len(set(names)):
        raise ValueError("Inventory contains duplicate node names")
    if sum(1 for node in nodes if node.role == "head" and node.enabled) != 1:
        raise ValueError("Inventory must contain exactly one enabled head node")
    return nodes if include_disabled else [node for node in nodes if node.enabled]


def select_nodes(nodes: Sequence[Node], names: Sequence[str], workers_only: bool = False) -> List[Node]:
    selected = list(nodes)
    if workers_only:
        selected = [node for node in selected if node.role == "worker"]
    if names:
        wanted = set(names)
        selected = [node for node in selected if node.name in wanted]
        found = {node.name for node in selected}
        missing = wanted.difference(found)
        if missing:
            raise ValueError(f"Unknown or disabled nodes: {', '.join(sorted(missing))}")
    return selected


def _identity_path(node: Node) -> Optional[Path]:
    raw = node.identity_file.strip()
    if not raw and DEFAULT_IDENTITY.exists():
        return DEFAULT_IDENTITY
    if not raw:
        return None
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def ssh_base(node: Node) -> List[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        str(node.ssh_port),
    ]
    identity = _identity_path(node)
    if identity is not None:
        command.extend(["-i", str(identity), "-o", "IdentitiesOnly=yes"])
    command.append(node.ssh_target)
    return command


def run_on_node(
    node: Node,
    args: Sequence[str],
    timeout: int = 120,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    if node.is_local:
        command = list(args)
    else:
        remote_command = " ".join(shlex.quote(part) for part in args)
        command = ssh_base(node) + [remote_command]
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def discover_node(node: Node, timeout: int = 20) -> Dict[str, Any]:
    """Inspect a key-authenticated Linux node without requiring project files."""
    result: Dict[str, Any] = {
        "name": node.name,
        "ssh": False,
        "project": False,
        "platform_kind": "unknown",
        "missing_packages": [],
        "sudo_nopasswd": False,
        "error": "",
    }
    try:
        proc = run_on_node(
            node,
            ["sh", "-lc", DISCOVERY_SCRIPT, "cluster-discovery", node.project_dir],
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
        return result
    if proc.returncode != 0:
        result["error"] = (proc.stderr or "SSH discovery failed").strip()
        return result
    result["ssh"] = True
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    result["project"] = str(result.get("project_exists", "false")).lower() == "true"
    result["sudo_nopasswd"] = str(result.get("sudo_nopasswd", "false")).lower() == "true"
    result["missing_packages"] = str(result.get("missing_packages", "")).split()
    disk_free_kb = str(result.get("disk_free_kb", ""))
    result["disk_free_gb"] = round(int(disk_free_kb) / (1024 * 1024), 2) if disk_free_kb.isdigit() else None
    return result


def bootstrap_system_one(node: Node) -> Dict[str, Any]:
    discovery = discover_node(node)
    if not discovery["ssh"]:
        return {"name": node.name, "ok": False, "stdout": "", "stderr": discovery["error"]}
    if discovery.get("platform_kind") not in {"jetson", "raspberry-pi"}:
        return {
            "name": node.name,
            "ok": False,
            "stdout": "",
            "stderr": f"Unsupported platform: {discovery.get('board_model', 'unknown')}",
        }
    if discovery.get("architecture") not in {"aarch64", "arm64"}:
        return {
            "name": node.name,
            "ok": False,
            "stdout": "",
            "stderr": f"64-bit ARM OS is required; detected {discovery.get('architecture')}",
        }
    missing = [item for item in discovery["missing_packages"] if item in SYSTEM_PACKAGE_ALLOWLIST]
    if not missing:
        return {
            "name": node.name,
            "ok": True,
            "stdout": f"platform={discovery['platform_kind']} system dependencies ready",
            "stderr": "",
            "discovery": discovery,
        }
    if not discovery["sudo_nopasswd"]:
        manual = (
            "sudo apt-get update && sudo apt-get install -y --no-install-recommends "
            + " ".join(missing)
        )
        return {
            "name": node.name,
            "ok": False,
            "stdout": "",
            "stderr": "Passwordless sudo is unavailable. Run once on the node: " + manual,
            "discovery": discovery,
        }
    prefix: List[str] = []
    identity = run_on_node(node, ["id", "-u"], timeout=10)
    if identity.stdout.strip() != "0":
        prefix = ["sudo", "-n"]
    update = run_on_node(node, prefix + ["apt-get", "update"], timeout=600)
    if update.returncode != 0:
        return {"name": node.name, "ok": False, "stdout": update.stdout, "stderr": update.stderr}
    install = run_on_node(
        node,
        prefix + ["apt-get", "install", "-y", "--no-install-recommends", *missing],
        timeout=1200,
    )
    return {
        "name": node.name,
        "ok": install.returncode == 0,
        "stdout": install.stdout.strip(),
        "stderr": install.stderr.strip(),
        "discovery": discovery,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _readiness_check(
    check_id: str,
    label: str,
    status: str,
    detail: str,
    auto_fixable: bool = False,
) -> Dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "status": status,
        "detail": detail,
        "auto_fixable": auto_fixable,
    }


def _expected_backend(platform_kind: str) -> str:
    if platform_kind == "jetson":
        return "cuda"
    if platform_kind == "raspberry-pi":
        return "openblas"
    return "unknown"


def _backend_report(kind: str, verified: bool = False) -> Dict[str, Any]:
    return {"kind": kind[:64], "verified": bool(verified)}


def _safe_missing_packages(values: Any) -> List[str]:
    if not isinstance(values, list):
        values = str(values or "").split()
    return sorted(
        {
            str(value)
            for value in values
            if isinstance(value, str) and value in SYSTEM_PACKAGE_ALLOWLIST
        }
    )


def _manual_commands_for(discovery: Dict[str, Any]) -> List[str]:
    """Return only locally constructed commands from the fixed package allowlist."""
    missing = _safe_missing_packages(discovery.get("missing_packages", []))
    if missing and not discovery.get("sudo_nopasswd", False):
        return [
            "sudo apt-get update && sudo apt-get install -y --no-install-recommends "
            + " ".join(missing)
        ]
    return []


def _discovery_readiness(node: Node, discovery: Dict[str, Any]) -> Dict[str, Any]:
    platform_kind = str(discovery.get("platform_kind") or "unknown")
    architecture = str(discovery.get("architecture") or "unknown")
    project_exists = bool(discovery.get("project", False))
    missing = _safe_missing_packages(discovery.get("missing_packages", []))
    ssh_ready = bool(discovery.get("ssh", False))
    platform_ready = platform_kind in {"jetson", "raspberry-pi"}
    architecture_ready = architecture in {"aarch64", "arm64"}
    manual_commands = _manual_commands_for(discovery)

    if not ssh_ready:
        status = "unavailable"
    elif not platform_ready or not architecture_ready:
        status = "blocked"
    elif manual_commands:
        status = "manual"
    else:
        status = "repairable"

    checks = [
        _readiness_check(
            "ssh",
            "SSH connection",
            "pass" if ssh_ready else "fail",
            "Key-based SSH connection is available"
            if ssh_ready
            else str(discovery.get("error") or "SSH host is unreachable or key authentication failed"),
        ),
        _readiness_check(
            "platform",
            "Supported board",
            "pass" if platform_ready else "fail",
            str(discovery.get("board_model") or platform_kind),
        ),
        _readiness_check(
            "architecture",
            "64-bit ARM OS",
            "pass" if architecture_ready else "fail",
            architecture,
        ),
        _readiness_check(
            "system_packages",
            "System build packages",
            "pass" if not missing else "missing",
            "Installed" if not missing else "Missing: " + ", ".join(missing),
            auto_fixable=bool(missing and discovery.get("sudo_nopasswd", False)),
        ),
        _readiness_check(
            "project",
            "Benchmark project",
            "pass" if project_exists else "missing",
            node.project_dir if project_exists else f"Project will be synchronized to {node.project_dir}",
            auto_fixable=ssh_ready and node.role == "worker",
        ),
        _readiness_check(
            "virtualenv",
            "Project virtual environment",
            "unknown",
            "Run the project preflight after synchronizing the project",
            auto_fixable=True,
        ),
        _readiness_check(
            "llm_backend",
            "LLM inference backend",
            "unknown",
            f"Expected backend: {_expected_backend(platform_kind)}",
            auto_fixable=platform_ready,
        ),
    ]
    return {
        "schema_version": 1,
        "node": node.name,
        "status": status,
        "checked_at": _utc_now(),
        "platform": platform_kind,
        "board_model": str(discovery.get("board_model") or "")[:240],
        "architecture": architecture[:32],
        "os": str(discovery.get("os") or "")[:240],
        "python": str(discovery.get("python") or "")[:120],
        "disk_free_gb": discovery.get("disk_free_gb"),
        "project_dir": node.project_dir,
        "venv_path": f"{node.project_dir}/.venv",
        "checks": checks,
        "missing_system_packages": missing,
        "manual_commands": manual_commands,
        "backend": _backend_report(_expected_backend(platform_kind)),
        "model_count": 0,
    }


def _extract_marker_json(output: str, marker: str) -> Optional[Dict[str, Any]]:
    for line in reversed(output.splitlines()):
        if not line.startswith(marker):
            continue
        try:
            value = json.loads(line[len(marker) :])
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None
    return None


def _normalize_worker_report(
    node: Node,
    raw: Dict[str, Any],
    discovery: Dict[str, Any],
    returncode: int,
    stdout: str,
    stderr: str,
) -> Dict[str, Any]:
    schema_version = raw.get("schema_version")
    structured_backend = raw.get("backend")
    marker_valid = schema_version == 1 and isinstance(structured_backend, dict)
    platform_kind = str(raw.get("platform") or discovery.get("platform_kind") or "unknown")
    missing = _safe_missing_packages(
        raw.get("missing_system_packages", discovery.get("missing_packages", []))
    )
    allowed_check_statuses = {"pass", "fail", "warn", "missing", "unknown"}
    checks: List[Dict[str, Any]] = []
    raw_checks = raw.get("checks")
    if isinstance(raw_checks, list):
        for index, item in enumerate(raw_checks[:40]):
            if not isinstance(item, dict):
                continue
            check_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(item.get("id") or f"check_{index}"))[:64]
            check_status = str(item.get("status") or "unknown")
            checks.append(
                _readiness_check(
                    check_id,
                    str(item.get("label") or check_id)[:120],
                    check_status if check_status in allowed_check_statuses else "unknown",
                    str(item.get("detail") or "")[:2000],
                    bool(item.get("auto_fixable", False)),
                )
            )
    if not checks:
        detail = (stderr or stdout or "Worker preflight did not return a structured report").strip()
        checks = [
            _readiness_check(
                "worker_preflight",
                "Worker runtime preflight",
                "pass" if returncode == 0 else "fail",
                detail[-2000:],
                auto_fixable=returncode != 0,
            )
        ]

    raw_status = str(raw.get("status") or "needs_setup")
    status = raw_status if raw_status in READINESS_STATUSES else ("ready" if returncode == 0 else "failed")
    if not marker_valid:
        status = "needs_setup"
        checks.append(
            _readiness_check(
                "readiness_schema",
                "Structured readiness schema",
                "fail",
                "worker_setup.sh schema v1 report is missing; synchronize the current project code",
                auto_fixable=True,
            )
        )
    if returncode != 0 and status == "ready":
        status = "needs_setup"
    manual_commands = _manual_commands_for({**discovery, "missing_packages": missing})
    if manual_commands and status not in {"unavailable", "failed"}:
        status = "manual"
    try:
        model_count = max(0, int(raw.get("model_count", 0)))
    except (TypeError, ValueError):
        model_count = 0
    raw_backend = structured_backend
    if isinstance(raw_backend, dict):
        backend = _backend_report(
            str(raw_backend.get("kind") or _expected_backend(platform_kind)),
            bool(raw_backend.get("verified", False)),
        )
    else:
        backend = _backend_report(_expected_backend(platform_kind), False)
    try:
        disk_free_gb: Optional[float] = round(float(raw.get("disk_free_gb")), 2)
    except (TypeError, ValueError):
        disk_free_gb = discovery.get("disk_free_gb")
    return {
        "schema_version": 1,
        "node": node.name,
        "status": status,
        "checked_at": str(raw.get("checked_at") or _utc_now()),
        "platform": platform_kind,
        "board_model": str(raw.get("board_model") or discovery.get("board_model") or "")[:240],
        "architecture": str(raw.get("architecture") or discovery.get("architecture") or "")[:32],
        "os": str(raw.get("os") or discovery.get("os") or "")[:240],
        "python": str(raw.get("python") or discovery.get("python") or "")[:120],
        "disk_free_gb": disk_free_gb,
        "project_dir": node.project_dir,
        "venv_path": f"{node.project_dir}/.venv",
        "checks": checks,
        "missing_system_packages": missing,
        # Never trust executable text returned by a remote node. Commands are
        # reconstructed locally from the fixed package allowlist above.
        "manual_commands": manual_commands,
        "backend": backend,
        "model_count": model_count,
    }


def check_environment_one(node: Node) -> Dict[str, Any]:
    """Return a stable readiness report, including for a node without project files."""
    discovery = discover_node(node, timeout=20)
    if not discovery.get("ssh") or not discovery.get("project"):
        return _discovery_readiness(node, discovery)

    script = f"{node.project_dir}/cluster/worker_setup.sh"
    try:
        proc = run_on_node(
            node,
            [script, "--check-only", "--project-dir", node.project_dir],
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report = _discovery_readiness(node, discovery)
        report["status"] = "failed"
        report["checks"].append(
            _readiness_check("worker_preflight", "Worker runtime preflight", "fail", str(exc))
        )
        return report
    raw = _extract_marker_json(
        "\n".join(part for part in (proc.stdout, proc.stderr) if part),
        WORKER_READINESS_MARKER,
    ) or {}
    return _normalize_worker_report(
        node,
        raw,
        discovery,
        proc.returncode,
        proc.stdout,
        proc.stderr,
    )


def _append_install_failure(report: Dict[str, Any], detail: str) -> Dict[str, Any]:
    if report.get("status") == "ready":
        report["status"] = "failed"
    report.setdefault("checks", []).append(
        _readiness_check(
            "environment_install",
            "Environment installation",
            "fail",
            detail[-2000:] or "Installation failed",
        )
    )
    return report


def install_environment_one(node: Node) -> Dict[str, Any]:
    """Install only the fixed cluster runtime, then always run a fresh check."""
    discovery = discover_node(node, timeout=20)
    if not discovery.get("ssh"):
        return check_environment_one(node)

    # Bootstrap every node before worker_setup. In particular, a minimal head
    # may not have util-linux/flock yet, which worker_setup needs to serialize
    # installation safely. Only remote workers need a project code sync.
    bootstrap = bootstrap_system_one(node)
    if not bootstrap.get("ok"):
        return _append_install_failure(
            check_environment_one(node),
            str(bootstrap.get("stderr") or bootstrap.get("stdout") or "System bootstrap failed"),
        )
    if node.role == "worker":
        sync = sync_code_one(node)
        if not sync.get("ok"):
            return _append_install_failure(
                check_environment_one(node),
                str(sync.get("stderr") or sync.get("stdout") or "Project synchronization failed"),
            )

    setup = _setup_one(node)
    lifecycle: Optional[Dict[str, Any]] = None
    if setup.get("ok"):
        # A running API has imported the old Python/native libraries. Restart it
        # so live execution matches the freshly verified environment report.
        lifecycle = _lifecycle_one(node, "restart")
    report = check_environment_one(node)
    if not setup.get("ok"):
        return _append_install_failure(
            report,
            str(setup.get("stderr") or setup.get("stdout") or "Runtime setup failed"),
        )
    if lifecycle is not None and not lifecycle.get("ok"):
        return _append_install_failure(
            report,
            str(
                lifecycle.get("stderr")
                or lifecycle.get("stdout")
                or "Worker API restart failed after environment installation"
            ),
        )
    return report


def _print_environment_reports(reports: Sequence[Dict[str, Any]]) -> None:
    for report in reports:
        print(f"[{report['node']}] {str(report['status']).upper()}", flush=True)
        print(
            ENVIRONMENT_MARKER + json.dumps(report, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )


def command_environment_check(nodes: Sequence[Node], _args: argparse.Namespace) -> int:
    if not nodes:
        reports: List[Dict[str, Any]] = []
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
            futures = {executor.submit(check_environment_one, node): node for node in nodes}
            reports = []
            for future in concurrent.futures.as_completed(futures):
                node = futures[future]
                try:
                    reports.append(future.result())
                except Exception as exc:
                    reports.append(_exception_environment_report(node, "Environment check", exc))
        order = {node.name: index for index, node in enumerate(nodes)}
        reports.sort(key=lambda report: order[report["node"]])
    _print_environment_reports(reports)
    return 0 if all(report["status"] == "ready" for report in reports) else 1


def command_environment_install(nodes: Sequence[Node], args: argparse.Namespace) -> int:
    if not getattr(args, "confirmed", False):
        print("Environment installation requires --confirmed", file=sys.stderr)
        return 2
    # Installation can include apt and a native llama-cpp-python build. Keep it
    # serial so a small head does not build several remote nodes at once.
    reports = []
    for node in nodes:
        print(f"[{node.name}] installing the fixed benchmark environment", flush=True)
        try:
            report = install_environment_one(node)
        except Exception as exc:
            report = _exception_environment_report(node, "Environment installation", exc)
        reports.append(report)
        # Persist partial progress immediately through the dashboard parser.
        _print_environment_reports([report])
    return 0 if all(report["status"] == "ready" for report in reports) else 1


def _exception_environment_report(
    node: Node, label: str, exc: BaseException
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "node": node.name,
        "status": "failed",
        "checked_at": _utc_now(),
        "platform": node.platform,
        "project_dir": node.project_dir,
        "venv_path": f"{node.project_dir}/.venv",
        "checks": [
            _readiness_check(
                "environment_operation",
                label,
                "fail",
                str(exc)[-2000:] or type(exc).__name__,
            )
        ],
        "missing_system_packages": [],
        "manual_commands": [],
        "backend": _backend_report(_expected_backend(node.platform), False),
        "model_count": 0,
    }


def request_json(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json", **worker_auth_headers()}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_worker_token() -> str:
    with _worker_token_lock:
        configured = os.getenv("CLUSTER_API_TOKEN", "").strip()
        if configured:
            DEFAULT_WORKER_TOKEN.parent.mkdir(parents=True, exist_ok=True)
            if not DEFAULT_WORKER_TOKEN.exists() or DEFAULT_WORKER_TOKEN.read_text(encoding="utf-8").strip() != configured:
                temporary = DEFAULT_WORKER_TOKEN.with_suffix(".tmp")
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(configured + "\n")
                os.replace(temporary, DEFAULT_WORKER_TOKEN)
            return configured
        if not DEFAULT_WORKER_TOKEN.exists():
            DEFAULT_WORKER_TOKEN.parent.mkdir(parents=True, exist_ok=True)
            temporary = DEFAULT_WORKER_TOKEN.with_suffix(".tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(secrets.token_urlsafe(32) + "\n")
            os.replace(temporary, DEFAULT_WORKER_TOKEN)
        DEFAULT_WORKER_TOKEN.chmod(0o600)
        return DEFAULT_WORKER_TOKEN.read_text(encoding="utf-8").strip()


def worker_auth_headers() -> Dict[str, str]:
    if not worker_auth_enabled():
        return {}
    token = ensure_worker_token()
    return {"X-Cluster-Worker-Token": token} if token else {}


def cluster_settings() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "worker_api_auth": False,
        "dashboard_token_auth": False,
    }
    try:
        stored = json.loads(DEFAULT_SETTINGS.read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            defaults.update({key: value for key, value in stored.items() if key in defaults})
    except (OSError, ValueError):
        pass
    return defaults


def worker_auth_enabled() -> bool:
    override = os.getenv("CLUSTER_WORKER_AUTH", "").strip().lower()
    if override:
        return override in {"1", "true", "yes", "on", "enabled"}
    return bool(cluster_settings().get("worker_api_auth", False))


def status_one(node: Node) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": node.name,
        "role": node.role,
        "host": node.host,
        "ssh": False,
        "project": False,
        "api": False,
        "loaded_model": None,
        "error": "",
    }
    discovery = discover_node(node, timeout=12)
    result["ssh"] = discovery["ssh"]
    result["project"] = discovery["project"]
    if not discovery["ssh"]:
        result["error"] = discovery["error"]
    elif not discovery["project"]:
        result["error"] = "project directory missing"

    try:
        health = request_json(f"{node.api_url}/cluster/health", timeout=3.0)
        result["api"] = health.get("ok") is True
        current = health.get("current") or {}
        result["loaded_model"] = current.get("model_id")
    except (OSError, ValueError, urllib.error.URLError) as exc:
        if not result["error"]:
            result["error"] = f"API: {exc}"
    return result


def run_parallel(nodes: Sequence[Node], function: Any) -> List[Any]:
    if not nodes:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = {executor.submit(function, node): node for node in nodes}
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    order = {node.name: index for index, node in enumerate(nodes)}
    return sorted(results, key=lambda item: order[item["name"]])


def print_status(results: Sequence[Dict[str, Any]]) -> None:
    print(f"{'NODE':<20} {'ROLE':<7} {'SSH':<5} {'PROJECT':<8} {'API':<5} MODEL")
    for item in results:
        print(
            f"{item['name']:<20} {item['role']:<7} "
            f"{str(item['ssh']):<5} {str(item['project']):<8} "
            f"{str(item['api']):<5} {item['loaded_model'] or '-'}"
        )
        if item["error"]:
            print(f"  error: {item['error']}")


def command_inventory(nodes: Sequence[Node], _args: argparse.Namespace) -> int:
    print(f"{'NODE':<20} {'ROLE':<7} {'HOST':<16} {'SSH':<5} {'API':<5} PROJECT")
    for node in nodes:
        print(
            f"{node.name:<20} {node.role:<7} {node.host:<16} "
            f"{node.ssh_port:<5} {node.api_port:<5} {node.project_dir}"
        )
    return 0


def command_status(nodes: Sequence[Node], args: argparse.Namespace) -> int:
    results = run_parallel(nodes, status_one)
    print_status(results)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["ssh"] and item["project"] for item in results) else 1


def _doctor_one(node: Node) -> Dict[str, Any]:
    script = f"{node.project_dir}/cluster/worker_setup.sh"
    try:
        proc = run_on_node(
            node,
            [script, "--check-only", "--project-dir", node.project_dir],
            timeout=60,
        )
        return {
            "name": node.name,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"name": node.name, "ok": False, "stdout": "", "stderr": str(exc)}


def _setup_one(node: Node) -> Dict[str, Any]:
    script = f"{node.project_dir}/cluster/worker_setup.sh"
    try:
        proc = run_on_node(
            node,
            [script, "--install", "--project-dir", node.project_dir],
            timeout=3600,
        )
        return {
            "name": node.name,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"name": node.name, "ok": False, "stdout": "", "stderr": str(exc)}


def command_doctor(nodes: Sequence[Node], _args: argparse.Namespace) -> int:
    results = run_parallel(nodes, _doctor_one)
    for item in results:
        print(f"[{item['name']}] {'OK' if item['ok'] else 'FAIL'}")
        if item["stdout"]:
            print(item["stdout"])
        if item["stderr"]:
            print(item["stderr"], file=sys.stderr)
    return 0 if all(item["ok"] for item in results) else 1


def command_setup(nodes: Sequence[Node], _args: argparse.Namespace) -> int:
    workers = [node for node in nodes if node.role == "worker"]
    if not workers:
        print("No enabled worker nodes; nothing to set up.")
        return 0
    results = run_parallel(workers, _setup_one)
    for item in results:
        print(f"[{item['name']}] {'OK' if item['ok'] else 'FAIL'}")
        if item["stdout"]:
            print(item["stdout"])
        if item["stderr"]:
            print(item["stderr"], file=sys.stderr)
    return 0 if all(item["ok"] for item in results) else 1


def _rsync_ssh(node: Node) -> str:
    parts = [
        "ssh",
        "-p",
        str(node.ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    identity = _identity_path(node)
    if identity is not None:
        parts.extend(["-i", str(identity), "-o", "IdentitiesOnly=yes"])
    return " ".join(shlex.quote(part) for part in parts)


def sync_code_one(node: Node, dry_run: bool = False) -> Dict[str, Any]:
    if node.is_local:
        return {"name": node.name, "ok": True, "stdout": "local head; skipped", "stderr": ""}
    mkdir = run_on_node(node, ["mkdir", "-p", node.project_dir], timeout=30)
    if mkdir.returncode != 0:
        return {"name": node.name, "ok": False, "stdout": mkdir.stdout, "stderr": mkdir.stderr}

    command = [
        "rsync",
        "-az",
        "--itemize-changes",
        "--exclude=.git/",
        "--exclude=.venv/",
        "--exclude=models/",
        "--exclude=outputs/",
        "--exclude=.run/",
        "--exclude=__pycache__/",
        "--exclude=cluster/nodes.local.csv",
        "--exclude=cluster/results/",
        "-e",
        _rsync_ssh(node),
    ]
    if dry_run:
        command.append("--dry-run")
    command.extend([f"{PROJECT_ROOT}/", f"{node.ssh_target}:{node.project_dir}/"])
    proc = subprocess.run(command, text=True, capture_output=True, timeout=600)
    return {
        "name": node.name,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def sync_worker_token_one(node: Node) -> Dict[str, Any]:
    ensure_worker_token()
    if node.is_local:
        return {"name": node.name, "ok": True, "stdout": "local token ready", "stderr": ""}
    remote_runtime = f"{node.project_dir}/.run/cluster"
    mkdir = run_on_node(node, ["mkdir", "-p", remote_runtime], timeout=30)
    if mkdir.returncode != 0:
        return {"name": node.name, "ok": False, "stdout": mkdir.stdout, "stderr": mkdir.stderr}
    command = [
        "rsync",
        "-a",
        "--chmod=F600",
        "-e",
        _rsync_ssh(node),
        str(DEFAULT_WORKER_TOKEN),
        f"{node.ssh_target}:{remote_runtime}/worker.token",
    ]
    proc = subprocess.run(command, text=True, capture_output=True, timeout=60)
    return {
        "name": node.name,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def command_sync_code(nodes: Sequence[Node], args: argparse.Namespace) -> int:
    workers = [node for node in nodes if node.role == "worker"]
    results = [sync_code_one(node, dry_run=args.dry_run) for node in workers]
    if not results:
        print("No enabled worker nodes; nothing to sync.")
        return 0
    for item in results:
        print(f"[{item['name']}] {'OK' if item['ok'] else 'FAIL'}")
        if item["stdout"]:
            print(item["stdout"])
        if item["stderr"]:
            print(item["stderr"], file=sys.stderr)
    return 0 if all(item["ok"] for item in results) else 1


def sync_models_one(node: Node, model_paths: Sequence[str], dry_run: bool = False) -> Dict[str, Any]:
    if node.is_local:
        return {"name": node.name, "ok": True, "stdout": "local head; skipped", "stderr": ""}
    target_models = f"{node.project_dir}/models"
    mkdir = run_on_node(node, ["mkdir", "-p", target_models], timeout=30)
    if mkdir.returncode != 0:
        return {"name": node.name, "ok": False, "stdout": mkdir.stdout, "stderr": mkdir.stderr}

    stdout: List[str] = []
    stderr: List[str] = []
    ok = True
    for relative in model_paths:
        source = (PROJECT_ROOT / "models" / relative).resolve()
        try:
            source.relative_to((PROJECT_ROOT / "models").resolve())
        except ValueError:
            return {"name": node.name, "ok": False, "stdout": "", "stderr": f"Unsafe model path: {relative}"}
        if not source.is_file() or source.suffix.lower() != ".gguf":
            return {"name": node.name, "ok": False, "stdout": "", "stderr": f"Model not found: {relative}"}

        remote_parent = f"{target_models}/{Path(relative).parent.as_posix()}"
        parent_result = run_on_node(node, ["mkdir", "-p", remote_parent], timeout=30)
        if parent_result.returncode != 0:
            return {"name": node.name, "ok": False, "stdout": parent_result.stdout, "stderr": parent_result.stderr}

        command = [
            "rsync",
            "-ah",
            "--partial",
            "--append-verify",
            "--info=progress2",
            "-e",
            _rsync_ssh(node),
        ]
        if dry_run:
            command.append("--dry-run")
        command.extend([str(source), f"{node.ssh_target}:{remote_parent}/"])
        proc = subprocess.run(command, text=True, capture_output=True, timeout=7200)
        stdout.append(proc.stdout.strip())
        stderr.append(proc.stderr.strip())
        ok = ok and proc.returncode == 0
        if proc.returncode != 0:
            break
    return {
        "name": node.name,
        "ok": ok,
        "stdout": "\n".join(item for item in stdout if item),
        "stderr": "\n".join(item for item in stderr if item),
    }


def all_model_paths() -> List[str]:
    model_root = PROJECT_ROOT / "models"
    return sorted(path.relative_to(model_root).as_posix() for path in model_root.rglob("*.gguf"))


def command_sync_models(nodes: Sequence[Node], args: argparse.Namespace) -> int:
    paths = args.model or all_model_paths()
    if not paths:
        print("No GGUF models found on head.", file=sys.stderr)
        return 1
    workers = [node for node in nodes if node.role == "worker"]
    if not workers:
        print("No enabled worker nodes; nothing to sync.")
        return 0
    for node in workers:
        item = sync_models_one(node, paths, dry_run=args.dry_run)
        print(f"[{item['name']}] {'OK' if item['ok'] else 'FAIL'}")
        if item["stdout"]:
            print(item["stdout"])
        if item["stderr"]:
            print(item["stderr"], file=sys.stderr)
        if not item["ok"]:
            return 1
    return 0


def command_prepare(nodes: Sequence[Node], args: argparse.Namespace) -> int:
    workers = [node for node in nodes if node.role == "worker"]
    if not workers:
        print("No enabled worker nodes; nothing to prepare.")
        return 0
    model_paths = args.model or []
    for node in workers:
        print(f"[{node.name}] discovering platform and system dependencies", flush=True)
        bootstrap_result = bootstrap_system_one(node)
        if bootstrap_result["stdout"]:
            print(bootstrap_result["stdout"])
        if not bootstrap_result["ok"]:
            print(bootstrap_result["stderr"], file=sys.stderr)
            return 1

        print(f"[{node.name}] syncing project code", flush=True)
        code_result = sync_code_one(node)
        if not code_result["ok"]:
            print(code_result["stderr"], file=sys.stderr)
            return 1

        print(f"[{node.name}] checking/installing runtime", flush=True)
        setup_result = _setup_one(node)
        if setup_result["stdout"]:
            print(setup_result["stdout"])
        if not setup_result["ok"]:
            print(setup_result["stderr"], file=sys.stderr)
            return 1

        if worker_auth_enabled():
            print(f"[{node.name}] provisioning worker API credential", flush=True)
            token_result = sync_worker_token_one(node)
            if not token_result["ok"]:
                print(token_result["stderr"], file=sys.stderr)
                return 1

        if model_paths:
            print(f"[{node.name}] syncing {len(model_paths)} selected model(s)", flush=True)
            model_result = sync_models_one(node, model_paths)
            if model_result["stdout"]:
                print(model_result["stdout"])
            if not model_result["ok"]:
                print(model_result["stderr"], file=sys.stderr)
                return 1

        print(f"[{node.name}] starting worker API", flush=True)
        start_result = _lifecycle_one(node, "start")
        if start_result["stdout"]:
            print(start_result["stdout"])
        if not start_result["ok"]:
            print(start_result["stderr"], file=sys.stderr)
            return 1
        print(f"[{node.name}] ready", flush=True)
    return 0


def _prepare_rpc_one(node: Node) -> Dict[str, Any]:
    script = f"{node.project_dir}/cluster/rpc/runtime.sh"
    try:
        process = run_on_node(node, [script, "prepare"], timeout=7200)
        return {
            "name": node.name,
            "ok": process.returncode == 0,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"name": node.name, "ok": False, "stdout": "", "stderr": str(exc)}


def command_prepare_rpc(nodes: Sequence[Node], _args: argparse.Namespace) -> int:
    """Build the pinned native llama.cpp RPC runtime on selected nodes."""
    for node in nodes:
        print(f"[{node.name}] syncing current project before RPC build", flush=True)
        if not node.is_local:
            sync = sync_code_one(node)
            if not sync["ok"]:
                print(sync["stderr"], file=sys.stderr)
                return 1
        print(f"[{node.name}] building pinned llama.cpp RPC runtime", flush=True)
        result = _prepare_rpc_one(node)
        if result["stdout"]:
            print(result["stdout"])
        if not result["ok"]:
            print(result["stderr"], file=sys.stderr)
            return 1
    return 0


def _lifecycle_one(node: Node, action: str) -> Dict[str, Any]:
    if action == "restart":
        stopped = _lifecycle_one(node, "stop")
        if not stopped["ok"]:
            return stopped
        if worker_auth_enabled():
            token = sync_worker_token_one(node)
            if not token["ok"]:
                return token
        return _lifecycle_one(node, "start")
    script = f"{node.project_dir}/cluster/worker/{action}.sh"
    env_command = [
        "env",
        f"PORT={node.api_port}",
        "HOST=0.0.0.0",
        f"CLUSTER_NODE_NAME={node.name}",
        f"CLUSTER_NODE_ROLE={node.role}",
        f"CLUSTER_PLATFORM={node.platform}",
        f"CLUSTER_WORKER_AUTH={'true' if worker_auth_enabled() else 'false'}",
        script,
    ]
    try:
        proc = run_on_node(node, env_command, timeout=120)
        return {
            "name": node.name,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"name": node.name, "ok": False, "stdout": "", "stderr": str(exc)}


def command_lifecycle(nodes: Sequence[Node], action: str) -> int:
    results = run_parallel(nodes, lambda node: _lifecycle_one(node, action))
    for item in results:
        print(f"[{item['name']}] {'OK' if item['ok'] else 'FAIL'}")
        if item["stdout"]:
            print(item["stdout"])
        if item["stderr"]:
            print(item["stderr"], file=sys.stderr)
    return 0 if all(item["ok"] for item in results) else 1


def _select_model_one(node: Node, model_id: str, n_ctx: int, n_gpu_layers: int) -> Dict[str, Any]:
    try:
        result = request_json(
            f"{node.api_url}/api/select-model",
            method="POST",
            payload={"model_id": model_id, "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers},
            timeout=900.0,
        )
        current = result.get("current") or {}
        return {
            "name": node.name,
            "ok": result.get("ok") is True,
            "model": current.get("model_id"),
            "n_ctx": current.get("n_ctx"),
            "n_gpu_layers": current.get("n_gpu_layers"),
            "error": "",
        }
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"name": node.name, "ok": False, "model": None, "error": str(exc)}


def command_select_model(nodes: Sequence[Node], args: argparse.Namespace) -> int:
    pi_nodes = [node.name for node in nodes if node.platform == "raspberry-pi"]
    if pi_nodes and args.n_gpu_layers != 0:
        print(
            "Raspberry Pi nodes require --n-gpu-layers 0: " + ", ".join(pi_nodes),
            file=sys.stderr,
        )
        return 1
    results = run_parallel(
        nodes,
        lambda node: _select_model_one(node, args.model_id, args.n_ctx, args.n_gpu_layers),
    )
    for item in results:
        if item["ok"]:
            print(
                f"[{item['name']}] OK model={item['model']} "
                f"n_ctx={item['n_ctx']} n_gpu_layers={item['n_gpu_layers']}"
            )
        else:
            print(f"[{item['name']}] FAIL {item['error']}", file=sys.stderr)
    return 0 if all(item["ok"] for item in results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(os.getenv("CLUSTER_INVENTORY", DEFAULT_INVENTORY)),
    )
    parser.add_argument("--node", action="append", default=[], help="Limit to a node name; repeatable")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inventory", help="Print enabled nodes")
    status_parser = subparsers.add_parser("status", help="Check SSH, project and API status")
    status_parser.add_argument("--json-out")
    subparsers.add_parser("doctor", help="Run environment checks on enabled nodes")
    subparsers.add_parser(
        "environment-check",
        help="Emit structured LLM environment readiness for selected nodes",
    )
    environment_install_parser = subparsers.add_parser(
        "environment-install",
        help="Install the fixed project runtime, then recheck readiness",
    )
    environment_install_parser.add_argument(
        "--confirmed",
        action="store_true",
        help="Confirm fixed package, virtualenv and native backend installation",
    )
    subparsers.add_parser("discover", help="Discover platform and bootstrap prerequisites")
    subparsers.add_parser("setup", help="Install the worker Python/CUDA runtime")

    sync_code_parser = subparsers.add_parser("sync-code", help="Rsync code from head to workers")
    sync_code_parser.add_argument("--dry-run", action="store_true")

    sync_models_parser = subparsers.add_parser("sync-models", help="Rsync GGUF models to workers")
    sync_models_parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Relative path under models/; repeatable. Default: all models",
    )
    sync_models_parser.add_argument("--dry-run", action="store_true")

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Sync code, install runtime, sync selected models and start workers",
    )
    prepare_parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Relative model path to sync; repeatable",
    )
    subparsers.add_parser(
        "prepare-rpc",
        help="Build the pinned native llama.cpp RPC model-parallel runtime",
    )

    subparsers.add_parser("start", help="Start API servers on enabled nodes")
    subparsers.add_parser("stop", help="Stop API servers on enabled nodes")
    subparsers.add_parser("restart", help="Restart API servers with current settings")

    select_parser = subparsers.add_parser("select-model", help="Load the same model on enabled nodes")
    select_parser.add_argument("--model-id", required=True)
    select_parser.add_argument("--n-ctx", type=int, default=1024)
    select_parser.add_argument("--n-gpu-layers", type=int, default=20)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        nodes = load_nodes(args.inventory)
        nodes = select_nodes(nodes, args.node)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if args.command == "inventory":
        return command_inventory(nodes, args)
    if args.command == "status":
        return command_status(nodes, args)
    if args.command == "doctor":
        return command_doctor(nodes, args)
    if args.command == "environment-check":
        return command_environment_check(nodes, args)
    if args.command == "environment-install":
        return command_environment_install(nodes, args)
    if args.command == "discover":
        discovered = run_parallel(nodes, discover_node)
        print(json.dumps(discovered, ensure_ascii=False, indent=2))
        return 0 if all(item["ssh"] for item in discovered) else 1
    if args.command == "setup":
        return command_setup(nodes, args)
    if args.command == "sync-code":
        return command_sync_code(nodes, args)
    if args.command == "sync-models":
        return command_sync_models(nodes, args)
    if args.command == "prepare":
        return command_prepare(nodes, args)
    if args.command == "prepare-rpc":
        return command_prepare_rpc(nodes, args)
    if args.command == "start":
        return command_lifecycle(nodes, "start")
    if args.command == "stop":
        return command_lifecycle(nodes, "stop")
    if args.command == "restart":
        return command_lifecycle(nodes, "restart")
    if args.command == "select-model":
        return command_select_model(nodes, args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
