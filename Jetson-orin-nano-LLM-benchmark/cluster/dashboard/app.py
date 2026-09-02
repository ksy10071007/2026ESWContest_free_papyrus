#!/usr/bin/env python3
"""FastAPI control plane for the Jetson head/worker LLM benchmark cluster."""

from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import hashlib
import ipaddress
import json
import os
import queue
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator, model_validator

from cluster.benchmark.runner import (
    DEFAULT_RESULTS_DIR,
    ExperimentConfig,
    experiment_strategy_catalog,
    normalize_model_ids,
    run_experiment,
    strategy_work_units,
    validate_strategy,
)
from cluster.clusterctl import (
    DEFAULT_INVENTORY,
    Node,
    discover_node,
    load_nodes,
    request_json,
    run_on_node,
    select_nodes,
)


DASHBOARD_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = DASHBOARD_DIR.parent
PROJECT_ROOT = CLUSTER_DIR.parent
RUNTIME_DIR = Path(os.getenv("CLUSTER_RUNTIME_DIR", PROJECT_ROOT / ".run" / "cluster"))
INVENTORY_PATH = Path(os.getenv("CLUSTER_INVENTORY", DEFAULT_INVENTORY))
RESULTS_DIR = Path(os.getenv("CLUSTER_RESULTS_DIR", DEFAULT_RESULTS_DIR))
EXPERIMENTS_DIR = RUNTIME_DIR / "experiments"
DEFAULTS_PATH = CLUSTER_DIR / "config" / "experiment_defaults.json"
EXAMPLE_INVENTORY = CLUSTER_DIR / "config" / "nodes.example.csv"
TOKEN_PATH = RUNTIME_DIR / "dashboard.token"
SETTINGS_PATH = RUNTIME_DIR / "settings.json"
ENVIRONMENT_DIR = RUNTIME_DIR / "environment"
ENVIRONMENT_MARKER = "CLUSTER_ENVIRONMENT_JSON="


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_runtime() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    RUNTIME_DIR.chmod(0o700)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "_suites").mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ENVIRONMENT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    ENVIRONMENT_DIR.chmod(0o700)
    if not INVENTORY_PATH.exists():
        raise RuntimeError(
            f"Cluster inventory is missing: {INVENTORY_PATH}. "
            "Run ./cluster/setup_head.sh before starting the dashboard."
        )
    try:
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = ""
    if not token:
        temporary = TOKEN_PATH.with_suffix(f".tmp.{uuid.uuid4().hex}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(24) + "\n")
        os.replace(temporary, TOKEN_PATH)
    TOKEN_PATH.chmod(0o600)
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.write_text(
            '{\n  "worker_api_auth": false,\n  "dashboard_token_auth": false\n}\n',
            encoding="utf-8",
        )
    SETTINGS_PATH.chmod(0o600)


ensure_runtime()
DASHBOARD_TOKEN = TOKEN_PATH.read_text(encoding="utf-8").strip()


def supplied_dashboard_token(request: Request) -> str:
    return request.headers.get("X-Cluster-Token") or request.query_params.get("token", "")


def dashboard_token_is_valid(supplied: str) -> bool:
    return bool(supplied) and secrets.compare_digest(supplied, DASHBOARD_TOKEN)


def verify_token(request: Request) -> None:
    if not read_settings()["dashboard_token_auth"]:
        return
    if not dashboard_token_is_valid(supplied_dashboard_token(request)):
        raise HTTPException(status_code=401, detail="Dashboard access token is missing or invalid")


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: List[queue.Queue[Dict[str, Any]]] = []

    def publish(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, "at": utc_now(), **payload}
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(event)
                except (queue.Empty, queue.Full):
                    pass

    def stream(self, supplied_token: str = "") -> Generator[str, None, None]:
        subscriber: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.append(subscriber)
        try:
            yield f"data: {json.dumps({'type': 'connected', 'at': utc_now()})}\n\n"
            while True:
                try:
                    event = subscriber.get(timeout=15.0)
                    if (
                        read_settings()["dashboard_token_auth"]
                        and not dashboard_token_is_valid(supplied_token)
                    ):
                        yield f"data: {json.dumps({'type': 'auth_required', 'at': utc_now()})}\n\n"
                        return
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    if (
                        read_settings()["dashboard_token_auth"]
                        and not dashboard_token_is_valid(supplied_token)
                    ):
                        yield f"data: {json.dumps({'type': 'auth_required', 'at': utc_now()})}\n\n"
                        return
                    yield ": keepalive\n\n"
        finally:
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)


events = EventBus()


class NodePayload(BaseModel):
    name: str = Field(min_length=1, max_length=40, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    role: str = "worker"
    host: str = Field(min_length=1, max_length=45)
    user: str = Field(min_length=1, max_length=64, pattern=r"^[a-z_][a-zA-Z0-9_-]*$")
    ssh_port: int = Field(22, ge=1, le=65535)
    api_port: int = Field(8000, ge=1, le=65535)
    project_dir: str = Field(min_length=2, max_length=512)
    enabled: bool = True
    identity_file: str = Field("", max_length=512)
    platform: str = "auto"

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"head", "worker"}:
            raise ValueError("role must be head or worker")
        return value

    @field_validator("project_dir")
    @classmethod
    def validate_project_dir(cls, value: str) -> str:
        if (
            not value.startswith(("/home/", "/opt/", "/srv/"))
            or ".." in Path(value).parts
            or not re.fullmatch(r"/[a-zA-Z0-9._/-]+", value)
        ):
            raise ValueError("project_dir must be a safe absolute path")
        normalized = str(Path(value))
        parts = Path(normalized).parts
        if normalized in {"/home", "/opt", "/srv"} or (
            len(parts) >= 2 and parts[1] == "home" and len(parts) < 4
        ):
            raise ValueError("project_dir must name a dedicated project directory")
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        try:
            address = ipaddress.ip_address(value.strip())
        except ValueError as exc:
            raise ValueError("host must be a private IPv4 address") from exc
        allowed = (
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("127.0.0.0/8"),
        )
        if address.version != 4 or not any(address in network for network in allowed):
            raise ValueError("host must belong to the head node's private LAN")
        return str(address)

    @field_validator("identity_file")
    @classmethod
    def validate_identity_file(cls, value: str) -> str:
        if value:
            raise ValueError("identity_file is managed by the head node")
        return ""

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"auto", "jetson", "raspberry-pi"}:
            raise ValueError("platform must be auto, jetson or raspberry-pi")
        return value


class ActionPayload(BaseModel):
    action: str
    node_names: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)


class ExperimentPayload(BaseModel):
    experiment_id: str = Field("", max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$|^$")
    name: str = "cluster-load-test"
    node_names: List[str] = Field(min_length=1, max_length=4)
    # model_id remains accepted for older dashboard/CLI clients.  model_ids is
    # authoritative when present and model_id is normalized to its first item.
    model_id: str = ""
    model_ids: List[str] = Field(default_factory=list, max_length=32)
    continue_on_model_error: bool = True
    model_cooldown_s: float = Field(2.0, ge=0.0, le=300.0)
    n_ctx: int = Field(1024, ge=128, le=4096)
    n_gpu_layers: int = Field(30, ge=0, le=120)
    requests: int = Field(20, ge=1, le=10_000)
    concurrency: int = Field(4, ge=1, le=256)
    max_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    seed: int = Field(42, ge=-1, le=2_147_483_647)
    warmup_requests: int = Field(1, ge=0, le=10)
    prompt: str = Field(min_length=1, max_length=20_000)
    require_uniform_config: bool = True
    execution_strategy: str = "replicated_round_robin"
    sweep_mode: str = "cumulative"
    rpc_split_mode: str = "layer"
    rpc_split_policy: str = "auto"
    rpc_tensor_split: List[float] = Field(default_factory=list, max_length=4)
    acknowledge_experimental_rpc: bool = False

    @model_validator(mode="after")
    def normalize_models(self) -> "ExperimentPayload":
        models = normalize_model_ids(self.model_id, self.model_ids)
        self.model_ids = models
        self.model_id = models[0]
        return self


class ClusterSettingsPayload(BaseModel):
    worker_api_auth: Optional[bool] = None
    dashboard_token_auth: Optional[bool] = None
    dashboard_token: str = Field("", max_length=256)


inventory_lock = threading.RLock()
settings_lock = threading.RLock()


def read_settings() -> Dict[str, Any]:
    with settings_lock:
        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("settings must be a JSON object")
        except FileNotFoundError:
            raw = {}
        except (OSError, ValueError):
            # A damaged existing settings file must not silently disable a
            # dashboard protection that may previously have been enabled.
            return {"worker_api_auth": False, "dashboard_token_auth": True}
        worker_value = raw.get("worker_api_auth", False)
        dashboard_value = raw.get("dashboard_token_auth", False)
        return {
            "worker_api_auth": worker_value if isinstance(worker_value, bool) else False,
            "dashboard_token_auth": (
                dashboard_value
                if isinstance(dashboard_value, bool)
                else "dashboard_token_auth" in raw
            ),
        }


def write_settings(settings: Dict[str, Any]) -> None:
    with settings_lock:
        temporary = SETTINGS_PATH.with_suffix(f".tmp.{uuid.uuid4().hex}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, SETTINGS_PATH)


def read_all_nodes() -> List[Node]:
    with inventory_lock:
        return load_nodes(INVENTORY_PATH, include_disabled=True)


def write_all_nodes(nodes: Sequence[Node]) -> None:
    enabled_heads = [node for node in nodes if node.role == "head" and node.enabled]
    if len(enabled_heads) != 1:
        raise ValueError("Exactly one enabled head node is required")
    if len({node.name for node in nodes}) != len(nodes):
        raise ValueError("Node names must be unique")
    endpoints = [(node.host, node.ssh_port) for node in nodes]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("Each physical host and SSH port can be registered only once")
    if sum(1 for node in nodes if node.enabled) > 4:
        raise ValueError("At most four nodes can be enabled in one cluster")
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = INVENTORY_PATH.with_suffix(f".tmp.{uuid.uuid4().hex}")
    fieldnames = [
        "name",
        "role",
        "host",
        "user",
        "ssh_port",
        "api_port",
        "project_dir",
        "enabled",
        "identity_file",
        "platform",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for node in nodes:
            row = asdict(node)
            row["enabled"] = "true" if node.enabled else "false"
            writer.writerow(row)
    os.replace(temporary, INVENTORY_PATH)


def serialize_node(node: Node) -> Dict[str, Any]:
    item = asdict(node)
    item.pop("identity_file", None)
    item["api_url"] = node.api_url
    return item


def _environment_path(node_name: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}", node_name):
        raise ValueError("Invalid node name for environment report")
    return ENVIRONMENT_DIR / f"{node_name}.json"


def _node_environment_fingerprint(node: Node) -> str:
    identity = {
        "name": node.name,
        "role": node.role,
        "host": node.host,
        "user": node.user,
        "ssh_port": node.ssh_port,
        "api_port": node.api_port,
        "project_dir": node.project_dir,
        "platform": node.platform,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _invalidate_environment_report(node_name: str) -> None:
    try:
        _environment_path(node_name).unlink()
    except FileNotFoundError:
        pass


def _environment_placeholder(node: Node) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "node": node.name,
        "status": "not_checked",
        "checked_at": None,
        "received_at": None,
        "inventory_fingerprint": _node_environment_fingerprint(node),
        "platform": node.platform,
        "board_model": "",
        "architecture": "",
        "os": "",
        "python": "",
        "disk_free_gb": None,
        "project_dir": node.project_dir,
        "venv_path": f"{node.project_dir}/.venv",
        "checks": [],
        "missing_system_packages": [],
        "manual_commands": [],
        "backend": {"kind": "unknown", "verified": False},
        "model_count": 0,
    }


def normalize_environment_report(raw: Dict[str, Any], node: Node) -> Dict[str, Any]:
    allowed_statuses = {
        "ready",
        "needs_setup",
        "manual",
        "unavailable",
        "failed",
        "not_checked",
        "repairable",
        "blocked",
        "checking",
    }
    status = str(raw.get("status") or "failed")
    if status not in allowed_statuses:
        status = "failed"
    checks = []
    for index, item in enumerate(raw.get("checks") or []):
        if index >= 40 or not isinstance(item, dict):
            break
        check_status = str(item.get("status") or "unknown")
        if check_status not in {"pass", "fail", "warn", "missing", "unknown", "checking"}:
            check_status = "unknown"
        check_id = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            str(item.get("id") or f"check_{index}"),
        )[:64]
        checks.append(
            {
                "id": check_id,
                "label": str(item.get("label") or check_id)[:120],
                "status": check_status,
                "detail": str(item.get("detail") or "")[:2000],
                "auto_fixable": bool(item.get("auto_fixable", False)),
            }
        )
    missing = [
        str(item)[:80]
        for item in (raw.get("missing_system_packages") or [])[:40]
        if isinstance(item, str)
    ]
    manual = [
        str(item)[:2000]
        for item in (raw.get("manual_commands") or [])[:10]
        if isinstance(item, str)
    ]
    try:
        model_count = max(0, int(raw.get("model_count", 0)))
    except (TypeError, ValueError):
        model_count = 0
    raw_backend = raw.get("backend")
    if isinstance(raw_backend, dict):
        backend: Any = {
            "kind": str(raw_backend.get("kind") or "unknown")[:64],
            "verified": bool(raw_backend.get("verified", False)),
        }
    else:
        backend = {
            "kind": str(raw_backend or "unknown")[:64],
            # Old reports used a scalar only after successful verification.
            "verified": bool(raw_backend and raw_backend != "unknown"),
        }
    try:
        disk_free_gb: Optional[float] = round(float(raw.get("disk_free_gb")), 2)
    except (TypeError, ValueError):
        disk_free_gb = None
    return {
        "schema_version": 1,
        "node": node.name,
        "status": status,
        # Never manufacture a fresh timestamp for an incomplete/legacy file:
        # experiment admission treats a missing timestamp as stale and asks
        # the user to run a real preflight again.
        "checked_at": str(raw.get("checked_at")) if raw.get("checked_at") else None,
        "received_at": str(raw.get("received_at")) if raw.get("received_at") else None,
        "inventory_fingerprint": str(raw.get("inventory_fingerprint") or "")[:64],
        "platform": str(raw.get("platform") or node.platform)[:80],
        "board_model": str(raw.get("board_model") or "")[:240],
        "architecture": str(raw.get("architecture") or "")[:32],
        "os": str(raw.get("os") or "")[:240],
        "python": str(raw.get("python") or "")[:120],
        "disk_free_gb": disk_free_gb,
        "project_dir": node.project_dir,
        "venv_path": f"{node.project_dir}/.venv",
        "checks": checks,
        "missing_system_packages": missing,
        "manual_commands": manual,
        "backend": backend,
        "model_count": model_count,
    }


def write_environment_report(raw: Dict[str, Any]) -> Dict[str, Any]:
    node_name = str(raw.get("node") or "")
    nodes = {node.name: node for node in read_all_nodes()}
    if node_name not in nodes:
        raise ValueError("Environment report references an unknown node")
    received = {
        **raw,
        "received_at": utc_now(),
        "inventory_fingerprint": _node_environment_fingerprint(nodes[node_name]),
    }
    report = normalize_environment_report(received, nodes[node_name])
    ENVIRONMENT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    ENVIRONMENT_DIR.chmod(0o700)
    target = _environment_path(node_name)
    temporary = target.with_suffix(f".tmp.{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        target.chmod(0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return report


def read_environment_reports() -> List[Dict[str, Any]]:
    reports = []
    for node in read_all_nodes():
        try:
            raw = json.loads(_environment_path(node.name).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("environment report must be an object")
            if raw.get("inventory_fingerprint") != _node_environment_fingerprint(node):
                placeholder = _environment_placeholder(node)
                placeholder["status"] = "not_checked"
                placeholder["checks"] = [
                    {
                        "id": "inventory_identity",
                        "label": "Node inventory identity",
                        "status": "fail",
                        "detail": "Node address or project identity changed; run a fresh environment check",
                        "auto_fixable": True,
                    }
                ]
                reports.append(placeholder)
                continue
            reports.append(normalize_environment_report(raw, node))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            reports.append(_environment_placeholder(node))
    return reports


def validate_experiment_environment(
    nodes: Sequence[Node],
    live_status: Dict[str, Dict[str, Any]],
    model_ids: Sequence[str],
    execution_strategy: str,
) -> None:
    """Reject experiments whose most recent persisted preflight is unsafe or stale."""
    reports = {item["node"]: item for item in read_environment_reports()}
    now = datetime.now(timezone.utc)
    problems: List[str] = []
    for node in nodes:
        report = reports.get(node.name) or _environment_placeholder(node)
        checked_at = report.get("checked_at")
        received_at = report.get("received_at")
        age_hours: Optional[float] = None
        if received_at:
            try:
                parsed = datetime.fromisoformat(str(received_at).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                age_hours = (now - parsed.astimezone(timezone.utc)).total_seconds() / 3600
            except ValueError:
                pass
        if report.get("status") != "ready":
            problems.append(
                f"{node.name}: 환경 상태 {report.get('status') or 'not_checked'} "
                "(환경 점검 또는 자동 구성을 실행하세요)"
            )
        elif age_hours is None or age_hours < -0.08 or age_hours > 24:
            problems.append(f"{node.name}: 환경 점검 결과가 24시간을 초과했으므로 다시 점검하세요")
        else:
            backend = report.get("backend")
            if isinstance(backend, dict):
                backend_verified = bool(backend.get("verified")) and backend.get("kind") not in {
                    "",
                    "unknown",
                    None,
                }
            else:
                backend_verified = backend not in {"", "unknown", None}
            if not backend_verified:
                problems.append(f"{node.name}: LLM 백엔드 검증 결과가 없습니다")
            detected_platform = report.get("platform")
            if node.platform != "auto" and detected_platform != node.platform:
                problems.append(
                    f"{node.name}: 인벤토리 플랫폼 {node.platform}와 실제 보드 "
                    f"{detected_platform or 'unknown'}이 다릅니다"
                )
        if checked_at:
            try:
                worker_time = datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
                if worker_time.tzinfo is None:
                    worker_time = worker_time.replace(tzinfo=timezone.utc)
                if (worker_time.astimezone(timezone.utc) - now).total_seconds() > 300:
                    problems.append(f"{node.name}: 노드 시계가 head보다 5분 이상 빠릅니다 (NTP를 확인하세요)")
            except ValueError:
                pass

        live = live_status.get(node.name) or {}
        if live.get("api") is not True:
            problems.append(f"{node.name}: 워커 API가 오프라인입니다 (노드 시작 후 다시 시도하세요)")

        # RPC model-parallel loads the GGUF only on its head coordinator. All
        # replicated strategies require every selected node to have every model.
        requires_models = execution_strategy != "model_parallel_rpc" or node.role == "head"
        if requires_models and live.get("api") is True:
            available = set(live.get("model_ids") or [])
            missing_models = [model_id for model_id in model_ids if model_id not in available]
            if missing_models:
                problems.append(
                    f"{node.name}: 모델 없음 - " + ", ".join(missing_models[:4])
                    + (" 외" if len(missing_models) > 4 else "")
                )
    if problems:
        raise ValueError("실험 환경 준비가 필요합니다: " + " / ".join(problems))


def list_models() -> List[Dict[str, Any]]:
    root = PROJECT_ROOT / "models"
    models = []
    if root.exists():
        for path in sorted(root.rglob("*.gguf")):
            models.append(
                {
                    "id": path.relative_to(root).as_posix(),
                    "name": path.name,
                    "size_gb": round(path.stat().st_size / (1024**3), 2),
                }
            )
    return models


def probe_node(node: Node) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        **serialize_node(node),
        "ssh": False,
        "project": False,
        "api": False,
        "current": {},
        "metrics": {},
        "model_count": 0,
        "model_ids": [],
        "node_info": {},
        "profile": {},
        "capabilities": {},
        "error": "",
        "checked_at": utc_now(),
    }
    if not node.enabled:
        result["error"] = "disabled"
        return result
    try:
        health = request_json(f"{node.api_url}/cluster/health", timeout=4.0)
        reported = health.get("node") or {}
        result["api"] = (
            health.get("ok") is True
            and int(health.get("telemetry_version") or 0) >= 2
            and reported.get("name") == node.name
        )
        result["current"] = health.get("current") or {}
        result["metrics"] = health.get("metrics") or {}
        result["model_count"] = int(health.get("model_count") or 0)
        result["model_ids"] = health.get("model_ids") or []
        result["node_info"] = reported
        result["profile"] = health.get("profile") or {}
        result["capabilities"] = health.get("capabilities") or {}
        result["telemetry_version"] = health.get("telemetry_version")
        if not result["api"]:
            raise ValueError("worker API identity or telemetry schema mismatch")
        result["ssh"] = True
        result["project"] = True
    except Exception as exc:
        discovery = discover_node(node, timeout=8)
        result["ssh"] = discovery["ssh"]
        result["project"] = discovery["project"]
        result["discovery"] = discovery
        if not discovery["ssh"]:
            result["error"] = "SSH key authentication failed or the host is unreachable"
        elif not discovery["project"]:
            result["error"] = "SSH connected; project is not installed yet"
        else:
            result["error"] = f"Worker API is offline: {exc}"
    return result


def _private_scan_networks() -> List[Dict[str, str]]:
    try:
        process = subprocess.run(
            ["ip", "-j", "-4", "addr", "show", "up", "scope", "global"],
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        )
        interfaces = json.loads(process.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    allowed = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
    )
    found: List[Dict[str, str]] = []
    seen = set()
    for interface in interfaces:
        interface_name = str(interface.get("ifname", ""))
        if interface_name.startswith(("docker", "br-", "veth", "virbr", "tailscale")):
            continue
        for info in interface.get("addr_info", []):
            raw = info.get("local", "")
            try:
                address = ipaddress.ip_address(raw)
            except ValueError:
                continue
            if address.version != 4 or not any(address in network for network in allowed):
                continue
            prefix = max(int(info.get("prefixlen", 24)), 24)
            network = ipaddress.ip_network(f"{address}/{prefix}", strict=False)
            key = str(network)
            if key not in seen:
                seen.add(key)
                found.append({"interface": interface_name, "local_ip": str(address), "network": key})
    return found


def _port_open(host: str, port: int = 22) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.22):
            return True
    except OSError:
        return False


def _ssh_fingerprint(host: str, port: int = 22) -> str:
    try:
        scan = subprocess.run(
            ["ssh-keyscan", "-T", "2", "-p", str(port), host],
            text=True,
            capture_output=True,
            timeout=4,
        )
        key_line = next((line for line in scan.stdout.splitlines() if line and not line.startswith("#")), "")
        if not key_line:
            return ""
        fingerprint = subprocess.run(
            ["ssh-keygen", "-lf", "-"],
            input=key_line + "\n",
            text=True,
            capture_output=True,
            timeout=3,
        )
        return fingerprint.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


scan_lock = threading.Lock()
scan_cache: Dict[str, Any] = {"at": 0.0, "result": None}


def scan_lan_devices(force: bool = False) -> Dict[str, Any]:
    with scan_lock:
        if not force and scan_cache["result"] is not None and time.monotonic() - scan_cache["at"] < 15:
            return json.loads(json.dumps(scan_cache["result"]))
    networks = _private_scan_networks()
    candidates: List[str] = []
    local_ips = {item["local_ip"] for item in networks}
    for item in networks:
        network = ipaddress.ip_network(item["network"])
        candidates.extend(str(address) for address in network.hosts())
    candidates = sorted(set(candidates), key=lambda value: tuple(int(part) for part in value.split(".")))
    open_hosts: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(64, max(1, len(candidates)))) as executor:
        futures = {executor.submit(_port_open, host): host for host in candidates}
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                open_hosts.append(futures[future])
    existing = {node.host: node for node in read_all_nodes()}
    head_node = next((node for node in read_all_nodes() if node.role == "head"), None)
    devices = []
    for host in sorted(open_hosts, key=lambda value: tuple(int(part) for part in value.split("."))):
        known = existing.get(host) or (head_node if host in local_ips else None)
        devices.append(
            {
                "host": host,
                "ssh_port": 22,
                "fingerprint": _ssh_fingerprint(host),
                "known_node": known.name if known else "",
                "is_head": host in local_ips,
            }
        )
    result = {"networks": networks, "devices": devices, "scanned_at": utc_now()}
    with scan_lock:
        scan_cache["at"] = time.monotonic()
        scan_cache["result"] = result
    return json.loads(json.dumps(result))


def probe_candidate(node: Node) -> Dict[str, Any]:
    discovery = discover_node(node, timeout=20)
    configured = node.platform
    detected = discovery.get("platform_kind", "unknown")
    warnings: List[str] = []
    if configured != "auto" and discovery["ssh"] and configured != detected:
        warnings.append(f"configured platform {configured} differs from detected {detected}")
    if discovery["ssh"] and detected == "raspberry-pi" and discovery.get("architecture") not in {"aarch64", "arm64"}:
        warnings.append("Raspberry Pi requires a 64-bit OS")
    if discovery["ssh"] and not discovery.get("sudo_nopasswd") and discovery.get("missing_packages"):
        warnings.append("system dependencies require one manual sudo command")
    compatible = (
        discovery["ssh"]
        and detected in {"jetson", "raspberry-pi"}
        and discovery.get("architecture") in {"aarch64", "arm64"}
        and (configured == "auto" or configured == detected)
    )
    if discovery["ssh"] and detected not in {"jetson", "raspberry-pi"}:
        warnings.append("only NVIDIA Jetson and Raspberry Pi are supported")
    if discovery["ssh"] and discovery.get("architecture") not in {"aarch64", "arm64"}:
        warnings.append("a 64-bit ARM operating system is required")
    return {
        "ok": compatible,
        "ssh_ok": discovery["ssh"],
        "stage": "ready_to_register" if compatible else "incompatible" if discovery["ssh"] else "pairing_required",
        "node": serialize_node(node),
        "fingerprint": _ssh_fingerprint(node.host, node.ssh_port),
        "discovery": discovery,
        "warnings": warnings,
    }


class StatusMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="cluster-status", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def refresh_now(self) -> None:
        try:
            nodes = read_all_nodes()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(nodes))) as executor:
                snapshot = list(executor.map(probe_node, nodes))
            with self._lock:
                changed = snapshot != self._snapshot
                self._snapshot = snapshot
            events.publish("cluster_status", nodes=snapshot, changed=changed)
        except Exception as exc:
            events.publish("monitor_error", message=str(exc))

    def _run(self) -> None:
        while not self._stop.is_set():
            self.refresh_now()
            self._stop.wait(5.0)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self._snapshot))


status_monitor = StatusMonitor()


class ActionManager:
    ALLOWED = {
        "doctor",
        "setup",
        "prepare",
        "prepare-rpc",
        "sync-code",
        "sync-models",
        "start",
        "stop",
        "restart",
        "select-model",
        "environment-check",
        "environment-install",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._actions: Dict[str, Dict[str, Any]] = {}

    def start(self, payload: ActionPayload) -> Dict[str, Any]:
        if payload.action not in self.ALLOWED:
            raise ValueError(f"Unsupported action: {payload.action}")
        enabled = load_nodes(INVENTORY_PATH)
        selected = select_nodes(enabled, payload.node_names)
        if not selected:
            raise ValueError("Select at least one enabled node")
        selected_names = {node.name for node in selected}
        experiment = experiments.active() if "experiments" in globals() else None
        if experiment and experiment.get("status") in {"queued", "running"}:
            overlap = selected_names.intersection(experiment.get("nodes") or [])
            if overlap:
                raise ValueError("Nodes are busy with an experiment: " + ", ".join(sorted(overlap)))
        action_id = datetime.now().strftime("%H%M%S") + "_" + uuid.uuid4().hex[:6]
        record = {
            "id": action_id,
            "action": payload.action,
            "nodes": [node.name for node in selected],
            "status": "queued",
            "started_at": utc_now(),
            "finished_at": None,
            "exit_code": None,
            "log": [],
        }
        if payload.action in {"environment-check", "environment-install"}:
            record["inventory_fingerprints"] = {
                node.name: _node_environment_fingerprint(node) for node in selected
            }
        with self._lock:
            for action in self._actions.values():
                if action.get("status") in {"queued", "running"} and selected_names.intersection(action.get("nodes") or []):
                    raise ValueError("A selected node already has a running control action")
            self._actions[action_id] = record
        if payload.action in {"environment-check", "environment-install"}:
            checking_reports = []
            for node in selected:
                pending = _environment_placeholder(node)
                pending.update(
                    {
                        "status": "checking",
                        "checked_at": None,
                        "checks": [
                            {
                                "id": "environment_operation",
                                "label": "Environment operation",
                                "status": "checking",
                                "detail": f"{payload.action} is running",
                                "auto_fixable": False,
                            }
                        ],
                    }
                )
                checking_reports.append(write_environment_report(pending))
            events.publish("environment_changed", environment=read_environment_reports(), reports=checking_reports)
        thread = threading.Thread(
            target=self._run,
            args=(action_id, payload),
            name=f"cluster-action-{action_id}",
            daemon=True,
        )
        thread.start()
        return dict(record)

    def _run(self, action_id: str, payload: ActionPayload) -> None:
        environment_reported_nodes: set[str] = set()
        with self._lock:
            expected_environment_fingerprints = dict(
                self._actions[action_id].get("inventory_fingerprints") or {}
            )

        def persist_missing_environment_reports(detail: str) -> None:
            if payload.action not in {"environment-check", "environment-install"}:
                return
            inventory = {node.name: node for node in read_all_nodes()}
            for node_name in payload.node_names:
                if node_name in environment_reported_nodes or node_name not in inventory:
                    continue
                raw = _environment_placeholder(inventory[node_name])
                raw.update(
                    {
                        "status": "failed",
                        "checked_at": utc_now(),
                        "checks": [
                            {
                                "id": "environment_operation",
                                "label": "Environment operation",
                                "status": "fail",
                                "detail": detail[-2000:] or "Environment process exited without a report",
                                "auto_fixable": True,
                            }
                        ],
                    }
                )
                report = write_environment_report(raw)
                environment_reported_nodes.add(node_name)
                events.publish(
                    "environment_changed",
                    environment=read_environment_reports(),
                    report=report,
                )

        command = [
            sys.executable,
            "-m",
            "cluster.clusterctl",
            "--inventory",
            str(INVENTORY_PATH),
        ]
        for node_name in payload.node_names:
            command.extend(["--node", node_name])
        command.append(payload.action)
        if payload.action in {"sync-models", "prepare"}:
            for model in payload.options.get("models", []):
                command.extend(["--model", str(model)])
        elif payload.action == "prepare-rpc":
            pass
        elif payload.action == "select-model":
            command.extend(
                [
                    "--model-id",
                    str(payload.options.get("model_id", "")),
                    "--n-ctx",
                    str(payload.options.get("n_ctx", 1024)),
                    "--n-gpu-layers",
                    str(payload.options.get("n_gpu_layers", 30)),
                ]
            )
        elif payload.action == "environment-install":
            command.append("--confirmed")

        with self._lock:
            self._actions[action_id]["status"] = "running"
        events.publish("action_started", action=self.get(action_id))
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                clean = line.rstrip()
                if clean.startswith(ENVIRONMENT_MARKER):
                    try:
                        raw_report = json.loads(clean[len(ENVIRONMENT_MARKER) :])
                        if isinstance(raw_report, dict):
                            report_node = str(raw_report.get("node") or "")
                            current_nodes = {node.name: node for node in read_all_nodes()}
                            expected_fingerprint = expected_environment_fingerprints.get(report_node)
                            current_node = current_nodes.get(report_node)
                            if (
                                not expected_fingerprint
                                or current_node is None
                                or _node_environment_fingerprint(current_node) != expected_fingerprint
                            ):
                                raise ValueError(
                                    f"inventory identity changed while checking {report_node or 'unknown node'}"
                                )
                            report = write_environment_report(raw_report)
                            environment_reported_nodes.add(report["node"])
                            events.publish(
                                "environment_changed",
                                environment=read_environment_reports(),
                                report=report,
                            )
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        clean = f"[environment-report-error] {exc}"
                with self._lock:
                    log = self._actions[action_id]["log"]
                    log.append(clean)
                    if len(log) > 500:
                        del log[:-500]
                events.publish("action_log", action_id=action_id, line=clean)
            exit_code = process.wait()
            persist_missing_environment_reports(
                f"Environment process exited with code {exit_code} without a structured report"
            )
            with self._lock:
                record = self._actions[action_id]
                record["exit_code"] = exit_code
                record["status"] = "completed" if exit_code == 0 else "failed"
                record["finished_at"] = utc_now()
        except Exception as exc:
            persist_missing_environment_reports(str(exc))
            with self._lock:
                record = self._actions[action_id]
                record["status"] = "failed"
                record["finished_at"] = utc_now()
                record["log"].append(str(exc))
        events.publish("action_finished", action=self.get(action_id))
        status_monitor.refresh_now()

    def get(self, action_id: str) -> Dict[str, Any]:
        with self._lock:
            if action_id not in self._actions:
                raise KeyError(action_id)
            return json.loads(json.dumps(self._actions[action_id]))

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            values = list(self._actions.values())
        return json.loads(json.dumps(values[-20:][::-1]))

    def busy_nodes(self) -> List[str]:
        with self._lock:
            return sorted(
                {
                    node
                    for action in self._actions.values()
                    if action.get("status") in {"queued", "running"}
                    for node in action.get("nodes", [])
                }
            )


actions = ActionManager()


def _suite_model_records(
    model_ids: Sequence[str],
    summaries: Sequence[Dict[str, Any]],
    errors: Sequence[Dict[str, Any]],
    attempted_models: int,
    suite_status: str,
    cleanup_statuses: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    summaries_by_index = {
        int(summary.get("model_index", 0)): summary
        for summary in summaries
        if str(summary.get("model_index", "")).isdigit()
    }
    terminal = suite_status in {"completed", "partial", "failed", "cancelled"}
    records = []
    for index, model_id in enumerate(model_ids, start=1):
        summary = summaries_by_index.get(index)
        model_errors = [
            error
            for error in errors
            if error.get("model_index") == index or error.get("model_id") == model_id
        ]
        cleanup_errors = [error for error in model_errors if error.get("stage") == "unload"]
        attempted = index <= attempted_models or summary is not None
        if summary is not None:
            status = summary.get("status", "failed")
        elif attempted:
            status = "failed" if terminal else "running"
        else:
            status = "unrun"
        cleanup_status = (cleanup_statuses or {}).get(index)
        if cleanup_errors:
            cleanup_status = "failed"
        elif not cleanup_status:
            cleanup_status = (
                "completed"
                if summary is not None
                else "pending"
                if attempted and not terminal
                else "unrun"
            )
        records.append(
            {
                "model_id": model_id,
                "model_index": index,
                "attempted": attempted,
                "status": status,
                "run_id": summary.get("run_id") if summary else None,
                "cleanup_status": cleanup_status,
                "errors": model_errors,
            }
        )
    return records


def _suite_document(
    *,
    suite_id: str,
    experiment_id: str,
    name: str,
    status: str,
    model_ids: Sequence[str],
    attempted_models: int,
    completed_models: int,
    total_work_units: int,
    completed_work_units: int,
    continue_on_model_error: bool,
    model_cooldown_s: float,
    started_at: str,
    summaries: Sequence[Dict[str, Any]],
    errors: Sequence[Dict[str, Any]],
    finished_at: Optional[str] = None,
    cleanup_statuses: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    document = {
        "schema_version": 1,
        "artifact_type": "experiment_suite",
        "suite_id": suite_id,
        "experiment_id": experiment_id,
        "name": name,
        "status": status,
        "model_ids": list(model_ids),
        "model_count": len(model_ids),
        "attempted_models": attempted_models,
        "completed_models": completed_models,
        "total_work_units": total_work_units,
        "completed_work_units": completed_work_units,
        "continue_on_model_error": continue_on_model_error,
        "model_cooldown_s": model_cooldown_s,
        "started_at": started_at,
        "finished_at": finished_at,
        "updated_at": utc_now(),
        "summaries": list(summaries),
        "errors": list(errors),
    }
    document["models"] = _suite_model_records(
        model_ids, summaries, errors, attempted_models, status, cleanup_statuses
    )
    return document


def write_suite_summary(summary: Dict[str, Any]) -> Path:
    suite_id = str(summary.get("suite_id") or "")
    if not suite_id or not suite_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("suite_id contains unsupported characters")
    suites_dir = RESULTS_DIR / "_suites"
    suites_dir.mkdir(parents=True, exist_ok=True)
    path = suites_dir / f"{suite_id}.json"
    temporary = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def reconcile_interrupted_suites() -> int:
    """Finalize suite artifacts left nonterminal by a previous dashboard process."""
    reconciled = 0
    suites_dir = RESULTS_DIR / "_suites"
    if not suites_dir.exists():
        return reconciled
    for path in suites_dir.glob("*.json"):
        try:
            suite = json.loads(path.read_text(encoding="utf-8"))
            suite_id = str(suite.get("suite_id") or "")
            if (
                suite.get("artifact_type") != "experiment_suite"
                or suite.get("status") not in {"queued", "running", "cancelling"}
                or path.stem != suite_id
                or not suite_id.replace("-", "").replace("_", "").isalnum()
            ):
                continue
            error = {
                "stage": "dashboard_restart",
                "error": "Suite was interrupted by a dashboard restart",
            }
            errors = list(suite.get("errors") or [])
            errors.append(error)
            suite.update(
                {
                    "status": "failed",
                    "interrupted": True,
                    "interrupted_from_status": suite.get("status"),
                    "finished_at": utc_now(),
                    "errors": errors,
                }
            )
            suite["updated_at"] = suite["finished_at"]
            suite["models"] = _suite_model_records(
                suite.get("model_ids") or [],
                suite.get("summaries") or [],
                errors,
                int(suite.get("attempted_models") or 0),
                "failed",
                {
                    int(model["model_index"]): str(model.get("cleanup_status") or "pending")
                    for model in suite.get("models") or []
                    if isinstance(model, dict)
                    and str(model.get("model_index", "")).isdigit()
                },
            )
            write_suite_summary(suite)
            reconciled += 1
        except (OSError, TypeError, ValueError):
            continue
    return reconciled


def read_suite_summaries(limit: int = 100) -> List[Dict[str, Any]]:
    suites = []
    suites_dir = RESULTS_DIR / "_suites"
    if suites_dir.exists():
        paths = sorted(suites_dir.glob("*.json"), reverse=True)
        for path in paths if limit <= 0 else paths[:limit]:
            try:
                suite = json.loads(path.read_text(encoding="utf-8"))
                if suite.get("artifact_type") == "experiment_suite" and suite.get("suite_id"):
                    suites.append(suite)
            except (OSError, ValueError):
                continue
    return suites


def _with_suite_metadata(
    summary: Dict[str, Any], suites_by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    suite = suites_by_id.get(str(summary.get("suite_id") or ""))
    if not suite:
        return summary
    return {
        **summary,
        "suite_status": suite.get("status"),
        "suite_started_at": suite.get("started_at"),
        "suite_finished_at": suite.get("finished_at"),
        "suite_attempted_models": suite.get("attempted_models"),
        "suite_completed_models": suite.get("completed_models"),
        "suite_models": suite.get("models", []),
        "suite_errors": suite.get("errors", []),
    }


def read_run_summaries(limit: int = 100) -> List[Dict[str, Any]]:
    summaries = []
    suites_by_id = {
        str(suite["suite_id"]): suite for suite in read_suite_summaries(limit=0)
    }
    if RESULTS_DIR.exists():
        paths = sorted(RESULTS_DIR.glob("*/summary.json"), reverse=True)
        for path in paths[:limit]:
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                summaries.append(_with_suite_metadata(summary, suites_by_id))
            except (OSError, ValueError):
                continue
    return summaries


reconcile_interrupted_suites()


def _experiment_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "experiment"


experiment_catalog_lock = threading.RLock()


def save_experiment_definition(payload: ExperimentPayload) -> Dict[str, Any]:
    with experiment_catalog_lock:
        experiment_id = payload.experiment_id
        if not experiment_id:
            experiment_id = f"{_experiment_slug(payload.name)}-{uuid.uuid4().hex[:6]}"
        path = EXPERIMENTS_DIR / f"{experiment_id}.json"
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
        if existing:
            previous_strategy = (existing.get("default_config") or {}).get(
                "execution_strategy", "replicated_round_robin"
            )
            if previous_strategy != payload.execution_strategy:
                raise ValueError(
                    "한 실험 묶음에는 하나의 실행 방식만 사용할 수 있습니다. 새 실험 묶음을 만드세요."
                )
        now = utc_now()
        definition = {
            "experiment_id": experiment_id,
            "name": payload.name,
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "archived": False,
            "default_config": {
                key: value
                for key, value in payload.model_dump().items()
                if key not in {"experiment_id", "name"}
            },
        }
        temporary = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
        temporary.write_text(json.dumps(definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return definition


def read_experiment_groups() -> List[Dict[str, Any]]:
    definitions: Dict[str, Dict[str, Any]] = {}
    with experiment_catalog_lock:
        for path in sorted(EXPERIMENTS_DIR.glob("*.json")):
            try:
                definition = json.loads(path.read_text(encoding="utf-8"))
                definitions[definition["experiment_id"]] = definition
            except (OSError, ValueError, KeyError):
                continue
    for run in read_run_summaries(limit=500):
        experiment_id = run.get("experiment_id")
        if not experiment_id:
            experiment_id = f"legacy-{_experiment_slug(str(run.get('name') or 'unnamed'))}"
        group = definitions.setdefault(
            experiment_id,
            {
                "experiment_id": experiment_id,
                "name": run.get("name") or experiment_id,
                "created_at": run.get("started_at") or run.get("finished_at"),
                "updated_at": run.get("finished_at"),
                "archived": False,
                "default_config": {},
                "legacy": not bool(run.get("experiment_id")),
            },
        )
        group.setdefault("runs", []).append(run)
        if str(run.get("finished_at", "")) > str(group.get("updated_at", "")):
            group["updated_at"] = run.get("finished_at")
    groups = []
    for definition in definitions.values():
        runs = sorted(definition.get("runs", []), key=lambda item: item.get("finished_at", ""), reverse=True)
        definition = {**definition, "runs": runs, "run_count": len(runs)}
        definition["latest_run"] = runs[0] if runs else None
        groups.append(definition)
    return sorted(groups, key=lambda item: item.get("updated_at") or "", reverse=True)


class ExperimentManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Optional[Dict[str, Any]] = None
        self._cancel: Optional[threading.Event] = None

    def start(self, payload: ExperimentPayload) -> Dict[str, Any]:
        payload_data = payload.model_dump()
        config = ExperimentConfig.from_dict(payload_data)
        config.validate()
        selected = select_nodes(load_nodes(INVENTORY_PATH), config.node_names)
        if len(selected) != len(config.node_names):
            raise ValueError("Some selected nodes are unavailable")
        validate_strategy(selected, config)
        available_models = {item["id"] for item in list_models()}
        missing_models = [model_id for model_id in payload.model_ids if model_id not in available_models]
        if missing_models:
            raise ValueError("Unknown model_ids: " + ", ".join(missing_models))
        suite_id = "suite_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        per_model_total = strategy_work_units(config, len(selected))
        suite_started_at = utc_now()
        with self._lock:
            if self._active and self._active.get("status") in {"queued", "running"}:
                raise ValueError("Another experiment is already running")
            active = {
                "id": suite_id,
                "suite_id": suite_id,
                "experiment_id": config.experiment_id,
                "name": config.name,
                "status": "queued",
                "phase": "queued",
                "completed": 0,
                "total": per_model_total * len(payload.model_ids),
                "model_completed": 0,
                "model_total": per_model_total,
                "strategy": config.execution_strategy,
                "started_at": suite_started_at,
                "nodes": config.node_names,
                "model_ids": list(payload.model_ids),
                "current_model": payload.model_ids[0],
                "model_index": 0,
                "model_count": len(payload.model_ids),
                "completed_models": 0,
                "summaries": [],
                "errors": [],
                "latest": None,
                "error": "",
            }
            write_suite_summary(
                _suite_document(
                    suite_id=suite_id,
                    experiment_id=config.experiment_id,
                    name=config.name,
                    status="queued",
                    model_ids=payload.model_ids,
                    attempted_models=0,
                    completed_models=0,
                    total_work_units=per_model_total * len(payload.model_ids),
                    completed_work_units=0,
                    continue_on_model_error=payload.continue_on_model_error,
                    model_cooldown_s=payload.model_cooldown_s,
                    started_at=suite_started_at,
                    summaries=[],
                    errors=[],
                )
            )
            self._active = active
            self._cancel = threading.Event()
            cancel_event = self._cancel
            record = dict(self._active)
        thread = threading.Thread(
            target=self._run,
            args=(config, list(payload.model_ids), payload.continue_on_model_error, payload.model_cooldown_s, cancel_event),
            name="cluster-experiment-suite",
            daemon=True,
        )
        thread.start()
        return record

    def _handle_event(self, event: Dict[str, Any], completed_offset: int = 0) -> None:
        with self._lock:
            if self._active is None:
                return
            if event.get("run_id"):
                self._active["current_run_id"] = event["run_id"]
            event_type = event.get("type")
            if event_type == "run_started":
                self._active["status"] = "running"
            elif event_type == "phase":
                self._active["phase"] = event.get("phase")
            elif event_type == "request_completed":
                model_completed = int(event.get("completed", 0))
                self._active["model_completed"] = model_completed
                self._active["completed"] = completed_offset + model_completed
                self._active["latest"] = event.get("result")
            elif event_type == "run_finished":
                # run_finished is model-scoped. The suite remains running until
                # every selected model (or the configured stop condition) ends.
                self._active["current_summary"] = event.get("summary")
            elif event_type == "run_failed":
                self._active["error"] = event.get("error", "")
        events.publish("experiment_event", event=event, active=self.active())

    def _publish_suite_event(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, "at": utc_now(), **payload}
        events.publish("experiment_event", event=event, active=self.active())

    @staticmethod
    def _unload_models(node_names: Sequence[str]) -> List[str]:
        """Best-effort concurrent unload with explicit errors for suite isolation."""
        nodes = select_nodes(load_nodes(INVENTORY_PATH), node_names)
        errors: List[str] = []

        def unload(node: Node) -> None:
            request_json(f"{node.api_url}/api/unload-model", method="POST", payload={}, timeout=60.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(nodes))) as executor:
            futures = {executor.submit(unload, node): node for node in nodes}
            for future in concurrent.futures.as_completed(futures):
                node = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors.append(f"{node.name}: {exc}")
        missing = sorted(set(node_names) - {node.name for node in nodes})
        errors.extend(f"{name}: unavailable" for name in missing)
        return errors

    def _run(
        self,
        base_config: ExperimentConfig,
        model_ids: List[str],
        continue_on_model_error: bool,
        model_cooldown_s: float,
        cancel_event: threading.Event,
    ) -> None:
        suite_id = self._active["suite_id"] if self._active else ""
        model_count = len(model_ids)
        per_model_total = strategy_work_units(base_config, len(base_config.node_names))
        completed_offset = 0
        summaries: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        cleanup_statuses: Dict[int, str] = {}
        completed_models = 0
        attempted_models = 0
        suite_started_at = (
            str(self._active.get("started_at")) if self._active else utc_now()
        )

        def persist_suite(status: str, finished_at: Optional[str] = None) -> Dict[str, Any]:
            suite_summary = _suite_document(
                suite_id=suite_id,
                experiment_id=base_config.experiment_id,
                name=base_config.name,
                status=status,
                model_ids=model_ids,
                attempted_models=attempted_models,
                completed_models=completed_models,
                total_work_units=per_model_total * model_count,
                completed_work_units=completed_offset,
                continue_on_model_error=continue_on_model_error,
                model_cooldown_s=model_cooldown_s,
                started_at=suite_started_at,
                finished_at=finished_at,
                summaries=summaries,
                errors=errors,
                cleanup_statuses=cleanup_statuses,
            )
            write_suite_summary(suite_summary)
            return suite_summary

        try:
            with self._lock:
                if self._active:
                    self._active.update({"status": "running", "phase": "suite", "started_at": suite_started_at})
            persist_suite("running")
            self._publish_suite_event(
                "suite_started",
                suite_id=suite_id,
                experiment_id=base_config.experiment_id,
                model_ids=model_ids,
                model_count=model_count,
                total_work_units=per_model_total * model_count,
            )

            for index, model_id in enumerate(model_ids, start=1):
                if cancel_event.is_set():
                    break
                attempted_models += 1
                config = ExperimentConfig.from_dict(
                    {
                        **asdict(base_config),
                        "model_id": model_id,
                        "suite_id": suite_id,
                        "model_index": index,
                        "model_count": model_count,
                    }
                )
                config.validate()
                local_completed = 0
                captured_summary: Optional[Dict[str, Any]] = None
                model_failed = False

                with self._lock:
                    if self._active:
                        self._active.update(
                            {
                                "phase": "model_starting",
                                "current_model": model_id,
                                "model_index": index,
                                "model_completed": 0,
                                "current_run_id": "",
                                "current_summary": None,
                                "error": "",
                            }
                        )
                self._publish_suite_event(
                    "model_started",
                    suite_id=suite_id,
                    experiment_id=config.experiment_id,
                    model_id=model_id,
                    model_index=index,
                    model_count=model_count,
                    total_work_units=per_model_total,
                )
                persist_suite("running")

                def handle_model_event(event: Dict[str, Any]) -> None:
                    nonlocal local_completed, captured_summary
                    if event.get("type") == "request_completed":
                        local_completed = max(local_completed, int(event.get("completed", 0)))
                    if event.get("type") in {"run_finished", "run_failed"} and event.get("summary"):
                        captured_summary = event["summary"]
                    self._handle_event(event, completed_offset)

                try:
                    summary = run_experiment(
                        config,
                        inventory_path=INVENTORY_PATH,
                        results_root=RESULTS_DIR,
                        progress=handle_model_event,
                        cancel_event=cancel_event,
                    )
                    captured_summary = summary
                    summaries.append(summary)
                    if summary.get("status") == "completed":
                        completed_models += 1
                    self._publish_suite_event(
                        "model_finished",
                        suite_id=suite_id,
                        experiment_id=config.experiment_id,
                        model_id=model_id,
                        model_index=index,
                        model_count=model_count,
                        status=summary.get("status", "completed"),
                        summary=summary,
                    )
                except Exception as exc:
                    model_failed = True
                    failure_summary = captured_summary or {
                        "suite_id": suite_id,
                        "experiment_id": config.experiment_id,
                        "model_id": model_id,
                        "model_index": index,
                        "model_count": model_count,
                        "status": "failed",
                        "error": str(exc),
                    }
                    summaries.append(failure_summary)
                    if cancel_event.is_set():
                        self._publish_suite_event(
                            "model_finished",
                            suite_id=suite_id,
                            experiment_id=config.experiment_id,
                            model_id=model_id,
                            model_index=index,
                            model_count=model_count,
                            status="cancelled",
                            summary=failure_summary,
                        )
                    else:
                        error = {
                            "model_id": model_id,
                            "model_index": index,
                            "stage": "benchmark",
                            "error": str(exc),
                        }
                        errors.append(error)
                        self._publish_suite_event(
                            "model_failed",
                            suite_id=suite_id,
                            experiment_id=config.experiment_id,
                            model_id=model_id,
                            model_index=index,
                            model_count=model_count,
                            error=str(exc),
                            summary=failure_summary,
                        )
                finally:
                    completed_offset += local_completed
                    with self._lock:
                        if self._active:
                            self._active.update(
                                {
                                    "completed": completed_offset,
                                    "completed_models": completed_models,
                                    "summaries": list(summaries),
                                    "errors": list(errors),
                                }
                            )

                cleanup_statuses[index] = "pending"
                persist_suite("cancelling" if cancel_event.is_set() else "running")

                # Every attempted model is unloaded, including the final
                # successful model, so a finished suite leaves no allocation.
                cleanup_failed = False
                with self._lock:
                    if self._active:
                        self._active["phase"] = "model_cleanup"
                try:
                    cleanup_errors = self._unload_models(config.node_names)
                except Exception as exc:
                    cleanup_errors = [str(exc)]
                if cleanup_errors:
                    cleanup_failed = True
                    cleanup_statuses[index] = "failed"
                    cleanup_error = {
                        "model_id": model_id,
                        "model_index": index,
                        "stage": "unload",
                        "error": "; ".join(cleanup_errors),
                    }
                    errors.append(cleanup_error)
                    self._publish_suite_event(
                        "model_failed",
                        suite_id=suite_id,
                        experiment_id=config.experiment_id,
                        model_id=model_id,
                        model_index=index,
                        model_count=model_count,
                        stage="unload",
                        error=cleanup_error["error"],
                    )
                    with self._lock:
                        if self._active:
                            self._active["errors"] = list(errors)
                else:
                    cleanup_statuses[index] = "completed"
                persist_suite("cancelling" if cancel_event.is_set() else "running")

                should_continue = (
                    index < model_count
                    and not cancel_event.is_set()
                    and not cleanup_failed
                    and (not model_failed or continue_on_model_error)
                )
                if not should_continue:
                    break
                if model_cooldown_s > 0:
                    with self._lock:
                        if self._active:
                            self._active["phase"] = "model_cooldown"
                    self._publish_suite_event(
                        "model_cooldown",
                        suite_id=suite_id,
                        after_model_id=model_id,
                        seconds=model_cooldown_s,
                    )
                    if cancel_event.wait(model_cooldown_s):
                        break

            if cancel_event.is_set():
                final_status = "cancelled"
            elif errors:
                final_status = "partial" if completed_models else "failed"
            elif attempted_models == model_count:
                final_status = "completed"
            else:
                final_status = "failed"
            suite_summary = persist_suite(final_status, utc_now())
            with self._lock:
                if self._active:
                    self._active.update(
                        {
                            "status": final_status,
                            "phase": "finished",
                            "summary": suite_summary,
                            "summaries": summaries,
                            "errors": errors,
                            "completed_models": completed_models,
                            "finished_at": suite_summary["finished_at"],
                            "error": errors[-1]["error"] if errors else "",
                        }
                    )
            self._publish_suite_event("suite_finished", **suite_summary)
        except Exception as exc:
            error = {"stage": "suite", "error": str(exc)}
            errors.append(error)
            suite_summary = persist_suite("failed", utc_now())
            with self._lock:
                if self._active:
                    self._active.update(
                        {
                            "status": "failed",
                            "phase": "finished",
                            "error": str(exc),
                            "errors": errors,
                            "summary": suite_summary,
                            "summaries": summaries,
                            "finished_at": suite_summary["finished_at"],
                        }
                    )
            self._publish_suite_event("suite_finished", **suite_summary)
        finally:
            status_monitor.refresh_now()

    def cancel(self) -> Dict[str, Any]:
        with self._lock:
            if not self._active or self._active.get("status") not in {"queued", "running"}:
                raise ValueError("No running experiment")
            assert self._cancel is not None
            self._cancel.set()
            self._active["phase"] = "cancelling"
            return json.loads(json.dumps(self._active))

    def active(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self._active)) if self._active else None


experiments = ExperimentManager()


app = FastAPI(title="MediFlow LLM Cluster Lab", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR / "static")), name="cluster-static")
templates = Jinja2Templates(directory=str(DASHBOARD_DIR / "templates"))
status_monitor.start()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/dashboard/health")
async def dashboard_health() -> Dict[str, Any]:
    return {"ok": True, "service": "cluster-dashboard"}


@app.get("/api/bootstrap", dependencies=[Depends(verify_token)])
async def bootstrap() -> Dict[str, Any]:
    defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    public_key = ""
    public_key_path = Path.home() / ".ssh" / "id_ed25519_llm_cluster.pub"
    if public_key_path.exists():
        public_key = public_key_path.read_text(encoding="utf-8").strip()
    return {
        "nodes": [serialize_node(node) for node in read_all_nodes()],
        "status": status_monitor.snapshot(),
        "models": list_models(),
        "defaults": defaults,
        "active_experiment": experiments.active(),
        "runs": read_run_summaries(),
        "suites": read_suite_summaries(),
        "experiment_groups": read_experiment_groups(),
        "actions": actions.list(),
        "environment": read_environment_reports(),
        "settings": read_settings(),
        "experiment_strategies": experiment_strategy_catalog(),
        "onboarding": {
            "public_key": public_key,
        },
    }


@app.get("/api/events", dependencies=[Depends(verify_token)])
async def event_stream(request: Request) -> StreamingResponse:
    return StreamingResponse(
        events.stream(supplied_dashboard_token(request)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/settings", dependencies=[Depends(verify_token)])
async def get_settings() -> Dict[str, Any]:
    return {"settings": read_settings()}


@app.put("/api/settings", dependencies=[Depends(verify_token)])
async def update_settings(payload: ClusterSettingsPayload, request: Request) -> Dict[str, Any]:
    action: Optional[Dict[str, Any]] = None
    with settings_lock:
        previous = read_settings()
        supplied = payload.dashboard_token or supplied_dashboard_token(request)
        if previous["dashboard_token_auth"] and not dashboard_token_is_valid(supplied):
            raise HTTPException(
                status_code=401,
                detail="Dashboard access token is missing or invalid",
            )
        updated = dict(previous)
        if payload.worker_api_auth is not None:
            updated["worker_api_auth"] = payload.worker_api_auth
        if payload.dashboard_token_auth is not None:
            updated["dashboard_token_auth"] = payload.dashboard_token_auth
        if not previous["dashboard_token_auth"] and updated["dashboard_token_auth"]:
            if not dashboard_token_is_valid(supplied):
                raise HTTPException(
                    status_code=403,
                    detail="Enabling dashboard token auth requires the current dashboard token",
                )
        write_settings(updated)
        if previous["worker_api_auth"] != updated["worker_api_auth"]:
            try:
                enabled_names = [node.name for node in load_nodes(INVENTORY_PATH)]
                action = actions.start(
                    ActionPayload(action="restart", node_names=enabled_names, options={})
                )
            except ValueError as exc:
                write_settings(previous)
                raise HTTPException(status_code=409, detail=str(exc)) from exc
    events.publish("settings_changed", settings=updated, action=action)
    return {"ok": True, "settings": updated, "action": action}


@app.get("/api/status", dependencies=[Depends(verify_token)])
async def get_status() -> Dict[str, Any]:
    return {"nodes": status_monitor.snapshot(), "at": utc_now()}


@app.post("/api/status/refresh", dependencies=[Depends(verify_token)])
async def refresh_status() -> Dict[str, Any]:
    threading.Thread(target=status_monitor.refresh_now, daemon=True).start()
    return {"ok": True}


@app.post("/api/network/scan", dependencies=[Depends(verify_token)])
async def scan_network(force: bool = False) -> Dict[str, Any]:
    return await asyncio.to_thread(scan_lan_devices, force)


@app.post("/api/nodes/probe", dependencies=[Depends(verify_token)])
async def probe_unregistered_node(payload: NodePayload) -> Dict[str, Any]:
    if payload.role != "worker":
        raise HTTPException(status_code=400, detail="Only worker candidates can be probed")
    node = Node(**payload.model_dump())
    return await asyncio.to_thread(probe_candidate, node)


@app.post("/api/nodes", dependencies=[Depends(verify_token)])
async def upsert_node(payload: NodePayload) -> Dict[str, Any]:
    node = Node(**payload.model_dump())
    with inventory_lock:
        nodes = read_all_nodes()
        existing_index = next((i for i, item in enumerate(nodes) if item.name == node.name), None)
        if existing_index is not None and node.name in set(actions.busy_nodes()):
            raise HTTPException(status_code=409, detail="Node has a running control action")
        identity_changed = existing_index is None
        if existing_index is None:
            if node.role == "head":
                raise HTTPException(status_code=400, detail="A head node already exists")
            nodes.append(node)
        else:
            existing = nodes[existing_index]
            if existing.role == "head" and node.role != "head":
                raise HTTPException(status_code=400, detail="The head role cannot be changed")
            identity_changed = _node_environment_fingerprint(existing) != _node_environment_fingerprint(node)
            nodes[existing_index] = node
        try:
            write_all_nodes(nodes)
            if identity_changed:
                _invalidate_environment_report(node.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    events.publish("inventory_changed", nodes=[serialize_node(item) for item in nodes])
    threading.Thread(target=status_monitor.refresh_now, daemon=True).start()
    return {"ok": True, "node": serialize_node(node)}


@app.delete("/api/nodes/{node_name}", dependencies=[Depends(verify_token)])
async def delete_node(node_name: str) -> Dict[str, Any]:
    with inventory_lock:
        if node_name in set(actions.busy_nodes()):
            raise HTTPException(status_code=409, detail="Node has a running control action")
        nodes = read_all_nodes()
        target = next((node for node in nodes if node.name == node_name), None)
        if target is None:
            raise HTTPException(status_code=404, detail="Node not found")
        if target.role == "head":
            raise HTTPException(status_code=400, detail="The head node cannot be removed")
        nodes = [node for node in nodes if node.name != node_name]
        write_all_nodes(nodes)
        _invalidate_environment_report(node_name)
    events.publish("inventory_changed", nodes=[serialize_node(item) for item in nodes])
    return {"ok": True}


@app.post("/api/actions", dependencies=[Depends(verify_token)])
async def start_action(payload: ActionPayload) -> Dict[str, Any]:
    if payload.action in {"setup", "prepare", "prepare-rpc", "environment-install"} and payload.options.get("confirmed") is not True:
        raise HTTPException(status_code=400, detail="Worker setup requires explicit confirmation")
    try:
        record = actions.start(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "action": record}


@app.get("/api/actions", dependencies=[Depends(verify_token)])
async def list_actions() -> Dict[str, Any]:
    return {"actions": actions.list()}


@app.get("/api/environment", dependencies=[Depends(verify_token)])
async def get_environment() -> Dict[str, Any]:
    return {"environment": read_environment_reports()}


@app.post("/api/experiments", dependencies=[Depends(verify_token)])
async def start_experiment(payload: ExperimentPayload) -> Dict[str, Any]:
    try:
        current = experiments.active()
        if current and current.get("status") in {"queued", "running"}:
            raise ValueError("Another experiment is already running")
        busy = set(actions.busy_nodes()).intersection(payload.node_names)
        if busy:
            raise ValueError("Nodes have a running control action: " + ", ".join(sorted(busy)))
        status_by_name = {item.get("name"): item for item in status_monitor.snapshot()}
        inventory_by_name = {item.name: item for item in read_all_nodes()}
        selected_nodes = [inventory_by_name[name] for name in payload.node_names if name in inventory_by_name]
        strategy_config = ExperimentConfig.from_dict(payload.model_dump())
        validate_strategy(selected_nodes, strategy_config)
        validate_experiment_environment(
            selected_nodes,
            status_by_name,
            payload.model_ids,
            payload.execution_strategy,
        )
        if payload.execution_strategy == "model_parallel_rpc" and read_settings()["worker_api_auth"]:
            raise ValueError(
                "워커 API 보안 모드에서는 인증 없는 llama.cpp RPC 포트를 열지 않습니다. "
                "SSH 터널 모드가 추가되기 전에는 신뢰 LAN에서만 보안을 끄고 실행하세요."
            )
        pi_nodes = []
        readiness_platforms = {
            item.get("node"): item.get("platform") for item in read_environment_reports()
        }
        for name in payload.node_names:
            detected = (status_by_name.get(name, {}).get("profile") or {}).get("platform_kind")
            configured = inventory_by_name.get(name).platform if inventory_by_name.get(name) else "auto"
            if (
                detected == "raspberry-pi"
                or configured == "raspberry-pi"
                or readiness_platforms.get(name) == "raspberry-pi"
            ):
                pi_nodes.append(name)
        if pi_nodes and payload.n_gpu_layers != 0 and payload.execution_strategy != "model_parallel_rpc":
            raise ValueError(
                "Raspberry Pi nodes require n_gpu_layers=0: " + ", ".join(str(item) for item in pi_nodes)
            )
        catalog_ids = {item["id"] for item in list_models()}
        missing_models = [model_id for model_id in payload.model_ids if model_id not in catalog_ids]
        if missing_models:
            raise ValueError("Unknown model_ids: " + ", ".join(missing_models))
        definition = save_experiment_definition(payload)
        linked_payload = payload.model_copy(update={"experiment_id": definition["experiment_id"]})
        active = experiments.start(linked_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "experiment": active, "definition": definition}


@app.get("/api/experiments", dependencies=[Depends(verify_token)])
async def list_experiments() -> Dict[str, Any]:
    return {
        "active": experiments.active(),
        "runs": read_run_summaries(),
        "suites": read_suite_summaries(),
        "experiment_groups": read_experiment_groups(),
    }


@app.get("/api/experiment-groups", dependencies=[Depends(verify_token)])
async def list_experiment_groups() -> Dict[str, Any]:
    return {"experiment_groups": read_experiment_groups()}


@app.post("/api/experiments/cancel", dependencies=[Depends(verify_token)])
async def cancel_experiment() -> Dict[str, Any]:
    try:
        active = experiments.cancel()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "experiment": active}


@app.get("/api/runs/{run_id}", dependencies=[Depends(verify_token)])
async def get_run(run_id: str) -> Dict[str, Any]:
    if not run_id.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid run id")
    summary_path = RESULTS_DIR / run_id / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    suites_by_id = {
        str(suite["suite_id"]): suite for suite in read_suite_summaries(limit=0)
    }
    return _with_suite_metadata(summary, suites_by_id)


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})
