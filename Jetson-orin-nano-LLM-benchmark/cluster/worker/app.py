#!/usr/bin/env python3
"""Cluster worker API layered on top of the existing local LLM web app."""

from __future__ import annotations

import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psutil
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field

from web.app import ChatStreamRequest, app, as_sse, manager

try:
    from jtop import jtop
except ImportError:  # pragma: no cover
    jtop = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODE_NAME = os.getenv("CLUSTER_NODE_NAME", socket.gethostname())
NODE_ROLE = os.getenv("CLUSTER_NODE_ROLE", "worker")
CLUSTER_API_TOKEN = os.getenv("CLUSTER_API_TOKEN", "").strip()
WORKER_API_AUTH_ENABLED = os.getenv("CLUSTER_WORKER_AUTH", "false").strip().lower() in {
    "1", "true", "yes", "on", "enabled"
}


@app.middleware("http")
async def require_cluster_api_token(request: Request, call_next: Any) -> Any:
    if not WORKER_API_AUTH_ENABLED:
        return await call_next(request)
    supplied = request.headers.get("X-Cluster-Worker-Token", "")
    if not CLUSTER_API_TOKEN or not supplied or not secrets.compare_digest(supplied, CLUSTER_API_TOKEN):
        return JSONResponse(status_code=401, content={"detail": "Worker API authentication failed"})
    return await call_next(request)


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: Any, digits: int = 2) -> Optional[float]:
    number = _safe_float(value)
    return round(number, digits) if number is not None else None


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return ""


def detect_platform_kind() -> str:
    override = os.getenv("CLUSTER_PLATFORM", "").strip().lower()
    if override in {"jetson", "raspberry-pi", "generic-linux"}:
        return override
    board = _read_text("/proc/device-tree/model").lower()
    if "raspberry pi" in board:
        return "raspberry-pi"
    if Path("/etc/nv_tegra_release").exists() or shutil.which("nvpmodel"):
        return "jetson"
    return "generic-linux"


def _os_release() -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in _read_text("/etc/os-release").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


PLATFORM_KIND = detect_platform_kind()


def runtime_backend() -> Dict[str, Any]:
    try:
        import llama_cpp as llama_package
        from llama_cpp import llama_cpp

        info = llama_cpp.llama_print_system_info().decode("utf-8", errors="replace")
        supports_gpu = bool(llama_cpp.llama_supports_gpu_offload())
        kind = "cuda" if supports_gpu else "cpu"
        if not supports_gpu:
            try:
                for candidate in Path(llama_package.__file__).resolve().parent.rglob("*.so"):
                    linked = subprocess.check_output(
                        ["ldd", str(candidate)], text=True, stderr=subprocess.DEVNULL, timeout=3
                    )
                    if "openblas" in linked.lower():
                        kind = "openblas"
                        break
            except (OSError, subprocess.SubprocessError):
                pass
        return {
            "verified": True,
            "kind": kind,
            "gpu_offload": supports_gpu,
            "llama_cpp_python": getattr(llama_package, "__version__", "unknown"),
            "system_info": " ".join(info.split())[:1000],
        }
    except Exception as exc:  # pragma: no cover - native runtime-specific
        return {
            "verified": False,
            "kind": "unknown",
            "gpu_offload": False,
            "error": str(exc),
        }


RUNTIME_BACKEND = runtime_backend()


def system_profile() -> Dict[str, Any]:
    os_release = _os_release()
    cpu_model = ""
    for line in _read_text("/proc/cpuinfo").splitlines():
        if line.lower().startswith(("model name", "hardware")) and ":" in line:
            cpu_model = line.split(":", 1)[1].strip()
            if cpu_model:
                break
    board_model = _read_text("/proc/device-tree/model") or platform.machine()
    memory = psutil.virtual_memory()
    l4t = _read_text("/etc/nv_tegra_release").splitlines()
    cuda_version = ""
    nvcc = Path("/usr/local/cuda/bin/nvcc")
    if nvcc.exists():
        try:
            cuda_version = subprocess.check_output(
                [str(nvcc), "--version"], text=True, stderr=subprocess.DEVNULL, timeout=3
            ).splitlines()[-1].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "platform_kind": PLATFORM_KIND,
        "board_model": board_model,
        "os": os_release.get("PRETTY_NAME") or platform.platform(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": cpu_model or platform.processor() or platform.machine(),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "inference_threads": int(
            os.getenv("LLM_N_THREADS", str(min(6, psutil.cpu_count(logical=True) or 1)))
        ),
        "memory_total_mb": round(memory.total / (1024 * 1024), 2),
        "accelerator": RUNTIME_BACKEND["kind"],
        "runtime_backend": RUNTIME_BACKEND,
        "l4t": l4t[0] if l4t else "",
        "cuda": cuda_version,
        "git_commit": _git_commit(),
    }


def _temperature_snapshot() -> Dict[str, Optional[float]]:
    values: Dict[str, Optional[float]] = {}
    if PLATFORM_KIND == "raspberry-pi" and shutil.which("vcgencmd"):
        try:
            output = subprocess.check_output(
                ["vcgencmd", "measure_temp"], text=True, stderr=subprocess.DEVNULL, timeout=2
            )
            match = re.search(r"(-?\d+(?:\.\d+)?)", output)
            if match:
                values["soc"] = round(float(match.group(1)), 2)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    try:
        for group, entries in psutil.sensors_temperatures().items():
            for index, entry in enumerate(entries):
                label = entry.label or f"sensor{index}"
                values[f"{group}:{label}"] = _round_optional(entry.current)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    if not values:
        raw = _read_text("/sys/class/thermal/thermal_zone0/temp")
        temperature = _safe_float(raw)
        if temperature is not None:
            values["system"] = round(temperature / 1000 if temperature > 1000 else temperature, 2)
    return values


class SystemMetricsSampler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._jetson_snapshot: Dict[str, Any] = {}
        self._error: Optional[str] = None
        self._last_net = psutil.net_io_counters()
        self._last_net_at = time.monotonic()
        psutil.cpu_percent(interval=None, percpu=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        if PLATFORM_KIND == "jetson":
            self._thread = threading.Thread(target=self._run_jtop, name="jtop-sampler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run_jtop(self) -> None:
        if jtop is None:
            with self._lock:
                self._error = "jtop is not installed; generic telemetry remains available"
            return
        while not self._stop.is_set():
            try:
                with jtop(interval=1.0) as jetson:
                    while not self._stop.is_set() and jetson.ok():
                        stats = dict(jetson.stats)
                        cpu_values = [_safe_float(value) for key, value in stats.items() if re.fullmatch(r"CPU\d+", key)]
                        cpu_values = [value for value in cpu_values if value is not None]
                        temperatures = {
                            key.removeprefix("Temp ").lower(): _round_optional(value)
                            for key, value in stats.items() if key.startswith("Temp ")
                        }
                        fans = {
                            key.removeprefix("Fan "): _round_optional(value)
                            for key, value in stats.items() if key.startswith("Fan ")
                        }
                        rails = {
                            key.removeprefix("Power "): _round_optional((_safe_float(value) or 0) / 1000)
                            for key, value in stats.items() if key.startswith("Power ") and key != "Power TOT"
                        }
                        engines = {
                            key: _round_optional(value)
                            for key, value in stats.items()
                            if key in {"GPU", "EMC", "APE", "NVDEC", "NVENC", "NVJPG", "OFA", "SE", "VIC"}
                        }
                        power_total = _safe_float(stats.get("Power TOT"))
                        snapshot = {
                            "cpu_cores_pct": cpu_values,
                            "gpu_pct": _round_optional(stats.get("GPU")),
                            "power_w": _round_optional(power_total / 1000) if power_total is not None else None,
                            "cpu_temp_c": temperatures.get("cpu"),
                            "gpu_temp_c": temperatures.get("gpu"),
                            "fan_pct": next(iter(fans.values()), None),
                            "power_mode": stats.get("nvp model"),
                            "jetson_clocks": stats.get("jetson_clocks"),
                            "jetson_temperatures": temperatures,
                            "jetson_fans": fans,
                            "jetson_power_rails_w": rails,
                            "jetson_engines": engines,
                        }
                        with self._lock:
                            self._jetson_snapshot = snapshot
                            self._error = None
            except Exception as exc:  # pragma: no cover - hardware-specific
                with self._lock:
                    self._error = str(exc)
                self._stop.wait(3.0)

    def snapshot(self) -> Dict[str, Any]:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage(str(PROJECT_ROOT))
        cores = psutil.cpu_percent(interval=None, percpu=True)
        cpu_average = round(sum(cores) / len(cores), 2) if cores else None
        try:
            frequency = psutil.cpu_freq()
        except (OSError, RuntimeError):
            frequency = None
        temperatures = _temperature_snapshot()
        net = psutil.net_io_counters()
        now = time.monotonic()
        with self._lock:
            sampled = dict(self._jetson_snapshot)
            error = self._error
            elapsed = max(now - self._last_net_at, 0.001)
            receive_rate = (net.bytes_recv - self._last_net.bytes_recv) / elapsed
            send_rate = (net.bytes_sent - self._last_net.bytes_sent) / elapsed
            self._last_net = net
            self._last_net_at = now

        jetson_cores = sampled.get("cpu_cores_pct")
        if jetson_cores:
            cores = jetson_cores
            cpu_average = round(sum(cores) / len(cores), 2)
        jetson_temperatures = sampled.get("jetson_temperatures") or {}
        temperatures.update({f"jetson:{key}": value for key, value in jetson_temperatures.items()})
        cpu_temp = sampled.get("cpu_temp_c")
        if cpu_temp is None and temperatures:
            cpu_temp = next(iter(temperatures.values()))

        sampled_at = datetime.now(timezone.utc).isoformat()
        sampled.update({
            "sampled_at": sampled_at,
            "platform_kind": PLATFORM_KIND,
            "cpu_pct": cpu_average,
            "ram_pct": round(memory.percent, 2),
            "swap_pct": round(swap.percent, 2),
            "ram_used_mb": round(memory.used / (1024 * 1024), 2),
            "ram_available_mb": round(memory.available / (1024 * 1024), 2),
            "swap_used_mb": round(swap.used / (1024 * 1024), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "load_1m": round(os.getloadavg()[0], 2),
            "cpu_temp_c": cpu_temp,
            "cpu": {
                "average_pct": cpu_average,
                "cores_pct": [round(value, 2) for value in cores],
                "frequency_mhz": _round_optional(frequency.current) if frequency else None,
                "frequency_min_mhz": _round_optional(frequency.min) if frequency else None,
                "frequency_max_mhz": _round_optional(frequency.max) if frequency else None,
                "load_1m": round(os.getloadavg()[0], 2),
                "load_5m": round(os.getloadavg()[1], 2),
                "load_15m": round(os.getloadavg()[2], 2),
            },
            "memory": {
                "percent": round(memory.percent, 2),
                "used_mb": round(memory.used / (1024 * 1024), 2),
                "available_mb": round(memory.available / (1024 * 1024), 2),
                "total_mb": round(memory.total / (1024 * 1024), 2),
            },
            "swap": {
                "percent": round(swap.percent, 2),
                "used_mb": round(swap.used / (1024 * 1024), 2),
                "total_mb": round(swap.total / (1024 * 1024), 2),
            },
            "disk": {
                "percent": round(disk.percent, 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2),
            },
            "network": {
                "bytes_received": net.bytes_recv,
                "bytes_sent": net.bytes_sent,
                "receive_bytes_s": round(max(receive_rate, 0), 2),
                "send_bytes_s": round(max(send_rate, 0), 2),
            },
            "temperatures_c": temperatures,
            "uptime_s": round(time.time() - psutil.boot_time(), 1),
            "accelerator": {
                "type": "cuda" if PLATFORM_KIND == "jetson" else "cpu",
                "utilization_pct": sampled.get("gpu_pct"),
                "engines": sampled.get("jetson_engines", {}),
            },
            "power": {
                "total_w": sampled.get("power_w"),
                "rails_w": sampled.get("jetson_power_rails_w", {}),
                "mode": sampled.get("power_mode"),
                "jetson_clocks": sampled.get("jetson_clocks"),
            },
            "fans": sampled.get("jetson_fans", {}),
        })
        if error:
            sampled["sampler_error"] = error
        return sampled

    @property
    def jtop_active(self) -> bool:
        with self._lock:
            return bool(self._jetson_snapshot) and self._error is None


sampler = SystemMetricsSampler()
sampler.start()
NODE_PROFILE = system_profile()


class ClusterChatRequest(ChatStreamRequest):
    seed: int = Field(42, ge=-1, le=2_147_483_647)


@app.get("/cluster/health")
async def cluster_health() -> Dict[str, Any]:
    models = manager.list_models()
    return {
        "ok": True,
        "node": {
            "name": NODE_NAME,
            "role": NODE_ROLE,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "platform_kind": PLATFORM_KIND,
            "git_commit": _git_commit(),
        },
        "profile": NODE_PROFILE,
        "capabilities": {
            "telemetry": "jtop+psutil" if sampler.jtop_active else "psutil",
            "gpu_offload": RUNTIME_BACKEND["gpu_offload"],
            "backend_verified": RUNTIME_BACKEND["verified"],
            "cpu_inference": True,
            "worker_api_auth": WORKER_API_AUTH_ENABLED,
        },
        "worker_api_auth": WORKER_API_AUTH_ENABLED,
        "telemetry_version": 2,
        "current": manager.current_model_info(),
        "model_count": len(models),
        "model_ids": [str(item["id"]) for item in models],
        "metrics": sampler.snapshot(),
    }


@app.get("/cluster/models")
async def cluster_models() -> Dict[str, Any]:
    return {
        "ok": True,
        "node": NODE_NAME,
        "models": manager.list_models(),
    }


@app.post("/cluster/chat/stream")
async def cluster_chat_stream(payload: ClusterChatRequest) -> StreamingResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")

    def event_generator() -> Iterable[str]:
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        pieces: List[str] = []
        chunks = 0
        try:
            with manager.lock:
                set_seed = getattr(manager.llm, "set_seed", None)
                if callable(set_seed):
                    set_seed(payload.seed)
                for token in manager.stream_chat(
                    message=message,
                    history=payload.history,
                    max_tokens=payload.max_tokens,
                    temperature=payload.temperature,
                    top_p=payload.top_p,
                ):
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    pieces.append(token)
                    chunks += 1
                    yield as_sse("token", {"text": token})

            finished = time.perf_counter()
            text = "".join(pieces)
            token_count = chunks
            tokenizer = getattr(manager.llm, "tokenize", None)
            if callable(tokenizer) and text:
                try:
                    token_count = len(tokenizer(text.encode("utf-8"), add_bos=False))
                except Exception:
                    pass

            ttft_s = (first_token_at - started) if first_token_at else finished - started
            generation_s = max(finished - (first_token_at or finished), 0.0)
            yield as_sse(
                "done",
                {
                    "metrics": {
                        "ttft_s": round(ttft_s, 6),
                        "generation_s": round(generation_s, 6),
                        "e2e_s": round(finished - started, 6),
                        "generated_tokens": token_count,
                        "stream_chunks": chunks,
                        "output_chars": len(text),
                    }
                },
            )
        except Exception as exc:
            yield as_sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
