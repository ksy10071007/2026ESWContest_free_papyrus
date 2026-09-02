#!/usr/bin/env python3
"""Run reproducible streaming LLM load experiments across selected nodes."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from cluster.clusterctl import (
    DEFAULT_INVENTORY,
    Node,
    load_nodes,
    request_json,
    run_on_node,
    select_nodes,
    worker_auth_headers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / ".run" / "cluster" / "results"
ProgressCallback = Callable[[Dict[str, Any]], None]

EXECUTION_STRATEGIES = {
    "single_node",
    "replicated_round_robin",
    "broadcast_compare",
    "node_sweep",
    "model_parallel_rpc",
}


def experiment_strategy_catalog() -> List[Dict[str, Any]]:
    """Return backend-owned descriptions used by the dashboard and reports."""
    return [
        {
            "id": "single_node",
            "label": "단일 노드 기준선",
            "model_placement": "full_model_single",
            "request_mapping": "all_to_one",
            "min_nodes": 1,
            "max_nodes": 1,
            "experimental": False,
            "summary": "선택한 1대에 전체 모델을 올리고 장치 자체 성능을 측정합니다.",
        },
        {
            "id": "replicated_round_robin",
            "label": "복제 모델 · 요청 분산",
            "model_placement": "full_model_per_node",
            "request_mapping": "round_robin",
            "min_nodes": 1,
            "max_nodes": 4,
            "experimental": False,
            "summary": "각 노드가 전체 모델을 따로 로드하고 여러 사용자 요청을 순서대로 나눕니다.",
        },
        {
            "id": "broadcast_compare",
            "label": "동일 요청 전체 전송",
            "model_placement": "full_model_per_node",
            "request_mapping": "broadcast",
            "min_nodes": 2,
            "max_nodes": 4,
            "experimental": False,
            "summary": "같은 요청을 모든 복제본에 보내 지연과 출력 일치도를 비교합니다.",
        },
        {
            "id": "node_sweep",
            "label": "노드 수 확장 스윕",
            "model_placement": "full_model_per_node",
            "request_mapping": "scenario_round_robin",
            "min_nodes": 2,
            "max_nodes": 4,
            "experimental": False,
            "summary": "1대, 2대, … 조건을 반복해 노드 추가에 따른 속도 향상과 효율을 비교합니다.",
        },
        {
            "id": "model_parallel_rpc",
            "label": "모델 분할 추론 · RPC",
            "model_placement": "sharded_model",
            "request_mapping": "one_coordinator",
            "min_nodes": 2,
            "max_nodes": 4,
            "experimental": True,
            "summary": "한 모델의 가중치와 계산을 여러 노드 장치에 분할하고 각 토큰을 함께 계산합니다.",
        },
    ]


@dataclass
class ExperimentConfig:
    experiment_id: str = ""
    name: str = "cluster-load-test"
    node_names: List[str] = field(default_factory=list)
    model_id: str = "qwen2.5-1.5b/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    n_ctx: int = 1024
    n_gpu_layers: int = 30
    requests: int = 20
    concurrency: int = 4
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9
    seed: int = 42
    warmup_requests: int = 1
    prompt: str = "엣지 장치에서 의료 LLM을 실행할 때의 장점과 한계를 한 문단으로 설명해줘."
    require_uniform_config: bool = True
    request_timeout_s: float = 600.0
    execution_strategy: str = "replicated_round_robin"
    sweep_mode: str = "cumulative"
    rpc_split_mode: str = "layer"
    rpc_split_policy: str = "auto"
    rpc_tensor_split: List[float] = field(default_factory=list)
    acknowledge_experimental_rpc: bool = False
    # The dashboard fills these fields when one user action expands into a
    # sequence of independent per-model runs.  CLI configs that omit them keep
    # the original single-run behaviour.
    suite_id: str = ""
    model_index: int = 1
    model_count: int = 1

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperimentConfig":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in raw.items() if key in known})

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty")
        if self.experiment_id and not self.experiment_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("experiment_id contains unsupported characters")
        if not self.node_names:
            raise ValueError("Select at least one node")
        validate_model_id(self.model_id)
        if self.suite_id and not self.suite_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("suite_id contains unsupported characters")
        if self.model_count < 1 or not 1 <= self.model_index <= self.model_count:
            raise ValueError("model_index must be between 1 and model_count")
        if not 128 <= self.n_ctx <= 4096:
            raise ValueError("n_ctx must be between 128 and 4096")
        if not 0 <= self.n_gpu_layers <= 120:
            raise ValueError("n_gpu_layers must be between 0 and 120")
        if not 1 <= self.requests <= 10_000:
            raise ValueError("requests must be between 1 and 10000")
        if not 1 <= self.concurrency <= 256:
            raise ValueError("concurrency must be between 1 and 256")
        if not 1 <= self.max_tokens <= 1024:
            raise ValueError("max_tokens must be between 1 and 1024")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
        if not 0 <= self.warmup_requests <= 10:
            raise ValueError("warmup_requests must be between 0 and 10")
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if self.execution_strategy not in EXECUTION_STRATEGIES:
            raise ValueError(f"Unsupported execution_strategy: {self.execution_strategy}")
        if self.sweep_mode not in {"cumulative", "individual"}:
            raise ValueError("sweep_mode must be cumulative or individual")
        if self.rpc_split_mode not in {"layer", "row"}:
            raise ValueError("rpc_split_mode must be layer or row")
        if self.rpc_split_policy not in {"auto", "equal", "custom"}:
            raise ValueError("rpc_split_policy must be auto, equal or custom")
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in self.rpc_tensor_split):
            raise ValueError("rpc_tensor_split values must be positive finite numbers")


def validate_model_id(model_id: str) -> str:
    """Validate and return a repository-relative GGUF model identifier."""
    if (
        not isinstance(model_id, str)
        or not model_id.endswith(".gguf")
        or model_id.startswith("/")
        or ".." in Path(model_id).parts
    ):
        raise ValueError("model_id must be a safe relative GGUF path")
    return model_id


def normalize_model_ids(model_id: str, model_ids: Sequence[str]) -> List[str]:
    """Normalize legacy single-model and suite payloads without ambiguity."""
    normalized = list(model_ids) if model_ids else ([model_id] if model_id else [])
    if not normalized:
        raise ValueError("Select at least one model")
    for item in normalized:
        validate_model_id(item)
    if len(set(normalized)) != len(normalized):
        raise ValueError("model_ids must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class RequestTask:
    request_id: int
    logical_request_id: int
    scenario_id: str
    target_node: str
    replica_index: int = 0


@dataclass(frozen=True)
class StrategyScenario:
    scenario_id: str
    label: str
    node_names: List[str]
    tasks: List[RequestTask]


def strategy_work_units(config: ExperimentConfig, node_count: int) -> int:
    if config.execution_strategy == "broadcast_compare":
        return config.requests * node_count
    if config.execution_strategy == "node_sweep":
        # Each scenario receives the same total logical workload. Cumulative
        # changes only which node subset shares those requests.
        return config.requests * node_count
    return config.requests


def benchmark_parameters(config: ExperimentConfig) -> Dict[str, Any]:
    """Return report-safe workload metadata without persisting prompt text."""
    return {
        "model_id": config.model_id,
        "n_ctx": config.n_ctx,
        # Retained for schema-v2 readers; requested/effective fields below are
        # authoritative for new report generators.
        "n_gpu_layers": config.n_gpu_layers,
        "requested_n_gpu_layers": config.n_gpu_layers,
        "effective_n_gpu_layers": "all" if config.execution_strategy == "model_parallel_rpc" else None,
        "requests_per_scenario": config.requests,
        "concurrency": config.concurrency,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
        "warmup_requests": config.warmup_requests,
        "require_uniform_config": config.require_uniform_config,
        "prompt_sha256": hashlib.sha256(config.prompt.encode("utf-8")).hexdigest(),
        "prompt_chars": len(config.prompt),
    }


def validate_strategy(nodes: Sequence[Node], config: ExperimentConfig) -> None:
    count = len(nodes)
    if config.execution_strategy == "single_node" and count != 1:
        raise ValueError("단일 노드 기준선은 정확히 1대의 노드가 필요합니다")
    if config.execution_strategy in {"broadcast_compare", "node_sweep", "model_parallel_rpc"} and count < 2:
        raise ValueError("선택한 실험 방식은 2대 이상의 노드가 필요합니다")
    if config.execution_strategy == "model_parallel_rpc":
        heads = [node for node in nodes if node.role == "head"]
        workers = [node for node in nodes if node.role == "worker"]
        if len(heads) != 1 or not workers:
            raise ValueError("모델 분할 RPC는 coordinator인 head 1대와 worker 1대 이상을 선택해야 합니다")
        if not config.acknowledge_experimental_rpc:
            raise ValueError("모델 분할 RPC의 실험적 특성과 LAN 보안 경고를 확인해야 합니다")
        if config.rpc_split_policy == "custom" and len(config.rpc_tensor_split) != count:
            raise ValueError("사용자 지정 분할 비율 수는 선택한 노드 수와 같아야 합니다")


def build_strategy_scenarios(config: ExperimentConfig, nodes: Sequence[Node]) -> List[StrategyScenario]:
    """Build a deterministic logical-to-physical request plan."""
    validate_strategy(nodes, config)
    scenarios: List[tuple[str, str, Sequence[Node], str]] = []
    if config.execution_strategy == "node_sweep":
        if config.sweep_mode == "cumulative":
            for size in range(1, len(nodes) + 1):
                scenarios.append((f"nodes-{size}", f"누적 {size}대", nodes[:size], "round_robin"))
        else:
            for index, node in enumerate(nodes, start=1):
                scenarios.append((f"node-{index}", f"개별 · {node.name}", [node], "round_robin"))
    elif config.execution_strategy == "broadcast_compare":
        scenarios.append(("broadcast", "동일 요청 전체 전송", nodes, "broadcast"))
    elif config.execution_strategy == "model_parallel_rpc":
        coordinator = next(node for node in nodes if node.role == "head")
        scenarios.append(("rpc-sharded", "RPC 모델 분할", nodes, f"coordinator:{coordinator.name}"))
    else:
        scenarios.append(("main", "단일 노드 기준선" if config.execution_strategy == "single_node" else "요청 분산", nodes, "round_robin"))

    request_id = 0
    built: List[StrategyScenario] = []
    for scenario_id, label, scenario_nodes, mapping in scenarios:
        tasks: List[RequestTask] = []
        for logical_id in range(1, config.requests + 1):
            if mapping == "broadcast":
                targets = list(enumerate(scenario_nodes))
            elif mapping.startswith("coordinator:"):
                coordinator_name = mapping.split(":", 1)[1]
                targets = [(0, next(node for node in scenario_nodes if node.name == coordinator_name))]
            else:
                targets = [((logical_id - 1) % len(scenario_nodes), scenario_nodes[(logical_id - 1) % len(scenario_nodes)])]
            for replica_index, target in targets:
                request_id += 1
                tasks.append(RequestTask(request_id, logical_id, scenario_id, target.name, replica_index))
        built.append(StrategyScenario(scenario_id, label, [node.name for node in scenario_nodes], tasks))
    return built


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _emit(callback: Optional[ProgressCallback], event_type: str, **payload: Any) -> None:
    if callback:
        callback({"type": event_type, "at": utc_now(), **payload})


def _stream_request(
    node: Node,
    config: ExperimentConfig,
    task: RequestTask,
    warmup: bool = False,
) -> Dict[str, Any]:
    payload = {
        "message": config.prompt,
        "history": [],
        "max_tokens": min(config.max_tokens, 16) if warmup else config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
    }
    request = urllib.request.Request(
        f"{node.api_url}/cluster/chat/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream", **worker_auth_headers()},
        method="POST",
    )
    started_wall = utc_now()
    started = time.perf_counter()
    first_token_at: Optional[float] = None
    output_parts: List[str] = []
    server_metrics: Dict[str, Any] = {}
    error = ""
    ok = False
    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                event_type = event.get("type")
                if event_type == "token":
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    output_parts.append(str(event.get("text", "")))
                elif event_type == "done":
                    server_metrics = event.get("metrics") or {}
                    ok = True
                elif event_type == "error":
                    error = str(event.get("message", "worker error"))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        error = str(exc)
    finished = time.perf_counter()
    ttft_s = (first_token_at - started) if first_token_at else None
    e2e_s = finished - started
    generated_tokens = int(server_metrics.get("generated_tokens") or 0)
    generation_s = float(server_metrics.get("generation_s") or 0.0)
    output = "".join(output_parts)
    return {
        "request_id": task.request_id,
        "logical_request_id": task.logical_request_id,
        "scenario_id": task.scenario_id,
        "replica_index": task.replica_index,
        "node": node.name,
        "assigned_node": node.name,
        "node_host": node.host,
        "started_at": started_wall,
        "ok": ok,
        "ttft_s": round(ttft_s, 6) if ttft_s is not None else None,
        "e2e_s": round(e2e_s, 6),
        "server_ttft_s": server_metrics.get("ttft_s"),
        "server_generation_s": server_metrics.get("generation_s"),
        "generated_tokens": generated_tokens,
        "tokens_per_s": round(generated_tokens / generation_s, 6) if generation_s > 0 else None,
        "output_chars": len(output),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest() if ok else "",
        "error": error,
        "warmup": warmup,
    }


def _stream_rpc_request(
    coordinator: Node,
    coordinator_url: str,
    config: ExperimentConfig,
    task: RequestTask,
) -> Dict[str, Any]:
    """Stream one request from the native llama.cpp RPC coordinator."""
    payload = {
        "messages": [{"role": "user", "content": config.prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{coordinator_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started_wall = utc_now()
    started = time.perf_counter()
    first_token_at: Optional[float] = None
    output_parts: List[str] = []
    generated_tokens = 0
    error = ""
    ok = False
    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                raw_event = line[6:]
                if raw_event == "[DONE]":
                    ok = True
                    continue
                event = json.loads(raw_event)
                usage = event.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    generated_tokens = int(usage["completion_tokens"])
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    token = str(delta.get("content") or choices[0].get("text") or "")
                    if token:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output_parts.append(token)
                    if choices[0].get("finish_reason") is not None:
                        ok = True
    except (OSError, ValueError, urllib.error.URLError) as exc:
        error = str(exc)
    finished = time.perf_counter()
    output = "".join(output_parts)
    if ok and generated_tokens <= 0:
        # Older llama-server builds may omit streaming usage. Preserve the run,
        # but record an explicit estimate rather than pretending it is exact.
        generated_tokens = len(output_parts)
    generation_s = finished - (first_token_at or started)
    return {
        "request_id": task.request_id,
        "logical_request_id": task.logical_request_id,
        "scenario_id": task.scenario_id,
        "replica_index": task.replica_index,
        "node": coordinator.name,
        "assigned_node": coordinator.name,
        "node_host": coordinator.host,
        "started_at": started_wall,
        "ok": ok,
        "ttft_s": round(first_token_at - started, 6) if first_token_at else None,
        "e2e_s": round(finished - started, 6),
        "server_ttft_s": None,
        "server_generation_s": round(generation_s, 6),
        "generated_tokens": generated_tokens,
        "tokens_per_s": round(generated_tokens / generation_s, 6) if generation_s > 0 else None,
        "output_chars": len(output),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest() if ok else "",
        "error": error,
        "warmup": False,
        "token_count_source": "server_usage" if generated_tokens and generated_tokens != len(output_parts) else "stream_chunk_estimate",
    }


def _load_model(node: Node, config: ExperimentConfig) -> Dict[str, Any]:
    result = request_json(
        f"{node.api_url}/api/select-model",
        method="POST",
        payload={
            "model_id": config.model_id,
            "n_ctx": config.n_ctx,
            "n_gpu_layers": config.n_gpu_layers,
        },
        timeout=900.0,
    )
    current = result.get("current") or {}
    if result.get("ok") is not True:
        raise RuntimeError(f"{node.name} rejected model selection")
    health = request_json(f"{node.api_url}/cluster/health", timeout=10.0)
    profile = health.get("profile") or {}
    return {
        "node": node.name,
        **current,
        "platform_kind": profile.get("platform_kind"),
        "runtime_backend": (profile.get("runtime_backend") or {}).get("kind"),
        "inference_threads": profile.get("inference_threads"),
    }


def _validate_uniform(loaded: Sequence[Dict[str, Any]], config: ExperimentConfig) -> List[str]:
    warnings: List[str] = []
    keys = ("model_id", "n_ctx", "n_gpu_layers", "n_batch")
    for key in keys:
        values = {str(item.get(key)) for item in loaded}
        if len(values) > 1:
            warnings.append(f"nodes differ in actual {key}: {', '.join(sorted(values))}")
    for item in loaded:
        if item.get("n_ctx") != config.n_ctx:
            warnings.append(f"{item['node']} adjusted n_ctx to {item.get('n_ctx')}")
        if item.get("n_gpu_layers") != config.n_gpu_layers:
            warnings.append(
                f"{item['node']} adjusted n_gpu_layers to {item.get('n_gpu_layers')}"
            )
    return warnings


def validate_platform_layers(nodes: Sequence[Node], config: ExperimentConfig) -> None:
    if config.execution_strategy == "model_parallel_rpc":
        return
    if config.n_gpu_layers == 0:
        return
    pi_nodes: List[str] = []
    for node in nodes:
        kind = node.platform
        if kind == "auto":
            try:
                health = request_json(f"{node.api_url}/cluster/health", timeout=5.0)
                kind = str((health.get("profile") or {}).get("platform_kind") or "auto")
            except Exception:
                kind = "auto"
        if kind == "raspberry-pi":
            pi_nodes.append(node.name)
    if pi_nodes:
        raise ValueError(
            "Raspberry Pi nodes require n_gpu_layers=0: " + ", ".join(pi_nodes)
        )


RPC_SERVER_PORT = 50052
RPC_COORDINATOR_PORT = 18080


def _rpc_runtime_command(node: Node, action: str, *arguments: str, timeout: int = 120) -> Dict[str, Any]:
    script = f"{node.project_dir}/cluster/rpc/runtime.sh"
    process = run_on_node(node, [script, action, *arguments], timeout=timeout)
    return {
        "node": node.name,
        "ok": process.returncode == 0,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }


def rpc_preflight(nodes: Sequence[Node]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for node in nodes:
        item = _rpc_runtime_command(node, "check", timeout=20)
        if not item["ok"]:
            item["error"] = item["stderr"] or item["stdout"] or "native RPC runtime unavailable"
        results.append(item)
    return results


def _rpc_platform_from_check(node: Node, check: Dict[str, Any]) -> str:
    """Resolve the platform from the pinned runtime check before topology setup."""
    output = f"{check.get('stdout', '')}\n{check.get('stderr', '')}"
    if "platform=raspberry-pi" in output:
        return "raspberry-pi"
    if "platform=jetson" in output:
        return "jetson"
    return node.platform


def _start_rpc_topology(
    nodes: Sequence[Node],
    config: ExperimentConfig,
    emit: Callable[..., None],
) -> tuple[Node, str, Dict[str, Any], List[Node]]:
    coordinator = next(node for node in nodes if node.role == "head")
    workers = [node for node in nodes if node.role == "worker"]
    checks = rpc_preflight(nodes)
    missing = [item for item in checks if not item["ok"]]
    if missing:
        details = "; ".join(f"{item['node']}: {item.get('error', 'not ready')}" for item in missing)
        raise RuntimeError("RPC runtime is not ready. Run 'RPC 환경 준비' first. " + details)

    # Free memory held by the replicated Python worker before native sharding.
    for node in nodes:
        try:
            request_json(f"{node.api_url}/api/unload-model", method="POST", payload={}, timeout=60.0)
        except Exception as exc:
            raise RuntimeError(f"Failed to unload the replicated model on {node.name}: {exc}") from exc

    checks_by_name = {item["node"]: item for item in checks}
    coordinator_platform = _rpc_platform_from_check(coordinator, checks_by_name[coordinator.name])
    endpoints: List[str] = []
    started_workers: List[Node] = []
    rpc_device_nodes: List[Node] = []
    try:
        for worker in workers:
            emit("rpc_started", node=worker.name, role="device", port=RPC_SERVER_PORT)
            started = _rpc_runtime_command(worker, "start-worker", str(RPC_SERVER_PORT), timeout=60)
            if not started["ok"]:
                raise RuntimeError(f"RPC device failed on {worker.name}: {started['stderr'] or started['stdout']}")
            started_workers.append(worker)
            rpc_device_nodes.append(worker)
            endpoints.append(f"{worker.host}:{RPC_SERVER_PORT}")

        # llama.cpp does not use the coordinator's local CPU as an offload
        # device.  On a Raspberry Pi head, expose that CPU through a loopback
        # RPC device so it genuinely receives a model shard and participates.
        if coordinator_platform == "raspberry-pi":
            emit("rpc_started", node=coordinator.name, role="loopback_cpu_device", port=RPC_SERVER_PORT)
            started = _rpc_runtime_command(
                coordinator,
                "start-worker",
                str(RPC_SERVER_PORT),
                "127.0.0.1",
                timeout=60,
            )
            if not started["ok"]:
                raise RuntimeError(
                    f"RPC loopback CPU device failed on {coordinator.name}: "
                    f"{started['stderr'] or started['stdout']}"
                )
            started_workers.append(coordinator)
            rpc_device_nodes.append(coordinator)
            endpoints.append(f"127.0.0.1:{RPC_SERVER_PORT}")

        model_path = (PROJECT_ROOT / "models" / config.model_id).resolve()
        try:
            model_path.relative_to((PROJECT_ROOT / "models").resolve())
        except ValueError as exc:
            raise RuntimeError("Unsafe coordinator model path") from exc
        if not model_path.is_file():
            raise RuntimeError(f"Coordinator model is missing: {config.model_id}")

        split_values: List[float] = []
        if config.rpc_split_policy == "equal":
            split_values = [1.0] * len(nodes)
        elif config.rpc_split_policy == "custom":
            requested_by_node = {
                node.name: float(value) for node, value in zip(nodes, config.rpc_tensor_split)
            }
            # llama.cpp registers RPC devices first in endpoint order and the
            # coordinator's local device last. Reorder user values accordingly.
            split_values = [requested_by_node[node.name] for node in rpc_device_nodes]
            if coordinator_platform != "raspberry-pi":
                split_values.append(requested_by_node[coordinator.name])
        split_csv = ",".join(f"{value:g}" for value in split_values) or "-"
        endpoint_csv = ",".join(endpoints)
        emit(
            "rpc_started",
            node=coordinator.name,
            role="coordinator",
            port=RPC_COORDINATOR_PORT,
            endpoints=endpoints,
        )
        load_started = time.perf_counter()
        started = _rpc_runtime_command(
            coordinator,
            "start-coordinator",
            str(RPC_COORDINATOR_PORT),
            str(model_path),
            str(config.n_ctx),
            "999",
            endpoint_csv,
            config.rpc_split_mode,
            split_csv,
            timeout=900,
        )
        if not started["ok"]:
            raise RuntimeError(f"RPC coordinator failed: {started['stderr'] or started['stdout']}")
        load_s = time.perf_counter() - load_started
        commit_check = run_on_node(
            coordinator,
            ["git", "-C", f"{coordinator.project_dir}/.run/cluster/llama.cpp-src", "rev-parse", "HEAD"],
            timeout=20,
        )
        topology = {
            "engine": "llama.cpp-rpc",
            "runtime_commit": commit_check.stdout.strip() if commit_check.returncode == 0 else "unknown",
            "coordinator": coordinator.name,
            "participants": [node.name for node in nodes],
            "rpc_workers": [node.name for node in workers],
            "rpc_device_nodes": [node.name for node in rpc_device_nodes],
            "rpc_endpoints": endpoints,
            "split_mode": config.rpc_split_mode,
            "split_policy": config.rpc_split_policy,
            "tensor_split": split_values,
            "resolved_device_order": [node.name for node in rpc_device_nodes]
            + ([] if coordinator_platform == "raspberry-pi" else [coordinator.name]),
            "requested_gpu_layers": "all",
            "model_load_s": round(load_s, 6),
            "transport": "TCP LAN",
            "rpc_security": "unauthenticated_ephemeral_private_lan",
            "coordinator_slots": 1,
            "client_concurrency": config.concurrency,
        }
        return coordinator, f"http://127.0.0.1:{RPC_COORDINATOR_PORT}", topology, started_workers
    except Exception as exc:
        cleanup_errors = _stop_rpc_topology(coordinator, started_workers)
        if cleanup_errors:
            raise RuntimeError(
                f"{exc}; RPC cleanup also failed: {'; '.join(cleanup_errors)}"
            ) from exc
        raise


def _stop_rpc_topology(coordinator: Node, workers: Sequence[Node]) -> List[str]:
    errors: List[str] = []
    try:
        result = _rpc_runtime_command(coordinator, "stop-coordinator", str(RPC_COORDINATOR_PORT), timeout=30)
        if not result["ok"]:
            errors.append(f"{coordinator.name} coordinator: {result['stderr'] or result['stdout']}")
    except Exception as exc:
        errors.append(f"{coordinator.name} coordinator: {exc}")
    for worker in workers:
        try:
            result = _rpc_runtime_command(worker, "stop-worker", str(RPC_SERVER_PORT), timeout=30)
            if not result["ok"]:
                errors.append(f"{worker.name} RPC device: {result['stderr'] or result['stdout']}")
        except Exception as exc:
            errors.append(f"{worker.name} RPC device: {exc}")
    return errors


def _aggregate(records: Sequence[Dict[str, Any]], wall_s: float) -> Dict[str, Any]:
    successful = [item for item in records if item["ok"]]
    ttft = [float(item["ttft_s"]) for item in successful if item["ttft_s"] is not None]
    e2e = [float(item["e2e_s"]) for item in successful]
    total_tokens = sum(int(item["generated_tokens"]) for item in successful)
    per_node: Dict[str, Dict[str, Any]] = {}
    for item in records:
        bucket = per_node.setdefault(
            item["node"],
            {
                "requests": 0,
                "successful": 0,
                "tokens": 0,
                "ttft_s": [],
                "e2e_s": [],
                "tokens_per_s_samples": [],
            },
        )
        bucket["requests"] += 1
        if item["ok"]:
            bucket["successful"] += 1
            bucket["tokens"] += int(item["generated_tokens"])
            bucket["e2e_s"].append(float(item["e2e_s"]))
            if item.get("ttft_s") is not None:
                bucket["ttft_s"].append(float(item["ttft_s"]))
            if item.get("tokens_per_s") is not None:
                bucket["tokens_per_s_samples"].append(float(item["tokens_per_s"]))
    for bucket in per_node.values():
        node_ttft = bucket.pop("ttft_s")
        node_e2e = bucket.pop("e2e_s")
        node_tps = bucket.pop("tokens_per_s_samples")
        bucket["failed"] = bucket["requests"] - bucket["successful"]
        bucket["success_rate"] = round(bucket["successful"] / bucket["requests"], 6) if bucket["requests"] else 0.0
        bucket["effective_tokens_per_s"] = round(bucket["tokens"] / wall_s, 6) if wall_s > 0 else 0.0
        bucket["average_generation_tokens_per_s"] = round(sum(node_tps) / len(node_tps), 6) if node_tps else None
        bucket["ttft_p50_s"] = percentile(node_ttft, 0.50)
        bucket["ttft_p95_s"] = percentile(node_ttft, 0.95)
        bucket["e2e_p50_s"] = percentile(node_e2e, 0.50)
        bucket["e2e_p95_s"] = percentile(node_e2e, 0.95)
    logical_groups: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
    for item in records:
        logical_groups.setdefault(
            (str(item.get("scenario_id") or "main"), int(item.get("logical_request_id") or item["request_id"])),
            [],
        ).append(item)
    all_success = sum(1 for group in logical_groups.values() if group and all(item["ok"] for item in group))
    comparable_groups = [group for group in logical_groups.values() if len(group) > 1 and all(item["ok"] for item in group)]
    agreement = sum(
        1
        for group in comparable_groups
        if len({str(item.get("output_sha256") or "") for item in group}) == 1
    )
    return {
        "requests": len(records),
        "logical_requests": len(logical_groups),
        "physical_requests": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "success_rate": round(len(successful) / len(records), 6) if records else 0.0,
        "wall_s": round(wall_s, 6),
        "requests_per_s": round(len(successful) / wall_s, 6) if wall_s > 0 else 0.0,
        "total_generated_tokens": total_tokens,
        "cluster_tokens_per_s": round(total_tokens / wall_s, 6) if wall_s > 0 else 0.0,
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p95_s": percentile(ttft, 0.95),
        "e2e_p50_s": percentile(e2e, 0.50),
        "e2e_p95_s": percentile(e2e, 0.95),
        "all_replicas_success_rate": round(all_success / len(logical_groups), 6) if logical_groups else 0.0,
        "answer_agreement_rate": round(agreement / len(comparable_groups), 6) if comparable_groups else None,
        "per_node": per_node,
    }


def _failure_record(task: RequestTask, node: Node, exc: Exception) -> Dict[str, Any]:
    return {
        "request_id": task.request_id,
        "logical_request_id": task.logical_request_id,
        "scenario_id": task.scenario_id,
        "replica_index": task.replica_index,
        "node": node.name,
        "assigned_node": node.name,
        "node_host": node.host,
        "started_at": utc_now(),
        "ok": False,
        "ttft_s": None,
        "e2e_s": 0.0,
        "server_ttft_s": None,
        "server_generation_s": None,
        "generated_tokens": 0,
        "tokens_per_s": None,
        "output_chars": 0,
        "output_sha256": "",
        "error": str(exc),
        "warmup": False,
    }


def _measure_scenario(
    scenario: StrategyScenario,
    nodes_by_name: Dict[str, Node],
    config: ExperimentConfig,
    emit: Callable[..., None],
    cancel_event: threading.Event,
    completed_offset: int,
    total_work_units: int,
    rpc_coordinator: Optional[Node] = None,
    rpc_url: str = "",
) -> tuple[List[Dict[str, Any]], float]:
    records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    # Broadcast concurrency is expressed in logical request groups; each group
    # fans out to every selected node. Other strategies use physical calls.
    physical_concurrency = config.concurrency
    if config.execution_strategy == "broadcast_compare":
        physical_concurrency = min(len(scenario.tasks), config.concurrency * len(scenario.node_names))
    else:
        physical_concurrency = min(len(scenario.tasks), config.concurrency)
    max_workers = max(1, physical_concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: Dict[concurrent.futures.Future[Dict[str, Any]], tuple[RequestTask, Node, int]] = {}
        batches: List[List[RequestTask]] = []
        if config.execution_strategy == "broadcast_compare":
            by_logical_id: Dict[int, List[RequestTask]] = {}
            for task in scenario.tasks:
                by_logical_id.setdefault(task.logical_request_id, []).append(task)
            batches = list(by_logical_id.values())
            batch_slots = min(config.concurrency, len(batches))
        else:
            batches = [[task] for task in scenario.tasks]
            batch_slots = min(max_workers, len(batches))
        batch_iterator = iter(enumerate(batches))
        pending_by_batch: Dict[int, int] = {}

        def submit_batch(batch_id: int, tasks: Sequence[RequestTask]) -> None:
            pending_by_batch[batch_id] = len(tasks)
            for task in tasks:
                target = nodes_by_name[task.target_node]
                if rpc_coordinator is not None:
                    future = executor.submit(_stream_rpc_request, rpc_coordinator, rpc_url, config, task)
                else:
                    future = executor.submit(_stream_request, target, config, task)
                futures[future] = (task, target, batch_id)

        for _ in range(batch_slots):
            if cancel_event.is_set():
                break
            try:
                submit_batch(*next(batch_iterator))
            except StopIteration:
                break

        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            completed_batches: List[int] = []
            for future in done:
                task, target, batch_id = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = _failure_record(task, target, exc)
                records.append(result)
                emit(
                    "request_completed",
                    completed=completed_offset + len(records),
                    total=total_work_units,
                    result=result,
                )
                pending_by_batch[batch_id] -= 1
                if pending_by_batch[batch_id] == 0:
                    del pending_by_batch[batch_id]
                    completed_batches.append(batch_id)
            if not cancel_event.is_set():
                for _ in completed_batches:
                    try:
                        submit_batch(*next(batch_iterator))
                    except StopIteration:
                        break
    return records, time.perf_counter() - started


def run_experiment(
    config: ExperimentConfig,
    inventory_path: Path = DEFAULT_INVENTORY,
    results_root: Path = DEFAULT_RESULTS_DIR,
    progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    config.validate()
    cancel_event = cancel_event or threading.Event()
    all_nodes = load_nodes(inventory_path)
    selected = select_nodes(all_nodes, config.node_names)
    if len(selected) != len(config.node_names):
        raise ValueError("Some selected nodes are unavailable")
    selected_by_name = {node.name: node for node in selected}
    nodes = [selected_by_name[name] for name in config.node_names]
    validate_strategy(nodes, config)
    validate_platform_layers(nodes, config)
    scenarios = build_strategy_scenarios(config, nodes)
    total_work_units = sum(len(scenario.tasks) for scenario in scenarios)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    events_path = run_dir / "events.jsonl"
    events_lock = threading.Lock()

    def emit(event_type: str, **payload: Any) -> None:
        event = {
            "type": event_type,
            "at": utc_now(),
            "run_id": run_id,
            "suite_id": config.suite_id,
            "model_id": config.model_id,
            "model_index": config.model_index,
            "model_count": config.model_count,
            **payload,
        }
        with events_lock:
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        if progress:
            progress(event)

    emit(
        "run_started",
        config=asdict(config),
        nodes=[node.name for node in nodes],
        strategy=config.execution_strategy,
        total_work_units=total_work_units,
    )
    loaded: List[Dict[str, Any]] = []
    warnings: List[str] = []
    rpc_coordinator: Optional[Node] = None
    rpc_workers: List[Node] = []
    rpc_url = ""
    topology: Dict[str, Any] = {}
    try:
        if config.execution_strategy == "model_parallel_rpc":
            emit("phase", phase="rpc_preflight", message="RPC 모델 분할 런타임과 노드 연결을 확인하는 중")
            rpc_coordinator, rpc_url, topology, rpc_workers = _start_rpc_topology(nodes, config, emit)
            loaded = [
                {
                    "node": node.name,
                    "loaded": True,
                    "model_id": config.model_id,
                    "placement": "sharded_participant",
                    "runtime_backend": "llama.cpp-rpc",
                    "coordinator": node.name == rpc_coordinator.name,
                }
                for node in nodes
            ]
            warnings.append("llama.cpp RPC는 proof-of-concept이며 인증 없는 사설 LAN 전용 실험 경로입니다")
        else:
            emit("phase", phase="loading_model", message="선택한 노드에 전체 모델 복제본을 로드하는 중")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
                futures = {executor.submit(_load_model, node, config): node for node in nodes}
                for future in concurrent.futures.as_completed(futures):
                    node = futures[future]
                    try:
                        info = future.result()
                        loaded.append(info)
                        emit("node_model_loaded", node=node.name, actual=info)
                    except Exception as exc:
                        emit("node_error", node=node.name, error=str(exc))
                        raise RuntimeError(f"Failed to load model on {node.name}: {exc}") from exc

            warnings.extend(_validate_uniform(loaded, config))
            if warnings and config.require_uniform_config:
                raise RuntimeError("Uniform configuration check failed: " + "; ".join(warnings))

        for warning in warnings:
            emit("warning", message=warning)

        if cancel_event.is_set():
            raise RuntimeError("Experiment cancelled before warmup")

        if config.warmup_requests:
            emit("phase", phase="warmup", message="측정 전 워밍업 실행 중")
            if rpc_coordinator is not None:
                for warmup_index in range(config.warmup_requests):
                    if cancel_event.is_set():
                        break
                    task = RequestTask(-(warmup_index + 1), warmup_index + 1, "warmup", rpc_coordinator.name)
                    result = _stream_rpc_request(rpc_coordinator, rpc_url, config, task)
                    if not result["ok"]:
                        raise RuntimeError(f"RPC warmup failed: {result['error']}")
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
                    # Keep at most one warmup in flight per node. This avoids
                    # queuing node_count * warmup_requests calls that would all
                    # survive a cancellation request.
                    warmup_jobs: Dict[concurrent.futures.Future[Dict[str, Any]], tuple[int, Node, int]] = {}

                    def submit_warmup(node_index: int, node: Node, warmup_index: int) -> None:
                        task = RequestTask(
                            -(node_index * config.warmup_requests + warmup_index + 1),
                            warmup_index + 1,
                            "warmup",
                            node.name,
                            node_index,
                        )
                        future = executor.submit(_stream_request, node, config, task, True)
                        warmup_jobs[future] = (node_index, node, warmup_index)

                    for node_index, node in enumerate(nodes):
                        if cancel_event.is_set():
                            break
                        submit_warmup(node_index, node, 0)

                    while warmup_jobs:
                        done, _ = concurrent.futures.wait(
                            warmup_jobs,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            node_index, node, warmup_index = warmup_jobs.pop(future)
                            result = future.result()
                            if not result["ok"]:
                                raise RuntimeError(f"Warmup failed on {result['node']}: {result['error']}")
                            next_index = warmup_index + 1
                            if not cancel_event.is_set() and next_index < config.warmup_requests:
                                submit_warmup(node_index, node, next_index)

        if cancel_event.is_set():
            raise RuntimeError("Experiment cancelled before measurement")

        emit("phase", phase="measurement", message="선택한 실험 전략으로 부하를 측정하는 중")
        records: List[Dict[str, Any]] = []
        scenario_summaries: List[Dict[str, Any]] = []
        wall_started = time.perf_counter()
        nodes_by_name = {node.name: node for node in nodes}
        for scenario in scenarios:
            if cancel_event.is_set():
                break
            emit(
                "scenario_started",
                scenario_id=scenario.scenario_id,
                label=scenario.label,
                nodes=scenario.node_names,
                physical_requests=len(scenario.tasks),
            )
            scenario_records, scenario_wall_s = _measure_scenario(
                scenario,
                nodes_by_name,
                config,
                emit,
                cancel_event,
                len(records),
                total_work_units,
                rpc_coordinator,
                rpc_url,
            )
            records.extend(scenario_records)
            scenario_summary = _aggregate(scenario_records, scenario_wall_s)
            scenario_summary.update(
                {
                    "scenario_id": scenario.scenario_id,
                    "label": scenario.label,
                    "nodes": scenario.node_names,
                }
            )
            scenario_summaries.append(scenario_summary)
            emit("scenario_finished", scenario_id=scenario.scenario_id, summary=scenario_summary)
        wall_s = time.perf_counter() - wall_started
        records.sort(key=lambda item: item["request_id"])
        summary = _aggregate(records, wall_s)
        if (
            config.execution_strategy == "node_sweep"
            and config.sweep_mode == "cumulative"
            and scenario_summaries
        ):
            baseline = float(scenario_summaries[0].get("cluster_tokens_per_s") or 0.0)
            for scenario_summary in scenario_summaries:
                throughput = float(scenario_summary.get("cluster_tokens_per_s") or 0.0)
                node_count = max(1, len(scenario_summary.get("nodes") or []))
                scenario_summary["speedup_vs_baseline"] = round(throughput / baseline, 6) if baseline > 0 else None
                scenario_summary["scaling_efficiency"] = round(throughput / baseline / node_count, 6) if baseline > 0 else None

        # A completed RPC result must also mean every ephemeral unauthenticated
        # device process was confirmed stopped.  Otherwise fail the run and let
        # the finalizer retry, instead of publishing a misleading success.
        if rpc_coordinator is not None:
            emit("phase", phase="rpc_cleanup", message="RPC 모델 분할 프로세스를 종료하는 중")
            cleanup_errors = _stop_rpc_topology(rpc_coordinator, rpc_workers)
            if cleanup_errors:
                raise RuntimeError("RPC cleanup failed: " + "; ".join(cleanup_errors))
            rpc_coordinator = None
            rpc_workers = []
        summary.update(
            {
                "schema_version": 2,
                "run_id": run_id,
                "suite_id": config.suite_id,
                "experiment_id": config.experiment_id,
                "name": config.name,
                "model_id": config.model_id,
                "model_index": config.model_index,
                "model_count": config.model_count,
                "execution_strategy": config.execution_strategy,
                "model_placement": "sharded" if config.execution_strategy == "model_parallel_rpc" else "replicated",
                "status": "cancelled" if cancel_event.is_set() else "completed",
                "started_at": json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])["at"],
                "finished_at": utc_now(),
                "nodes": [node.name for node in nodes],
                "actual_model_config": loaded,
                "benchmark_parameters": benchmark_parameters(config),
                "warnings": warnings,
                "scenario_summaries": scenario_summaries,
                "topology": topology,
                "result_dir": str(run_dir),
            }
        )

        fieldnames = list(records[0].keys()) if records else []
        if fieldnames:
            with (run_dir / "requests.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        emit("run_finished", summary=summary)
        return summary
    except Exception as exc:
        cancelled = cancel_event.is_set()
        failure = {
            "schema_version": 2,
            "run_id": run_id,
            "suite_id": config.suite_id,
            "experiment_id": config.experiment_id,
            "name": config.name,
            "model_id": config.model_id,
            "model_index": config.model_index,
            "model_count": config.model_count,
            "execution_strategy": config.execution_strategy,
            "model_placement": "sharded" if config.execution_strategy == "model_parallel_rpc" else "replicated",
            "status": "cancelled" if cancelled else "failed",
            "finished_at": utc_now(),
            "nodes": [node.name for node in nodes],
            "actual_model_config": loaded,
            "benchmark_parameters": benchmark_parameters(config),
            "error": str(exc),
            "result_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if cancelled:
            emit("run_finished", summary=failure)
            return failure
        emit("run_failed", error=str(exc), summary=failure)
        raise
    finally:
        if rpc_coordinator is not None:
            emit("phase", phase="rpc_cleanup", message="RPC 모델 분할 프로세스를 종료하는 중")
            cleanup_errors = _stop_rpc_topology(rpc_coordinator, rpc_workers)
            if cleanup_errors:
                emit("rpc_cleanup_failed", errors=cleanup_errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    config = ExperimentConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    try:
        summary = run_experiment(
            config,
            inventory_path=args.inventory,
            results_root=args.results_dir,
            progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
