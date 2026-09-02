#!/usr/bin/env python3
"""Benchmark local GGUF LLM inference on Jetson-class hardware.

Primary backend: llama-cpp-python (GGUF), built with CUDA enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    from jtop import jtop

    JTOP_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    jtop = None
    JTOP_IMPORT_ERROR = str(exc)


DEFAULT_PROMPTS = [
    "Explain edge AI in one paragraph for a beginner.",
    "Write 5 bullet points on optimizing LLM inference on embedded devices.",
    "Summarize why quantization helps on memory-constrained GPUs.",
]


@dataclass
class PromptResult:
    prompt: str
    output: str
    ttft_s: float
    generation_time_s: float
    generated_tokens: int
    tps: float
    peak_ram_vram_used_mb: Optional[float]
    peak_ram_vram_used_pct: Optional[float]
    avg_gpu_util_pct: Optional[float]
    avg_power_w: Optional[float]
    peak_gpu_temp_c: Optional[float]
    peak_cpu_temp_c: Optional[float]
    hw_sample_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local GGUF LLM inference.")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--model-name", default=None, help="Optional friendly name for report")
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Optional text file with one prompt per line",
    )
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of prompts to run")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens per prompt")
    parser.add_argument("--n-ctx", type=int, default=1024, help="Context window")
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=35,
        help="Number of layers to offload to GPU (reduce if OOM)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="CPU threads for llama.cpp",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=512,
        help="Batch size for llama context allocation (lower can reduce memory pressure)",
    )
    parser.add_argument(
        "--n-ubatch",
        type=int,
        default=512,
        help="Micro-batch size for llama context allocation (lower can reduce memory pressure)",
    )
    parser.add_argument(
        "--op-offload",
        choices=["auto", "on", "off"],
        default="auto",
        help="Operator offload mode for llama.cpp (auto/on/off)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one short warmup generation before measurements",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save machine-readable benchmark summary JSON",
    )
    return parser.parse_args()


def get_process_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_peak_rss_mb_linux() -> float:
    # On Linux ru_maxrss is in KB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return statistics.mean(present)


def _max_or_none(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return max(present)


def _format_optional(value: Optional[float], suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}{suffix}"


def _flatten_numeric(
    value: Any,
    prefix: str = "",
    out: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if out is None:
        out = {}

    if isinstance(value, dict):
        for key, item in value.items():
            key_str = str(key).strip().lower().replace(" ", "_")
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            _flatten_numeric(item, prefix=next_prefix, out=out)
        return out

    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            _flatten_numeric(item, prefix=next_prefix, out=out)
        return out

    num = _to_float(value)
    if num is not None and prefix:
        out[prefix] = num
    return out


def _find_numeric(
    flat: Dict[str, float],
    required_tokens: List[str],
    any_tokens: Optional[List[str]] = None,
    exclude_tokens: Optional[List[str]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Optional[float]:
    for key, value in flat.items():
        if required_tokens and any(token not in key for token in required_tokens):
            continue
        if any_tokens and not any(token in key for token in any_tokens):
            continue
        if exclude_tokens and any(token in key for token in exclude_tokens):
            continue
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        return value
    return None


def _to_mb(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    # Heuristic unit conversion:
    # - very large values are typically bytes
    # - mid-size values can be KB from low-level counters
    if value > 512 * 1024 * 1024:
        return value / (1024.0 * 1024.0)
    if value > 512 * 1024:
        return value / 1024.0
    return value


def _to_watts(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    # Heuristic unit conversion:
    # - common rail readings are mW
    # - some APIs expose uW
    if value > 100000:
        return value / 1_000_000.0
    if value > 1000:
        return value / 1000.0
    return value


def _extract_jetson_sample(jetson_obj: Any) -> Dict[str, Optional[float]]:
    stats_obj = jetson_obj.stats if isinstance(getattr(jetson_obj, "stats", None), dict) else {}
    stats_flat = _flatten_numeric(stats_obj)

    memory_obj = getattr(jetson_obj, "memory", None)
    memory_flat = _flatten_numeric(memory_obj, prefix="memory") if isinstance(memory_obj, dict) else {}

    gpu_obj = getattr(jetson_obj, "gpu", None)
    gpu_flat = _flatten_numeric(gpu_obj, prefix="gpu") if isinstance(gpu_obj, dict) else {}

    power_obj = getattr(jetson_obj, "power", None)
    power_flat = _flatten_numeric(power_obj, prefix="power") if isinstance(power_obj, dict) else {}

    temp_obj = getattr(jetson_obj, "temperature", None)
    temp_flat = _flatten_numeric(temp_obj, prefix="temp") if isinstance(temp_obj, dict) else {}

    ram_used_raw = _find_numeric(memory_flat, ["ram"], any_tokens=["used"])
    if ram_used_raw is None:
        ram_used_raw = _find_numeric(stats_flat, ["ram"], any_tokens=["used"])

    ram_total_raw = _find_numeric(memory_flat, ["ram"], any_tokens=["tot", "total"])
    if ram_total_raw is None:
        ram_total_raw = _find_numeric(stats_flat, ["ram"], any_tokens=["tot", "total"])

    ram_pct = _find_numeric(
        memory_flat,
        ["ram"],
        any_tokens=["perc", "percent", "util", "usage", "use"],
        exclude_tokens=["used"],
        min_value=0.0,
        max_value=100.0,
    )
    if ram_pct is None:
        ram_pct = _find_numeric(
            stats_flat,
            ["ram"],
            any_tokens=["perc", "percent", "util", "usage", "use"],
            exclude_tokens=["used"],
            min_value=0.0,
            max_value=100.0,
        )

    # Some jtop versions expose RAM as ratio [0, 1] under key "RAM" in stats.
    if ram_pct is None:
        ram_ratio_or_pct = _find_numeric(
            stats_flat,
            ["ram"],
            exclude_tokens=["temp", "power"],
            min_value=0.0,
            max_value=100.0,
        )
        if ram_ratio_or_pct is not None:
            ram_pct = ram_ratio_or_pct * 100.0 if ram_ratio_or_pct <= 1.0 else ram_ratio_or_pct

    ram_used_mb = _to_mb(ram_used_raw)
    ram_total_mb = _to_mb(ram_total_raw)
    if ram_pct is None and ram_used_mb is not None and ram_total_mb and ram_total_mb > 0:
        ram_pct = max(0.0, min(100.0, (ram_used_mb / ram_total_mb) * 100.0))

    # Unified memory fallback: estimate used MB from ratio and system RAM total.
    if ram_used_mb is None and ram_pct is not None and psutil is not None:
        total_sys_mb = psutil.virtual_memory().total / (1024.0 * 1024.0)
        ram_used_mb = total_sys_mb * (ram_pct / 100.0)

    gpu_util_pct = _find_numeric(
        gpu_flat,
        ["gpu"],
        any_tokens=["load", "util", "usage"],
        min_value=0.0,
        max_value=100.0,
    )
    if gpu_util_pct is None:
        gpu_util_pct = _find_numeric(
            stats_flat,
            ["gpu"],
            any_tokens=["load", "util", "usage"],
            min_value=0.0,
            max_value=100.0,
        )

    # Some jtop versions expose GPU as ratio [0, 1] under key "GPU" in stats.
    if gpu_util_pct is None:
        gpu_ratio_or_pct = _find_numeric(
            stats_flat,
            ["gpu"],
            exclude_tokens=["temp", "power", "freq", "mhz", "clock"],
            min_value=0.0,
            max_value=100.0,
        )
        if gpu_ratio_or_pct is not None:
            gpu_util_pct = gpu_ratio_or_pct * 100.0 if gpu_ratio_or_pct <= 1.0 else gpu_ratio_or_pct

    power_raw = None
    for key, value in power_flat.items():
        if ("tot" in key or "total" in key or "vdd_in" in key) and (
            "power" in key or "pwr" in key or "cur" in key or "instant" in key
        ):
            power_raw = value
            break
    if power_raw is None:
        for key, value in stats_flat.items():
            if ("power" in key or "pwr" in key) and (
                "tot" in key or "total" in key or "in" in key
            ):
                power_raw = value
                break
    if power_raw is None:
        power_raw = _find_numeric(power_flat, [], any_tokens=["power", "pwr"], exclude_tokens=["avg"])

    power_w = _to_watts(power_raw)

    temp_source = temp_flat if temp_flat else stats_flat
    gpu_temp_candidates = [
        value
        for key, value in temp_source.items()
        if "gpu" in key and ("temp" in key or "temperature" in key)
    ]
    cpu_temp_candidates = [
        value
        for key, value in temp_source.items()
        if "cpu" in key and ("temp" in key or "temperature" in key)
    ]

    return {
        "ram_vram_used_mb": ram_used_mb,
        "ram_vram_used_pct": ram_pct,
        "gpu_util_pct": gpu_util_pct,
        "power_w": power_w,
        "gpu_temp_c": max(gpu_temp_candidates) if gpu_temp_candidates else None,
        "cpu_temp_c": max(cpu_temp_candidates) if cpu_temp_candidates else None,
    }


class HardwareMonitor:
    """Poll Jetson hardware telemetry in a background thread during generation."""

    def __init__(self, poll_interval_s: float = 0.5):
        self.poll_interval_s = poll_interval_s
        self._enabled = jtop is not None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._sample_count = 0
        self._ram_vram_used_mb: List[float] = []
        self._ram_vram_used_pct: List[float] = []
        self._gpu_util_pct: List[float] = []
        self._power_w: List[float] = []
        self._gpu_temp_c: List[float] = []
        self._cpu_temp_c: List[float] = []
        self._error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run,
            name="jetson-hardware-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> Dict[str, Optional[float]]:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(1.0, self.poll_interval_s * 4.0))
        return self._build_summary()

    def _run(self) -> None:
        if jtop is None:
            return

        try:
            with jtop() as jetson_obj:
                while jetson_obj.ok() and not self._stop_event.is_set():
                    sample = _extract_jetson_sample(jetson_obj)
                    with self._lock:
                        self._sample_count += 1
                        if sample["ram_vram_used_mb"] is not None:
                            self._ram_vram_used_mb.append(sample["ram_vram_used_mb"])
                        if sample["ram_vram_used_pct"] is not None:
                            self._ram_vram_used_pct.append(sample["ram_vram_used_pct"])
                        if sample["gpu_util_pct"] is not None:
                            self._gpu_util_pct.append(sample["gpu_util_pct"])
                        if sample["power_w"] is not None:
                            self._power_w.append(sample["power_w"])
                        if sample["gpu_temp_c"] is not None:
                            self._gpu_temp_c.append(sample["gpu_temp_c"])
                        if sample["cpu_temp_c"] is not None:
                            self._cpu_temp_c.append(sample["cpu_temp_c"])

                    if self._stop_event.wait(self.poll_interval_s):
                        break
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._error = str(exc)

    def _build_summary(self) -> Dict[str, Optional[float]]:
        with self._lock:
            return {
                "monitor_enabled": self._enabled,
                "sample_count": self._sample_count,
                "peak_ram_vram_used_mb": _max_or_none(self._ram_vram_used_mb),
                "peak_ram_vram_used_pct": _max_or_none(self._ram_vram_used_pct),
                "avg_gpu_util_pct": _mean_or_none(self._gpu_util_pct),
                "avg_power_w": _mean_or_none(self._power_w),
                "peak_gpu_temp_c": _max_or_none(self._gpu_temp_c),
                "peak_cpu_temp_c": _max_or_none(self._cpu_temp_c),
                "monitor_error": self._error,
            }


def load_prompts(prompts_file: Optional[str], num_prompts: int) -> List[str]:
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {prompts_file}")
    else:
        prompts = DEFAULT_PROMPTS

    if num_prompts <= len(prompts):
        return prompts[:num_prompts]

    # Repeat prompts if user asks for more than available.
    expanded = []
    while len(expanded) < num_prompts:
        expanded.extend(prompts)
    return expanded[:num_prompts]


def load_model(
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    n_batch: int,
    n_ubatch: int,
    op_offload: Optional[bool],
    seed: int,
) -> Tuple[Any, float, Dict[str, Any]]:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install it with CUDA build flags first."
        ) from exc

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # First attempt preserves user-requested settings. If context creation fails
    # due transient/peak allocation pressure, retry with a smaller batch and
    # explicit op-offload disable, which is more stable on Jetson memory budgets.
    attempts: List[Dict[str, Any]] = [
        {
            "n_batch": max(1, int(n_batch)),
            "n_ubatch": max(1, int(n_ubatch)),
            "op_offload": op_offload,
            "label": "requested",
        }
    ]

    safe_batch = max(1, min(32, int(n_ctx)))
    safe_ubatch = max(1, min(32, int(n_ctx)))
    safe_attempt = {
        "n_batch": safe_batch,
        "n_ubatch": safe_ubatch,
        "op_offload": False,
        "label": "memory_safe_fallback",
    }
    if (
        attempts[0]["n_batch"] != safe_attempt["n_batch"]
        or attempts[0]["n_ubatch"] != safe_attempt["n_ubatch"]
        or attempts[0]["op_offload"] is not False
    ):
        attempts.append(safe_attempt)

    last_exc: Optional[Exception] = None
    for idx, attempt in enumerate(attempts, start=1):
        try:
            start = time.perf_counter()
            llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                n_batch=attempt["n_batch"],
                n_ubatch=attempt["n_ubatch"],
                op_offload=attempt["op_offload"],
                seed=seed,
                verbose=False,
            )
            load_time_s = time.perf_counter() - start
            if idx > 1:
                print(
                    "load_model fallback applied: "
                    f"n_batch={attempt['n_batch']}, "
                    f"n_ubatch={attempt['n_ubatch']}, "
                    f"op_offload={attempt['op_offload']}"
                )
            return llm, load_time_s, attempt
        except Exception as exc:
            last_exc = exc
            is_ctx_alloc_error = "Failed to create llama_context" in str(exc)
            if not is_ctx_alloc_error or idx == len(attempts):
                raise
            print(
                "Model context allocation failed; retrying with memory-safe settings "
                f"({idx}/{len(attempts)})"
            )

    assert last_exc is not None
    raise last_exc


def tokenize_count(llm, text: str) -> int:
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=False)
    return len(tokens)


def run_single_prompt(
    llm,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> PromptResult:
    request_start = time.perf_counter()
    first_token_at = None
    output_chunks: List[str] = []

    monitor = HardwareMonitor(poll_interval_s=0.5)
    monitor.start()

    try:
        stream = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )

        for chunk in stream:
            text_piece = chunk.get("choices", [{}])[0].get("text", "")
            if not text_piece:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            output_chunks.append(text_piece)
    finally:
        hardware_metrics = monitor.stop()

    finished_at = time.perf_counter()
    output_text = "".join(output_chunks)

    if first_token_at is None:
        # No generated token case.
        ttft_s = finished_at - request_start
        generation_time_s = 0.0
    else:
        ttft_s = first_token_at - request_start
        generation_time_s = finished_at - first_token_at

    generated_tokens = tokenize_count(llm, output_text) if output_text else 0
    tps = generated_tokens / generation_time_s if generation_time_s > 0 else 0.0

    return PromptResult(
        prompt=prompt,
        output=output_text,
        ttft_s=ttft_s,
        generation_time_s=generation_time_s,
        generated_tokens=generated_tokens,
        tps=tps,
        peak_ram_vram_used_mb=hardware_metrics.get("peak_ram_vram_used_mb"),
        peak_ram_vram_used_pct=hardware_metrics.get("peak_ram_vram_used_pct"),
        avg_gpu_util_pct=hardware_metrics.get("avg_gpu_util_pct"),
        avg_power_w=hardware_metrics.get("avg_power_w"),
        peak_gpu_temp_c=hardware_metrics.get("peak_gpu_temp_c"),
        peak_cpu_temp_c=hardware_metrics.get("peak_cpu_temp_c"),
        hw_sample_count=int(hardware_metrics.get("sample_count") or 0),
    )


def format_mb(value: Optional[float]) -> str:
    return f"{value:.2f} MB" if value is not None else "N/A (install psutil)"


def main() -> int:
    args = parse_args()

    op_offload_map = {"auto": None, "on": True, "off": False}
    op_offload = op_offload_map[args.op_offload]

    if jtop is None:
        print(
            "Warning: jetson-stats (jtop) is not installed; "
            "hardware monitoring will be skipped."
        )
        if JTOP_IMPORT_ERROR:
            print(f"Import detail: {JTOP_IMPORT_ERROR}")

    prompts = load_prompts(args.prompts_file, args.num_prompts)

    mem_before_load = get_process_rss_mb()

    llm, load_time_s, load_cfg = load_model(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        op_offload=op_offload,
        seed=args.seed,
    )

    mem_after_load = get_process_rss_mb()

    if args.warmup:
        _ = run_single_prompt(
            llm=llm,
            prompt="Warmup: say hello.",
            max_tokens=min(16, args.max_tokens),
            temperature=0.0,
            top_p=1.0,
        )

    results: List[PromptResult] = []
    for idx, prompt in enumerate(prompts, start=1):
        result = run_single_prompt(
            llm=llm,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        results.append(result)

        print(f"\\n--- Prompt {idx} ---")
        print(f"Prompt: {prompt}")
        print(f"TTFT: {result.ttft_s:.3f} s")
        print(f"Generation time: {result.generation_time_s:.3f} s")
        print(f"Generated tokens: {result.generated_tokens}")
        print(f"TPS: {result.tps:.2f} tokens/s")
        print(f"Jetson samples: {result.hw_sample_count}")
        print(
            "Jetson RAM/VRAM peak: "
            f"{_format_optional(result.peak_ram_vram_used_mb, ' MB')} "
            f"({_format_optional(result.peak_ram_vram_used_pct, '%')})"
        )
        print(f"Jetson avg GPU util: {_format_optional(result.avg_gpu_util_pct, '%')}")
        print(f"Jetson avg power: {_format_optional(result.avg_power_w, ' W')}")
        print(f"Jetson peak GPU temp: {_format_optional(result.peak_gpu_temp_c, ' C')}")
        print(f"Jetson peak CPU temp: {_format_optional(result.peak_cpu_temp_c, ' C')}")
        print(f"Output preview: {result.output[:200].replace(chr(10), ' ')}")

    mem_after_run = get_process_rss_mb()
    peak_rss_mb = get_peak_rss_mb_linux()

    avg_ttft = statistics.mean(r.ttft_s for r in results)
    avg_tps = statistics.mean(r.tps for r in results)
    total_generated_tokens = sum(r.generated_tokens for r in results)
    avg_generation_time_s = statistics.mean(r.generation_time_s for r in results)
    total_hw_samples = sum(r.hw_sample_count for r in results)

    jetson_hw_summary = {
        "monitor_enabled": jtop is not None,
        "sample_count_total": total_hw_samples,
        "peak_ram_vram_used_mb": _max_or_none([r.peak_ram_vram_used_mb for r in results]),
        "peak_ram_vram_used_pct": _max_or_none([r.peak_ram_vram_used_pct for r in results]),
        "avg_gpu_util_pct": _mean_or_none([r.avg_gpu_util_pct for r in results]),
        "avg_power_w": _mean_or_none([r.avg_power_w for r in results]),
        "peak_gpu_temp_c": _max_or_none([r.peak_gpu_temp_c for r in results]),
        "peak_cpu_temp_c": _max_or_none([r.peak_cpu_temp_c for r in results]),
    }

    summary = {
        "model_name": args.model_name or os.path.basename(args.model),
        "model_path": os.path.abspath(args.model),
        "model_load_time_s": load_time_s,
        "avg_ttft_s": avg_ttft,
        "avg_tps": avg_tps,
        "avg_generation_time_s": avg_generation_time_s,
        "total_generated_tokens": total_generated_tokens,
        "num_prompts": len(results),
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "n_batch": load_cfg["n_batch"],
        "n_ubatch": load_cfg["n_ubatch"],
        "op_offload": load_cfg["op_offload"],
        "load_profile": load_cfg["label"],
        "max_tokens": args.max_tokens,
        "memory": {
            "process_rss_before_load_mb": mem_before_load,
            "process_rss_after_load_mb": mem_after_load,
            "process_rss_after_run_mb": mem_after_run,
            "peak_process_rss_mb": peak_rss_mb,
        },
        "jetson_hardware": jetson_hw_summary,
        "per_prompt": [
            {
                "prompt": r.prompt,
                "ttft_s": r.ttft_s,
                "generation_time_s": r.generation_time_s,
                "generated_tokens": r.generated_tokens,
                "tps": r.tps,
                "peak_ram_vram_used_mb": r.peak_ram_vram_used_mb,
                "peak_ram_vram_used_pct": r.peak_ram_vram_used_pct,
                "avg_gpu_util_pct": r.avg_gpu_util_pct,
                "avg_power_w": r.avg_power_w,
                "peak_gpu_temp_c": r.peak_gpu_temp_c,
                "peak_cpu_temp_c": r.peak_cpu_temp_c,
                "hw_sample_count": r.hw_sample_count,
            }
            for r in results
        ],
    }

    print("\\n=== Benchmark Summary ===")
    print(f"Model: {summary['model_name']}")
    print(f"Model load time: {load_time_s:.3f} s")
    print(f"Average TTFT: {avg_ttft:.3f} s")
    print(f"Average TPS: {avg_tps:.2f} tokens/s")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Process RSS before load: {format_mb(mem_before_load)}")
    print(f"Process RSS after load: {format_mb(mem_after_load)}")
    print(f"Process RSS after run: {format_mb(mem_after_run)}")
    print(f"Peak process RSS: {peak_rss_mb:.2f} MB")
    print(f"Jetson sample count: {total_hw_samples}")
    print(
        "Jetson RAM/VRAM peak: "
        f"{_format_optional(jetson_hw_summary['peak_ram_vram_used_mb'], ' MB')} "
        f"({_format_optional(jetson_hw_summary['peak_ram_vram_used_pct'], '%')})"
    )
    print(
        "Jetson avg GPU util: "
        f"{_format_optional(jetson_hw_summary['avg_gpu_util_pct'], '%')}"
    )
    print(
        "Jetson avg power: "
        f"{_format_optional(jetson_hw_summary['avg_power_w'], ' W')}"
    )
    print(
        "Jetson peak GPU temp: "
        f"{_format_optional(jetson_hw_summary['peak_gpu_temp_c'], ' C')}"
    )
    print(
        "Jetson peak CPU temp: "
        f"{_format_optional(jetson_hw_summary['peak_cpu_temp_c'], ' C')}"
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved JSON summary: {args.json_out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
