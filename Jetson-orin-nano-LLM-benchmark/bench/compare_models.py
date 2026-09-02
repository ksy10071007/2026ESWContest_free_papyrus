#!/usr/bin/env python3
"""Run benchmark.py across multiple GGUF models and compare results.

This script executes each model in a separate process to avoid memory carry-over.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelSpec:
    name: str
    path: str
    n_gpu_layers: Optional[int] = None
    n_ctx: Optional[int] = None
    max_tokens: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple GGUF models on same prompts.")
    parser.add_argument(
        "--models-file",
        required=True,
        help="CSV file with columns: name,path[,n_gpu_layers,n_ctx,max_tokens]",
    )
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-gpu-layers", type=int, default=35)
    parser.add_argument("--n-threads", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument(
        "--op-offload",
        choices=["auto", "on", "off"],
        default="auto",
        help="Operator offload mode passed through to benchmark.py",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--output-dir", default="outputs/general_benchmark")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def to_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return int(value)


def load_model_specs(models_file: str) -> List[ModelSpec]:
    if not os.path.isfile(models_file):
        raise FileNotFoundError(f"Models CSV not found: {models_file}")

    specs: List[ModelSpec] = []
    with open(models_file, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Models CSV is empty.")

        fieldnames = {name.strip() for name in reader.fieldnames}
        required = {"name", "path"}
        if not required.issubset(fieldnames):
            raise ValueError("Models CSV must include headers: name,path")

        for row in reader:
            name = (row.get("name") or "").strip()
            path = (row.get("path") or "").strip()
            if not name or not path:
                continue
            specs.append(
                ModelSpec(
                    name=name,
                    path=path,
                    n_gpu_layers=to_optional_int(row.get("n_gpu_layers")),
                    n_ctx=to_optional_int(row.get("n_ctx")),
                    max_tokens=to_optional_int(row.get("max_tokens")),
                )
            )

    if not specs:
        raise ValueError(f"No valid model rows found in {models_file}")

    return specs


def build_cmd(
    benchmark_script: str,
    spec: ModelSpec,
    args: argparse.Namespace,
    json_out: str,
    n_gpu_layers_override: Optional[int] = None,
) -> List[str]:
    model_max_tokens = spec.max_tokens if spec.max_tokens is not None else args.max_tokens
    model_n_ctx = spec.n_ctx if spec.n_ctx is not None else args.n_ctx
    model_n_gpu_layers = (
        spec.n_gpu_layers if spec.n_gpu_layers is not None else args.n_gpu_layers
    )
    if n_gpu_layers_override is not None:
        model_n_gpu_layers = n_gpu_layers_override

    cmd = [
        sys.executable,
        benchmark_script,
        "--model",
        spec.path,
        "--model-name",
        spec.name,
        "--num-prompts",
        str(args.num_prompts),
        "--max-tokens",
        str(model_max_tokens),
        "--n-ctx",
        str(model_n_ctx),
        "--n-gpu-layers",
        str(model_n_gpu_layers),
        "--n-threads",
        str(args.n_threads),
        "--n-batch",
        str(args.n_batch),
        "--n-ubatch",
        str(args.n_ubatch),
        "--op-offload",
        args.op_offload,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--json-out",
        json_out,
    ]

    if args.prompts_file:
        cmd.extend(["--prompts-file", args.prompts_file])
    if args.warmup:
        cmd.append("--warmup")

    return cmd


def build_gpu_retry_schedule(initial_layers: int) -> List[int]:
    """Create a conservative retry schedule that degrades GPU offload on failure."""
    schedule = [max(0, initial_layers)]

    current = max(0, initial_layers)
    while current > 0:
        # Large step down first, then converge to CPU-only fallback.
        current = max(0, current - 4)
        if current not in schedule:
            schedule.append(current)

    return schedule


def format_optional_float(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def main() -> int:
    args = parse_args()
    specs = load_model_specs(args.models_file)
    benchmark_script = os.path.join(os.path.dirname(__file__), "benchmark.py")
    output_csv = args.output_csv or os.path.join(args.output_dir, "comparison_results.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results = []

    for idx, spec in enumerate(specs, start=1):
        print(f"\n=== [{idx}/{len(specs)}] Benchmarking: {spec.name} ===")
        started = time.perf_counter()

        with tempfile.NamedTemporaryFile(
            prefix="bench_", suffix=".json", delete=False
        ) as temp_file:
            json_out = temp_file.name

        initial_n_gpu_layers = (
            spec.n_gpu_layers if spec.n_gpu_layers is not None else args.n_gpu_layers
        )
        retry_schedule = build_gpu_retry_schedule(initial_n_gpu_layers)

        proc = None
        selected_n_gpu_layers = None

        for attempt, n_gpu_layers in enumerate(retry_schedule, start=1):
            if attempt > 1:
                print(
                    f"Retrying {spec.name} with lower n_gpu_layers={n_gpu_layers} "
                    f"(attempt {attempt}/{len(retry_schedule)})"
                )

            cmd = build_cmd(
                benchmark_script=benchmark_script,
                spec=spec,
                args=args,
                json_out=json_out,
                n_gpu_layers_override=n_gpu_layers,
            )
            proc = subprocess.run(cmd, text=True, capture_output=True)

            if proc.stdout:
                print(proc.stdout.strip())
            if proc.stderr:
                print(proc.stderr.strip(), file=sys.stderr)

            if proc.returncode == 0:
                selected_n_gpu_layers = n_gpu_layers
                break

        elapsed = time.perf_counter() - started

        if proc is None or proc.returncode != 0:
            error_row = {
                "name": spec.name,
                "path": spec.path,
                "status": "failed",
                "error": f"benchmark.py exited with code {proc.returncode}",
                "model_load_time_s": "",
                "avg_ttft_s": "",
                "avg_tps": "",
                "peak_process_rss_mb": "",
                "avg_gpu_util_pct": "",
                "avg_power_w": "",
                "peak_ram_vram_used_mb": "",
                "peak_ram_vram_used_pct": "",
                "peak_gpu_temp_c": "",
                "peak_cpu_temp_c": "",
                "elapsed_wall_s": f"{elapsed:.3f}",
            }
            results.append(error_row)
            print(f"FAILED: {spec.name}")
            if args.fail_fast:
                break
            continue

        try:
            with open(json_out, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except Exception as exc:
            error_row = {
                "name": spec.name,
                "path": spec.path,
                "status": "failed",
                "error": f"failed to read JSON summary: {exc}",
                "model_load_time_s": "",
                "avg_ttft_s": "",
                "avg_tps": "",
                "peak_process_rss_mb": "",
                "avg_gpu_util_pct": "",
                "avg_power_w": "",
                "peak_ram_vram_used_mb": "",
                "peak_ram_vram_used_pct": "",
                "peak_gpu_temp_c": "",
                "peak_cpu_temp_c": "",
                "elapsed_wall_s": f"{elapsed:.3f}",
            }
            results.append(error_row)
            print(f"FAILED: {spec.name}")
            if args.fail_fast:
                break
            continue
        finally:
            try:
                os.remove(json_out)
            except OSError:
                pass

        ok_row = {
            "name": summary.get("model_name", spec.name),
            "path": summary.get("model_path", spec.path),
            "status": "ok",
            "error": "",
            "model_load_time_s": f"{summary.get('model_load_time_s', 0.0):.3f}",
            "avg_ttft_s": f"{summary.get('avg_ttft_s', 0.0):.3f}",
            "avg_tps": f"{summary.get('avg_tps', 0.0):.3f}",
            "peak_process_rss_mb": f"{summary.get('memory', {}).get('peak_process_rss_mb', 0.0):.2f}",
            "avg_gpu_util_pct": format_optional_float(
                summary.get("jetson_hardware", {}).get("avg_gpu_util_pct"),
                precision=3,
            ),
            "avg_power_w": format_optional_float(
                summary.get("jetson_hardware", {}).get("avg_power_w"),
                precision=3,
            ),
            "peak_ram_vram_used_mb": format_optional_float(
                summary.get("jetson_hardware", {}).get("peak_ram_vram_used_mb"),
                precision=2,
            ),
            "peak_ram_vram_used_pct": format_optional_float(
                summary.get("jetson_hardware", {}).get("peak_ram_vram_used_pct"),
                precision=2,
            ),
            "peak_gpu_temp_c": format_optional_float(
                summary.get("jetson_hardware", {}).get("peak_gpu_temp_c"),
                precision=2,
            ),
            "peak_cpu_temp_c": format_optional_float(
                summary.get("jetson_hardware", {}).get("peak_cpu_temp_c"),
                precision=2,
            ),
            "elapsed_wall_s": f"{elapsed:.3f}",
        }
        results.append(ok_row)
        if selected_n_gpu_layers is not None and selected_n_gpu_layers != initial_n_gpu_layers:
            print(
                f"Recovered {spec.name} by lowering n_gpu_layers "
                f"{initial_n_gpu_layers} -> {selected_n_gpu_layers}"
            )
        print(
            f"OK: {ok_row['name']} | TTFT={ok_row['avg_ttft_s']}s | "
            f"TPS={ok_row['avg_tps']} | PeakRSS={ok_row['peak_process_rss_mb']}MB | "
            f"GPU%={ok_row['avg_gpu_util_pct'] or 'N/A'} | "
            f"PowerW={ok_row['avg_power_w'] or 'N/A'}"
        )

    out_fields = [
        "name",
        "path",
        "status",
        "error",
        "model_load_time_s",
        "avg_ttft_s",
        "avg_tps",
        "peak_process_rss_mb",
        "avg_gpu_util_pct",
        "avg_power_w",
        "peak_ram_vram_used_mb",
        "peak_ram_vram_used_pct",
        "peak_gpu_temp_c",
        "peak_cpu_temp_c",
        "elapsed_wall_s",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(results)

    ok_results = [r for r in results if r["status"] == "ok"]
    ok_sorted = sorted(ok_results, key=lambda row: float(row["avg_tps"]), reverse=True)

    print("\n=== Comparison Summary (sorted by TPS desc) ===")
    if not ok_sorted:
        print("No successful benchmark runs.")
    else:
        for i, row in enumerate(ok_sorted, start=1):
            print(
                f"{i}. {row['name']}: TPS={row['avg_tps']} | TTFT={row['avg_ttft_s']}s | PeakRSS={row['peak_process_rss_mb']}MB"
            )

    print(f"\nSaved comparison CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
