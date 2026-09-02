#!/usr/bin/env python3
"""Jetson CUDA, model, database, USB UVC, and service preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
ENV_PATH = PROJECT_ROOT / '.env'


@dataclass
class Check:
    state: str
    label: str
    detail: str = ''


class Reporter:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, state: str, label: str, detail: str = '') -> None:
        check = Check(state, label, detail)
        self.checks.append(check)
        suffix = f': {detail}' if detail else ''
        print(f'{state}: {label}{suffix}')

    def pass_(self, label: str, detail: str = '') -> None:
        self.add('PASS', label, detail)

    def fail(self, label: str, detail: str) -> None:
        self.add('FAIL', label, detail)

    def skip(self, label: str, detail: str) -> None:
        self.add('SKIP', label, detail)

    @property
    def failed(self) -> bool:
        return any(check.state == 'FAIL' for check in self.checks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='MediFlow Jetson deployment preflight')
    parser.add_argument('--allow-no-camera', action='store_true')
    parser.add_argument('--service-smoke-test', action='store_true')
    return parser.parse_args()


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        from dotenv import dotenv_values
        return {
            str(key): str(value or '')
            for key, value in dotenv_values(path).items()
            if key is not None
        }
    except Exception:
        values = {}
        for raw_line in path.read_text(encoding='utf-8-sig').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values


def effective_setting(env_file: dict[str, str], name: str, default: str = '') -> str:
    value = os.environ.get(name)
    if value is None:
        value = env_file.get(name, default)
    return str(value or '').strip()


def command_output(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    return completed.stdout.strip()


def check_system(reporter: Reporter) -> None:
    failures = []
    machine = platform.machine()
    if machine != 'aarch64':
        failures.append(f'architecture={machine}')
    l4t_release = Path('/etc/nv_tegra_release')
    l4t_core = command_output(['dpkg-query', '-W', '-f=${Version}', 'nvidia-l4t-core'])
    if not l4t_release.is_file() and not l4t_core:
        failures.append('NVIDIA L4T not detected')
    memory_gib = 0.0
    for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
        if line.startswith('MemTotal:'):
            memory_gib = int(line.split()[1]) / 1024 / 1024
            break
    disk_gib = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    details = (
        f'arch={machine}, Python={platform.python_version()}, '
        f'L4T={l4t_core or "detected"}, RAM={memory_gib:.1f}GiB, free={disk_gib:.1f}GiB'
    )
    if failures:
        reporter.fail('system', '; '.join(failures) + '; ' + details)
    else:
        reporter.pass_('system', details)

    video_paths = sorted(Path('/dev').glob('video*'))
    ports = []
    for port in (5000, 5001):
        with socket.socket() as probe:
            ports.append(f'{port}={"open" if probe.connect_ex(("127.0.0.1", port)) == 0 else "free"}')
    print(f'[INFO] USB video nodes={", ".join(map(str, video_paths)) or "none"}; ports={", ".join(ports)}')


def check_environment(reporter: Reporter, env_file: dict[str, str]) -> None:
    if not ENV_PATH.is_file():
        reporter.fail('environment', f'missing {ENV_PATH}')
        return
    required_values = {
        'MODEL_DEVICE': 'jetson',
        'TORCH_DEVICE': 'cuda',
    }
    problems = []
    for key, expected in required_values.items():
        actual = effective_setting(env_file, key)
        if actual.lower() != expected:
            problems.append(f'{key} must be {expected} (current={actual or "missing"})')
    for key in ('HASH_PEPPER', 'EYE_APP_SECRET_KEY'):
        if not effective_setting(env_file, key):
            problems.append(f'{key} is missing or empty')
    cuda_index = effective_setting(env_file, 'CUDA_DEVICE_INDEX', '0')
    try:
        if int(cuda_index) < 0:
            raise ValueError
    except ValueError:
        problems.append('CUDA_DEVICE_INDEX must be a non-negative integer')
    if problems:
        reporter.fail('environment', '; '.join(problems))
    else:
        reporter.pass_('environment', 'required deployment keys are set; secret values hidden')


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return 'unknown'


def check_dependencies(reporter: Reporter) -> None:
    imports = {
        'flask': 'flask',
        'python-dotenv': 'dotenv',
        'requests': 'requests',
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'Pillow': 'PIL',
        'mediapipe': 'mediapipe',
        'grad-cam': 'pytorch_grad_cam',
        'qrcode': 'qrcode',
        'reportlab': 'reportlab',
        'torch': 'torch',
        'torchvision': 'torchvision',
    }
    loaded = []
    try:
        for package_name, module_name in imports.items():
            importlib.import_module(module_name)
            loaded.append(f'{package_name}={package_version(package_name)}')
    except Exception as exc:
        reporter.fail('python dependencies', f'{module_name}: {exc}')
        return
    reporter.pass_('python dependencies', ', '.join(loaded))


def check_cuda(reporter: Reporter):
    try:
        import torch
        import torchvision
        assert torch.cuda.is_available(), 'torch.cuda.is_available() is false'
        assert torch.cuda.device_count() >= 1, 'no CUDA devices detected'
        device_index = int(os.environ.get('CUDA_DEVICE_INDEX', '0'))
        assert device_index < torch.cuda.device_count(), 'configured CUDA index is unavailable'
        device = torch.device(f'cuda:{device_index}')
        tensor = torch.arange(1024, device=device, dtype=torch.float32)
        result = (tensor * 2).sum()
        torch.cuda.synchronize(device)
        assert result.is_cuda
        properties = torch.cuda.get_device_properties(device)
        detail = (
            f'torch={torch.__version__}, torchvision={torchvision.__version__}, '
            f'torch_cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version()}, '
            f'device={properties.name}, memory={properties.total_memory / 1024**3:.1f}GiB, '
            f'capability={properties.major}.{properties.minor}'
        )
        reporter.pass_('CUDA tensor operation', detail)
        return device
    except Exception as exc:
        reporter.fail('CUDA tensor operation', str(exc))
        return None


def check_model(reporter: Reporter, device) -> None:
    checkpoint_path = PROJECT_ROOT / 'models' / 'Augmented_EffNet_V1_B0_best.pth'
    sums_path = PROJECT_ROOT / 'models' / 'SHA256SUMS'
    try:
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
            raise RuntimeError(f'missing or empty checkpoint: {checkpoint_path}')
        expected = None
        for line in sums_path.read_text(encoding='ascii').splitlines():
            digest, filename = line.split(maxsplit=1)
            if filename.lstrip('*') == checkpoint_path.name:
                expected = digest
                break
        if expected is None:
            raise RuntimeError('checkpoint is absent from SHA256SUMS')
        actual = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'SHA-256 mismatch: expected={expected}, actual={actual}')

        import numpy as np
        from modules.classifier import DiseaseClassifier
        classifier = DiseaseClassifier(str(checkpoint_path))
        parameter_device = next(classifier.model.parameters()).device
        if device is None or parameter_device != device:
            raise RuntimeError(f'model parameter device is {parameter_device}, expected {device}')
        dummy = np.random.default_rng(7).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        result = classifier.classify_with_details(dummy, generate_cam=True)
        probabilities = result['probabilities']
        if len(probabilities) != 5 or abs(sum(probabilities) - 1.0) > 1e-4:
            raise RuntimeError('classifier probability output is invalid')
        reporter.pass_('classifier on cuda:0', f'parameter_device={parameter_device}, sha256={actual}')
        if result['heatmap_image'] is None or result['heatmap_image'].shape != (224, 224, 3):
            reporter.fail('Grad-CAM', 'no valid 224x224x3 overlay was generated')
        else:
            reporter.pass_('Grad-CAM', f'shape={result["heatmap_image"].shape}')
    except Exception as exc:
        reporter.fail('classifier on cuda:0', str(exc))
        reporter.fail('Grad-CAM', 'classifier initialization or inference failed')


def check_mediapipe(reporter: Reporter) -> None:
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        face_mesh.close()
        reporter.pass_('MediaPipe Face Mesh', 'initialization passed; no repository face fixture available')
    except Exception as exc:
        reporter.fail('MediaPipe Face Mesh', str(exc))


def check_database(reporter: Reporter) -> None:
    try:
        from database.db import identifier_hash_id, init_db
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / 'preflight.db')
            init_db(db_path)
            identifier = identifier_hash_id('preflight-user')
            connection = sqlite3.connect(db_path)
            try:
                connection.execute('PRAGMA foreign_keys=ON')
                connection.execute(
                    'INSERT INTO users (phone_hash, display_name) VALUES (?, ?)',
                    (identifier, 'preflight'),
                )
                row = connection.execute(
                    'SELECT phone_hash FROM users WHERE phone_hash=?',
                    (identifier,),
                ).fetchone()
                connection.commit()
            finally:
                connection.close()
            assert row is not None and row[0] == identifier
        reporter.pass_('database temporary test', 'schema, hash, insert, and read passed')
    except Exception as exc:
        reporter.fail('database temporary test', str(exc))


def camera_formats(device_path: Path) -> str:
    if shutil.which('v4l2-ctl') is None:
        return 'v4l2-ctl unavailable'
    output = command_output(['v4l2-ctl', f'--device={device_path}', '--list-formats-ext'])
    formats = []
    for token in ('MJPG', 'YUYV', '640x480'):
        if token in output:
            formats.append(token)
    return ','.join(formats) or 'no MJPG/YUYV 640x480 capability detected'


def check_cameras(reporter: Reporter, env_file: dict[str, str], allow_no_camera: bool) -> None:
    from utils.uvc_camera import open_usb_uvc_camera
    indices = []
    for key, default in (('CAMERA_DEVICE_INDEX', '0'), ('MICROSCOPE_CAMERA_DEVICE_INDEX', '1')):
        try:
            index = int(effective_setting(env_file, key, default))
        except ValueError:
            reporter.fail(f'USB camera from {key}', 'index must be an integer')
            continue
        if index not in indices:
            indices.append(index)

    for index in indices:
        path = Path(f'/dev/video{index}')
        label = f'USB camera {path}'
        if not path.exists():
            if allow_no_camera:
                reporter.skip(label, 'device is not connected (--allow-no-camera)')
            else:
                reporter.fail(label, 'device does not exist')
            continue
        if not os.access(path, os.R_OK | os.W_OK):
            reporter.fail(label, 'current user lacks read/write access')
            continue
        name_path = Path(f'/sys/class/video4linux/video{index}/name')
        camera_name = name_path.read_text(encoding='utf-8').strip() if name_path.is_file() else 'unknown'
        capture = None
        try:
            capture, frame, metadata = open_usb_uvc_camera(index)
            reporter.pass_(
                label,
                f'name={camera_name}, backend={metadata["backend"]}, frame={frame.shape}, '
                f'format_support={camera_formats(path)}',
            )
        except Exception as exc:
            reporter.fail(label, str(exc))
        finally:
            if capture is not None:
                capture.release()


def check_optional_features(reporter: Reporter, env_file: dict[str, str]) -> None:
    if effective_setting(env_file, 'OPENAI_API_KEY') or effective_setting(env_file, 'GEMINI_API_KEY'):
        reporter.pass_('optional LLM credentials', 'at least one provider key is configured; API call not performed')
    else:
        reporter.skip('optional LLM credentials', 'chat provider key is not configured')
    if effective_setting(env_file, 'KAKAO_CLIENT_ID') and effective_setting(env_file, 'KAKAO_REFRESH_TOKEN'):
        reporter.pass_('optional Kakao credentials', 'configuration present; external API call not performed')
    else:
        reporter.skip('optional Kakao credentials', 'OAuth/send credentials are incomplete')


def check_service_smoke(reporter: Reporter) -> None:
    manager = PROJECT_ROOT / 'scripts' / 'mediflow-kiosk'
    status = subprocess.run(
        [str(manager), 'status'],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )
    already_running = '[MediFlow Kiosk] status: running' in status.stdout
    started_here = False
    try:
        if not already_running:
            started = subprocess.run(
                [str(manager), 'start'],
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=120,
            )
            if started.returncode != 0:
                raise RuntimeError(started.stdout.strip())
            started_here = True
        import urllib.request
        with urllib.request.urlopen('http://127.0.0.1:5000/status', timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f'health HTTP status={response.status}')
        reporter.pass_('service health check', 'existing service preserved' if already_running else 'started and reached /status')
    except Exception as exc:
        reporter.fail('service health check', str(exc))
    finally:
        if started_here:
            subprocess.run(
                [str(manager), 'stop'],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    env_file = parse_env_file(ENV_PATH)
    for key, value in env_file.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault('MODEL_DEVICE', 'jetson')
    os.environ.setdefault('TORCH_DEVICE', 'cuda')
    os.environ.setdefault('CUDA_DEVICE_INDEX', '0')

    reporter = Reporter()
    check_system(reporter)
    check_environment(reporter, env_file)
    check_dependencies(reporter)
    device = check_cuda(reporter)
    check_model(reporter, device)
    check_mediapipe(reporter)
    check_database(reporter)
    check_cameras(reporter, env_file, args.allow_no_camera)
    check_optional_features(reporter, env_file)
    if args.service_smoke_test:
        check_service_smoke(reporter)
    else:
        reporter.skip('service health check', 'use --service-smoke-test to run')

    print()
    print(f'OVERALL: {"NOT READY" if reporter.failed else "READY"}')
    return 1 if reporter.failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
