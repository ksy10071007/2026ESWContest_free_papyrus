"""
[설정] 눈병 진단 시스템 중앙 관리
- IP주소, 모델 경로, 임계값, 클래스명
"""

import os
import torch


def _get_env_str(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    value = str(value).strip()
    return value if value != '' else default


def _get_env_int(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == '':
        return int(default)
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _get_env_float(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == '':
        return float(default)
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def _get_env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return bool(default)

    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if text in ('0', 'false', 'no', 'n', 'off'):
        return False
    return bool(default)

# ========================================
# [1] 기본 경로 설정
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def _cuda_failure_message(requested, cuda_index, reason):
    return (
        f"PyTorch device request failed: requested={requested}, cuda_index={cuda_index}, "
        f"reason={reason}, torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
        f"cuda_available={torch.cuda.is_available()}, cuda_device_count={torch.cuda.device_count()}. "
        "Jetson에서는 현재 JetPack/L4T와 호환되는 NVIDIA PyTorch 및 torchvision "
        "빌드가 설치되어 있는지 확인하세요."
    )


def _probe_cuda_device(device):
    """Run a small real CUDA operation before accepting a CUDA device."""
    with torch.cuda.device(device):
        probe = torch.ones(4, device=device, dtype=torch.float32)
        result = (probe * 2).sum()
        torch.cuda.synchronize(device)
    if float(result.cpu()) != 8.0:
        raise RuntimeError('CUDA tensor probe returned an unexpected result')


def resolve_torch_device(requested=None, cuda_index=None):
    """Resolve cpu/auto/cuda policy and fail explicitly for unusable CUDA."""
    requested_value = requested if requested is not None else os.getenv('TORCH_DEVICE', 'auto')
    requested_value = str(requested_value).strip().lower()
    if requested_value not in ('auto', 'cuda', 'cpu'):
        raise ValueError(
            f"Invalid TORCH_DEVICE={requested_value!r}; expected one of: auto, cuda, cpu"
        )

    if cuda_index is None:
        raw_index = os.getenv('CUDA_DEVICE_INDEX', '0')
        try:
            cuda_index = int(str(raw_index).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid CUDA_DEVICE_INDEX={raw_index!r}; expected a non-negative integer") from exc
    if cuda_index < 0:
        raise ValueError(f"Invalid CUDA_DEVICE_INDEX={cuda_index}; expected a non-negative integer")

    if requested_value == 'cpu':
        return torch.device('cpu')

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    if not cuda_available or device_count <= cuda_index:
        if requested_value == 'auto':
            return torch.device('cpu')
        reason = 'CUDA is unavailable' if not cuda_available else 'requested GPU index was not detected'
        raise RuntimeError(_cuda_failure_message(requested_value, cuda_index, reason))

    device = torch.device(f'cuda:{cuda_index}')
    try:
        _probe_cuda_device(device)
    except Exception as exc:
        if requested_value == 'auto':
            print(f"[Inference] CUDA probe failed in auto mode; using CPU: {exc}")
            return torch.device('cpu')
        raise RuntimeError(_cuda_failure_message(requested_value, cuda_index, str(exc))) from exc
    return device


# ========================================
# [1-1] PyTorch inference device
# ========================================
TORCH_DEVICE_REQUESTED = _get_env_str('TORCH_DEVICE', 'auto').lower()
CUDA_DEVICE_INDEX = _get_env_int('CUDA_DEVICE_INDEX', 0)
CUDA_EMPTY_CACHE_AFTER_ANALYSIS = _get_env_bool('CUDA_EMPTY_CACHE_AFTER_ANALYSIS', False)
DEVICE = resolve_torch_device(TORCH_DEVICE_REQUESTED, CUDA_DEVICE_INDEX)

print(
    f"[Inference] requested_device={TORCH_DEVICE_REQUESTED} "
    f"resolved_device={DEVICE}"
)
if DEVICE.type == 'cuda':
    print(f"[Inference] cuda_device={torch.cuda.get_device_name(DEVICE.index or 0)}")

# ========================================
# [2] 모델 경로
# ========================================
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, 'set_1000_YOLO26s_best.pt')  # YOLO eye detector 모델
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, 'Augmented_EffNet_V1_B0_best.pth')

# ========================================
# [3] 서버 설정
# ========================================
SERVER_IP = _get_env_str('SERVER_IP', _get_env_str('SERVER_HOST', '0.0.0.0'))
SERVER_PORT = _get_env_int('SERVER_PORT', 5000)
DEBUG_MODE = _get_env_bool('DEBUG_MODE', False)

# ========================================
# [3-1] 카메라 설정
# ========================================
CAMERA_DEVICE_INDEX = _get_env_int('CAMERA_DEVICE_INDEX', 0)
MICROSCOPE_CAMERA_DEVICE_INDEX = _get_env_int(
    'MICROSCOPE_CAMERA_DEVICE_INDEX',
    CAMERA_DEVICE_INDEX,
)

# ========================================
# [4] YOLO 검출 임계값
# ========================================
YOLO_CONF_THRESHOLD = _get_env_float('MEDIAPIPE_CONF_THRESHOLD', 0.5)
YOLO_IOU_THRESHOLD = _get_env_float('MEDIAPIPE_IOU_THRESHOLD', 0.45)
YOLO_INPUT_SIZE = _get_env_int('MEDIAPIPE_INPUT_SIZE', 640)
YOLO_STATUS_CONF_THRESHOLD = _get_env_float('MEDIAPIPE_STATUS_CONF_THRESHOLD', 0.25)

# ========================================
# [5] 분류 모델 설정
# ========================================
CLASSIFIER_INPUT_SIZE = (224, 224)       # EfficientNet 입력 크기
CLASSIFIER_CONFIDENCE_THRESHOLD = _get_env_float('CLASSIFIER_CONFIDENCE_THRESHOLD', 0.7)

# ========================================
# [6] 질환 분류 클래스 (5개)
# ========================================
DISEASE_CLASSES = {
    0: '결막염 (Conjunctivitis)',
    1: '다래끼 (Eyelid)',
    2: '백내장 (Cataract)',
    3: '일반 (Normal)',
    4: '포도막염 (Uveitis)'
}

# ========================================
# [7] 홍채 제거 설정
# ========================================
IRIS_REMOVAL_ENABLED = _get_env_bool('IRIS_REMOVAL_ENABLED', True)
IRIS_THRESHOLD = _get_env_float('IRIS_THRESHOLD', 0.3)

# ========================================
# [8] 로깅 설정
# ========================================
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FORMAT = 'csv'               # 'csv' 또는 'db'
os.makedirs(LOG_DIR, exist_ok=True)

# ========================================
# [9] 이미지 처리 설정
# ========================================
IMAGE_SAVE_DIR = os.path.join(BASE_DIR, 'web', 'static', 'captures')
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# ========================================
# [10] 자동 촬영 설정
# ========================================
# 중심점 거리 임계값: 중심점이 가이드라인 중심으로부터 
# 30픽셀 이내일 때 자동 촬영 준비
AUTO_DIST_THRESHOLD = _get_env_int('AUTO_DIST_THRESHOLD', 30)

# 눈 크기 비율 임계값: 가이드라인 대비 
# 눈의 크기가 이 범위 내에 있을 때 적절한 위치로 판단
AUTO_SCALE_MIN = _get_env_float('AUTO_SCALE_MIN', 0.8)
AUTO_SCALE_MAX = _get_env_float('AUTO_SCALE_MAX', 1.1)

# 자동 촬영 대기 프레임: 조건을 만족한 후 
# 이 프레임 수만큼 유지되면 자동 촬영
AUTO_CAPTURE_HOLD_FRAMES = _get_env_int('AUTO_CAPTURE_HOLD_FRAMES', 10)
