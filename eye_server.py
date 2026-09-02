    
def normalize_selected_eye(value):
    text = str(value or '').strip().upper()
    if text in ('L', 'LEFT', 'LEFT_EYE'):
        return 'L'
    if text in ('R', 'RIGHT', 'RIGHT_EYE'):
        return 'R'
    return None

def apply_selected_eye_mapping(analysis, selected_eye):
    side = normalize_selected_eye(selected_eye)
    if side not in ('L', 'R'):
        return analysis

    left_eye = analysis.get('left_eye')
    right_eye = analysis.get('right_eye')
    has_left = bool(left_eye)
    has_right = bool(right_eye)

    if has_left and has_right:
        return analysis

    detected_side = None
    detected_payload = None
    if has_left:
        detected_side = 'L'
        detected_payload = left_eye
    elif has_right:
        detected_side = 'R'
        detected_payload = right_eye
    else:
        return analysis

    if detected_side == side:
        return analysis

    meta = analysis.get('meta') or {}

    # fallback ROI로 만든 임시 검출은 반대쪽 눈으로 강제 리매핑하지 않는다.
    if bool((detected_payload or {}).get('fallback')):
        meta['selected_eye'] = side
        meta['side_remapped_by_selected_eye'] = False
        meta['side_remap_skipped_reason'] = 'fallback_detection'
        analysis['meta'] = meta
        return analysis

    # 검출 신뢰도가 낮으면 리매핑하지 않아 반대쪽 눈 허위 생성 방지
    det_conf = (detected_payload or {}).get('detection_confidence')
    if det_conf is not None:
        try:
            conf_pct = float(det_conf)
            if conf_pct <= 1.0:
                conf_pct *= 100.0
            min_conf_pct = max(float(config.YOLO_STATUS_CONF_THRESHOLD) * 100.0, 35.0)
            if conf_pct < min_conf_pct:
                meta['selected_eye'] = side
                meta['side_remapped_by_selected_eye'] = False
                meta['side_remap_skipped_reason'] = 'low_detection_confidence'
                meta['detected_confidence_pct'] = round(conf_pct, 2)
                analysis['meta'] = meta
                return analysis
        except Exception:
            pass

    # bbox 위치가 선택한 눈과 명확히 반대면 리매핑하지 않는다.
    bbox = (detected_payload or {}).get('bbox')
    source_resolution = (meta or {}).get('source_resolution') or []
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and isinstance(source_resolution, (list, tuple)) and len(source_resolution) >= 1:
        try:
            source_w = float(source_resolution[0])
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            left_boundary = source_w * 0.45
            right_boundary = source_w * 0.55
            bbox_side = None
            if center_x < left_boundary:
                bbox_side = 'L'
            elif center_x > right_boundary:
                bbox_side = 'R'

            if bbox_side and bbox_side != side:
                meta['selected_eye'] = side
                meta['side_remapped_by_selected_eye'] = False
                meta['side_remap_skipped_reason'] = 'bbox_side_mismatch'
                meta['bbox_side'] = bbox_side
                analysis['meta'] = meta
                return analysis
        except Exception:
            pass

    analysis['left_eye'] = detected_payload if side == 'L' else None
    analysis['right_eye'] = detected_payload if side == 'R' else None

    meta['selected_eye'] = side
    meta['detected_side_before_remap'] = detected_side
    meta['side_remapped_by_selected_eye'] = True
    analysis['meta'] = meta

    return analysis
"""
Eye Disease Detection Server
웹 인터페이스와 AI 파이프라인의 오케스트레이터
"""

from dotenv import load_dotenv, set_key
load_dotenv()

from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
import torch
import time
import os
import json
import base64
import threading
import atexit
import gc
import sqlite3
import re
import sys
import io
import importlib
import requests
import random
import uuid
import socket
import subprocess
import hmac
from datetime import datetime
from PIL import Image, ImageOps

import config as config
from model_loader import initialize_models, get_models
from utils.chat_prompt import build_chat_system_prompt
from utils.image_proc import resize_image, enhance_contrast
from utils.service_control import service_manager_argv
from utils.uvc_camera import UvcCameraError, open_usb_uvc_camera
from database.db import get_conn, identifier_hash_id, init_db, migrate_legacy_history

MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles
MP_FACE_MESH_LOCK = threading.Lock()

# Pillow fallback statistics (thread-safe)
PILLOW_FALLBACK_SUCCESS = 0
PILLOW_FALLBACK_FAIL = 0
PILLOW_FALLBACK_LOCK = threading.Lock()

# ========================================
# [1] Flask 앱 설정
# ========================================
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app_secret = os.getenv('EYE_APP_SECRET_KEY', '').strip()
if not app_secret:
    # 고정 기본 키를 제거하고, 미설정 시 재시작마다 바뀌는 임시키를 사용한다.
    app_secret = os.urandom(32).hex()
    print('[WARNING] EYE_APP_SECRET_KEY 미설정: 임시 세션 키를 사용합니다. 운영 환경에서는 .env에 반드시 설정하세요.')

app.secret_key = app_secret
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', '0').strip().lower() in ('1', 'true', 'yes', 'on')

# 전역 변수
current_frame = None  # 실시간 카메라 프레임
model_manager = None  # 싱글톤 모델 매니저
models_initialized = False  # 모델 초기화 완료 여부
camera_thread = None  # 카메라 스레드
camera_running = False  # 카메라 스레드 실행 상태
camera_session_count = 0  # capture 페이지 활성 세션 수
camera_session_lock = threading.Lock()
active_camera_role = None
active_camera_device_index = None
debug_boxes_cache = []
debug_frame_counter = 0
YOLO_STREAM_DEBUG_OVERLAY = os.getenv('YOLO_STREAM_DEBUG_OVERLAY', '0') == '1'

DATABASE_PATH = os.getenv(
    'EYE_DATABASE_PATH',
    os.path.join(config.BASE_DIR, 'database', 'database.db')
)
LEGACY_HISTORY_DB_PATH = os.path.join(config.BASE_DIR, 'database', 'history.db')
REPORT_EXPORT_DIR = os.path.join(config.BASE_DIR, 'web', 'static', 'reports')
KAKAO_BRIDGE_URL = os.getenv('KAKAO_BRIDGE_URL', 'http://127.0.0.1:5001').strip().rstrip('/')
SCREENING_CONFIG_PATH = os.path.join(config.BASE_DIR, 'config', 'screening_modalities.json')

ADMIN_EDITABLE_CONFIG_KEYS = {
    'SERVER_IP': str,
    'SERVER_PORT': int,
    'DEBUG_MODE': bool,
    'CAMERA_DEVICE_INDEX': int,
    'MICROSCOPE_CAMERA_DEVICE_INDEX': int,
    'YOLO_CONF_THRESHOLD': float,
    'YOLO_IOU_THRESHOLD': float,
    'YOLO_STATUS_CONF_THRESHOLD': float,
    'YOLO_INPUT_SIZE': int,
    'CLASSIFIER_CONFIDENCE_THRESHOLD': float,
    'IRIS_REMOVAL_ENABLED': bool,
    'IRIS_THRESHOLD': float,
    'AUTO_DIST_THRESHOLD': int,
    'AUTO_SCALE_MIN': float,
    'AUTO_SCALE_MAX': float,
    'AUTO_CAPTURE_HOLD_FRAMES': int
}

MOBILE_PIN_STORE = {}
MOBILE_PIN_LOCK = threading.Lock()
MOBILE_PIN_TTL_SECONDS = 300
ENV_FILE_PATH = os.path.join(config.BASE_DIR, '.env')


CAMERA_ROLE_LABELS = {
    'webcam': '일반 웹캠',
    'usb_microscope': 'USB 현미경 카메라',
}


def normalize_camera_role(value):
    role = str(value or 'webcam').strip().lower()
    return role if role in CAMERA_ROLE_LABELS else 'webcam'


def get_camera_device_index(camera_role='webcam'):
    role = normalize_camera_role(camera_role)
    if role == 'usb_microscope':
        return int(getattr(
            config,
            'MICROSCOPE_CAMERA_DEVICE_INDEX',
            getattr(config, 'CAMERA_DEVICE_INDEX', 0),
        ))
    return int(getattr(config, 'CAMERA_DEVICE_INDEX', 0))


def describe_camera(camera_role='webcam'):
    role = normalize_camera_role(camera_role)
    camera_index = get_camera_device_index(role)
    camera_path = f'/dev/video{camera_index}'
    camera_name = ''
    try:
        with open(f'/sys/class/video4linux/video{camera_index}/name', 'r', encoding='utf-8') as name_file:
            camera_name = name_file.read().strip()
    except OSError:
        pass

    return {
        'role': role,
        'label': CAMERA_ROLE_LABELS[role],
        'device_index': camera_index,
        'device_path': camera_path,
        'connected': os.path.exists(camera_path),
        'name': camera_name or '카메라 미연결',
    }


def load_screening_config():
    """Load the UI-facing modality registry without coupling it to model code."""
    try:
        with open(SCREENING_CONFIG_PATH, 'r', encoding='utf-8') as config_file:
            payload = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f'[WARNING] 스크리닝 설정 로드 실패: {exc}')
        payload = {
            'service_name': '종합 건강 스크리닝',
            'version': 1,
            'modalities': [],
        }

    cameras = {
        role: describe_camera(role)
        for role in CAMERA_ROLE_LABELS
    }
    payload['cameras'] = cameras
    payload['camera'] = cameras['webcam']
    return payload


def _read_env_int(name, default):
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return int(default)

ADMIN_LOGIN_NAME = os.getenv('ADMIN_LOGIN_NAME', 'admin').strip().lower() or 'admin'
ADMIN_LOGIN_PASSWORD = os.getenv('ADMIN_LOGIN_PASSWORD', '').strip()
ADMIN_LOGIN_MAX_ATTEMPTS = _read_env_int('ADMIN_LOGIN_MAX_ATTEMPTS', 5)
ADMIN_LOGIN_WINDOW_SECONDS = _read_env_int('ADMIN_LOGIN_WINDOW_SECONDS', 300)
ADMIN_LOGIN_LOCK_SECONDS = _read_env_int('ADMIN_LOGIN_LOCK_SECONDS', 300)

ADMIN_LOGIN_ATTEMPTS = {}
ADMIN_LOGIN_ATTEMPTS_LOCK = threading.Lock()

ADMIN_LLM_EDITABLE_KEYS = {
    'LLM_PROVIDER': str,
    'OPENAI_MODEL': str,
    'GEMINI_MODEL': str,
    'OPENAI_API_KEY': str,
    'GEMINI_API_KEY': str
}


def is_admin_session():
    return bool(session.get('is_admin', False))


def get_client_ip_address():
    forwarded_for = str(request.headers.get('X-Forwarded-For', '')).strip()
    if forwarded_for:
        return forwarded_for.split(',')[0].strip() or 'unknown'
    return str(request.remote_addr or 'unknown').strip() or 'unknown'


def _cleanup_admin_attempts_locked(now_ts):
    stale_ips = []
    for ip, payload in ADMIN_LOGIN_ATTEMPTS.items():
        lock_until = float(payload.get('lock_until', 0))
        latest = float(payload.get('latest', 0))
        if now_ts > lock_until and (now_ts - latest) > (ADMIN_LOGIN_WINDOW_SECONDS * 2):
            stale_ips.append(ip)

    for ip in stale_ips:
        ADMIN_LOGIN_ATTEMPTS.pop(ip, None)


def check_admin_login_rate_limit(ip_addr):
    now_ts = time.time()
    with ADMIN_LOGIN_ATTEMPTS_LOCK:
        _cleanup_admin_attempts_locked(now_ts)
        payload = ADMIN_LOGIN_ATTEMPTS.get(ip_addr)
        if not payload:
            return True, 0

        lock_until = float(payload.get('lock_until', 0))
        if now_ts < lock_until:
            return False, int(lock_until - now_ts)

    return True, 0


def record_admin_login_attempt(ip_addr, success):
    now_ts = time.time()
    with ADMIN_LOGIN_ATTEMPTS_LOCK:
        payload = ADMIN_LOGIN_ATTEMPTS.get(ip_addr, {
            'failures': [],
            'lock_until': 0,
            'latest': now_ts,
        })

        payload['latest'] = now_ts
        if success:
            payload['failures'] = []
            payload['lock_until'] = 0
            ADMIN_LOGIN_ATTEMPTS[ip_addr] = payload
            return

        recent_failures = [
            ts for ts in payload.get('failures', [])
            if (now_ts - float(ts)) <= ADMIN_LOGIN_WINDOW_SECONDS
        ]
        recent_failures.append(now_ts)
        payload['failures'] = recent_failures

        if len(recent_failures) >= ADMIN_LOGIN_MAX_ATTEMPTS:
            payload['lock_until'] = now_ts + ADMIN_LOGIN_LOCK_SECONDS
            payload['failures'] = []

        ADMIN_LOGIN_ATTEMPTS[ip_addr] = payload


def create_admin_csrf_token():
    token = uuid.uuid4().hex
    session['admin_csrf_token'] = token
    return token


def is_valid_admin_csrf_token(incoming_token):
    saved = str(session.get('admin_csrf_token', '')).strip()
    incoming = str(incoming_token or '').strip()
    if not saved or not incoming:
        return False
    return hmac.compare_digest(saved, incoming)


def require_admin_csrf():
    if not is_admin_session():
        return jsonify({'status': 'error', 'message': '관리자 권한이 필요합니다.'}), 403

    header_token = request.headers.get('X-Admin-CSRF', '')
    body_token = ''
    try:
        body = request.json or {}
        if isinstance(body, dict):
            body_token = body.get('csrf_token', '')
    except Exception:
        body_token = ''

    if not is_valid_admin_csrf_token(header_token or body_token):
        return jsonify({'status': 'error', 'message': '유효하지 않은 관리자 요청입니다(CSRF).'}), 403

    return None


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if text in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise ValueError('bool 값이 아닙니다')


def cast_config_value(key, value):
    target_type = ADMIN_EDITABLE_CONFIG_KEYS.get(key)
    if target_type is None:
        raise ValueError(f'수정 불가 항목: {key}')

    if target_type is bool:
        return normalize_bool(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return str(value)


def _env_serialize_value(value):
    if isinstance(value, bool):
        return '1' if value else '0'
    return str(value)


def apply_admin_config_updates(updates):
    if not updates:
        return {}

    os.makedirs(config.BASE_DIR, exist_ok=True)
    if not os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, 'a', encoding='utf-8'):
            pass

    normalized_updates = {}
    for key, value in updates.items():
        if key not in ADMIN_EDITABLE_CONFIG_KEYS:
            continue

        env_value = _env_serialize_value(value)
        set_key(ENV_FILE_PATH, key, env_value)
        os.environ[key] = env_value
        normalized_updates[key] = value

        # SERVER_HOST를 사용하는 기존 환경과의 호환 유지
        if key == 'SERVER_IP':
            set_key(ENV_FILE_PATH, 'SERVER_HOST', env_value)
            os.environ['SERVER_HOST'] = env_value

    return normalized_updates


def get_admin_config_snapshot():
    snapshot = {}
    for key in ADMIN_EDITABLE_CONFIG_KEYS:
        snapshot[key] = getattr(config, key, None)
    return snapshot


def mask_secret_value(value):
    text = str(value or '').strip()
    if not text:
        return ''
    if len(text) <= 8:
        return '*' * len(text)
    return f"{text[:4]}{'*' * (len(text) - 8)}{text[-4:]}"


def get_admin_llm_settings_snapshot():
    openai_key = os.getenv('OPENAI_API_KEY', '')
    gemini_key = os.getenv('GEMINI_API_KEY', '')

    provider = os.getenv('LLM_PROVIDER', 'openai').strip().lower()
    if provider not in ('openai', 'gemini'):
        provider = 'openai'

    return {
        'LLM_PROVIDER': provider,
        'OPENAI_MODEL': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        'GEMINI_MODEL': os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
        'OPENAI_API_KEY_MASKED': mask_secret_value(openai_key),
        'GEMINI_API_KEY_MASKED': mask_secret_value(gemini_key),
        'OPENAI_API_KEY_CONFIGURED': bool(str(openai_key).strip()),
        'GEMINI_API_KEY_CONFIGURED': bool(str(gemini_key).strip())
    }


def apply_admin_llm_updates(updates):
    if not updates:
        return {}

    os.makedirs(config.BASE_DIR, exist_ok=True)
    if not os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, 'a', encoding='utf-8'):
            pass

    normalized_updates = {}
    for key, raw_value in updates.items():
        if key not in ADMIN_LLM_EDITABLE_KEYS:
            continue

        value = str(raw_value or '').strip()
        if key == 'LLM_PROVIDER':
            value = value.lower()
            if value not in ('openai', 'gemini'):
                raise ValueError('LLM_PROVIDER는 openai 또는 gemini만 허용됩니다.')

        # API 키는 빈 값이면 "변경 안 함"으로 처리
        if key in ('OPENAI_API_KEY', 'GEMINI_API_KEY') and value == '':
            continue

        set_key(ENV_FILE_PATH, key, value)
        os.environ[key] = value
        normalized_updates[key] = value

    return normalized_updates


def cleanup_expired_mobile_pins():
    now_ts = time.time()
    expired_keys = []

    with MOBILE_PIN_LOCK:
        for request_id, payload in MOBILE_PIN_STORE.items():
            created_at = float(payload.get('created_at_ts', 0))
            if (now_ts - created_at) > MOBILE_PIN_TTL_SECONDS:
                expired_keys.append(request_id)

        for request_id in expired_keys:
            MOBILE_PIN_STORE.pop(request_id, None)


def get_mobile_entry(request_id):
    if not request_id:
        return None

    cleanup_expired_mobile_pins()
    with MOBILE_PIN_LOCK:
        payload = MOBILE_PIN_STORE.get(request_id)
        if not payload:
            return None
        return dict(payload)


def is_verified_mobile_request(request_id):
    payload = get_mobile_entry(request_id)
    if not payload:
        return False
    return bool(payload.get('verified', False))


def resolve_mobile_base_url():
    external_base_url = os.environ.get('EXTERNAL_BASE_URL', '').strip().rstrip('/')
    if external_base_url:
        return external_base_url

    server_ip = str(getattr(config, 'SERVER_IP', '')).strip()
    server_port = int(getattr(config, 'SERVER_PORT', 5000))
    if server_ip and server_ip not in ('0.0.0.0', '127.0.0.1', 'localhost'):
        return f"http://{server_ip}:{server_port}"

    host_from_request = request.host.split(':')[0].strip()
    if host_from_request and host_from_request not in ('0.0.0.0', '127.0.0.1', 'localhost'):
        return f"http://{host_from_request}:{server_port}"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(('8.8.8.8', 80))
        local_ip = sock.getsockname()[0]
        sock.close()
        if local_ip:
            return f"http://{local_ip}:{server_port}"
    except Exception:
        pass

    return request.host_url.rstrip('/')

# ========================================
# [2] 촬영 상태 관리 (왼쪽/오른쪽 눈 시퀀스)
# ========================================
class CaptureState:
    """촬영 상태 관리 클래스"""
    
    # 촬영 상태 조건 체크 프레임 카운터
    ACCEPTED_FRAMES = 0
    
    # 현재 촬영 상태
    current_eye = "LEFT_EYE"  # "LEFT_EYE" 또는 "RIGHT_EYE"
    
    # 촬영 완료된 눈
    captured_eyes = {
        "LEFT_EYE": False,
        "RIGHT_EYE": False
    }
    
    # 자동 촬영 준비 상태
    auto_capture_ready = False
    state_lock = threading.Lock()
    
    @classmethod
    def reset(cls):
        """촬영 시퀀스 초기화"""
        with cls.state_lock:
            cls.ACCEPTED_FRAMES = 0
            cls.current_eye = "LEFT_EYE"
            cls.captured_eyes = {"LEFT_EYE": False, "RIGHT_EYE": False}
            cls.auto_capture_ready = False
    
    @classmethod
    def move_to_next_eye(cls):
        """다음 눈으로 이동"""
        with cls.state_lock:
            if cls.current_eye == "LEFT_EYE":
                cls.current_eye = "RIGHT_EYE"
            else:
                cls.current_eye = "LEFT_EYE"
            cls.ACCEPTED_FRAMES = 0
            cls.auto_capture_ready = False
    
    @classmethod
    def mark_captured(cls, eye_type):
        """촬영 완료 표시"""
        with cls.state_lock:
            cls.captured_eyes[eye_type] = True

    @classmethod
    def next_accepted_frame(cls):
        with cls.state_lock:
            cls.ACCEPTED_FRAMES += 1
            return cls.ACCEPTED_FRAMES

    @classmethod
    def reset_accepted_frames(cls):
        with cls.state_lock:
            cls.ACCEPTED_FRAMES = 0

    @classmethod
    def get_snapshot(cls):
        with cls.state_lock:
            return {
                'current_eye': cls.current_eye,
                'captured_eyes': cls.captured_eyes.copy(),
                'auto_capture_ready': bool(cls.auto_capture_ready),
            }


def init_history_db():
    """통합 DB를 초기화하고 기존 history.db를 안전하게 가져온다."""
    init_db(DATABASE_PATH)
    migration = migrate_legacy_history(
        LEGACY_HISTORY_DB_PATH,
        target_db_path=DATABASE_PATH,
        source_name='history.db-v1',
    )
    if migration['diagnoses'] or migration['surveys']:
        print(
            '[DB] legacy migration completed: '
            f"diagnoses={migration['diagnoses']} surveys={migration['surveys']}"
        )


def normalize_user_id(user_id):
    """파일 시스템/DB 저장용 사용자 식별자 정규화"""
    if not user_id:
        return 'anonymous'

    value = str(user_id).strip()
    if not value:
        return 'anonymous'

    # 전화번호 형식은 하이픈 유무와 무관하게 동일한 식별자로 통일한다.
    digits = re.sub(r'\D+', '', value)
    if len(digits) == 11 and digits.startswith('01'):
        value = f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"

    normalized = re.sub(r'[^0-9A-Za-z가-힣._-]+', '_', value)
    return normalized[:80] if normalized else 'anonymous'


def save_history_record(user_id, source_image_bgr, analysis):
    """분석 결과와 원본 이미지를 사용자별로 로컬 저장 + DB 기록"""
    safe_user_id = normalize_user_id(user_id)
    user_hash = identifier_hash_id(safe_user_id)

    user_dir = os.path.join(config.IMAGE_SAVE_DIR, 'users', user_hash)
    os.makedirs(user_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'diagnosis_{timestamp}.jpg'
    image_path = os.path.join(user_dir, filename)

    write_ok = cv2.imwrite(image_path, source_image_bgr)
    if not write_ok:
        raise RuntimeError('로컬 이미지 저장 실패')

    image_url = f"/static/captures/users/{user_hash}/{filename}"

    conn = get_conn(DATABASE_PATH)
    try:
        conn.execute(
            'INSERT OR IGNORE INTO users (phone_hash) VALUES (?)',
            (user_hash,)
        )
        user_row = conn.execute(
            'SELECT id FROM users WHERE phone_hash=?',
            (user_hash,)
        ).fetchone()
        if user_row is None:
            raise RuntimeError('사용자 레코드 생성 실패')

        status = str((analysis or {}).get('status') or 'done')
        pixel_metrics = (analysis or {}).get('pixel_metrics') or {}
        guide = (analysis or {}).get('guide') or {}
        impression = guide.get('summary') if isinstance(guide, dict) else None

        cur = conn.execute(
            """
            INSERT INTO diagnosis_sessions (
              user_id, ai_reading_json, pixel_metrics_json, survey_json,
              impression, status
            ) VALUES (?, ?, ?, '{}', ?, ?)
            """,
            (
                int(user_row['id']),
                json.dumps(analysis, ensure_ascii=False),
                json.dumps(pixel_metrics, ensure_ascii=False),
                impression,
                status,
            )
        )
        session_id = int(cur.lastrowid)
        conn.execute(
            """
            INSERT INTO session_assets
              (session_id, asset_type, file_path, public_url, mime_type)
            VALUES (?, 'image_raw', ?, ?, 'image/jpeg')
            """,
            (session_id, image_path, image_url)
        )
        conn.execute(
            """
            INSERT INTO event_logs (session_id, event_type, payload_json)
            VALUES (?, 'DIAG_SAVED', ?)
            """,
            (session_id, json.dumps({'status': status}, ensure_ascii=False))
        )
        conn.commit()
        return session_id, image_url
    except Exception:
        conn.rollback()
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except OSError:
            pass
        raise
    finally:
        conn.close()


def save_cam_image(user_id, eye_side, cam_image_bgr):
    """Grad-CAM 이미지를 사용자별로 저장하고 정적 URL 반환"""
    if cam_image_bgr is None:
        return None

    safe_user_id = identifier_hash_id(normalize_user_id(user_id))
    safe_eye_side = re.sub(r'[^0-9A-Za-z_-]+', '_', str(eye_side or 'eye')).lower()

    user_dir = os.path.join(config.IMAGE_SAVE_DIR, 'users', safe_user_id)
    os.makedirs(user_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{safe_eye_side}_cam_{timestamp}.jpg"
    image_path = os.path.join(user_dir, filename)

    write_ok = cv2.imwrite(image_path, cam_image_bgr)
    if not write_ok:
        return None

    return f"/static/captures/users/{safe_user_id}/{filename}"


def build_mediapipe_mesh_overlay(source_bgr):
    """원본 이미지 위에 MediaPipe Face Mesh를 시각화한다."""
    if source_bgr is None or source_bgr.size == 0:
        return None

    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    with MP_FACE_MESH_LOCK:
        results = MP_FACE_MESH.process(source_rgb)

    if not results.multi_face_landmarks:
        return None

    overlay = source_bgr.copy()
    for face_landmarks in results.multi_face_landmarks:
        MP_DRAWING.draw_landmarks(
            image=overlay,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=MP_DRAWING_STYLES.get_default_face_mesh_tesselation_style(),
        )
        MP_DRAWING.draw_landmarks(
            image=overlay,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=MP_DRAWING_STYLES.get_default_face_mesh_contours_style(),
        )
        MP_DRAWING.draw_landmarks(
            image=overlay,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=MP_DRAWING_STYLES.get_default_face_mesh_iris_connections_style(),
        )

    return overlay


# [LEGACY YOLO] def build_yolo_overlay_image(source_bgr, eye_crops):
# [LEGACY YOLO]     """원본 이미지 위에 YOLO 검출 기반 히트맵 오버레이를 생성한다."""
# [LEGACY YOLO]     if source_bgr is None or source_bgr.size == 0:
# [LEGACY YOLO]         return None
# [LEGACY YOLO]     if not eye_crops:
# [LEGACY YOLO]         return None
# [LEGACY YOLO]
# [LEGACY YOLO]     h, w = source_bgr.shape[:2]
# [LEGACY YOLO]     heat_mask = np.zeros((h, w), dtype=np.uint8)
# [LEGACY YOLO]
# [LEGACY YOLO]     for crop in eye_crops:
# [LEGACY YOLO]         bbox = crop.get('bbox')
# [LEGACY YOLO]         conf = float(crop.get('confidence', 0.0))
# [LEGACY YOLO]         if not bbox or len(bbox) != 4:
# [LEGACY YOLO]             continue
# [LEGACY YOLO]
# [LEGACY YOLO]         x1, y1, x2, y2 = [int(v) for v in bbox]
# [LEGACY YOLO]         x1 = max(0, min(w - 1, x1))
# [LEGACY YOLO]         x2 = max(0, min(w, x2))
# [LEGACY YOLO]         y1 = max(0, min(h - 1, y1))
# [LEGACY YOLO]         y2 = max(0, min(h, y2))
# [LEGACY YOLO]         if x2 <= x1 or y2 <= y1:
# [LEGACY YOLO]             continue
# [LEGACY YOLO]
# [LEGACY YOLO]         intensity = int(max(80, min(255, 80 + conf * 175)))
# [LEGACY YOLO]         region = heat_mask[y1:y2, x1:x2]
# [LEGACY YOLO]         np.maximum(region, intensity, out=region)
# [LEGACY YOLO]
# [LEGACY YOLO]     if not np.any(heat_mask > 0):
# [LEGACY YOLO]         return None
# [LEGACY YOLO]
# [LEGACY YOLO]     colored = cv2.applyColorMap(heat_mask, cv2.COLORMAP_JET)
# [LEGACY YOLO]     overlay = source_bgr.copy()
# [LEGACY YOLO]     active = heat_mask > 0
# [LEGACY YOLO]     overlay[active] = cv2.addWeighted(source_bgr[active], 0.55, colored[active], 0.45, 0)
# [LEGACY YOLO]
# [LEGACY YOLO]     for crop in eye_crops:
# [LEGACY YOLO]         bbox = crop.get('bbox')
# [LEGACY YOLO]         conf = float(crop.get('confidence', 0.0))
# [LEGACY YOLO]         if not bbox or len(bbox) != 4:
# [LEGACY YOLO]             continue
# [LEGACY YOLO]         x1, y1, x2, y2 = [int(v) for v in bbox]
# [LEGACY YOLO]         cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
# [LEGACY YOLO]         cv2.putText(
# [LEGACY YOLO]             overlay,
# [LEGACY YOLO]             f"YOLO {conf*100:.1f}%",
# [LEGACY YOLO]             (x1, max(18, y1 - 6)),
# [LEGACY YOLO]             cv2.FONT_HERSHEY_SIMPLEX,
# [LEGACY YOLO]             0.5,
# [LEGACY YOLO]             (0, 255, 255),
# [LEGACY YOLO]             1,
# [LEGACY YOLO]             cv2.LINE_AA
# [LEGACY YOLO]         )
# [LEGACY YOLO]
# [LEGACY YOLO]     return overlay


def list_history_records(user_id, limit=10):
    """사용자별 최근 히스토리 조회"""
    safe_user_id = normalize_user_id(user_id)
    user_hash = identifier_hash_id(safe_user_id)
    conn = get_conn(DATABASE_PATH)
    try:
        rows = conn.execute(
            """
            SELECT
              sessions.id,
              assets.public_url AS image_url,
              sessions.ai_reading_json AS analysis_json,
              sessions.diagnosed_at AS created_at
            FROM diagnosis_sessions AS sessions
            JOIN users ON users.id = sessions.user_id
            LEFT JOIN session_assets AS assets ON assets.id = (
              SELECT candidate.id
              FROM session_assets AS candidate
              WHERE candidate.session_id = sessions.id
                AND candidate.asset_type = 'image_raw'
              ORDER BY candidate.id ASC
              LIMIT 1
            )
            WHERE users.phone_hash=?
            ORDER BY datetime(sessions.diagnosed_at) DESC, sessions.id DESC
            LIMIT ?
            """,
            (user_hash, int(limit))
        ).fetchall()

        history = []
        for row in rows:
            analysis = {}
            try:
                analysis = json.loads(row['analysis_json']) if row['analysis_json'] else {}
            except Exception:
                analysis = {}

            history.append({
                'id': row['id'],
                'user_id': safe_user_id,
                'image_url': row['image_url'] or '',
                'created_at': row['created_at'],
                'analysis': analysis
            })

        return history
    finally:
        conn.close()


def delete_history_record(user_id, history_id):
    """사용자별 히스토리 단건 삭제 (+ 로컬 이미지 파일 삭제)"""
    safe_user_id = normalize_user_id(user_id)
    user_hash = identifier_hash_id(safe_user_id)
    conn = get_conn(DATABASE_PATH)
    try:
        row = conn.execute(
            """
            SELECT sessions.id, assets.file_path
            FROM diagnosis_sessions AS sessions
            JOIN users ON users.id = sessions.user_id
            LEFT JOIN session_assets AS assets ON assets.id = (
              SELECT candidate.id
              FROM session_assets AS candidate
              WHERE candidate.session_id = sessions.id
                AND candidate.asset_type = 'image_raw'
              ORDER BY candidate.id ASC
              LIMIT 1
            )
            WHERE sessions.id=? AND users.phone_hash=?
            """,
            (int(history_id), user_hash)
        ).fetchone()

        if row is None:
            return False

        image_path = row['file_path']

        conn.execute(
            'DELETE FROM diagnosis_sessions WHERE id=?',
            (int(history_id),)
        )
        conn.commit()

        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
        except Exception as file_error:
            print(f"[WARNING] 히스토리 이미지 삭제 실패(id={history_id}): {file_error}")

        return True
    finally:
        conn.close()


def normalize_survey_payload(payload):
    """설문 payload를 저장 가능한 형태로 정규화"""
    payload = payload or {}

    age_value = payload.get('age')
    age = None
    try:
        if age_value not in (None, ''):
            age = max(1, min(120, int(age_value)))
    except Exception:
        age = None

    symptoms = payload.get('symptoms')
    if not isinstance(symptoms, list):
        symptoms = []

    normalized_symptoms = []
    for symptom in symptoms[:30]:
        text = str(symptom).strip()
        if text:
            normalized_symptoms.append(text[:120])

    def clean_text(field_name, max_len=200):
        value = payload.get(field_name, '')
        return str(value).strip()[:max_len]

    def clean_vision(field_name):
        value = payload.get(field_name, '')
        text = str(value).strip()
        if not text:
            return ''
        try:
            number = float(text)
            number = max(0.0, min(3.0, number))
            return f"{number:.1f}"
        except Exception:
            return text[:10]

    screening_types = []
    for modality in payload.get('screening_types', []):
        modality_id = str(modality).strip().lower()
        if modality_id in ('eye', 'skin', 'scalp') and modality_id not in screening_types:
            screening_types.append(modality_id)

    def clean_section(field_name, allowed_fields):
        section = payload.get(field_name)
        if not isinstance(section, dict):
            return {}

        normalized = {}
        for key, max_len in allowed_fields.items():
            value = section.get(key, '')
            if isinstance(value, list):
                normalized[key] = [str(item).strip()[:120] for item in value[:30] if str(item).strip()]
            else:
                normalized[key] = str(value).strip()[:max_len]
        return normalized

    return {
        'screening_version': 1,
        'screening_types': screening_types,
        'age': age,
        'gender': clean_text('gender', 30),
        'wearing': clean_text('wearing', 40),
        'vision_left': clean_vision('vision_left'),
        'vision_right': clean_vision('vision_right'),
        'conditions': clean_text('conditions', 500),
        'surgery_history': clean_text('surgery_history', 300),
        'smoking': clean_text('smoking', 20),
        'drinking': clean_text('drinking', 20),
        'symptoms': normalized_symptoms,
        'other_notes': clean_text('other_notes', 500),
        'eye': clean_section('eye', {
            'wearing': 40,
            'vision_left': 10,
            'vision_right': 10,
            'surgery_history': 300,
            'symptoms': 0,
            'notes': 500,
        }),
        'skin': clean_section('skin', {
            'body_site': 80,
            'duration': 80,
            'symptoms': 0,
            'changes': 300,
            'notes': 500,
        }),
        'scalp': clean_section('scalp', {
            'scalp_site': 80,
            'duration': 80,
            'symptoms': 0,
            'treatment_history': 300,
            'notes': 500,
        }),
    }


def save_survey_record(user_id, survey_payload):
    """사용자 설문 기록 저장"""
    safe_user_id = normalize_user_id(user_id)
    user_hash = identifier_hash_id(safe_user_id)
    normalized_payload = normalize_survey_payload(survey_payload)

    conn = get_conn(DATABASE_PATH)
    try:
        conn.execute(
            'INSERT OR IGNORE INTO users (phone_hash) VALUES (?)',
            (user_hash,)
        )
        user_row = conn.execute(
            'SELECT id FROM users WHERE phone_hash=?',
            (user_hash,)
        ).fetchone()
        if user_row is None:
            raise RuntimeError('사용자 레코드 생성 실패')

        cur = conn.execute(
            """
            INSERT INTO survey_responses (user_id, survey_json)
            VALUES (?, ?)
            """,
            (
                int(user_row['id']),
                json.dumps(normalized_payload, ensure_ascii=False)
            )
        )
        conn.commit()
        return int(cur.lastrowid), normalized_payload
    finally:
        conn.close()


def list_survey_records(user_id, limit=50):
    """사용자별 최근 설문 기록 조회"""
    safe_user_id = normalize_user_id(user_id)
    user_hash = identifier_hash_id(safe_user_id)
    conn = get_conn(DATABASE_PATH)
    try:
        rows = conn.execute(
            """
            SELECT surveys.id, surveys.survey_json, surveys.created_at
            FROM survey_responses AS surveys
            JOIN users ON users.id = surveys.user_id
            WHERE users.phone_hash=?
            ORDER BY datetime(surveys.created_at) DESC, surveys.id DESC
            LIMIT ?
            """,
            (user_hash, int(limit))
        ).fetchall()

        surveys = []
        for row in rows:
            survey = {}
            try:
                survey = json.loads(row['survey_json']) if row['survey_json'] else {}
            except Exception:
                survey = {}

            surveys.append({
                'id': row['id'],
                'user_id': safe_user_id,
                'created_at': row['created_at'],
                'survey': survey
            })

        return surveys
    finally:
        conn.close()


def delete_survey_record(user_id, survey_id):
    """사용자별 설문 단건 삭제"""
    safe_user_id = normalize_user_id(user_id)
    user_hash = identifier_hash_id(safe_user_id)
    conn = get_conn(DATABASE_PATH)
    try:
        cur = conn.execute(
            """
            DELETE FROM survey_responses
            WHERE id=? AND user_id=(
              SELECT id FROM users WHERE phone_hash=?
            )
            """,
            (int(survey_id), user_hash)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def build_history_sessions(records):
    """
    report.html과 동일한 기준(3분 내 좌/우안 병합)으로 세션을 구성한다.
    - 이유: 프론트에서 보이는 최신 검사 단위와 백엔드 PDF 기준을 일치시키기 위함
    """
    sessions = []
    max_merge_gap_ms = 3 * 60 * 1000

    def to_ms(value):
        if not value:
            return 0
        try:
            return int(datetime.fromisoformat(str(value).replace(' ', 'T')).timestamp() * 1000)
        except Exception:
            return 0

    index = 0
    while index < len(records):
        first = records[index]
        session = {
            'created_at': first.get('created_at'),
            'records': [first],
            'analysis': first.get('analysis') or {}
        }

        has_left = bool((session['analysis'] or {}).get('left_eye'))
        has_right = bool((session['analysis'] or {}).get('right_eye'))
        needs_merge = not (has_left and has_right)

        if needs_merge and (index + 1) < len(records):
            second = records[index + 1]
            if abs(to_ms(first.get('created_at')) - to_ms(second.get('created_at'))) <= max_merge_gap_ms:
                session['records'].append(second)
                merged = session['analysis'] or {}
                second_analysis = second.get('analysis') or {}
                if not merged.get('left_eye') and second_analysis.get('left_eye'):
                    merged['left_eye'] = second_analysis.get('left_eye')
                if not merged.get('right_eye') and second_analysis.get('right_eye'):
                    merged['right_eye'] = second_analysis.get('right_eye')
                if not merged.get('guide') and second_analysis.get('guide'):
                    merged['guide'] = second_analysis.get('guide')
                session['analysis'] = merged
                index += 1

        sessions.append(session)
        index += 1

    return sessions


def summarize_session_analysis(analysis):
    """PDF/카카오 메시지에 공통으로 쓰는 요약 문자열 생성"""
    analysis = analysis or {}
    left = analysis.get('left_eye') or {}
    right = analysis.get('right_eye') or {}
    guide = analysis.get('guide') or {}

    def eye_text(label, eye_data):
        if not eye_data:
            return f"{label}: 데이터 없음"
        disease = eye_data.get('disease', '미상')
        confidence = eye_data.get('confidence', 0)
        redness = eye_data.get('redness', None)
        if isinstance(confidence, (int, float)) and confidence <= 1:
            confidence = confidence * 100
        conf_text = f"{float(confidence):.1f}%" if isinstance(confidence, (int, float)) else str(confidence)
        red_text = f"{float(redness):.4f}" if isinstance(redness, (int, float)) else 'N/A'
        return f"{label}: {disease} | 신뢰도 {conf_text} | 충혈도 {red_text}"

    return {
        'left_summary': eye_text('좌안', left),
        'right_summary': eye_text('우안', right),
        'guide_tag': guide.get('tag_text', '가이드 없음'),
        'guide_summary': guide.get('summary', '가이드 요약 없음')
    }


def cleanup_old_report_exports(user_id, keep_count=10, max_age_hours=24):
    """
    PDF 파일 누적 방지를 위한 보관 정책
    - 같은 사용자 파일은 최신 keep_count개만 유지
    - max_age_hours 초과 파일은 추가 삭제
    """
    safe_user_id = normalize_user_id(user_id)
    os.makedirs(REPORT_EXPORT_DIR, exist_ok=True)

    prefix = f"report_{safe_user_id}_"
    files = []
    for name in os.listdir(REPORT_EXPORT_DIR):
        if not name.startswith(prefix) or not name.endswith('.pdf'):
            continue
        path = os.path.join(REPORT_EXPORT_DIR, name)
        try:
            stat = os.stat(path)
            files.append((path, stat.st_mtime))
        except Exception:
            continue

    files.sort(key=lambda item: item[1], reverse=True)
    now_ts = time.time()
    ttl_seconds = max_age_hours * 3600

    for index, (path, mtime) in enumerate(files):
        over_count = index >= keep_count
        over_ttl = (now_ts - mtime) > ttl_seconds
        if over_count or over_ttl:
            try:
                os.remove(path)
            except Exception as error:
                print(f"[WARNING] 오래된 리포트 삭제 실패: {path} ({error})")


def build_report_eye_image_reader(image_path, bbox=None):
    """리포트용 눈 이미지를 원본 검사 사진에서 잘라 반환한다."""
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        from reportlab.lib.utils import ImageReader as ReportImageReader

        with Image.open(image_path) as source_image:
            source_image = ImageOps.exif_transpose(source_image).convert('RGB')
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    x1, y1, x2, y2 = [int(float(value)) for value in bbox]
                    left = max(0, min(source_image.width - 1, min(x1, x2)))
                    top = max(0, min(source_image.height - 1, min(y1, y2)))
                    right = max(left + 1, min(source_image.width, max(x1, x2)))
                    bottom = max(top + 1, min(source_image.height, max(y1, y2)))
                    source_image = source_image.crop((left, top, right, bottom))
                except Exception:
                    pass

            buffer = io.BytesIO()
            source_image = ImageOps.contain(
                source_image,
                (1600, 1600),
                method=Image.Resampling.LANCZOS,
            )
            source_image.save(buffer, format='JPEG', quality=95, subsampling=0, optimize=True)
            buffer.seek(0)
            return ReportImageReader(buffer)
    except Exception as error:
        print(f"[WARNING] 리포트 눈 이미지 준비 실패: {image_path} ({error})")
        return None


_REPORT_PDF_FONT_NAME = None


def get_report_pdf_font_name() -> str:
    global _REPORT_PDF_FONT_NAME
    if _REPORT_PDF_FONT_NAME:
        return _REPORT_PDF_FONT_NAME

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        _REPORT_PDF_FONT_NAME = 'Helvetica'
        return _REPORT_PDF_FONT_NAME

    font_candidates = [
        'Noto Sans CJK KR',
        'NanumGothic',
        'Apple SD Gothic Neo',
        'DejaVu Sans',
    ]

    for font_query in font_candidates:
        font_path = None
        try:
            completed = subprocess.run(
                ['fc-match', '-f', '%{file}\n', font_query],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            font_path = (completed.stdout or '').strip().splitlines()[0].strip() if (completed.stdout or '').strip() else None
        except Exception:
            font_path = None

        if not font_path or not os.path.exists(font_path):
            continue

        font_name = f"ReportFont-{font_query.replace(' ', '')}"
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path, subfontIndex=0))
            _REPORT_PDF_FONT_NAME = font_name
            return _REPORT_PDF_FONT_NAME
        except Exception:
            continue

    _REPORT_PDF_FONT_NAME = 'Helvetica'
    return _REPORT_PDF_FONT_NAME


def wrap_pdf_text(text: object, font_name: str, font_size: float, max_width: float) -> list[str]:
    from reportlab.pdfbase import pdfmetrics

    raw_text = '' if text is None else str(text)
    lines: list[str] = []

    for paragraph in raw_text.splitlines() or ['']:
        if paragraph == '':
            lines.append('')
            continue

        current_line = ''
        for word in paragraph.split(' '):
            candidate = word if not current_line else f'{current_line} {word}'
            if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
                current_line = candidate
                continue

            if current_line:
                lines.append(current_line)
                current_line = ''

            if pdfmetrics.stringWidth(word, font_name, font_size) <= max_width:
                current_line = word
                continue

            chunk = ''
            for char in word:
                candidate = f'{chunk}{char}'
                if not chunk or pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
                    chunk = candidate
                else:
                    lines.append(chunk)
                    chunk = char
            current_line = chunk

        if current_line:
            lines.append(current_line)

    return lines or ['']


def draw_pdf_text_block(c, x: float, y: float, text: object, font_name: str, font_size: float, max_width: float, line_gap: float = 4.0) -> float:
    c.setFont(font_name, font_size)
    lines = wrap_pdf_text(text, font_name, font_size, max_width)
    line_height = font_size + line_gap
    for line in lines:
        c.drawString(x, y, line)
        y -= line_height
    return y


def generate_session_pdf_report(user_id, session_data, latest_survey):
    """
    메인 서버 기준 PDF 생성
    - reportlab이 설치되지 않은 환경에서도 명확한 오류를 반환하도록 지연 import 사용
    """
    try:
        reportlab_pagesizes = importlib.import_module('reportlab.lib.pagesizes')
        reportlab_utils = importlib.import_module('reportlab.lib.utils')
        reportlab_canvas = importlib.import_module('reportlab.pdfgen.canvas')
        A4 = reportlab_pagesizes.A4
        ImageReader = reportlab_utils.ImageReader
        canvas = reportlab_canvas
    except Exception as import_error:
        raise RuntimeError('PDF 기능을 사용하려면 reportlab 설치가 필요합니다. (pip install reportlab)') from import_error

    safe_user_id = normalize_user_id(user_id)
    os.makedirs(REPORT_EXPORT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"report_{safe_user_id}_{timestamp}.pdf"
    pdf_path = os.path.join(REPORT_EXPORT_DIR, filename)
    pdf_url = f"/static/reports/{filename}"

    session_created_at = session_data.get('created_at') or '-'
    analysis = session_data.get('analysis') or {}
    summary = summarize_session_analysis(analysis)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    page_w, page_h = A4
    y = page_h - 48
    font_name = get_report_pdf_font_name()
    bold_font_name = font_name
    text_width = page_w - 72

    c.setFont(bold_font_name, 14)
    c.drawString(36, y, 'Eye Project - Diagnosis Report')
    y -= 20

    c.setFont(font_name, 10)
    c.drawString(36, y, f'User ID: {safe_user_id}')
    y -= 14
    c.drawString(36, y, f'Exam Time: {session_created_at}')
    y -= 20

    c.setFont(bold_font_name, 11)
    c.drawString(36, y, 'AI Summary')
    y -= 14

    y = draw_pdf_text_block(c, 44, y, summary['left_summary'], font_name, 10, text_width - 8)
    y = draw_pdf_text_block(c, 44, y, summary['right_summary'], font_name, 10, text_width - 8)
    y = draw_pdf_text_block(c, 44, y, f"Guide Tag: {summary['guide_tag']}", font_name, 10, text_width - 8)
    y = draw_pdf_text_block(c, 44, y, f"Guide: {summary['guide_summary'][:120]}", font_name, 10, text_width - 8)
    y -= 20

    if latest_survey:
        c.setFont(bold_font_name, 11)
        c.drawString(36, y, 'Latest Survey')
        y -= 14
        c.setFont(font_name, 10)
        age_text = latest_survey.get('age') if latest_survey.get('age') is not None else '-'
        y = draw_pdf_text_block(c, 44, y, f"Age: {age_text} | Gender: {latest_survey.get('gender', '-')}", font_name, 10, text_width - 8)
        symptoms = latest_survey.get('symptoms') or []
        symptom_text = ', '.join(symptoms[:5]) if isinstance(symptoms, list) and symptoms else '없음'
        y = draw_pdf_text_block(c, 44, y, f"Symptoms: {symptom_text}", font_name, 10, text_width - 8)
        y -= 8

    c.setFont(bold_font_name, 11)
    c.drawString(36, y, 'Captured Images')
    y -= 12

    image_drawn = 0
    for record in session_data.get('records', []):
        image_path = record.get('image_path')
        try:
            analysis = record.get('analysis') or {}
            eye_candidates = []
            if analysis.get('left_eye') and analysis['left_eye'].get('bbox'):
                eye_candidates.append(analysis['left_eye'].get('bbox'))
            if analysis.get('right_eye') and analysis['right_eye'].get('bbox'):
                eye_candidates.append(analysis['right_eye'].get('bbox'))

            if not eye_candidates:
                eye_candidates = [None]

            for bbox in eye_candidates:
                image_reader = build_report_eye_image_reader(image_path, bbox)
                if image_reader is None:
                    continue

                draw_w = 240
                draw_h = 150
                if y - draw_h < 40:
                    c.showPage()
                    y = page_h - 48
                c.drawImage(image_reader, 44, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto')
                y -= (draw_h + 14)
                image_drawn += 1
                if image_drawn >= 2:
                    break
            if image_drawn >= 2:
                break
        except Exception as image_error:
            print(f"[WARNING] PDF 이미지 삽입 실패: {image_path} ({image_error})")

    if image_drawn == 0:
        c.setFont(font_name, 10)
        c.drawString(44, y, '이미지 파일을 찾지 못해 텍스트 요약만 포함되었습니다.')

    c.showPage()
    c.save()

    cleanup_old_report_exports(safe_user_id)
    return pdf_path, pdf_url


def send_kakao_report_message(user_id, report_url):
    """
    카카오 나에게 보내기 API
    - 기본 경로: 카카오 연동 서버(database/app.py)로 전송 위임
    - 예외 fallback: 환경변수 KAKAO_ACCESS_TOKEN 직접 사용
    """
    token = os.environ.get('KAKAO_ACCESS_TOKEN')
    external_base_url = os.environ.get('EXTERNAL_BASE_URL', '').strip().rstrip('/')
    if external_base_url:
        open_url = f"{external_base_url}{report_url}"
    else:
        open_url = f"http://{config.SERVER_IP}:{config.SERVER_PORT}{report_url}"

    # 우선: 카카오 연동 서버(database/app.py)로 전송 위임
    if KAKAO_BRIDGE_URL:
        payload = {
            'phone': normalize_user_id(user_id),
            'text': f"안구 진단 보고서가 생성되었습니다.\n사용자: {normalize_user_id(user_id)}\n\n보고서 열기: {open_url}",
            'link_url': open_url,
        }

        try:
            bridge_response = requests.post(
                f"{KAKAO_BRIDGE_URL}/kakao/send_report",
                json=payload,
                timeout=10
            )
            bridge_data = bridge_response.json() if bridge_response.text else {}
            if bridge_response.status_code == 200 and bridge_data.get('status') == 'ok':
                return {
                    'open_url': open_url,
                    'kakao_response': bridge_data,
                }

            bridge_message = bridge_data.get('message') if isinstance(bridge_data, dict) else bridge_response.text
            raise RuntimeError(bridge_message or f"카카오 연동 서버 오류({bridge_response.status_code})")
        except Exception as bridge_error:
            token = os.environ.get('KAKAO_ACCESS_TOKEN')
            if not token:
                raise RuntimeError(str(bridge_error))

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    template_object = {
        'object_type': 'text',
        'text': f"안구 진단 보고서가 생성되었습니다.\n사용자: {normalize_user_id(user_id)}\n\n보고서 열기: {open_url}",
        'link': {
            'web_url': open_url,
            'mobile_web_url': open_url
        },
        'button_title': '보고서 열기'
    }

    response = requests.post(
        'https://kapi.kakao.com/v2/api/talk/memo/default/send',
        headers=headers,
        data={'template_object': json.dumps(template_object, ensure_ascii=False)},
        timeout=10
    )

    if response.status_code != 200:
        raise RuntimeError(f"Kakao send failed: {response.status_code} {response.text}")

    return {
        'open_url': open_url,
        'kakao_response': response.json() if response.text else {}
    }


def get_report_dependency_status():
    """
    보고서 기능 의존성 점검
    - reportlab: PDF 생성 필수
    - requests: 카카오 API 호출 필수
    """
    reportlab_installed = importlib.util.find_spec('reportlab') is not None
    requests_installed = importlib.util.find_spec('requests') is not None
    kakao_token_set = bool(os.environ.get('KAKAO_ACCESS_TOKEN'))
    kakao_bridge_enabled = bool(KAKAO_BRIDGE_URL)

    missing = []
    if not reportlab_installed:
        missing.append('reportlab')
    if not requests_installed:
        missing.append('requests')

    return {
        'reportlab_installed': reportlab_installed,
        'requests_installed': requests_installed,
        'kakao_token_configured': kakao_token_set,
        'kakao_bridge_enabled': kakao_bridge_enabled,
        'pdf_generation_ready': reportlab_installed,
        'kakao_send_ready': requests_installed and (kakao_token_set or kakao_bridge_enabled),
        'missing_packages': missing
    }


# ========================================
# [3] 자동 촬영 조건 확인 함수
# ========================================
def check_auto_capture(detections, eye_crop_bbox, guideline_bbox):
    """
    자동 촬영 조건 확인
    
    조건 1: 중심점 거리 확인 (AUTO_DIST_THRESHOLD)
    조건 2: 눈 크기 비율 확인 (AUTO_SCALE_MIN ~ AUTO_SCALE_MAX)
    
    Args:
        detections: YOLO 검출 결과 (중심점 좌표 포함)
        eye_crop_bbox: 눈 크로핑 바운딩박스 (x1, y1, x2, y2)
        guideline_bbox: 가이드라인 바운딩박스 (x1, y1, x2, y2)
    
    Returns:
        bool: 자동 촬영 조건 만족 여부
    """
    try:
        # 검출 결과가 없으면 False
        if not detections or len(detections) == 0:
            return False
        
        # 검출된 눈의 중심점 계산
        detection = detections[0]
        x_center = (detection.get('x1', 0) + detection.get('x2', 0)) / 2
        y_center = (detection.get('y1', 0) + detection.get('y2', 0)) / 2
        
        # 가이드라인 중심점 계산
        g_x1, g_y1, g_x2, g_y2 = guideline_bbox
        guideline_x_center = (g_x1 + g_x2) / 2
        guideline_y_center = (g_y1 + g_y2) / 2
        
        # 조건 1: 중심점 거리 확인
        dist = np.sqrt((x_center - guideline_x_center)**2 + (y_center - guideline_y_center)**2)
        if dist > config.AUTO_DIST_THRESHOLD:
            return False
        
        # 조건 2: 눈 크기 비율 확인
        eye_width = eye_crop_bbox[2] - eye_crop_bbox[0]
        eye_height = eye_crop_bbox[3] - eye_crop_bbox[1]
        eye_area = eye_width * eye_height
        
        guideline_width = g_x2 - g_x1
        guideline_height = g_y2 - g_y1
        guideline_area = guideline_width * guideline_height
        
        scale_ratio = eye_area / guideline_area if guideline_area > 0 else 0
        
        if scale_ratio < config.AUTO_SCALE_MIN or scale_ratio > config.AUTO_SCALE_MAX:
            return False
        
        return True
    
    except Exception as e:
        print(f"[WARNING] 자동 촬영 조건 확인 실패: {e}")
        return False


def should_auto_capture():
    """
    자동 촬영 프레임 누적 확인
    
    Returns:
        bool: 자동 촬영 실행 여부
    """
    accepted_frames = CaptureState.next_accepted_frame()
    
    # AUTO_CAPTURE_HOLD_FRAMES만큼 프레임이 조건을 만족하면 True
    if accepted_frames >= config.AUTO_CAPTURE_HOLD_FRAMES:
        CaptureState.reset_accepted_frames()
        return True
    
    return False


def _is_yolo_cuda_oom_error(exc):
    text = str(exc).lower()
    return ('out of memory' in text) or ('cuda error' in text and 'memory' in text)


# [LEGACY YOLO] def safe_yolo_detect(detector, image, conf_threshold=None):
# [LEGACY YOLO]     """YOLO 추론을 수행하되 CUDA OOM 발생 시 캐시 정리 후 1회 재시도한다."""
# [LEGACY YOLO]     try:
# [LEGACY YOLO]         return detector.detect(image, conf_threshold=conf_threshold)
# [LEGACY YOLO]     except Exception as e:
# [LEGACY YOLO]         if not _is_yolo_cuda_oom_error(e):
# [LEGACY YOLO]             raise
# [LEGACY YOLO]
# [LEGACY YOLO]         print(f"[WARNING] YOLO CUDA OOM 감지, 캐시 정리 후 재시도: {e}")
# [LEGACY YOLO]         try:
# [LEGACY YOLO]             gc.collect()
# [LEGACY YOLO]             if torch.cuda.is_available():
# [LEGACY YOLO]                 torch.cuda.empty_cache()
# [LEGACY YOLO]                 if hasattr(torch.cuda, 'ipc_collect'):
# [LEGACY YOLO]                     torch.cuda.ipc_collect()
# [LEGACY YOLO]         except Exception as cleanup_error:
# [LEGACY YOLO]             print(f"[WARNING] CUDA 캐시 정리 실패: {cleanup_error}")
# [LEGACY YOLO]
# [LEGACY YOLO]         try:
# [LEGACY YOLO]             return detector.detect(image, conf_threshold=conf_threshold)
# [LEGACY YOLO]         except Exception as retry_error:
# [LEGACY YOLO]             print(f"[WARNING] YOLO 재시도 실패: {retry_error}")
# [LEGACY YOLO]             return None

# ========================================
# [2] LED 하드웨어 설정 (Jetson)
# ========================================
try:
    import Jetson.GPIO as GPIO
    LED_PIN = 32
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    pwm_led = GPIO.PWM(LED_PIN, 1000)
    pwm_led.start(0)
    LED_AVAILABLE = True
except:
    print("[WARNING] Jetson GPIO not available. LED control disabled.")
    LED_AVAILABLE = False


# [LEGACY YOLO] ========================================
# [LEGACY YOLO] [2-1] YOLO 눈 감지 여부 확인
# [LEGACY YOLO] ========================================
def check_eye_detection(frame):
    """
    현재 프레임에서 눈이 감지되는지 확인 (MediaPipe 기반)
    
    Args:
        frame: 검사할 이미지 프레임
        
    Returns:
        tuple: (감지 여부, 신뢰도)
    """
    if frame is None:
        return False, 0
    
    try:
        manager = get_models()
        detector = manager.get_detector()
        
        # MediaPipe Face Mesh 검출
        detection_result = detector.detect(frame)
        
        if detection_result is None:
            return False, 0
        
        # MediaPipe은 얼굴 검출 시 랜드마크를 반환
        # 눈이 포함된 랜드마크가 있으므로 1.0으로 신뢰도 설정
        return True, 1.0
    except Exception as e:
        print(f"[WARNING] 눈 감지 확인 실패: {e}")
        return False, 0


# ========================================
# [2-2] EfficientNet을 이용한 질환 분석
# ========================================
def analyze_eye_image(base64_image_data):
    """
    Base64로 인코딩된 이미지를 받아 EfficientNet으로 질환 분류
    
    Args:
        base64_image_data: Base64로 인코딩된 이미지
        
    Returns:
        dict: 분석 결과
    """
    try:
        # 1. Base64 → OpenCV 이미지 변환
        image_data = base64.b64decode(base64_image_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return None, None, "이미지 디코딩 실패"
        
        # 2. 모델 매니저에서 분류기 가져오기
        manager = get_models()
        classifier = manager.get_classifier()
        
        # 3. 분류 수행
        disease_label, confidence, _ = classifier.classify(img_bgr)
        confidence_score = confidence * 100  # 퍼센티지
        
        return disease_label, confidence_score, None
        
    except Exception as e:
        print(f"[ERROR] 이미지 분석 실패: {e}")
        return None, None, str(e)


def decode_base64_image(base64_image_data):
    """Base64(Data URL 포함) 문자열을 OpenCV BGR 이미지로 디코딩"""
    if not base64_image_data:
        return None, "이미지 데이터가 비어있습니다"

    try:
        encoded = base64_image_data.split(',', 1)[1] if ',' in base64_image_data else base64_image_data
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # OpenCV로 디코딩 실패 시 Pillow로 폴백 시도 (AVIF 등 OpenCV가 직접 디코딩하지 못하는 형식 처리)
            try:
                bio = io.BytesIO(image_data)
                pil_img = Image.open(bio)
                pil_img = pil_img.convert('RGB')
                arr = np.array(pil_img)
                # PIL은 RGB, OpenCV는 BGR 사용
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                # 기록
                try:
                    with PILLOW_FALLBACK_LOCK:
                        global PILLOW_FALLBACK_SUCCESS
                        PILLOW_FALLBACK_SUCCESS += 1
                except Exception:
                    pass
                print('[INFO] decode_base64_image: OpenCV 디코드 실패, Pillow로 폴백 성공')
            except Exception as pil_err:
                try:
                    with PILLOW_FALLBACK_LOCK:
                        global PILLOW_FALLBACK_FAIL
                        PILLOW_FALLBACK_FAIL += 1
                except Exception:
                    pass
                print(f'[WARNING] decode_base64_image: Pillow 폴백 실패: {pil_err}')
                return None, "이미지 디코딩 실패"

        return img_bgr, None
    except Exception as e:
        return None, str(e)


def upscale_eye_crop_for_classifier(eye_crop):
    """작은 눈 크롭 해상도 방어를 위해 INTER_CUBIC 기반 224x224 리사이즈"""
    if eye_crop is None or eye_crop.size == 0:
        return None

    return cv2.resize(eye_crop, config.CLASSIFIER_INPUT_SIZE, interpolation=cv2.INTER_CUBIC)


def analyze_uploaded_single_eye_from_image(img_bgr, user_id='anonymous', selected_eye=None):
    """업로드 단안 이미지 분석: YOLO를 건너뛰고 EfficientNet/Analyzer만 수행"""
    try:
        manager = get_models()
        classifier = manager.get_classifier()
        analyzer = manager.get_analyzer()

        source_h, source_w = img_bgr.shape[:2]
        prepared_eye = upscale_eye_crop_for_classifier(img_bgr)
        if prepared_eye is None:
            return {
                'status': 'error',
                'message': '업로드 이미지 전처리에 실패했습니다.'
            }

        cls_start = time.time()
        classification = classifier.classify_with_details(prepared_eye, generate_cam=True)
        eye_analysis = analyzer.analyze(prepared_eye)
        cls_elapsed_ms = (time.time() - cls_start) * 1000.0

        side = normalize_selected_eye(selected_eye) or 'L'
        side_key = 'left_eye' if side == 'L' else 'right_eye'
        cam_image_url = save_cam_image(user_id, side_key, classification.get('heatmap_image'))

        eye_payload = {
            'disease': classification['disease'],
            'class': classification['class'],
            'confidence': round(float(classification['confidence']) * 100.0, 2),
            'redness': round(float(eye_analysis.get('redness', 0.0)), 4),
            'bbox': (0, 0, source_w, source_h),
            'cam_image_url': cam_image_url,
            'detection_confidence': None,
            'process_time_ms': round(cls_elapsed_ms, 1),
            'detection_method': 'upload_single_image'
        }

        result = {
            'status': 'success',
            'message': '업로드 단안 분석 완료',
            'left_eye': eye_payload if side_key == 'left_eye' else None,
            'right_eye': eye_payload if side_key == 'right_eye' else None,
            'mediapipe_cam_image_url': None,
            'meta': {
                'source_resolution': [source_w, source_h],
                'eyes_detected': 1,
                'yolo_skipped': True,
                'analysis_mode': 'upload_single_eye_no_yolo',
                'selected_eye': side,
                'process_time_ms': round(cls_elapsed_ms, 1)
            }
        }

        del prepared_eye
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
    except Exception as e:
        print(f"[ERROR] 업로드 단안 분석 실패: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }


def select_distinct_eye_crops(eye_crops, frame_width, frame_height):
    """
    MediaPipe 검출 결과를 좌/우안으로 정리한다.
    - MediaPipe는 이미 좌/우안을 명확하게 분리하므로 단순 정렬만 수행
    - 'side' 필드로 LEFT_EYE, RIGHT_EYE를 구분
    """
    # [LEGACY YOLO] if not eye_crops:
    # [LEGACY YOLO]     return []
    # [LEGACY YOLO]
    # [LEGACY YOLO] min_conf = max(float(config.YOLO_STATUS_CONF_THRESHOLD), 0.25)
    # [LEGACY YOLO] min_box_w = max(20.0, float(frame_width) * 0.05)
    # [LEGACY YOLO] min_box_h = max(14.0, float(frame_height) * 0.05)
    # [LEGACY YOLO]
    # [LEGACY YOLO] filtered = []
    # [LEGACY YOLO] for item in eye_crops:
    # [LEGACY YOLO]     conf = float(item.get('confidence', 0.0))
    # [LEGACY YOLO]     bbox = item.get('bbox') or (0, 0, 0, 0)
    # [LEGACY YOLO]     try:
    # [LEGACY YOLO]         bw = abs(float(bbox[2]) - float(bbox[0]))
    # [LEGACY YOLO]         bh = abs(float(bbox[3]) - float(bbox[1]))
    # [LEGACY YOLO]     except Exception:
    # [LEGACY YOLO]         continue
    # [LEGACY YOLO]
    # [LEGACY YOLO]     if conf < min_conf:
    # [LEGACY YOLO]         continue
    # [LEGACY YOLO]     if bw < min_box_w or bh < min_box_h:
    # [LEGACY YOLO]         continue
    # [LEGACY YOLO]     filtered.append(item)
    # [LEGACY YOLO]
    # [LEGACY YOLO] candidates = sorted(filtered, key=lambda item: float(item.get('confidence', 0.0)), reverse=True)[:6]
    # [LEGACY YOLO] if not candidates:
    # [LEGACY YOLO]     return []
    # [LEGACY YOLO] if len(candidates) == 1:
    # [LEGACY YOLO]     return [candidates[0]]
    # [LEGACY YOLO]
    # [LEGACY YOLO] def center_x(item):
    # [LEGACY YOLO]     x1, _, x2, _ = item['bbox']
    # [LEGACY YOLO]     return (x1 + x2) / 2.0
    # [LEGACY YOLO]
    # [LEGACY YOLO] best_pair = None
    # [LEGACY YOLO] best_sep = -1.0
    # [LEGACY YOLO] pair_min_conf = max(min_conf + 0.12, 0.38)
    # [LEGACY YOLO] for idx in range(len(candidates)):
    # [LEGACY YOLO]     for jdx in range(idx + 1, len(candidates)):
    # [LEGACY YOLO]         if float(candidates[idx].get('confidence', 0.0)) < pair_min_conf:
    # [LEGACY YOLO]             continue
    # [LEGACY YOLO]         if float(candidates[jdx].get('confidence', 0.0)) < pair_min_conf:
    # [LEGACY YOLO]             continue
    # [LEGACY YOLO]         sep = abs(center_x(candidates[idx]) - center_x(candidates[jdx]))
    # [LEGACY YOLO]         if sep > best_sep:
    # [LEGACY YOLO]             best_sep = sep
    # [LEGACY YOLO]             best_pair = (candidates[idx], candidates[jdx])
    # [LEGACY YOLO]
    # [LEGACY YOLO] # 좌/우로 볼 수 있는 최소 수평 간격(프레임 폭의 12%)
    # [LEGACY YOLO] min_sep = max(40.0, float(frame_width) * 0.12)
    # [LEGACY YOLO] if best_pair and best_sep >= min_sep:
    # [LEGACY YOLO]     return sorted([best_pair[0], best_pair[1]], key=center_x)
    # [LEGACY YOLO]
    # [LEGACY YOLO] # 서로 충분히 떨어진 2안이 아니면 단안으로 처리 (selected_eye 매핑이 후속 보정)
    # [LEGACY YOLO] return [candidates[0]]
    
    # MediaPipe: side 필드로 이미 구분되어 있음
    if not eye_crops:
        return []
    
    # side 필드가 있는 경우 (MediaPipe)
    if eye_crops and 'side' in eye_crops[0]:
        organized = {'LEFT_EYE': None, 'RIGHT_EYE': None}
        for crop in eye_crops:
            side = crop.get('side')
            if side in organized:
                organized[side] = crop
        
        # 좌안, 우안 순서로 반환
        result = []
        if organized['LEFT_EYE'] is not None:
            result.append(organized['LEFT_EYE'])
        if organized['RIGHT_EYE'] is not None:
            result.append(organized['RIGHT_EYE'])
        
        return result
    
    # side 필드 없는 경우 (호환성)
    return eye_crops


def build_selected_eye_fallback_crop(img_bgr, selected_eye):
    """
    [DEPRECATED] MediaPipe는 얼굴을 감지하지 못하면 None을 반환하므로 fallback crop이 필요 없음
    에러 메시지로 사용자에게 재촬영을 유도
    """
    # [LEGACY YOLO] side = normalize_selected_eye(selected_eye)
    # [LEGACY YOLO] if side not in ('L', 'R'):
    # [LEGACY YOLO]     return None
    # [LEGACY YOLO]
    # [LEGACY YOLO] h, w = img_bgr.shape[:2]
    # [LEGACY YOLO] crop_w = max(120, int(w * 0.38))
    # [LEGACY YOLO] crop_h = max(90, int(h * 0.36))
    # [LEGACY YOLO] center_x = int(w * 0.33) if side == 'L' else int(w * 0.67)
    # [LEGACY YOLO] center_y = int(h * 0.44)
    # [LEGACY YOLO]
    # [LEGACY YOLO] x1 = max(0, center_x - crop_w // 2)
    # [LEGACY YOLO] y1 = max(0, center_y - crop_h // 2)
    # [LEGACY YOLO] x2 = min(w, x1 + crop_w)
    # [LEGACY YOLO] y2 = min(h, y1 + crop_h)
    # [LEGACY YOLO]
    # [LEGACY YOLO] if x2 <= x1 or y2 <= y1:
    # [LEGACY YOLO]     return None
    # [LEGACY YOLO]
    # [LEGACY YOLO] crop = img_bgr[y1:y2, x1:x2]
    # [LEGACY YOLO] if crop is None or crop.size == 0:
    # [LEGACY YOLO]     return None
    # [LEGACY YOLO]
    # [LEGACY YOLO] return {
    # [LEGACY YOLO]     'image': crop,
    # [LEGACY YOLO]     'bbox': (x1, y1, x2, y2),
    # [LEGACY YOLO]     'confidence': 0.0,
    # [LEGACY YOLO]     'fallback': True,
    # [LEGACY YOLO]     'fallback_side': side
    # [LEGACY YOLO] }
    
    return None


def analyze_bilateral_from_image(img_bgr, user_id='anonymous', selected_eye=None):
    """
    단일 캡처 이미지(양안 포함) 분석
    1) MediaPipe Face Mesh로 얼굴 검출 및 눈 랜드마크 추출
    2) 좌안(OS)과 우안(OD)에 대해 정방형 크롭 생성
    3) EfficientNet으로 양안 순차 분류
    """
    try:
        manager = get_models()
        detector = manager.get_detector()
        classifier = manager.get_classifier()
        analyzer = manager.get_analyzer()

        source_h, source_w = img_bgr.shape[:2]

        # Step 1) MediaPipe Face Mesh 검출 + 눈 크롭
        detection_start = time.time()
        
        # MediaPipe에서 얼굴 검출 및 랜드마크 추출
        detection_result = detector.detect(img_bgr)
        
        if detection_result is None:
            # [ERROR] 얼굴을 인식할 수 없음
            return {
                'status': 'error',
                'message': '얼굴을 인식할 수 없습니다. 다시 촬영해주세요.',
                'left_eye': None,
                'right_eye': None,
                'mediapipe_cam_image_url': None,
                'meta': {
                    'source_resolution': [source_w, source_h],
                    'eyes_detected': 0,
                    'detection_method': 'mediapipe_face_mesh',
                    'detection_time_ms': round((time.time() - detection_start) * 1000.0, 1),
                    'face_detected': False
                }
            }
        
        # 양안 크롭 추출
        raw_eye_crops = detector.crop_eyes(img_bgr, detection_result)
        detection_elapsed_ms = (time.time() - detection_start) * 1000.0
        
        # MediaPipe은 이미 좌/우를 구분하여 반환하므로 select_distinct_eye_crops는 단순 정렬만 수행
        eye_crops = select_distinct_eye_crops(raw_eye_crops, source_w, source_h)
        
        # 메모리 정리
        del detection_result
        del raw_eye_crops
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Strict pipeline: 양안이 모두 검출되지 않으면 전체 분석을 중단한다.
        if len(eye_crops) != 2:
            detected_count = len(eye_crops)
            detected_sides = {str(crop.get('side', '')).upper() for crop in eye_crops}
            missing_parts = []
            if 'LEFT_EYE' not in detected_sides:
                missing_parts.append('left_eye')
            if 'RIGHT_EYE' not in detected_sides:
                missing_parts.append('right_eye')
            return {
                'status': 'error',
                'message': '좌안/우안이 모두 검출되지 않아 분석을 중단했습니다. 얼굴 전체가 프레임에 들어오도록 다시 촬영해주세요.',
                'left_eye': None,
                'right_eye': None,
                'mediapipe_mesh_image_url': None,
                'mediapipe_cam_image_url': None,
                'meta': {
                    'source_resolution': [source_w, source_h],
                    'eyes_detected': detected_count,
                    'missing_eye': missing_parts or ['left_eye', 'right_eye'],
                    'detection_method': 'mediapipe_face_mesh',
                    'detection_time_ms': round(detection_elapsed_ms, 1),
                    'face_detected': True
                }
            }

        # Step 2) 시각화 이미지 생성 (MediaPipe Face Mesh 오버레이)
        mesh_overlay_bgr = build_mediapipe_mesh_overlay(img_bgr)
        mediapipe_mesh_image_url = save_cam_image(user_id, 'mediapipe_mesh', mesh_overlay_bgr)

        # 좌/우 레이블링 (MediaPipe은 'side' 필드로 이미 구분됨)
        labeled_eyes = {}
        for crop in eye_crops:
            side_field = crop.get('side', 'LEFT_EYE')
            if 'LEFT_EYE' in side_field:
                labeled_eyes['left_eye'] = crop
            elif 'RIGHT_EYE' in side_field:
                labeled_eyes['right_eye'] = crop

        # Step 3) EfficientNet 순차 분류
        result = {
            'status': 'success',
            'message': '양안 분석 완료',
            'left_eye': None,
            'right_eye': None,
            'mediapipe_mesh_image_url': mediapipe_mesh_image_url,
            'mediapipe_cam_image_url': mediapipe_mesh_image_url,
            'meta': {
                'source_resolution': [source_w, source_h],
                'eyes_detected': 2,
                'detection_method': 'mediapipe_face_mesh',
                'detection_time_ms': round(detection_elapsed_ms, 1)
            }
        }

        # 양안 순차 분류
        for side in ['left_eye', 'right_eye']:
            eye_item = labeled_eyes.get(side)
            if eye_item is None:
                return {
                    'status': 'error',
                    'message': '좌안/우안이 모두 검출되지 않아 분석을 중단했습니다. 얼굴 전체가 프레임에 들어오도록 다시 촬영해주세요.',
                    'left_eye': None,
                    'right_eye': None,
                    'mediapipe_mesh_image_url': mediapipe_mesh_image_url,
                    'mediapipe_cam_image_url': mediapipe_mesh_image_url,
                    'meta': {
                        'source_resolution': [source_w, source_h],
                        'eyes_detected': len(eye_crops),
                        'detection_method': 'mediapipe_face_mesh',
                        'detection_time_ms': round(detection_elapsed_ms, 1),
                        'face_detected': True,
                        'missing_eye': [side]
                    }
                }

            cls_start = time.time()
            prepared_eye = upscale_eye_crop_for_classifier(eye_item['image'])
            if prepared_eye is None:
                return {
                    'status': 'error',
                    'message': '눈 크롭 생성에 실패했습니다. 다시 촬영해주세요.',
                    'left_eye': None,
                    'right_eye': None,
                    'mediapipe_mesh_image_url': mediapipe_mesh_image_url,
                    'mediapipe_cam_image_url': mediapipe_mesh_image_url,
                    'meta': {
                        'source_resolution': [source_w, source_h],
                        'eyes_detected': len(eye_crops),
                        'detection_method': 'mediapipe_face_mesh',
                        'detection_time_ms': round(detection_elapsed_ms, 1),
                        'face_detected': True,
                        'failed_eye': side
                    }
                }

            classification = classifier.classify_with_details(prepared_eye, generate_cam=True)
            eye_analysis = analyzer.analyze(prepared_eye)
            cls_elapsed_ms = (time.time() - cls_start) * 1000.0

            cam_image_url = save_cam_image(user_id, side, classification.get('heatmap_image'))

            result[side] = {
                'disease': classification['disease'],
                'class': classification['class'],
                'confidence': round(float(classification['confidence']) * 100.0, 2),
                'redness': round(float(eye_analysis.get('redness', 0.0)), 4),
                'bbox': eye_item['bbox'],
                'cam_image_url': cam_image_url,
                'detection_confidence': 100.0,  # MediaPipe는 신뢰도를 직접 제공하지 않음
                'process_time_ms': round(cls_elapsed_ms, 1)
            }

            del prepared_eye
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"[ERROR] 양안 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'분석 중 오류가 발생했습니다: {str(e)}'
        }


def analyze_bilateral_from_base64(base64_image_data, user_id='anonymous', selected_eye=None):
    img_bgr, decode_error = decode_base64_image(base64_image_data)
    if decode_error:
        return {
            'status': 'error',
            'message': decode_error
        }

    return analyze_bilateral_from_image(img_bgr, user_id=user_id, selected_eye=selected_eye)


def build_ai_guide(analysis):
    """분석 결과 기반 리포트 가이드 생성"""
    left = analysis.get('left_eye') or {}
    right = analysis.get('right_eye') or {}

    candidates = []
    if left.get('disease'):
        candidates.append({'side': 'left', 'disease': left.get('disease', ''), 'confidence': float(left.get('confidence', 0))})
    if right.get('disease'):
        candidates.append({'side': 'right', 'disease': right.get('disease', ''), 'confidence': float(right.get('confidence', 0))})

    if not candidates:
        return {
            'risk_level': 'safe',
            'tag_text': '안내: 유효한 분석 데이터 없음',
            'summary': '눈 영역 분석 데이터가 부족합니다. 조명을 보강하고 다시 촬영해 주세요.',
            'recommended_departments': ['일반 안과'],
            'daily_care': [
                '촬영 시 정면 응시 및 충분한 조명을 확보하세요.',
                '눈 자극(손 비비기, 렌즈 장시간 착용)을 피하세요.'
            ]
        }

    top = max(candidates, key=lambda x: x['confidence'])
    disease = str(top['disease'])
    conf = float(top['confidence'])

    if conf >= 80:
        risk_level = 'danger'
    elif conf >= 60:
        risk_level = 'warning'
    else:
        risk_level = 'safe'

    rule = {
        'tag': f'분석 결과: {disease} ({conf:.1f}%)',
        'summary': f"{top['side']} 눈에서 {disease} 가능성이 관측되었습니다.",
        'dept': ['일반 안과'],
        'care': [
            '건조한 환경을 피하고 실내 습도를 유지하세요.',
            '눈 비비기를 피하고 필요 시 인공눈물을 사용하세요.'
        ]
    }

    if '결막염' in disease or 'Conjunctivitis' in disease:
        rule = {
            'tag': f'주의: 결막염 의심 ({conf:.1f}%)',
            'summary': '충혈/가려움 등 결막염 관련 징후가 관찰되었습니다. 증상이 지속되면 진료를 권장합니다.',
            'dept': ['일반 안과', '안구건조증 클리닉'],
            'care': ['손 위생을 철저히 하고 눈 비비기를 피하세요.', '화면 사용 중 20-20-20 규칙으로 눈 피로를 줄이세요.']
        }
    elif '백내장' in disease or 'Cataract' in disease:
        rule = {
            'tag': f'주의: 백내장 징후 ({conf:.1f}%)',
            'summary': '수정체 혼탁 관련 징후가 감지되었습니다. 시야 흐림이 느껴지면 정밀검사가 필요합니다.',
            'dept': ['백내장/망막 전문 안과', '일반 안과'],
            'care': ['야간 눈부심/시야 흐림 증상을 기록해 진료 시 전달하세요.', '정기적인 시력검사 일정을 권장합니다.']
        }
    elif '포도막염' in disease or 'Uveitis' in disease:
        rule = {
            'tag': f'주의: 포도막염 의심 ({conf:.1f}%)',
            'summary': '염증성 징후가 관측되었습니다. 통증/광과민이 있으면 빠른 진료가 필요합니다.',
            'dept': ['염증성 안질환 클리닉', '일반 안과'],
            'care': ['강한 빛 노출을 줄이고 선글라스를 사용하세요.', '통증/충혈 악화 시 즉시 병원 방문을 권장합니다.']
        }
    elif '다래끼' in disease or 'Eyelid' in disease:
        rule = {
            'tag': f'주의: 다래끼 가능성 ({conf:.1f}%)',
            'summary': '눈꺼풀 주변 염증 징후가 감지되었습니다. 증상 지속 시 진료를 권장합니다.',
            'dept': ['일반 안과'],
            'care': ['눈꺼풀을 청결하게 유지하고 화장품 사용을 줄이세요.', '온찜질을 짧게 하루 1~2회 적용하세요.']
        }
    elif '일반' in disease or 'Normal' in disease:
        rule = {
            'tag': f'정상 소견 우세 ({conf:.1f}%)',
            'summary': '현재 촬영 기준으로 특이 소견이 상대적으로 낮습니다.',
            'dept': ['정기 검진용 일반 안과'],
            'care': ['장시간 화면 사용 시 주기적 휴식을 취하세요.', '충혈/통증이 생기면 조기 검진을 권장합니다.']
        }

    return {
        'risk_level': risk_level,
        'tag_text': rule['tag'],
        'summary': rule['summary'],
        'recommended_departments': rule['dept'],
        'daily_care': rule['care']
    }


# ========================================
# [3] GStreamer 파이프라인
# ========================================
def gstreamer_pipeline(cam_id=0):
    """
    Logitech C920 웹캠용 GStreamer 파이프라인
    
    Args:
        cam_id: 비디오 장치 ID (기본값: 0 = /dev/video0)
        
    Returns:
        GStreamer 파이프라인 문자열
    """
    return (
        f"v4l2src device=/dev/video{cam_id} ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegdec ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    )

# ========================================
# [4] 백그라운드 카메라 스레드 
# ========================================
def start_camera_thread(camera_role='webcam'):
    """
    카메라 프레임을 백그라운드에서 계속 읽고 current_frame 업데이트
    """
    global current_frame, camera_running, camera_thread
    global active_camera_role, active_camera_device_index

    requested_role = normalize_camera_role(camera_role)
    requested_device_index = get_camera_device_index(requested_role)

    if camera_running and camera_thread is not None and camera_thread.is_alive():
        if active_camera_device_index != requested_device_index:
            raise RuntimeError(
                f'{CAMERA_ROLE_LABELS.get(active_camera_role, "카메라")} 사용 중에는 '
                f'{CAMERA_ROLE_LABELS[requested_role]}로 전환할 수 없습니다.'
            )
        return

    active_camera_role = requested_role
    active_camera_device_index = requested_device_index
    
    def camera_worker():
        global current_frame, camera_running
        print(
            f"[카메라 스레드] 백그라운드 프레임 수집 시작... "
            f"role={requested_role}, device=/dev/video{requested_device_index}"
        )
        time.sleep(1) # 초기화 대기

        cam_id = requested_device_index
        try:
            cap, first_frame, camera_metadata = open_usb_uvc_camera(
                cam_id,
                gstreamer_pipeline=gstreamer_pipeline,
            )
        except UvcCameraError as error:
            print(f"[카메라 스레드] 카메라 열기 실패: {error}")
            camera_running = False
            return

        current_frame = first_frame.copy()
        print(
            '[카메라 스레드] USB UVC 연결 성공: '
            f"role={requested_role}, path={camera_metadata['role_path']}, "
            f"backend={camera_metadata['backend']}, "
            f"resolution={camera_metadata['width']}x{camera_metadata['height']}, "
            f"fps={camera_metadata['fps']:.1f}, fourcc={camera_metadata['fourcc'] or 'unknown'}"
        )
        
        print("[카메라 스레드] ✓ 카메라 초기화 완료, 프레임 수집 중...")
        frame_count = 1
        
        while camera_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[카메라 스레드] ✗ 프레임 읽기 실패!")
                time.sleep(1)
                continue
            
            current_frame = frame.copy()
            frame_count += 1
            
            # 잘 돌아가는지 확인하기 위한 하트비트 로그 (일단 켜둡니다)
            if frame_count % 30 == 0:
                print(f"[카메라 스레드] {frame_count}개 프레임 수집 완료")
            
            time.sleep(0.033)  # ~30fps
        
        cap.release()
        print(f"[카메라 스레드] 종료 ({frame_count}개 프레임 수집됨)")
    
    camera_running = True
    camera_thread = threading.Thread(target=camera_worker, daemon=True)
    camera_thread.start()
    print("[카메라 스레드] ✓ 시작")


def stop_camera_thread():
    """카메라 스레드 안전 종료"""
    global camera_running, camera_thread, current_frame
    global active_camera_role, active_camera_device_index
    camera_running = False

    if camera_thread is not None and camera_thread.is_alive():
        camera_thread.join(timeout=2.0)

    camera_thread = None
    current_frame = None
    active_camera_role = None
    active_camera_device_index = None


def acquire_camera_session(camera_role='webcam'):
    """capture 페이지 활성화 시 카메라 세션을 획득하고 필요하면 카메라를 시작한다."""
    global camera_session_count
    requested_role = normalize_camera_role(camera_role)
    requested_device_index = get_camera_device_index(requested_role)

    with camera_session_lock:
        if (
            camera_session_count > 0
            and active_camera_device_index is not None
            and active_camera_device_index != requested_device_index
        ):
            raise RuntimeError(
                f'{CAMERA_ROLE_LABELS.get(active_camera_role, "카메라")} 세션이 이미 사용 중입니다.'
            )
        camera_session_count += 1
        active_sessions = camera_session_count

    try:
        if not camera_running:
            start_camera_thread(requested_role)
    except Exception:
        with camera_session_lock:
            camera_session_count = max(0, camera_session_count - 1)
        raise

    return active_sessions


def release_camera_session(force=False):
    """capture 페이지 이탈 시 카메라 세션을 해제하고 마지막 세션이면 카메라를 종료한다."""
    global camera_session_count
    with camera_session_lock:
        if force:
            camera_session_count = 0
        elif camera_session_count > 0:
            camera_session_count -= 1
        active_sessions = camera_session_count

    if active_sessions == 0:
        stop_camera_thread()

    return active_sessions


def get_camera_session_count():
    """현재 활성 카메라 세션 수 조회"""
    with camera_session_lock:
        return camera_session_count


# ========================================
# [4] 카메라 프레임 스트림
# ========================================
def gen_frames(show_mesh=True):
    """
    백그라운드 스레드가 수집한 current_frame을 브라우저로 실시간 스트리밍
    """
    global current_frame, debug_boxes_cache, debug_frame_counter
    print("[gen_frames] 브라우저 스트리밍 시작...")
    
    while True:
        if current_frame is None:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy, 'Waiting for camera...', (170, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            ret, buffer = cv2.imencode('.jpg', dummy)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
            
        frame_to_send = current_frame.copy()
        if show_mesh:
            mesh_overlay = build_mediapipe_mesh_overlay(frame_to_send)
            if mesh_overlay is not None:
                frame_to_send = mesh_overlay

        # [LEGACY YOLO] 스트림 오버레이 YOLO는 GPU 메모리 사용량이 커서 기본 비활성화한다.
        # [LEGACY YOLO] if YOLO_STREAM_DEBUG_OVERLAY:
        # [LEGACY YOLO]     try:
        # [LEGACY YOLO]         debug_frame_counter += 1
        # [LEGACY YOLO]         if debug_frame_counter % 10 == 0 or len(debug_boxes_cache) == 0:
        # [LEGACY YOLO]             manager = get_models()
        # [LEGACY YOLO]             detector = manager.get_detector()
        # [LEGACY YOLO]             detections = safe_yolo_detect(
        # [LEGACY YOLO]                 detector,
        # [LEGACY YOLO]                 frame_to_send,
        # [LEGACY YOLO]                 conf_threshold=config.YOLO_STATUS_CONF_THRESHOLD
        # [LEGACY YOLO]             )
        # [LEGACY YOLO]
        # [LEGACY YOLO]             boxes = []
        # [LEGACY YOLO]             if detections is not None and hasattr(detections, 'boxes') and detections.boxes is not None:
        # [LEGACY YOLO]                 for box in detections.boxes:
        # [LEGACY YOLO]                     x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # [LEGACY YOLO]                     conf = float(box.conf.item()) if box.conf is not None else 0.0
        # [LEGACY YOLO]                     boxes.append((x1, y1, x2, y2, conf))
        # [LEGACY YOLO]
        # [LEGACY YOLO]             debug_boxes_cache = boxes
        # [LEGACY YOLO]
        # [LEGACY YOLO]         for (x1, y1, x2, y2, conf) in debug_boxes_cache:
        # [LEGACY YOLO]             cv2.rectangle(frame_to_send, (x1, y1), (x2, y2), (40, 205, 65), 2)
        # [LEGACY YOLO]             label = f"Eye {conf*100:.1f}%"
        # [LEGACY YOLO]             cv2.putText(
        # [LEGACY YOLO]                 frame_to_send,
        # [LEGACY YOLO]                 label,
        # [LEGACY YOLO]                 (x1, max(20, y1 - 8)),
        # [LEGACY YOLO]                 cv2.FONT_HERSHEY_SIMPLEX,
        # [LEGACY YOLO]                 0.55,
        # [LEGACY YOLO]                 (40, 205, 65),
        # [LEGACY YOLO]                 2
        # [LEGACY YOLO]             )
        # [LEGACY YOLO]     except Exception as e:
        # [LEGACY YOLO]         print(f"[WARNING] YOLO 박스 오버레이 실패: {e}")

        ret, buffer = cv2.imencode('.jpg', frame_to_send)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
               
        time.sleep(0.033)


# ========================================
# [5] 핵심 진단 파이프라인 (3단계)
# ========================================
def run_diagnosis_pipeline(snapshot, language='ko'):
    """
    3단계 진단 파이프라인 - i18n 언어 지원 추가
    
    Stage 1: YOLO 눈 검출 및 크롭
    Stage 2: EfficientNet 질환 분류
    Stage 3: 홍채 제거 및 충혈도 산출
    
    Args:
        snapshot (np.ndarray): 캡처된 프레임
        language (str): i18n 언어 코드 (ko|en|zh|vi)
        
    Returns:
        dict: 진단 결과 (language 필드 포함)
    """
    try:
        # 모델 매니저 가져오기
        manager = get_models()
        detector = manager.get_detector()
        classifier = manager.get_classifier()
        analyzer = manager.get_analyzer()
        logger = manager.get_logger()
        
        result = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'language': language,
            'left_eye': None,
            'right_eye': None,
            'pipeline_steps': []
        }
        
        # ============================================
        # Stage 1: 얼굴 및 눈 검출 (MediaPipe Face Mesh)
        # ============================================
        print("[Stage 1] MediaPipe Face Mesh 얼굴 및 눈 검출 시작...")
        start_time = time.time()
        
        # MediaPipe 기반 검출
        detection_result = detector.detect(snapshot)
        
        if detection_result is None:
            result['status'] = 'error'
            result['message'] = '얼굴을 인식할 수 없습니다. 다시 촬영해주세요.'
            result['pipeline_steps'].append({
                'stage': 'Detection (MediaPipe)',
                'status': 'failed',
                'reason': 'No face detected'
            })
            return result
        
        eye_crops = detector.crop_eyes(snapshot, detection_result)
        
        elapsed = time.time() - start_time
        result['pipeline_steps'].append({
            'stage': 'Detection (MediaPipe)',
            'status': 'completed',
            'time_ms': f"{elapsed*1000:.1f}",
            'eyes_detected': len(eye_crops)
        })
        print(f"  ✓ {len(eye_crops)}개 눈 검출 ({elapsed*1000:.1f}ms)")
        
        if len(eye_crops) == 0:
            result['status'] = 'error'
            result['message'] = '얼굴은 인식되었으나 눈을 감지하지 못했습니다. 다시 촬영해주세요.'
            return result
        
        # ============================================
        # Stage 2 & 3: 각 눈마다 분류 및 분석
        # ============================================
        # MediaPipe은 이미 좌/우를 구분하므로 순서대로 처리
        eye_sides = []
        for crop in eye_crops:
            side_field = crop.get('side', 'LEFT_EYE')
            if 'LEFT_EYE' in side_field:
                eye_sides.append(('left_eye', crop))
            elif 'RIGHT_EYE' in side_field:
                eye_sides.append(('right_eye', crop))
        
        for eye_side, eye_crop in eye_sides:  # MediaPipe에서는 이미 구분됨
            print(f"\n[Stage 2-3] {eye_side} 분류 및 분석...")
            
            crop_image = eye_crop['image']
            
            # Stage 2: 질환 분류
            print(f"  Stage 2: 질환 분류 중...")
            start_time = time.time()
            classification = classifier.classify_with_details(crop_image, generate_cam=True)
            elapsed_classify = time.time() - start_time
            print(f"    ✓ {classification['disease']} (신뢰도: {classification['confidence']*100:.1f}%)")
            
            # Stage 3: 눈 분석 (홍채 제거 + 충혈도)
            print(f"  Stage 3: 충혈도 분석 중...")
            start_time = time.time()
            analysis = analyzer.analyze(crop_image)
            elapsed_analyze = time.time() - start_time
            print(f"    ✓ 충혈도: {analysis['redness']:.3f}")

            cam_image_url = save_cam_image('anonymous', eye_side, classification.get('heatmap_image'))
            
            # 결과 저장
            result[eye_side] = {
                'detected': True,
                'disease': classification['disease'],
                'disease_class': classification['class'],
                'confidence': float(classification['confidence']),
                'probabilities': classification['probabilities'],
                'redness': float(analysis['redness']),
                'bbox': eye_crop['bbox'],
                'cam_image_url': cam_image_url,
                'detection_confidence': float(eye_crop['confidence']),
                'processing_time_ms': f"{elapsed_classify + elapsed_analyze:.1f}"
            }
            
            result['pipeline_steps'].append({
                'stage': f'Classification & Analysis ({eye_side})',
                'status': 'completed',
                'time_ms': f"{elapsed_classify + elapsed_analyze:.1f}",
                'disease': classification['disease'],
                'redness': f"{analysis['redness']:.3f}"
            })
        
        # ============================================
        # 결과 로깅
        # ============================================
        try:
            logger.log_result({
                'patient_id': request.form.get('patient_id', 'unknown'),
                'left_eye': result.get('left_eye'),
                'right_eye': result.get('right_eye'),
                'notes': request.form.get('notes', '')
            })
        except Exception as e:
            print(f"[WARNING] 로깅 실패: {e}")
        
        print(f"\n[진단 완료] 전체 소요시간: {time.time():.2f}초")
        return result
        
    except Exception as e:
        print(f"[ERROR] 진단 파이프라인 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }


# ========================================
# [6] Flask 라우트
# ========================================

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/api/screening/config', methods=['GET'])
def api_screening_config():
    """Return modality metadata used by the kiosk and mobile UI."""
    return jsonify({'status': 'ok', **load_screening_config()}), 200


@app.route('/api/generate_pin', methods=['POST'])
def api_generate_pin():
    """키오스크에서 모바일 접속용 4자리 PIN 발급"""
    try:
        cleanup_expired_mobile_pins()
        request_id = uuid.uuid4().hex[:12]
        pin_code = f"{random.randint(0, 9999):04d}"

        with MOBILE_PIN_LOCK:
            MOBILE_PIN_STORE[request_id] = {
                'pin': pin_code,
                'created_at_ts': time.time(),
                'mobile_connected': False,
                'verified': False
            }

        mobile_base = resolve_mobile_base_url()
        mobile_url = f"{mobile_base}/m?request_id={request_id}"

        return jsonify({
            'status': 'ok',
            'request_id': request_id,
            'mobile_url': mobile_url,
            'expires_in_sec': MOBILE_PIN_TTL_SECONDS
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/pin_status', methods=['GET'])
def api_pin_status():
    """키오스크 모달에서 PIN 상태(모바일 접속/인증 완료) 확인"""
    try:
        request_id = str(request.args.get('request_id', '')).strip()
        payload = get_mobile_entry(request_id)
        if not payload:
            return jsonify({'status': 'error', 'message': '유효하지 않거나 만료된 PIN입니다.'}), 404

        created_at_ts = float(payload.get('created_at_ts', 0))
        remaining = max(0, MOBILE_PIN_TTL_SECONDS - int(time.time() - created_at_ts))

        return jsonify({
            'status': 'ok',
            'request_id': request_id,
            'mobile_connected': bool(payload.get('mobile_connected', False)),
            'verified': bool(payload.get('verified', False)),
            'pin': payload.get('pin') if payload.get('mobile_connected', False) else None,
            'expires_in_sec': remaining
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/mobile_connected', methods=['POST'])
def api_mobile_connected():
    """모바일 /m 진입 확인 신호"""
    try:
        data = request.json or {}
        request_id = str(data.get('request_id', '')).strip()
        if not request_id:
            return jsonify({'status': 'error', 'message': 'request_id가 필요합니다.'}), 400

        cleanup_expired_mobile_pins()
        with MOBILE_PIN_LOCK:
            payload = MOBILE_PIN_STORE.get(request_id)
            if not payload:
                return jsonify({'status': 'error', 'message': '유효하지 않거나 만료된 PIN입니다.'}), 404
            payload['mobile_connected'] = True

        return jsonify({'status': 'ok', 'request_id': request_id}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/m')
def mobile_login_page():
    """모바일 PIN 입력 전용 화면"""
    request_id = str(request.args.get('request_id', '')).strip()
    return render_template('m_login.html', request_id=request_id)


@app.route('/api/verify_pin', methods=['POST'])
def api_verify_pin():
    """모바일 PIN 검증 및 세션 인증"""
    try:
        data = request.json or {}
        request_id = str(data.get('request_id', '')).strip()
        pin = ''.join(ch for ch in str(data.get('pin', '')) if ch.isdigit())[:4]

        if not request_id or len(pin) != 4:
            return jsonify({'status': 'error', 'message': 'request_id와 4자리 PIN이 필요합니다.'}), 400

        cleanup_expired_mobile_pins()
        with MOBILE_PIN_LOCK:
            payload = MOBILE_PIN_STORE.get(request_id)
            if not payload:
                return jsonify({'status': 'error', 'message': '유효하지 않거나 만료된 PIN입니다.'}), 404

            if str(payload.get('pin')) != pin:
                return jsonify({'status': 'error', 'message': 'PIN 번호가 올바르지 않습니다.'}), 401

            payload['verified'] = True

        session['mobile_verified'] = True
        session['mobile_request_id'] = request_id

        return jsonify({
            'status': 'ok',
            'verified': True,
            'request_id': request_id,
            'redirect_url': f"/m/dashboard?request_id={request_id}"
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/m/dashboard')
def mobile_dashboard_page():
    """모바일 인증 후 접근 가능한 대시보드"""
    request_id = str(request.args.get('request_id', '')).strip()

    if session.get('mobile_verified'):
        return render_template('m_dashboard.html')

    if request_id and is_verified_mobile_request(request_id):
        session['mobile_verified'] = True
        session['mobile_request_id'] = request_id
        return render_template('m_dashboard.html')

    return redirect(url_for('mobile_login_page', request_id=request_id))


@app.route('/video_feed')
def video_feed():
    """실시간 영상 스트림 (MJPEG)"""
    show_mesh = str(request.args.get('mesh', '1')).strip().lower() not in ('0', 'false', 'no', 'off')
    return Response(gen_frames(show_mesh=show_mesh),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/session/start', methods=['POST'])
def camera_session_start():
    """capture 화면 진입 시 카메라 세션 시작"""
    try:
        data = request.get_json(silent=True) or {}
        requested_role = normalize_camera_role(data.get('camera_role', 'webcam'))
        session_role = normalize_camera_role(session.get('capture_camera_role', requested_role))

        if not session.get('capture_camera_active', False):
            active_sessions = acquire_camera_session(requested_role)
            session['capture_camera_active'] = True
            session['capture_camera_role'] = requested_role
        else:
            if session_role != requested_role:
                return jsonify({
                    'status': 'error',
                    'message': '현재 카메라 세션을 종료한 뒤 다른 카메라를 시작해 주세요.'
                }), 409
            if not camera_running:
                start_camera_thread(requested_role)
            active_sessions = get_camera_session_count()

        return jsonify({
            'status': 'ok',
            'camera_running': camera_running,
            'active_sessions': active_sessions,
            'camera_role': requested_role,
            'camera': describe_camera(requested_role),
        }), 200
    except RuntimeError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 409
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/camera/session/stop', methods=['POST'])
def camera_session_stop():
    """capture 화면 이탈 시 카메라 세션 종료"""
    try:
        if session.get('capture_camera_active', False):
            active_sessions = release_camera_session()
            session['capture_camera_active'] = False
            session.pop('capture_camera_role', None)
        else:
            active_sessions = get_camera_session_count()

        return jsonify({
            'status': 'ok',
            'camera_running': camera_running,
            'active_sessions': active_sessions
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/video_frame')
def video_frame():
    """현재 카메라 프레임을 JPEG로 반환"""
    global current_frame
    try:
        if current_frame is None:
            # 검정색 더미 프레임 반환
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', dummy)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        
        show_mesh = str(request.args.get('mesh', '1')).strip().lower() not in ('0', 'false', 'no', 'off')
        frame_to_send = current_frame
        if show_mesh:
            mesh_overlay = build_mediapipe_mesh_overlay(current_frame)
            frame_to_send = mesh_overlay if mesh_overlay is not None else current_frame
        ret, buffer = cv2.imencode('.jpg', frame_to_send)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        print(f"[ERROR] /video_frame 실패: {e}")
        return Response(b'', status=500)


@app.route('/detect_status', methods=['GET'])
def detect_status():
    """
    현재 프레임에서 눈의 감지 여부만 반환 (HTML에서 상태 표시용)
    
    Returns:
        - detected: bool - 눈 감지 여부
        - confidence: float - 신뢰도 (0~1)
        - message: str - 상태 메시지
    """
    global current_frame
    try:
        if current_frame is None:
            return jsonify({
                'detected': False,
                'confidence': 0,
                'message': '카메라 연결을 기다리는 중...'
            }), 200
        
        detected, confidence = check_eye_detection(current_frame)
        
        return jsonify({
            'detected': detected,
            'confidence': round(confidence, 3),
            'message': '🟢 눈이 감지되었습니다!' if detected else '⚪ 눈을 맞춰주세요'
        }), 200
    
    except Exception as e:
        print(f"[ERROR] 눈 감지 상태 조회 실패: {e}")
        return jsonify({
            'detected': False,
            'confidence': 0,
            'error': str(e)
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    웹 브라우저에서 캡처한 단일 Base64 이미지를
    기본은 YOLO(양안 검출/크롭) -> EfficientNet(양안 순차 분류),
    업로드+비모바일일 때는 YOLO를 건너뛰고 단안 분류로 분석
    
    Request:
        {
            "image": "data:image/jpeg;base64,..."
        }
    
    Response:
        {
            "status": "success|warning|error",
            "message": "...",
            "left_eye": {...},
            "right_eye": {...},
            "meta": {...}
        }
    """
    try:
        data = request.json or {}
        if 'image' not in data:
            return jsonify({'error': '이미지 데이터가 없습니다'}), 400

        user_id = normalize_user_id(data.get('user_id'))
        selected_eye = data.get('selected_eye')
        capture_source = str(data.get('capture_source', '')).strip().lower()
        mobile_upload_only = bool(data.get('mobile_upload_only', False))
        img_bgr, decode_error = decode_base64_image(data['image'])
        if decode_error:
            return jsonify({'status': 'error', 'message': decode_error}), 400

        skip_yolo_for_upload = (capture_source == 'upload' and not mobile_upload_only)

        if skip_yolo_for_upload:
            analysis = analyze_uploaded_single_eye_from_image(
                img_bgr,
                user_id=user_id,
                selected_eye=selected_eye
            )
        else:
            analysis = analyze_bilateral_from_image(img_bgr, user_id=user_id, selected_eye=selected_eye)

        if analysis.get('status') == 'error':
            error_message = str(analysis.get('message', ''))
            soft_failure_messages = (
                '얼굴을 인식할 수 없습니다.',
                '얼굴은 인식되었으나 눈을 감지하지 못했습니다.',
            )
            if any(text in error_message for text in soft_failure_messages):
                analysis['status'] = 'warning'
                analysis['message'] = error_message
                analysis['meta'] = analysis.get('meta') or {}
                analysis['meta']['analysis_warning'] = error_message
                return jsonify(analysis), 200

            return jsonify(analysis), 400

        if not skip_yolo_for_upload:
            analysis = apply_selected_eye_mapping(analysis, selected_eye)

        meta = analysis.get('meta') or {}
        meta['capture_source'] = capture_source or 'camera'
        meta['mobile_upload_only'] = mobile_upload_only
        analysis['meta'] = meta

        analysis['guide'] = build_ai_guide(analysis)

        try:
            history_id, image_url = save_history_record(user_id, img_bgr, analysis)
            analysis['history_id'] = history_id
            analysis['saved_image_url'] = image_url
            analysis['user_id'] = user_id
        except Exception as save_error:
            analysis['save_warning'] = f'히스토리 저장 실패: {save_error}'

        return jsonify(analysis), 200
        
    except Exception as e:
        print(f"[ERROR] /analyze 라우트 실패: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def api_history():
    """사용자별 진단 히스토리 조회"""
    try:
        user_id = normalize_user_id(request.args.get('user_id'))
        limit = int(request.args.get('limit', 200))
        limit = max(1, min(limit, 1000))

        history = list_history_records(user_id, limit)
        return jsonify({
            'status': 'ok',
            'user_id': user_id,
            'count': len(history),
            'history': history
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/history/<int:history_id>', methods=['DELETE'])
def api_history_delete(history_id):
    """사용자별 진단 히스토리 삭제"""
    try:
        user_id = normalize_user_id(request.args.get('user_id'))
        if user_id == 'anonymous':
            return jsonify({'status': 'error', 'message': '사용자 식별자가 필요합니다.'}), 400

        deleted = delete_history_record(user_id, history_id)
        if not deleted:
            return jsonify({'status': 'error', 'message': '삭제할 기록을 찾지 못했습니다.'}), 404

        return jsonify({'status': 'ok', 'deleted_id': history_id}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/survey', methods=['POST'])
def api_survey():
    """사용자 문진 데이터 저장"""
    try:
        data = request.json or {}
        user_id = data.get('user_id')
        survey = data.get('survey')

        safe_user_id = normalize_user_id(user_id)
        if safe_user_id == 'anonymous':
            return jsonify({
                'status': 'error',
                'message': '사용자 식별자가 필요합니다.'
            }), 400

        row_id, normalized_survey = save_survey_record(safe_user_id, survey)
        return jsonify({
            'status': 'ok',
            'survey_id': row_id,
            'user_id': safe_user_id,
            'survey': normalized_survey
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/survey', methods=['GET'])
def api_survey_list():
    """사용자 문진 데이터 조회"""
    try:
        user_id = normalize_user_id(request.args.get('user_id'))
        if user_id == 'anonymous':
            return jsonify({'status': 'error', 'message': '사용자 식별자가 필요합니다.'}), 400

        limit = int(request.args.get('limit', 50))
        limit = max(1, min(limit, 500))

        surveys = list_survey_records(user_id, limit)
        return jsonify({
            'status': 'ok',
            'user_id': user_id,
            'count': len(surveys),
            'surveys': surveys
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/survey/<int:survey_id>', methods=['DELETE'])
def api_survey_delete(survey_id):
    """사용자 문진 데이터 삭제"""
    try:
        user_id = normalize_user_id(request.args.get('user_id'))
        if user_id == 'anonymous':
            return jsonify({'status': 'error', 'message': '사용자 식별자가 필요합니다.'}), 400

        deleted = delete_survey_record(user_id, survey_id)
        if not deleted:
            return jsonify({'status': 'error', 'message': '삭제할 설문을 찾지 못했습니다.'}), 404

        return jsonify({'status': 'ok', 'deleted_id': survey_id}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _normalize_diagnosis_result_for_chat(diagnosis_result):
    """채팅 프롬프트에 넣기 전에 진단 결과를 JSON 직렬화 가능한 형태로 정규화한다."""
    if diagnosis_result is None:
        return {}

    if isinstance(diagnosis_result, (dict, list)):
        try:
            # numpy scalar 등 비표준 타입이 섞여 있을 수 있어 round-trip 정규화
            return json.loads(json.dumps(diagnosis_result, ensure_ascii=False, default=str))
        except Exception:
            return {'raw_result': str(diagnosis_result)}

    return {'raw_result': str(diagnosis_result)}


def _call_openai_chat(system_prompt, user_message):
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY is not configured')

    model_name = os.getenv('OPENAI_MODEL', 'gpt-4o-mini').strip() or 'gpt-4o-mini'
    url = 'https://api.openai.com/v1/chat/completions'
    payload = {
        'model': model_name,
        'temperature': 0.4,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message}
        ]
    }

    response = requests.post(
        url,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        json=payload,
        timeout=35
    )

    if response.status_code != 200:
        detail = response.text[:400] if response.text else 'no detail'
        raise RuntimeError(f'OpenAI API error ({response.status_code}): {detail}')

    data = response.json()
    choices = data.get('choices') or []
    if not choices:
        raise RuntimeError('OpenAI API returned empty choices')

    message = (choices[0] or {}).get('message') or {}
    content = (message.get('content') or '').strip()
    if not content:
        raise RuntimeError('OpenAI API returned empty content')

    return content


def _call_gemini_chat(system_prompt, user_message):
    api_key = os.getenv('GEMINI_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY is not configured')

    prompt = (
        f"System instruction:\n{system_prompt}\n\n"
        f"User question:\n{user_message}"
    )
    payload = {
        'contents': [
            {
                'role': 'user',
                'parts': [{'text': prompt}]
            }
        ],
        'generationConfig': {
            'temperature': 0.4
        }
    }

    configured_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash').strip() or 'gemini-2.5-flash'
    candidate_models = []
    for model_name in [
        configured_model,
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-2.0-flash',
        'gemini-2.0-flash-001'
    ]:
        normalized = str(model_name or '').strip()
        if normalized and normalized not in candidate_models:
            candidate_models.append(normalized)

    last_error = None
    for model_name in candidate_models:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}'
        response = requests.post(
            url,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=35
        )

        if response.status_code != 200:
            detail = response.text[:400] if response.text else 'no detail'
            message = f'Gemini API error ({response.status_code}) for {model_name}: {detail}'
            last_error = message

            # 지원하지 않는 모델이면 다음 후보를 시도한다.
            if response.status_code == 404 and 'not found' in detail.lower():
                continue
            raise RuntimeError(message)

        data = response.json()
        candidates = data.get('candidates') or []
        if not candidates:
            raise RuntimeError('Gemini API returned empty candidates')

        parts = ((candidates[0] or {}).get('content') or {}).get('parts') or []
        text_chunks = []
        for part in parts:
            text = str((part or {}).get('text') or '').strip()
            if text:
                text_chunks.append(text)

        content = '\n'.join(text_chunks).strip()
        if not content:
            raise RuntimeError('Gemini API returned empty content')

        return content

    raise RuntimeError(last_error or 'Gemini API returned no supported model response')


def generate_llm_chat_reply(user_message, diagnosis_result):
    """LLM 공급자(OpenAI/Gemini)를 선택해 채팅 응답을 생성한다."""
    provider = os.getenv('LLM_PROVIDER', 'openai').strip().lower()
    normalized_result = _normalize_diagnosis_result_for_chat(diagnosis_result)
    system_prompt = build_chat_system_prompt(normalized_result)

    if provider == 'gemini':
        return _call_gemini_chat(system_prompt, user_message), 'gemini'

    # 기본값은 openai
    return _call_openai_chat(system_prompt, user_message), 'openai'


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    채팅 요청을 받아 LLM(OpenAI/Gemini)으로 상담 응답을 생성한다.

    Request JSON:
      - user_message: 사용자 질문
      - diagnosis_result: 진단 결과 JSON
    """
    try:
        data = request.get_json(silent=True) or {}
        user_message = str(data.get('user_message', '')).strip()
        diagnosis_result = data.get('diagnosis_result', {})

        if not user_message:
            return jsonify({
                'status': 'error',
                'message': 'user_message is required'
            }), 400

        if len(user_message) > 2000:
            return jsonify({
                'status': 'error',
                'message': 'user_message is too long (max 2000 chars)'
            }), 400

        reply_text, provider = generate_llm_chat_reply(user_message, diagnosis_result)
        return jsonify({
            'status': 'ok',
            'provider': provider,
            'reply': reply_text
        }), 200

    except RuntimeError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 503
    except Exception as e:
        print(f"[ERROR] /api/chat failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/report/share', methods=['POST'])
def api_report_share():
    """
    report 페이지의 "보고서" 버튼용 API
    1) 최신 검사 세션 기준 PDF 생성
    2) 카카오톡 전송(옵션)
    """
    try:
        data = request.json or {}
        user_id = normalize_user_id(data.get('user_id'))
        send_kakao = bool(data.get('send_kakao', True))

        if user_id == 'anonymous':
            return jsonify({'status': 'error', 'message': '사용자 식별자가 필요합니다.'}), 400

        history = list_history_records(user_id, limit=200)
        if not history:
            return jsonify({'status': 'error', 'message': '보고서를 만들 검사 기록이 없습니다.'}), 404

        sessions = build_history_sessions(history)
        latest_session = sessions[0] if sessions else None
        if not latest_session:
            return jsonify({'status': 'error', 'message': '유효한 검사 세션을 찾지 못했습니다.'}), 404

        surveys = list_survey_records(user_id, limit=1)
        latest_survey = (surveys[0].get('survey') if surveys else None)

        pdf_path, pdf_url = generate_session_pdf_report(user_id, latest_session, latest_survey)

        kakao_result = None
        kakao_error = None
        if send_kakao:
            try:
                kakao_result = send_kakao_report_message(user_id, pdf_url)
            except Exception as error:
                kakao_error = str(error)

        return jsonify({
            'status': 'ok',
            'user_id': user_id,
            'pdf_path': pdf_path,
            'pdf_url': pdf_url,
            'kakao_sent': bool(kakao_result),
            'kakao': kakao_result,
            'kakao_error': kakao_error,
            'message': '보고서 생성이 완료되었습니다.'
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/generate_report', methods=['POST'])
def api_generate_report():
    """
    LLM을 사용해 의료 진단 보고서 생성 (다국어 지원)
    i18n.js의 i18n.generateMedicalReport()에서 호출
    """
    try:
        data = request.json or {}
        language = data.get('language', 'ko')
        if language not in ['ko', 'en', 'zh', 'vi']:
            language = 'ko'
        
        diagnosis_result = data.get('diagnosis_result', {})
        if not diagnosis_result:
            return jsonify({
                'status': 'error',
                'message': '진단 결과가 필요합니다.',
                'language': language
            }), 400
        
        print(f"[/api/generate_report] Generating report in language: {language}")
        
        # 언어별 템플릿 (데모 보고서)
        templates = {
            'ko': """진단 보고서 (한국어)
질병: {disease}
신뢰도: {confidence:.1%}

진단:
안구 진단 검사가 완료되었습니다.
추가 진료가 필요한 경우 안과 의사와 상담하십시오.""",
            'en': """Medical Report (English)
Disease: {disease}
Confidence: {confidence:.1%}

Diagnosis:
Your eye examination has been completed.
Please consult an ophthalmologist if further treatment is needed.""",
            'zh': """医疗报告 (中文)
疾病: {disease}
置信度: {confidence:.1%}

诊断:
您的眼睛检查已完成。
如需进一步治疗,请咨询眼科医生。""",
            'vi': """Báo Cáo Y Tế (Tiếng Việt)
Bệnh: {disease}
Độ tin cậy: {confidence:.1%}

Chẩn đoán:
Kiểm tra mắt của bạn đã hoàn thành.
Vui lòng tham khảo ý kiến bác sĩ nhãn khoa nếu cần điều trị thêm."""
        }
        
        # 진단 데이터 추출
        left_eye = diagnosis_result.get('left_eye', {})
        disease = left_eye.get('disease', 'Normal')
        confidence = left_eye.get('confidence', 0.0)
        
        # 보고서 생성
        template = templates.get(language, templates['ko'])
        report_text = template.format(disease=disease, confidence=confidence)
        
        return jsonify({
            'status': 'ok',
            'report': report_text,
            'language': language,
            'generated_at': datetime.now().isoformat(),
            'model_version': '1.0'
        }), 200
    
    except Exception as e:
        print(f"[ERROR] /api/generate_report failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'language': data.get('language', 'ko') if data else 'ko'
        }), 500


@app.route('/diagnose', methods=['POST'])
def diagnose():
    """
    진단 실행 라우트 - i18n 언어 지원 추가
    클라이언트에서 '진단 시작' 버튼 클릭 시 호출
    Request Body: {"language": "ko"|"en"|"zh"|"vi"}
    """
    global current_frame
    
    if current_frame is None:
        return jsonify({
            'status': 'error',
            'message': '카메라 연결을 확인하세요.',
            'language': 'ko'
        }), 400
    
    try:
        # Language 파라미터 추출 (프론트엔드에서 i18n.js를 통해 전달)
        language = 'ko'
        if request.is_json and request.json:
            language = request.json.get('language', 'ko')
        elif request.form:
            language = request.form.get('language', 'ko')
        
        if language not in ['ko', 'en', 'zh', 'vi']:
            language = 'ko'
        
        print(f"[/diagnose] Request language: {language}")
        
        # 1. 조명 제어 (LED 플래시)
        if LED_AVAILABLE:
            pwm_led.ChangeDutyCycle(100)  # 최대 밝기
            time.sleep(0.3)  # 빛 안정화
        
        # 2. 스냅샷 캡처
        snapshot = current_frame.copy()
        
        # 3. 이미지 전처리 (선택사항)
        snapshot = enhance_contrast(snapshot)
        
        # 4. 조명 낮추기
        if LED_AVAILABLE:
            pwm_led.ChangeDutyCycle(10)
        
        # 5. 진단 파이프라인 실행 (language 전달)
        diagnosis_result = run_diagnosis_pipeline(snapshot, language=language)
        
        # 6. 촬영 상태 관리 및 UI 가이드라인 업데이트
        if diagnosis_result['status'] == 'success':
            state_snapshot = CaptureState.get_snapshot()
            current_eye = state_snapshot['current_eye']
            
            # 현재 촬영 눈 표시
            CaptureState.mark_captured(current_eye)
            state_snapshot = CaptureState.get_snapshot()
            diagnosis_result['current_eye'] = current_eye
            diagnosis_result['captured_eyes'] = state_snapshot['captured_eyes']
            
            # UI 가이드라인 텍스트 업데이트
            if current_eye == "LEFT_EYE":
                diagnosis_result['next_guide_text'] = "오른쪽 눈을 맞춰주세요. 진단 시작을 누르세요."
                CaptureState.move_to_next_eye()  # 오른쪽 눈으로 이동
            else:
                diagnosis_result['next_guide_text'] = "진단이 완료되었습니다."
                # 양쪽 눈 촬영 완료
                completed_snapshot = CaptureState.get_snapshot()
                captured = completed_snapshot['captured_eyes']
                if captured.get("LEFT_EYE") and captured.get("RIGHT_EYE"):
                    diagnosis_result['diagnosis_complete'] = True
                    CaptureState.reset()  # 시퀀스 초기화
        
        # 7. 스냅샷 저장
        if diagnosis_result['status'] == 'success':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            state_snapshot = CaptureState.get_snapshot()
            eye_label = state_snapshot['captured_eyes']
            img_path = os.path.join(
                config.IMAGE_SAVE_DIR, 
                f"diagnosis_{timestamp}_{state_snapshot['current_eye']}.jpg"
            )
            cv2.imwrite(img_path, snapshot)
            diagnosis_result['snapshot_path'] = img_path
        
        return jsonify(diagnosis_result)
    
    except Exception as e:
        print(f"[ERROR] /diagnose 라우트 실패: {e}")
        import traceback
        traceback.print_exc()
        
        if LED_AVAILABLE:
            pwm_led.ChangeDutyCycle(10)
        
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/capture/state', methods=['GET'])
def get_capture_state():
    """
    현재 촬영 상태 조회
    
    Returns:
        - current_eye: 현재 촬영 중인 눈 ("LEFT_EYE" 또는 "RIGHT_EYE")
        - captured_eyes: {"LEFT_EYE": bool, "RIGHT_EYE": bool}
        - auto_capture_ready: 자동 촬영 준비 상태
        - guide_text: UI에 표시할 가이드 텍스트
    """
    try:
        state_snapshot = CaptureState.get_snapshot()
        guide_text = "왼쪽 눈을 맞춰주세요. 진단 시작을 누르세요."
        if state_snapshot['current_eye'] == "RIGHT_EYE":
            guide_text = "오른쪽 눈을 맞춰주세요. 진단 시작을 누르세요."
        
        return jsonify({
            'status': 'ok',
            'current_eye': state_snapshot['current_eye'],
            'captured_eyes': state_snapshot['captured_eyes'],
            'auto_capture_ready': state_snapshot['auto_capture_ready'],
            'guide_text': guide_text
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/capture/reset', methods=['POST'])
def reset_capture_state():
    """
    촬영 시퀀스 초기화
    새로운 진단을 시작할 때 호출
    """
    try:
        CaptureState.reset()
        return jsonify({
            'status': 'ok',
            'message': '촬영 상태가 초기화되었습니다.'
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/report/dependencies', methods=['GET'])
def api_report_dependencies():
    """
    보고서(PDF/카카오) 기능 의존성 상태 조회
    운영 점검용 엔드포인트
    """
    return jsonify({
        'status': 'ok',
        'report_dependencies': get_report_dependency_status()
    }), 200


@app.route('/status')
def status():
    """서버 상태 확인"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'device': torch.cuda.get_device_name(0),
            'total_memory_gb': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}",
            'allocated_memory_gb': f"{torch.cuda.memory_allocated(0) / 1e9:.2f}",
            'reserved_memory_gb': f"{torch.cuda.memory_reserved(0) / 1e9:.2f}"
        }
    
    report_dep = get_report_dependency_status()

    model_parameter_device = None
    if model_manager is not None:
        classifier = model_manager.get_classifier()
        if classifier is not None:
            try:
                model_parameter_device = str(next(classifier.model.parameters()).device)
            except (AttributeError, StopIteration):
                model_parameter_device = None

    resolved_cuda_index = config.DEVICE.index if config.DEVICE.type == 'cuda' else None
    cuda_device_name = None
    if resolved_cuda_index is not None and torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(resolved_cuda_index)

    return jsonify({
        'status': 'running',
        'models_loaded': model_manager is not None,
        'camera_connected': current_frame is not None,
        'gpu_info': gpu_info,
        'inference': {
            'requested_device': config.TORCH_DEVICE_REQUESTED,
            'resolved_device': str(config.DEVICE),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_device_name': cuda_device_name,
            'torch_version': torch.__version__,
            'torch_cuda_version': torch.version.cuda,
            'model_parameter_device': model_parameter_device,
        },
        'report_feature': {
            'pdf_generation_ready': report_dep['pdf_generation_ready'],
            'kakao_send_ready': report_dep['kakao_send_ready'],
            'kakao_token_configured': report_dep['kakao_token_configured'],
            'missing_packages': report_dep['missing_packages']
        }
    })


@app.route('/login')
def login():
    """로그인 페이지"""
    return render_template('login.html')


@app.route('/admin/config')
def admin_config_page():
    """관리자 설정 페이지"""
    if not is_admin_session():
        return redirect(url_for('login'))
    return render_template('admin_config.html')


@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    """관리자 로그인 세션 발급"""
    try:
        data = request.json or {}
        identifier = str(data.get('identifier', '')).strip().lower()
        password = str(data.get('password', '')).strip()
        client_ip = get_client_ip_address()

        allowed, wait_sec = check_admin_login_rate_limit(client_ip)
        if not allowed:
            return jsonify({
                'status': 'error',
                'message': f'로그인 시도가 너무 많습니다. {wait_sec}초 후 다시 시도해 주세요.'
            }), 429

        if identifier == ADMIN_LOGIN_NAME:
            if not ADMIN_LOGIN_PASSWORD:
                session['is_admin'] = False
                return jsonify({
                    'status': 'error',
                    'message': 'ADMIN_LOGIN_PASSWORD가 설정되지 않았습니다. .env에 설정 후 재시작해 주세요.'
                }), 503

            if hmac.compare_digest(password, ADMIN_LOGIN_PASSWORD):
                session['is_admin'] = True
                csrf_token = create_admin_csrf_token()
                record_admin_login_attempt(client_ip, True)
                return jsonify({'status': 'ok', 'role': 'admin', 'csrf_token': csrf_token}), 200

            session['is_admin'] = False
            session.pop('admin_csrf_token', None)
            record_admin_login_attempt(client_ip, False)
            return jsonify({'status': 'error', 'message': '관리자 비밀번호가 올바르지 않습니다.'}), 401

        session['is_admin'] = False
        session.pop('admin_csrf_token', None)
        return jsonify({'status': 'ok', 'role': 'user'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/admin/config', methods=['GET'])
def api_admin_config_get():
    """관리자 설정 조회"""
    if not is_admin_session():
        return jsonify({'status': 'error', 'message': '관리자 권한이 필요합니다.'}), 403

    csrf_token = str(session.get('admin_csrf_token', '')).strip()
    if not csrf_token:
        csrf_token = create_admin_csrf_token()

    return jsonify({
        'status': 'ok',
        'config': get_admin_config_snapshot(),
        'editable_keys': list(ADMIN_EDITABLE_CONFIG_KEYS.keys()),
        'llm_settings': get_admin_llm_settings_snapshot(),
        'llm_editable_keys': list(ADMIN_LLM_EDITABLE_KEYS.keys()),
        'csrf_token': csrf_token
    }), 200


@app.route('/api/admin/config', methods=['POST'])
def api_admin_config_update():
    """관리자 설정 저장"""
    csrf_error = require_admin_csrf()
    if csrf_error:
        return csrf_error

    try:
        data = request.json or {}
        updates_raw = data.get('updates') or {}
        llm_updates_raw = data.get('llm_updates') or {}

        if not isinstance(updates_raw, dict):
            updates_raw = {}
        if not isinstance(llm_updates_raw, dict):
            llm_updates_raw = {}

        if not updates_raw and not llm_updates_raw:
            return jsonify({'status': 'error', 'message': '저장할 설정값이 없습니다.'}), 400

        casted_updates = {}
        for key, raw_value in updates_raw.items():
            if key not in ADMIN_EDITABLE_CONFIG_KEYS:
                continue
            casted_updates[key] = cast_config_value(key, raw_value)

        applied_llm_updates = apply_admin_llm_updates(llm_updates_raw)

        if not casted_updates and not applied_llm_updates:
            return jsonify({'status': 'error', 'message': '유효한 설정 항목이 없습니다.'}), 400

        persisted_updates = apply_admin_config_updates(casted_updates)
        for key, value in persisted_updates.items():
            setattr(config, key, value)

        safe_llm_updates = {}
        for key, value in applied_llm_updates.items():
            if key in ('OPENAI_API_KEY', 'GEMINI_API_KEY'):
                safe_llm_updates[key] = 'updated'
            else:
                safe_llm_updates[key] = value

        return jsonify({
            'status': 'ok',
            'message': '설정이 저장되었습니다. 일부 항목은 재시작 후 완전 적용됩니다.',
            'updated': persisted_updates,
            'llm_updated': safe_llm_updates
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/admin/logout', methods=['POST'])
def api_admin_logout():
    csrf_error = require_admin_csrf()
    if csrf_error:
        return csrf_error

    session['is_admin'] = False
    session.pop('admin_csrf_token', None)
    return jsonify({'status': 'ok', 'message': '관리자 세션이 종료되었습니다.'}), 200


@app.route('/api/admin/fallback_stats', methods=['GET'])
def api_admin_fallback_stats():
    """관리자 전용: Pillow 폴백 성공/실패 통계 조회"""
    if not is_admin_session():
        return jsonify({'status': 'error', 'message': '관리자 권한이 필요합니다.'}), 403

    try:
        with PILLOW_FALLBACK_LOCK:
            success = int(PILLOW_FALLBACK_SUCCESS)
            fail = int(PILLOW_FALLBACK_FAIL)

        return jsonify({
            'status': 'ok',
            'pillow_fallback_success': success,
            'pillow_fallback_fail': fail
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


def schedule_server_action(action):
    thread = threading.Thread(target=action, daemon=True)
    thread.start()


def launch_service_control(action):
    """Run the fixed service manager in a separate session without a shell."""
    project_dir = config.BASE_DIR
    log_dir = os.path.join(project_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'admin_server_control.log')
    argv = service_manager_argv(project_dir, action)

    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(
            f"\n[{datetime.now().isoformat()}] action={action} manager={argv[0]}\n"
        )
        log_file.flush()
        subprocess.Popen(
            argv,
            cwd=project_dir,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            close_fds=True,
            start_new_session=True,
        )


def shutdown_process_delayed():
    time.sleep(1.0)
    try:
        launch_service_control('stop')
    except Exception as error:
        print(f"[ERROR] service manager stop failed: {error}")


def restart_process_delayed():
    time.sleep(0.7)
    try:
        launch_service_control('restart')
    except Exception as error:
        print(f"[ERROR] service manager restart failed: {error}")


@app.route('/api/admin/server/restart', methods=['POST'])
def api_admin_server_restart():
    csrf_error = require_admin_csrf()
    if csrf_error:
        return csrf_error

    schedule_server_action(restart_process_delayed)
    return jsonify({
        'status': 'ok',
        'message': '서버 재시작 요청이 접수되었습니다. 잠시 후 다시 연결됩니다.'
    }), 200


@app.route('/api/admin/server/shutdown', methods=['POST'])
def api_admin_server_shutdown():
    csrf_error = require_admin_csrf()
    if csrf_error:
        return csrf_error

    schedule_server_action(shutdown_process_delayed)
    return jsonify({
        'status': 'ok',
        'message': '서버 종료 요청이 접수되었습니다.'
    }), 200


@app.route('/capture')
def capture():
    """촬영 페이지"""
    return render_template('capture.html')


@app.route('/screening')
def screening_select():
    """검사 항목 선택 페이지"""
    return render_template('screening_select.html')


@app.route('/screening/survey')
def screening_survey():
    """선택 검사 기반 통합 문진 페이지"""
    return render_template('screening_survey.html')


@app.route('/screening/capture')
def screening_capture():
    """피부·두피 촬영 및 업로드 페이지"""
    return render_template('screening_capture.html')


@app.route('/screening/result')
def screening_result():
    """모델 연결 전 피부·두피 결과 셸"""
    return render_template('screening_result.html')


@app.route('/screening/summary')
def screening_summary():
    """선택 검사 통합 요약 페이지"""
    return render_template('screening_summary.html')


@app.route('/result')
def result():
    """진단 결과 페이지"""
    return render_template('result.html')


@app.route('/report')
def report():
    """리포트 페이지"""
    return render_template('report.html')


@app.route('/report/pdf')
def report_pdf():
    """출력 전용 리포트 페이지"""
    return render_template('report_pdf.html')


@app.route('/report_pdf')
def report_pdf_legacy():
    """Legacy alias for older report PDF links."""
    return render_template('report_pdf.html')


@app.route('/training')
@app.route('/training.html')
def training():
    """안구 마사지 가이드 페이지"""
    return render_template('training.html')


@app.route('/survey')
def survey():
    """설문조사 페이지"""
    return render_template('survey.html')


# ========================================
# [7] 서버 시작/종료
# ========================================

@app.before_request
def initialize_on_first_request():
    """서버 시작 시 모델만 로드 (Flask 2.3+ 호환)"""
    global model_manager, models_initialized
    if not models_initialized:
        models_initialized = True
        init_history_db()
        print("\n" + "="*50)
        print("[Eye Disease Detection Server]")
        print("="*50)
        model_manager = initialize_models()

        print("\n✓ 서버 준비 완료! http://0.0.0.0:5000 에서 접속하세요\n")


def cleanup_resources():
    """프로세스 종료 시 자원 정리"""
    stop_camera_thread()

    if LED_AVAILABLE:
        pwm_led.stop()
        GPIO.cleanup()

    print("[Shutdown] 서버 종료")


atexit.register(cleanup_resources)


if __name__ == '__main__':
    print("Starting Eye Disease Detection Server...")

    # 서버 시작 전에 모델만 초기화
    if not models_initialized:
        models_initialized = True
        init_history_db()
        print("\n" + "="*50)
        print("[Eye Disease Detection Server]")
        print("="*50)
        model_manager = initialize_models()

        print("\n✓ 서버 준비 완료! http://0.0.0.0:5000 에서 접속하세요\n")
    
    app.run(
        host=config.SERVER_IP,
        port=config.SERVER_PORT,
        debug=config.DEBUG_MODE,
        threaded=True,
        use_reloader=False
    )
