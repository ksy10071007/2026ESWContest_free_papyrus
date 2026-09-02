"""USB UVC camera opening helpers for /dev/videoN devices."""

from __future__ import annotations

import time
from typing import Callable

import cv2


class UvcCameraError(RuntimeError):
    """Raised when no supported USB UVC capture path returns a frame."""


def opencv_has_gstreamer() -> bool:
    build_info = cv2.getBuildInformation()
    return any(
        line.strip().startswith('GStreamer:') and 'YES' in line
        for line in build_info.splitlines()
    )


def _fourcc_text(value: float) -> str:
    number = int(value)
    return ''.join(chr((number >> (8 * index)) & 0xFF) for index in range(4)).strip('\x00')


def _configure_capture(capture) -> None:
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)


def _read_valid_frame(capture, attempts: int):
    for _ in range(max(1, attempts)):
        ok, frame = capture.read()
        if ok and frame is not None and getattr(frame, 'size', 0) > 0:
            return frame
        time.sleep(0.05)
    return None


def open_usb_uvc_camera(
    device_index: int,
    gstreamer_pipeline: Callable[[int], str] | None = None,
    frame_attempts: int = 8,
):
    """Open a USB UVC device and return (capture, first_frame, metadata)."""
    if int(device_index) < 0:
        raise ValueError('USB camera index must be non-negative')

    index = int(device_index)
    device_path = f'/dev/video{index}'
    candidates = [
        ('v4l2-index', lambda: cv2.VideoCapture(index, cv2.CAP_V4L2)),
        ('v4l2-path', lambda: cv2.VideoCapture(device_path, cv2.CAP_V4L2)),
    ]
    if gstreamer_pipeline is not None and opencv_has_gstreamer():
        candidates.append(
            ('gstreamer-uvc', lambda: cv2.VideoCapture(
                gstreamer_pipeline(index),
                cv2.CAP_GSTREAMER,
            ))
        )
    candidates.append(('opencv-any', lambda: cv2.VideoCapture(device_path, cv2.CAP_ANY)))

    failures = []
    for backend_name, opener in candidates:
        capture = None
        try:
            capture = opener()
            if not capture.isOpened():
                failures.append(f'{backend_name}: open failed')
                capture.release()
                continue

            _configure_capture(capture)
            first_frame = _read_valid_frame(capture, frame_attempts)
            if first_frame is None:
                failures.append(f'{backend_name}: no valid frame')
                capture.release()
                continue

            metadata = {
                'role_path': device_path,
                'backend': backend_name,
                'width': int(first_frame.shape[1]),
                'height': int(first_frame.shape[0]),
                'fps': float(capture.get(cv2.CAP_PROP_FPS)),
                'fourcc': _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC)),
            }
            return capture, first_frame, metadata
        except Exception as exc:
            failures.append(f'{backend_name}: {exc}')
            if capture is not None:
                capture.release()

    raise UvcCameraError(
        f"USB UVC camera {device_path} did not return a valid frame; "
        + '; '.join(failures)
    )
