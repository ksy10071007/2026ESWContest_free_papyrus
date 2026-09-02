# MediFlow Kiosk Jetson USB GPU Runbook

이 런북은 새 NVIDIA Jetson에서 USB UVC 카메라와 CUDA로 MediFlow Kiosk를 수동 설치·점검·운영하는 절차입니다.

## 1. 지원 범위

- NVIDIA Jetson, JetPack/L4T, `aarch64`
- `/dev/videoN` USB UVC 웹캠과 USB 현미경
- PyTorch EfficientNet-B0 및 Grad-CAM CUDA 추론
- 명령 기반 수동 시작·중지

지원하지 않는 항목:

- CSI 카메라와 `nvarguscamerasrc`
- RTSP 네트워크 카메라
- systemd, cron, 데스크톱 autostart
- 검증되지 않은 NVIDIA wheel URL 자동 선택

## 2. 장비 정보 확인

```bash
uname -m
python3 --version
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core
df -h
free -h
```

아키텍처는 `aarch64`여야 합니다. JetPack/L4T와 Python 버전을 기록하고 NVIDIA 공식 호환 자료에서 해당 조합의 PyTorch와 torchvision wheel을 준비합니다.

## 3. 저장소 설치

```bash
git clone https://github.com/Phjrab/mediflow-kiosk-core.git
cd mediflow-kiosk-core
bash scripts/install_jetson.sh \
  --torch-wheel /path/to/jetpack-compatible-torch.whl \
  --torchvision-wheel /path/to/matching-torchvision.whl
```

이미 설치된 torch·torchvision이 CUDA 텐서와 torchvision 모델 생성 검사를 통과하면 wheel 인자는 생략합니다.

```bash
bash scripts/install_jetson.sh
```

apt 작업을 별도로 끝냈다면:

```bash
bash scripts/install_jetson.sh --skip-apt
```

설치 중 기존 `.env`가 발견되면 값은 변경되지 않습니다. 누락 경고가 표시되면 직접 보완합니다.

## 4. 환경 설정

```dotenv
MODEL_DEVICE=jetson
TORCH_DEVICE=cuda
CUDA_DEVICE_INDEX=0
CUDA_EMPTY_CACHE_AFTER_ANALYSIS=0
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
HASH_PEPPER=<영구 비밀값>
EYE_APP_SECRET_KEY=<영구 비밀값>
ADMIN_LOGIN_PASSWORD=<관리자 기능 사용 시 설정>
CAMERA_DEVICE_INDEX=0
MICROSCOPE_CAMERA_DEVICE_INDEX=1
```

주의:

- 기존 `HASH_PEPPER`를 변경하지 않습니다.
- `.env` 권한은 `600`을 권장합니다.
- LLM과 카카오 키는 해당 기능을 사용할 때만 필요합니다.
- 여러 네트워크 인터페이스가 있으면 `EXTERNAL_BASE_URL`로 표시 주소를 고정합니다.

## 5. GPU 확인

```bash
source venv/bin/activate
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')
x = torch.ones(4, device='cuda')
print('probe:', float((x * 2).sum().cpu()))
PY
```

`CUDA available: True`, 실제 GPU 이름, `probe: 8.0`이 나와야 합니다.

## 6. USB UVC 카메라 확인

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
v4l2-ctl --device=/dev/video0 --list-formats-ext
v4l2-ctl --device=/dev/video1 --list-formats-ext
```

일반 웹캠과 현미경의 실제 `/dev/videoN` 번호를 `.env`에 반영합니다. MJPEG 또는 YUYV 640x480 지원 여부와 현재 사용자의 읽기·쓰기 권한을 확인합니다.

## 7. 전체 사전점검

카메라 연결 전:

```bash
bash scripts/jetson_preflight.sh --allow-no-camera
```

카메라 연결 후 최종 판정:

```bash
bash scripts/jetson_preflight.sh
```

출력 예:

```text
PASS: system
PASS: environment
PASS: python dependencies
PASS: CUDA tensor operation
PASS: classifier on cuda:0
PASS: Grad-CAM
PASS: MediaPipe Face Mesh
PASS: database temporary test
PASS: USB camera /dev/video0
SKIP: optional Kakao credentials
PASS: optional LLM credentials
SKIP: service health check

OVERALL: READY
```

`--allow-no-camera`의 카메라 `SKIP`은 나머지 설치 검사만 허용합니다. 실제 배포 준비 완료는 카메라를 연결한 일반 preflight가 통과해야 합니다.

## 8. 서비스 시작과 확인

```bash
mediflow-kiosk start
mediflow-kiosk status
mediflow-kiosk logs
curl -fsS http://127.0.0.1:5000/status
curl -fsS http://127.0.0.1:5001/health
```

`/status`의 `inference`에서 다음을 확인합니다.

- `requested_device`: `cuda`
- `resolved_device`: `cuda:0`
- `cuda_available`: `true`
- `model_parameter_device`: `cuda:0`

브라우저에서 `http://<JETSON_IP>:5000/`에 접속합니다. 서버 시작은 브라우저를 자동으로 열지 않습니다.

## 9. 실제 기능 점검

1. 안구 검사를 선택합니다.
2. USB 웹캠 세션을 시작합니다.
3. 실시간 프레임과 얼굴·양안 검출을 확인합니다.
4. 분석을 한 번 실행하고 Grad-CAM 결과를 확인합니다.
5. 결과 저장 후 이력 화면에서 다시 조회합니다.
6. PDF를 생성해 한글 글꼴과 다운로드를 확인합니다.
7. 키가 설정된 경우에만 LLM 채팅과 카카오 전송을 별도 검사합니다.

피부·두피 모델은 아직 `not_configured`이므로 의료 결과 생성 성공 기준에 포함하지 않습니다.

## 10. 재시작과 종료

```bash
mediflow-kiosk restart
mediflow-kiosk status
mediflow-kiosk stop
mediflow-kiosk status
```

`restart`는 각 서비스의 새 PID와 프로세스 시작 시각이 확인된 뒤 성공합니다. `stop`은 관리자가 기록하고 검증한 PID만 종료합니다.

## 11. 선택적 서비스 스모크 테스트

서비스가 중지된 상태에서 시작·health check·종료까지 자동 검사:

```bash
bash scripts/jetson_preflight.sh --allow-no-camera --service-smoke-test
```

서비스가 이미 실행 중이면 health check만 수행하고 기존 서비스를 유지합니다.

## 12. 문제 해결

| 오류 | 가능한 원인 | 조치 |
| --- | --- | --- |
| `torch.cuda.is_available() == False` | CPU용 torch 또는 JetPack 불일치 | 현재 L4T용 NVIDIA wheel 확인 후 torch·torchvision 재설치 |
| `operator torchvision::nms does not exist` | torch/torchvision 불일치 | 공식 호환 조합을 함께 설치 |
| `No module named mediapipe` | 앱 의존성 미설치 | `venv/bin/python -m pip install -r requirements_jetson.txt` |
| `libGL.so.1` 누락 | OpenCV 런타임 누락 | `sudo apt-get install libgl1` |
| `/dev/videoN` 없음 | 미연결 또는 장치 번호 변경 | 재연결 후 `v4l2-ctl --list-devices` 확인 |
| 카메라 권한 없음 | `video` 그룹 미포함 | 장치 소유권 확인 후 사용자 그룹 정책 수정 및 재로그인 |
| 카메라 open 후 프레임 없음 | 지원 포맷 차이 또는 다른 프로세스 점유 | 포맷 목록과 점유 프로세스 확인 |
| `HASH_PEPPER environment variable is required` | `.env` 영구 키 누락 | 기존 데이터와 같은 pepper를 설정 |
| 체크포인트 SHA 불일치 | 파일 손상·교체 | 올바른 체크포인트 복구 후 `sha256sum -c models/SHA256SUMS` |
| 모델 시작 중 외부 다운로드 | `pretrained=True` 회귀 | `efficientnet_b0(weights=None)` 확인 |
| 관리자 restart 실패 | 비공식 경로로 실행 | 안전하게 기존 프로세스를 정리하고 `mediflow-kiosk start` 사용 |
| PDF 한글 깨짐 | Noto CJK 미설치 | `sudo apt-get install fonts-noto-cjk fontconfig` |

## 13. 운영 기록

배포 시 다음을 기록합니다.

```text
Jetson model:
JetPack/L4T:
Python:
torch:
torchvision:
torch CUDA runtime:
cuDNN:
MediaPipe:
OpenCV:
USB webcam name/index/format:
USB microscope name/index/format:
preflight date/result:
commit:
```

장시간 운영 전에는 반복 분석과 Grad-CAM으로 GPU 메모리 사용량, 온도, 전력 모드, 카메라 재연결 동작을 별도 검증합니다.
