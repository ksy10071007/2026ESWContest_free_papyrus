# MediFlow Kiosk

NVIDIA Jetson과 USB UVC 카메라에서 구동하는 엣지 AI 기반 종합 건강 스크리닝 키오스크입니다. 현재 운영 가능한 AI 경로는 안구 스크리닝이며, 피부와 두피는 선택·문진·촬영·결과 UI와 확장 가능한 모델 등록 구조까지 구현되어 있습니다.

> 이 시스템은 의료 진단을 확정하는 장비가 아니라 이상 징후 선별과 설명을 돕는 연구용 보조 도구입니다. 최종 판단은 의료 전문가의 진료로 확인해야 합니다.

## 현재 구현 기준

이 문서는 2026-09-01의 `main` 브랜치 실행 경로를 기준으로 합니다.

- 서비스 표시 이름: 종합 건강 스크리닝
- 검사 선택: 안구, 피부, 두피, 전체 검사
- 안구 모델: MediaPipe Face Mesh + PyTorch EfficientNet-B0 + Grad-CAM
- 피부·두피 모델: 미구현 상태로 UI에서 비활성 처리
- 카메라: `/dev/videoN` USB UVC 웹캠 및 USB 현미경만 지원
- 안구·피부 카메라 역할: 일반 USB 웹캠
- 두피 카메라 역할: USB 현미경
- 운영 웹 서버: `eye_server.py`, 기본 포트 `5000`
- PDF·카카오 브리지: `database/app.py`, 기본 포트 `5001`
- 운영 명령: `mediflow-kiosk start|stop|restart|status|logs`
- 운영 DB: `database/database.db`
- 스키마 단일 원본: `database/schema.sql`

CSI 카메라와 `nvarguscamerasrc`는 지원하지 않습니다. systemd, cron, 데스크톱 autostart도 설치하지 않으며 사용자가 명령을 실행할 때만 서비스가 시작됩니다.

## 주요 기능

- 데스크톱 키오스크와 모바일 화면에서 검사 항목 선택
- 검사별 문진과 카메라·업로드 입력
- 실시간 USB 카메라 스트림과 눈 정렬 상태
- MediaPipe Face Mesh 기반 좌안·우안 검출 및 224x224 크롭
- EfficientNet-B0 5개 클래스 분류: 결막염, 다래끼, 백내장, 정상, 포도막염
- 예측 클래스 Grad-CAM 히트맵과 충혈도 등 픽셀 지표
- 사용자별 진단·문진 이력, PDF 보고서, QR 및 선택적 카카오 공유
- OpenAI 또는 Gemini 기반 결과 질의응답
- 관리자 설정, 안전한 서비스 재시작·종료
- `/status`의 CUDA, 모델 파라미터 장치, 선택 기능 준비 상태

## 시스템 구성

```text
USB UVC 웹캠/현미경 또는 업로드 이미지
                    |
                    v
       검사 유형 선택 및 문진
                    |
       +------------+-------------+
       |            |             |
     안구          피부          두피
 MediaPipe +     모델 추가      모델 추가
 EfficientNet    예정           예정
       |
       v
 질환 분류 + Grad-CAM + 픽셀 분석
                    |
                    v
       eye_server.py :5000
       |        |          |
     SQLite   모바일 UI   LLM 설명
       |
 database/app.py :5001 -> PDF/선택적 Kakao
```

## 새 Jetson 설치

### 준비 사항

- NVIDIA Jetson, JetPack/L4T 설치, `aarch64` Ubuntu
- Python 3
- 최소 5 GiB 여유 공간
- JetPack 버전과 맞는 NVIDIA PyTorch 및 torchvision wheel 또는 이미 검증된 설치본
- `/dev/videoN`으로 표시되는 USB UVC 카메라

저장소는 검증하지 않은 NVIDIA wheel URL을 제공하지 않습니다. 새 장비의 JetPack/L4T/Python 조합에 맞는 wheel을 NVIDIA 공식 자료에서 확인한 뒤 로컬 경로나 사용자가 검증한 URL로 전달해야 합니다.

```bash
git clone https://github.com/Phjrab/mediflow-kiosk-core.git
cd mediflow-kiosk-core
bash scripts/install_jetson.sh \
  --torch-wheel /path/to/jetpack-compatible-torch.whl \
  --torchvision-wheel /path/to/matching-torchvision.whl
```

호환되는 PyTorch와 torchvision이 이미 설치되어 있고 CUDA 검사를 통과하면 wheel 인자는 생략할 수 있습니다.

```bash
bash scripts/install_jetson.sh
```

설치 스크립트는 다음 원칙을 지킵니다.

- 기존 `venv`, `.env`, `config.local.json`, DB, 이미지, 보고서를 보존
- 기존 `.env`의 `HASH_PEPPER`와 다른 값은 자동 변경하지 않음
- `.env`가 없을 때만 예제에서 생성하고 영구 비밀값을 한 번 생성
- PyTorch CUDA 검증 실패 시 wheel 인자 없이 일반 PyPI torch를 설치하지 않음
- `~/.local/bin/mediflow-kiosk` 심볼릭 링크 설치
- 부팅 자동 실행을 만들지 않음

사용 가능한 옵션은 다음 명령으로 확인합니다.

```bash
bash scripts/install_jetson.sh --help
```

## 기존 Jetson 갱신

기존 `.env`는 설치기가 수정하지 않습니다. 다음 배포 키가 없다면 직접 추가합니다.

```dotenv
MODEL_DEVICE=jetson
TORCH_DEVICE=cuda
CUDA_DEVICE_INDEX=0
CUDA_EMPTY_CACHE_AFTER_ANALYSIS=0
HASH_PEPPER=<기존 값을 유지하거나 신규 설치 시 한 번 생성>
EYE_APP_SECRET_KEY=<장기간 유지할 임의 비밀값>
ADMIN_LOGIN_PASSWORD=<관리자 기능 사용 시 설정>
CAMERA_DEVICE_INDEX=0
MICROSCOPE_CAMERA_DEVICE_INDEX=1
```

운영 데이터가 생성된 이후 `HASH_PEPPER`를 바꾸면 동일 사용자를 이전 기록과 연결할 수 없습니다. `.env`와 `config.local.json`은 Git에 커밋하지 않습니다.

## 사전점검

카메라를 포함한 전체 배포 판정:

```bash
bash scripts/jetson_preflight.sh
```

카메라가 아직 연결되지 않았을 때 나머지 항목만 검사:

```bash
bash scripts/jetson_preflight.sh --allow-no-camera
```

서비스를 잠시 시작해 `/status`까지 검사하고 원래 중지 상태로 되돌리기:

```bash
bash scripts/jetson_preflight.sh --allow-no-camera --service-smoke-test
```

이미 서비스가 실행 중이면 사전점검은 해당 서비스를 임의로 종료하지 않습니다. 필수 검사 실패 시 종료 코드는 0이 아니며 마지막에 `OVERALL: NOT READY`를 출력합니다.

검사 범위:

- Jetson/aarch64/L4T, RAM, 저장공간, 포트
- 필수 환경 설정 존재 여부(비밀값 내용은 표시하지 않음)
- Python 패키지와 버전
- 실제 CUDA 텐서 연산과 cuDNN/GPU 정보
- EfficientNet 체크포인트 SHA-256 및 strict 로딩
- `cuda:0` 더미 분류와 224x224 Grad-CAM
- MediaPipe Face Mesh 초기화
- 임시 DB 스키마·해시·쓰기·읽기
- USB UVC 장치 권한·포맷·실제 프레임
- 선택적 LLM·카카오 준비 상태

## 서비스 운영

어느 디렉터리에서든 다음 명령만 사용합니다.

```bash
mediflow-kiosk start
mediflow-kiosk status
mediflow-kiosk logs
mediflow-kiosk restart
mediflow-kiosk stop
```

`mediflow-kiosk`는 서비스별 PID의 사용자, 실행 파일, 명령행, 작업 경로, 부팅 ID와 시작 시각을 검증합니다. `pkill -f`나 `killall`을 사용하지 않으며 다른 프로젝트 프로세스를 종료하지 않습니다.

기본 접속 주소:

- 키오스크: `http://<jetson-ip>:5000/`
- 상태 API: `http://<jetson-ip>:5000/status`
- 카카오 브리지: `http://<jetson-ip>:5001/health`

표시 URL은 `EXTERNAL_BASE_URL`, 서버 설정, LAN 주소 순으로 계산됩니다. 여러 네트워크 인터페이스 중 특정 주소를 보여야 하면 `.env`에 `EXTERNAL_BASE_URL=http://<jetson-ip>:5000`을 설정합니다.

로그는 `runtime/log/eye_server.log`, `runtime/log/kakao_app.log`에 저장되며 `mediflow-kiosk logs`는 각 로그의 최근 80줄만 표시합니다.

### 선택적 로컬 브라우저

서버와 브라우저는 별도입니다. 기존 호환 래퍼로 브라우저까지 열 때만 다음을 사용합니다.

```bash
OPEN_KIOSK_BROWSER=1 ./start_services.sh
```

기본 `./start_services.sh`는 서비스만 시작하고, `./stop_services.sh`는 안전 관리자에 종료를 위임합니다. 선택적으로 연 브라우저는 사용자가 별도로 닫습니다.

## GPU 상태 확인

```bash
source venv/bin/activate
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')
PY

curl -fsS http://127.0.0.1:5000/status
```

`/status`의 `inference`에서 다음을 확인합니다.

```json
{
  "requested_device": "cuda",
  "resolved_device": "cuda:0",
  "cuda_available": true,
  "model_parameter_device": "cuda:0"
}
```

`TORCH_DEVICE=cuda`에서 CUDA가 없거나 지정 GPU 연산이 실패하면 서버는 CPU로 폴백하지 않고 JetPack 호환 PyTorch 설치 정보를 포함한 오류로 중단합니다. 개발 PC에서는 `TORCH_DEVICE=cpu` 또는 `auto`를 명시할 수 있습니다.

## USB 카메라 확인

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

카메라 열기 순서는 V4L2 인덱스, V4L2 장치 경로, OpenCV가 지원할 때만 USB GStreamer, `CAP_ANY`입니다. 열린 핸들만으로 성공 처리하지 않고 실제 프레임을 받아야 합니다.

카메라가 한 대뿐인 개발 환경에서는 두 인덱스를 같게 둘 수 있습니다. 두 장치를 연결하면 안구·피부용 `CAMERA_DEVICE_INDEX`와 두피용 `MICROSCOPE_CAMERA_DEVICE_INDEX`를 각각 지정합니다.

## 환경 변수

### 플랫폼·추론

| 변수 | Jetson 권장값 | 설명 |
| --- | --- | --- |
| `MODEL_DEVICE` | `jetson` | Jetson PyTorch 백엔드 선택 |
| `TORCH_DEVICE` | `cuda` | `cuda`, `auto`, `cpu` 정책 |
| `CUDA_DEVICE_INDEX` | `0` | 사용할 GPU 인덱스 |
| `CUDA_EMPTY_CACHE_AFTER_ANALYSIS` | `0` | OOM 분석 시에만 캐시 비우기 활성화 |
| `CAMERA_DEVICE_INDEX` | `0` | 안구·피부 USB 웹캠 |
| `MICROSCOPE_CAMERA_DEVICE_INDEX` | `1` | 두피 USB 현미경 |

### 서버·보안

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `SERVER_HOST` | `0.0.0.0` | Flask 바인딩 주소 |
| `SERVER_PORT` | `5000` | 키오스크 포트 |
| `EXTERNAL_BASE_URL` | 자동 감지 | 사용자에게 표시할 기준 URL |
| `HASH_PEPPER` | 없음 | 사용자 식별자 해시용 영구 비밀값 |
| `EYE_APP_SECRET_KEY` | 없음 | 관리자 세션 서명 키 |
| `ADMIN_LOGIN_PASSWORD` | 없음 | 관리자 로그인 사용 시 필요 |
| `SESSION_COOKIE_SECURE` | `0` | HTTPS 운영 시 `1` |

### 선택 기능

- LLM: `LLM_PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL`
- 카카오: `KAKAO_CLIENT_ID`, `KAKAO_CLIENT_SECRET`, `KAKAO_REDIRECT_URI`, `KAKAO_REFRESH_TOKEN`, `KAKAO_ACCESS_TOKEN`
- 브리지: `KAKAO_APP_HOST`, `KAKAO_APP_PORT`, `KAKAO_BRIDGE_URL`

LLM 키가 없으면 채팅만 사용할 수 없고, 카카오 설정이 없으면 카카오 공유만 제한됩니다. 메인 스크리닝 서버와 로컬 PDF 기능은 계속 시작할 수 있습니다. `database/app.py`의 설정 우선순위는 프로세스 환경, 프로젝트 루트 `.env`, 선택적 `config.local.json`, 코드 기본값입니다.

## 주요 페이지와 API

| 경로 | 용도 |
| --- | --- |
| `/` | 키오스크 메인 |
| `/screening` | 검사 선택 |
| `/screening/survey` | 통합 문진 |
| `/screening/capture` | 검사별 촬영 |
| `/screening/result` | 검사 결과 |
| `/screening/summary` | 전체 검사 요약 |
| `/m`, `/m/dashboard` | 모바일 연결과 대시보드 |
| `/admin/config` | 관리자 설정 |
| `/status` | 서비스·CUDA·기능 상태 |

주요 API:

- 카메라: `GET /video_feed`, `GET /video_frame`, `POST /camera/session/start`, `POST /camera/session/stop`
- 분석: `POST /analyze`, `POST /diagnose`, `GET /detect_status`
- 이력·문진: `GET /api/history`, `POST|GET /api/survey`
- AI·보고서: `POST /api/chat`, `POST /api/generate_report`, `POST /api/report/share`
- 관리자: `POST /api/admin/login`, `GET|POST /api/admin/config`, `POST /api/admin/server/restart`, `POST /api/admin/server/shutdown`

브라우저 건강 채팅 역할은 `config/llm_chat_role.txt`에 있고 검사별 준비 상태는 `config/screening_modalities.json`에 있습니다. `model_status=ready`인 검사 결과만 LLM 설명 대상으로 사용합니다.

## 데이터 저장

| 위치 | 내용 |
| --- | --- |
| `database/database.db` | 운영 사용자·세션·자산·문진·이벤트 |
| `database/schema.sql` | 스키마 단일 원본 |
| `database/history.db` | 레거시 마이그레이션 원본 |
| `web/static/captures/users/<hash>/` | 사용자 촬영 이미지 |
| `web/static/reports/`, `reports/` | 보고서 |
| `database/backups/` | 로컬 DB 백업 |

`.env`, `config.local.json`, DB, 사용자 이미지, 보고서, 런타임 로그는 Git 제외 대상입니다.

## Raspberry Pi 5 경로

RPi 백엔드는 ONNX Runtime 기반 연구 경로로 유지됩니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_rpi.txt
bash scripts/export_onnx_rpi.sh
bash scripts/rpi_preflight.sh
python server.py --device rpi
```

전체 키오스크의 공식 GPU·USB 배포 대상은 현재 Jetson입니다. RPi 세부 내용은 `docs/RPI5_UBUNTU_RUNBOOK.md`를 참고합니다.

## 테스트

```bash
source venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m compileall -q .
python -m unittest discover -s tests -v
python -m pytest -q
bash -n scripts/install_jetson.sh
bash -n scripts/jetson_preflight.sh
bash -n start_services.sh
bash -n stop_services.sh
git diff --check
```

## 문제 해결

| 오류 | 가능한 원인 | 조치 |
| --- | --- | --- |
| `torch.cuda.is_available() == False` | CPU용 torch 또는 JetPack 불일치 | 현재 L4T와 맞는 NVIDIA wheel로 torch·torchvision 재설치 |
| `operator torchvision::nms does not exist` | torch/torchvision 조합 불일치 | 공식 호환 버전 쌍으로 함께 재설치 |
| `No module named mediapipe` | Jetson 앱 의존성 누락 | `requirements_jetson.txt` 설치 후 preflight 재실행 |
| `libGL.so.1` 누락 | OpenCV 런타임 라이브러리 누락 | `sudo apt-get install libgl1` |
| 카메라 open 실패 | 잘못된 인덱스·권한·포맷 | `v4l2-ctl`, `/dev/videoN`, `video` 그룹 확인 |
| `HASH_PEPPER environment variable is required` | 영구 해시 키 누락 | `.env`에 한 번 설정하고 이후 값 유지 |
| 모델 시작 중 인터넷 접근 | 외부 pretrained 가중치 코드 | `efficientnet_b0(weights=None)` 유지 확인 |
| 체크포인트 SHA 실패 | 파일 손상 또는 다른 모델 | 올바른 모델 복구 후 `models/SHA256SUMS` 확인 |
| 관리자 restart 실패 | 관리자 외 경로로 시작한 프로세스 | 기존 프로세스를 안전하게 종료하고 `mediflow-kiosk start` 사용 |
| PDF 한글 깨짐 | 한글 폰트 누락 | `sudo apt-get install fonts-noto-cjk fontconfig` |

장시간 GPU 메모리 안정성, 실제 환자 데이터 정확도, 장치별 USB 포맷 차이는 별도 검증 대상입니다.

## 프로젝트 구조

```text
mediflow-kiosk-core/
├── eye_server.py
├── config.py
├── model_loader.py
├── database/
│   ├── app.py
│   ├── db.py
│   └── schema.sql
├── inference/
├── modules/
│   ├── detector.py
│   ├── classifier.py
│   └── analyzer.py
├── models/
│   ├── Augmented_EffNet_V1_B0_best.pth
│   └── SHA256SUMS
├── scripts/
│   ├── install_jetson.sh
│   ├── jetson_preflight.sh
│   ├── jetson_preflight.py
│   └── mediflow-kiosk
├── utils/
│   ├── service_control.py
│   └── uvc_camera.py
├── tests/
├── web/
└── docs/
    ├── JETSON_USB_GPU_RUNBOOK.md
    └── RPI5_UBUNTU_RUNBOOK.md
```

## Git 훅

Jetson 전용 파일의 의도치 않은 변경을 막는 선택적 훅:

```bash
bash scripts/install_git_hooks.sh
```

의도적인 Jetson 변경 커밋에서는 다음 환경변수를 사용합니다.

```bash
ALLOW_JETSON_CHANGES=1 git commit -m "Describe intentional Jetson change"
```
