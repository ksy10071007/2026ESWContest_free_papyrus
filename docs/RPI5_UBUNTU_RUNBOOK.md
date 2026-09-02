# Raspberry Pi 5 (Ubuntu) 실행/검증 가이드

이 문서는 Smart Eye Diagnosis System을 Raspberry Pi 5 + Ubuntu(64bit)에서 CPU 기반(ONNX Runtime)으로 실행하고 점검하는 절차입니다.

## 1. 사전 준비

- Ubuntu 22.04/24.04 (aarch64)
- Python 3.10+
- 카메라 장치 연결 (선택)
- 모델 파일 준비
  - `models/yolo.onnx`
  - `models/efficientnet.onnx`

> 참고: 현재 코드의 RPi 경로는 ONNX 분류를 우선 사용하며, YOLO ONNX 파싱은 모델 export 포맷에 따라 후속 튜닝이 필요할 수 있습니다.

## 2. 시스템 패키지 설치

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential libatlas-base-dev libjpeg-dev zlib1g-dev
```

## 3. 가상환경 및 의존성 설치

```bash
cd /path/to/eye_project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements_rpi.txt
```

## 4. 환경 변수 설정

`.env` 파일(또는 쉘 환경변수)에 아래 값 설정:

```bash
MODEL_DEVICE=rpi
MEDIAPIPE_ONNX_PATH=models/yolo.onnx
CLASSIFIER_ONNX_PATH=models/efficientnet.onnx
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
```

## 5. 빠른 프리플라이트 점검

```bash
bash scripts/rpi_preflight.sh
```

정상이라면 Python/패키지/모델 경로가 OK로 표시됩니다.

## 6. 서버 실행

```bash
source .venv/bin/activate
python server.py --device rpi
```

또는 `.env`를 사용할 경우:

```bash
source .venv/bin/activate
python server.py
```

## 7. API 스모크 테스트

### 헬스체크

```bash
curl http://127.0.0.1:5000/health
```

예상 응답 예시:

```json
{"status":"ok","device":"rpi"}
```

### 예측 테스트

- base64 이미지를 body에 넣어 `/predict` 호출

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,..."}'
```

## 8. 트러블슈팅

- `onnxruntime` import 실패
  - `python -c "import onnxruntime as ort; print(ort.__version__)"` 확인
- 모델 파일 없음
  - `models/yolo.onnx`, `models/efficientnet.onnx` 경로/권한 확인
- 성능 이슈
  - 입력 해상도 축소, 요청 동시성 제한, swap 설정 점검

## 9. PT -> ONNX 변환 권장 흐름

- Jetson/개발 PC에서 `.pt`를 ONNX로 export
- 생성된 `.onnx`를 Pi의 `models/`에 배치
- Pi에서는 ONNX Runtime만 사용
