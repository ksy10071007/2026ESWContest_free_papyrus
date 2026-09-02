# Edge LLM Head/Worker Cluster

NVIDIA Jetson Orin Nano 또는 Raspberry Pi 5를 최대 4대까지 묶어 동일한 GGUF
워크로드를 비교하는 head/worker 벤치마크 환경이다. 현재 Jetson Orin Nano가 **head**이며
대시보드, 노드 준비, 모델 배포, 실험 스케줄링, 결과 집계를 담당한다. Head도 추론 노드로
참여할 수 있다.

## 지원 플랫폼

| 플랫폼 | 런타임 | 필수 조건 | 권장 실험 설정 |
|---|---|---|---|
| Jetson Orin Nano | CUDA `llama-cpp-python` | JetPack/CUDA가 미리 설치된 64-bit Ubuntu/L4T | `n_gpu_layers` 30 또는 모델에 맞게 조정 |
| Raspberry Pi 5 | CPU/OpenBLAS `llama-cpp-python` | Raspberry Pi OS 64-bit 또는 Ubuntu 64-bit, `aarch64` | `n_gpu_layers=0`, 1B~3B Q4 GGUF |

Raspberry Pi는 장시간 부하에서 thermal throttling이 발생할 수 있으므로 Active Cooler 또는
팬을 권장한다. 메모리 여유가 작은 장치에서는 큰 swap이 실행을 가능하게 할 수 있지만
스토리지 I/O 때문에 벤치마크 값이 왜곡될 수 있어 1B/3B 모델부터 검증한다.

한 실험에 Jetson과 Pi를 함께 넣을 수는 있지만, Pi가 포함되면 대시보드는 모든 노드의
GPU 레이어를 0으로 맞춘다. 가속기 성능 비교는 Jetson 클러스터와 Pi 클러스터를 별도
실험으로 생성하는 것이 해석하기 쉽다.

## 최초 head 설정

```bash
cd <project-dir>
./cluster/setup_head.sh
./cluster/dashboard/start.sh
```

예: Jetson은 `/home/jetson_orin_nano/project/llm/local_llm_bench`, Raspberry Pi는
`/home/pi/local_llm_bench`처럼 실제 clone 경로를 사용한다.

생성되는 로컬 런타임 파일은 커밋하지 않는다.

- `.run/cluster/nodes.local.csv`: 실제 노드 인벤토리
- `~/.ssh/id_ed25519_llm_cluster`: head 전용 SSH 키
- `.run/cluster/dashboard.token`: 선택형 대시보드 인증을 켤 때 사용하는 접근 토큰
- `.run/cluster/settings.json`: 선택형 보안 설정(대시보드·worker API 인증 모두 기본 꺼짐)
- `.run/cluster/worker.token`: 인증을 켤 때 생성되는 비공개 head/worker 공유 토큰
- `.run/cluster/experiments/`: 실험 정의 카탈로그
- `.run/cluster/environment/<node>.json`: 노드별 최신 LLM 런타임 점검 결과
- `.run/cluster/results/`: 실행별 원시 결과와 요약

대시보드는 기본적으로 `http://HEAD_IP:8080`에서 실행한다. 내부 LAN 전용이며 인터넷에
직접 공개하지 않는다. 기본 상태에서는 토큰 없이 접속하며, `설정 → 대시보드 토큰 인증`을
켠 경우에만 `.run/cluster/dashboard.token` 값을 요구한다.
토큰 인증을 켜더라도 기본 HTTP 연결 자체가 암호화되는 것은 아니므로, 신뢰하지 않는 네트워크나
인터넷에서는 그대로 노출하지 말고 별도의 TLS/VPN 구간 안에서 사용한다.

## 대시보드에서 워커 찾기와 자동 준비

`+ 워커 연결`을 누르면 head가 연결된 사설 LAN의 최대 `/24` 범위에서 SSH 포트가 열린
기기만 제한적으로 찾는다. Docker·가상 브리지 인터페이스는 제외한다.

1. 검색된 기기 카드에서 워커를 선택한다.
2. SSH 사용자와 프로젝트 경로를 확인한다. Raspberry Pi OS 기본 사용자 구성이라면
   사용자명을 `pi`, 경로를 `/home/pi/local_llm_bench`처럼 해당 홈 아래로 바꾼다.
3. Head 공개 키를 워커의 `~/.ssh/authorized_keys`에 최초 한 번 등록한다.
4. `SSH 환경 확인`으로 키 인증, 보드, OS, `aarch64`, 디스크, NTP, 누락 패키지와
   passwordless sudo 가능 여부를 확인한다.
5. `저장 후 자동 준비`를 실행한다.
6. 시스템 의존성 확인 → 코드 동기화 → 플랫폼별 venv/llama 빌드 → 선택 모델 동기화 →
   worker API 시작 순서로 진행된다.

대시보드는 SSH 또는 sudo 비밀번호를 입력받거나 저장하지 않는다. 누락된 apt 패키지가
있고 `sudo -n`이 불가능하면 워커 콘솔에서 실행할 정확한 allowlist 명령을 표시한다.
명령을 한 번 실행한 뒤 환경 확인을 재시도한다. JetPack/CUDA 자체는 자동 설치하지 않고
NVIDIA 이미지에 정상 설치되어 있는지만 검증한다.

공통 apt 의존성:

```text
ca-certificates curl git rsync openssh-client iproute2 util-linux
build-essential cmake ninja-build pkg-config python3 python3-dev python3-venv
```

Raspberry Pi에는 `libopenblas-dev`가 추가된다. 설치 스크립트는 재실행 가능한 형태이며,
검증된 백엔드가 이미 있으면 다시 빌드하지 않는다. 실제 설치 없이 계획만 확인할 수도 있다.

```bash
./cluster/worker_setup.sh --plan-only --platform jetson
./cluster/worker_setup.sh --plan-only --platform raspberry-pi
```

## 노드별 LLM 런타임 점검과 자동 구성

개요의 `노드 실행 환경`은 head를 포함한 선택 노드를 독립적으로 점검한다.

- Jetson Orin / Raspberry Pi 5 보드와 64-bit ARM OS
- 고정 apt 허용 목록, 프로젝트 쓰기 경로와 디스크 여유
- `<project-dir>/.venv`와 `requirements-runtime.txt`의 고정 Python 버전
- `llama-cpp-python==0.3.20`과 Jetson CUDA 또는 Pi OpenBLAS/ARM 최적화
- GGUF 발견 개수와 Jetson jtop 고급 측정 사용 가능 여부

`선택 노드 환경 점검`은 읽기 전용이다. `선택 노드 자동 구성`을 명시적으로
확인하면 누락된 시스템 패키지를 고정 allowlist에서만 처리하고 프로젝트를
동기화한 뒤 프로젝트 안의 `.venv`에 Python/LLM 런타임을 설치한다. root 또는
`sudo -n`이 가능할 때만 apt를 자동 실행하며, 비밀번호가 필요하면 대시보드가
정확한 수동 명령만 표시한다. JetPack, CUDA, OS 이미지는 자동 설치하지 않는다.

결과는 `.run/cluster/environment/<node>.json`에 `0600`으로 원자적 저장되며,
노드 카드에 `READY`, `AUTO FIX`, `MANUAL`, `BLOCKED`로 표시된다. `READY`는
플랫폼별 LLM 런타임이 검증됐다는 뜻이다. 실험 시작 시점에는 다시 다음을
강제한다.

- 점검 결과가 24시간 이내이고 backend가 검증됐을 것
- worker API가 온라인일 것
- 복제 전략은 선택한 모든 노드에, RPC는 coordinator head에 선택 GGUF가 있을 것

따라서 런타임 구성 후 API가 중지돼 있으면 `서버 시작`, 모델이 없으면
`선택 모델 동기화`를 별도로 실행한다. 런타임 점검이 대용량 모델을 암묵적으로
복제하거나 서버를 자동 시작하지는 않는다.

CLI에서도 같은 점검과 설치 흐름을 사용할 수 있다.

```bash
python -m cluster.clusterctl environment-check
python -m cluster.clusterctl --node edge-worker-01 environment-install --confirmed

# 단일 노드 스크립트로 JSON 산출물까지 저장
./cluster/worker_setup.sh --check-only --project-dir "$PWD" \
  --report-json "$PWD/.run/cluster/environment/local-check.json"
```

## jtop형 노드 상태

노드 카드의 `상세 상태`를 누르면 다음 정보를 실시간으로 확인한다.

- 보드, OS/L4T, kernel, Python, 검증된 llama 백엔드, uptime
- 전체/코어별 CPU 사용률과 주파수, load average
- RAM, swap, 디스크 사용량
- 네트워크 송수신 속도
- 온도 센서, 팬, 총전력과 전력 레일
- Jetson GPU/EMC 및 하드웨어 엔진 상태
- 브라우저 세션의 최근 CPU/GPU/RAM 이력 그래프

Jetson은 jtop 서비스가 사용 가능할 때 `jtop + psutil`, 없으면 `psutil`로 안전하게
fallback한다. 고급 GPU/전력/팬 지표에는 운영체제 수준 `jetson-stats` 서비스가 필요하며
`systemctl is-active jtop.service`와 사용자 `jtop` 그룹 권한을 확인한다. Raspberry Pi는
`psutil + vcgencmd/sysfs`를 사용한다. Pi에서
신뢰할 수 없는 GPU 사용률과 보드 전체 전력 값은 0으로 꾸미지 않고 `N/A`로 표시한다.

## 실험과 결과 관리

### 실행 방식

대시보드의 `실행 방식`은 모델 배치와 요청 흐름을 함께 정의한다. 효율이 낮아 보이는
방식도 실제 오버헤드를 검증할 수 있도록 선택 가능하며, 실행 결과에 방식과 토폴로지를
항상 기록한다.

| 실행 방식 | 모델 배치 | 요청 흐름 | 해석 |
|---|---|---|---|
| 단일 노드 기준선 | 선택한 1대에 전체 모델 | 모든 요청을 1대가 처리 | 장치 자체 성능 |
| 복제 · 요청 분산 | 각 노드에 전체 모델 복제 | 총 요청을 round-robin 분배 | 여러 사용자 처리량 |
| 동일 요청 전체 전송 | 각 노드에 전체 모델 복제 | 같은 요청을 모든 노드에 전송 | 지연·출력 일치도 |
| 노드 수 스윕 | 각 비교 노드에 전체 모델 복제 | 1대→2대→… 조건 반복 | 확장 speedup·효율 |
| 모델 분할 · RPC | 모델 하나를 여러 장치에 분할 | 한 답변 계산에 모든 노드 참여 | 대형 모델 수용·LAN 오버헤드 |

기본값은 기존 동작과 호환되는 `replicated_round_robin`이다. 이 모드는 모델을 나누지
않는다. 노드 안의 Python 모델 관리자는 추론 요청을 직렬화하므로 동시성은 여러 노드의
병렬 처리와 노드별 대기열을 함께 측정한다.

`model_parallel_rpc`는 `llama.cpp`의 proof-of-concept RPC 백엔드를 사용한다. 선택한 모든
노드는 같은 고정 커밋 `f49e9178767d557a522618b16ce8694f9ddac628`으로 네이티브 런타임을
빌드한다. GGUF 원본은 coordinator인 head에만 필요하고 worker는 할당된 텐서와 KV cache를
RPC로 받아 계산한다. 먼저 대시보드의 `선택 노드 RPC 환경 준비`를 실행한다.

```bash
python -m cluster.clusterctl --node edge-head --node edge-worker-01 prepare-rpc
```

RPC 포트는 인증이 없으므로 신뢰하는 사설 LAN에서 실험 중에만 열고 항상 종료한다. 워커
API 토큰 보안이 켜져 있으면 RPC 실험은 차단한다. Jetson+Pi 혼합도 의도적으로 허용하지만
Pi는 원격 CPU 장치이므로 네트워크와 가장 느린 장치가 전체 토큰 생성의 병목이 될 수 있다.
RPC 결과의 `cluster_tokens_per_s`는 여러 노드가 함께 만든 단일 모델 처리량이며, 복제 방식의
노드별 처리량 합과 같은 의미가 아니다.

대시보드의 `실험 묶음`은 영속적인 `experiment_id`를 가진다. 새 실험을 만들거나 기존
실험을 선택해 반복 실행하면 각 run의 `summary.json`이 동일한 실험에 연결된다. 기존
결과처럼 `experiment_id`가 없는 파일은 이름 기준의 `legacy-*` 묶음으로 읽으며 원본을
변경하거나 삭제하지 않는다.

모델 선택기는 설치된 GGUF를 검색해 한 개 이상 체크할 수 있다. 여러 모델을 선택하면 한 번의
시작 동작이 `suite_id` 하나를 만들고 선택 순서대로 모델을 교체하며 독립 run을 실행한다.
각 모델은 별도 `config.json`, 요청 CSV와 `summary.json`을 가지면서 같은 실험과 suite에
연결된다. 모델 사이에는 전체 노드 unload와 설정한 cooldown을 적용한다. `모델 실패 후 계속`을
켜면 한 모델의 로드 또는 측정 실패를 기록하고 다음 모델로 진행하며, 끄면 즉시 suite를
중단한다. 취소는 현재 요청 묶음까지만 마친 뒤 다음 모델을 시작하지 않는다.

결과 화면에서 실험을 선택하면 해당 실험의 실행만 필터링해 다음을 표시한다.

- 최신 suite의 모델별 처리량과 TTFT p50/E2E p95 비교 또는 반복 실행 추세
- 최근 실행의 노드별 effective tokens/s 기여도
- 성공률, 요청 수, 전체 실행 표

화면 그래프는 hover/touch 툴팁, 키보드 방향키 탐색과 범례별 표시 전환을 지원한다. 각
그래프의 `PNG`는 현재 대시보드 테마를 고해상도로 저장한다. `논문용`은 동일 experiment ID와
실행 방식만 사용해 흰 배경, 색각 이상 친화 팔레트, 축·단위·재현 파라미터를 포함한 그림으로
다시 그린다. 1단 85 mm 또는 2단 180 mm, 벡터 SVG 또는 실제 pHYs 메타데이터가 포함된
300/600 DPI PNG를 선택할 수 있다. 실행 방식이 섞인 전체 결과는 논문 그래프로 내보내지 않는다.

각 실행은 `.run/cluster/results/<run-id>/`에 다음 파일을 보존한다.

- `config.json`: experiment ID를 포함한 요청 설정
- `events.jsonl`: 모델 로딩, 워밍업, 요청 완료, 경고의 시간순 기록
- `requests.csv`: 요청별 노드, TTFT, E2E, 토큰, 처리량과 오류
- `summary.json`: suite/모델 순서, 실행 방식, 토폴로지, 재현 파라미터,
  scenario별·클러스터별 p50/p95와 유효 처리량

노드가 설정을 자동 하향 조정하면 실제 `n_ctx`, `n_gpu_layers`, `n_batch`를 기록한다.
`동일 구성 강제`가 켜져 있고 노드별 실제 설정이 다르면 측정을 중단한다.

## CLI

### Head 서비스 관리

Head 장비에서는 설치 후 어느 디렉터리에서든 다음 명령으로 대시보드를
관리한다. `llm-cluster`는 프로젝트의 고정된 대시보드 프로세스만 PID, 실행 사용자, 실행 파일,
작업 디렉터리, 전체 명령행과 Linux 프로세스 시작 시각까지 확인한 뒤 제어한다.

```bash
llm-cluster start
llm-cluster stop
llm-cluster restart
llm-cluster status
llm-cluster logs
```

실행 파일은 프로젝트의 `scripts/llm-cluster`이며 사용자 전용
`~/.local/bin/llm-cluster` 심볼릭 링크로 노출한다. systemd unit이나 재부팅 자동 실행은
만들지 않는다. `logs`는 고정된 대시보드 로그의 최근 200줄만 보여준다. 워커 API(8000)는
노드 준비·시작·중지 기능이 별도로 관리하므로 이 명령이 신호를 보내지 않는다. 런타임 설정,
토큰, 인벤토리와 실험 결과는 기존 `.run/cluster` 위치에 그대로 보존한다.

### 클러스터 제어 및 실험

```bash
python -m cluster.clusterctl inventory
python -m cluster.clusterctl status
python -m cluster.clusterctl discover
python -m cluster.clusterctl doctor
python -m cluster.clusterctl environment-check
python -m cluster.clusterctl --node edge-worker-01 environment-install --confirmed
python -m cluster.clusterctl prepare \
  --model qwen2.5-1.5b/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
python -m cluster.clusterctl start
python -m cluster.clusterctl select-model \
  --model-id qwen2.5-1.5b/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --n-ctx 1024 --n-gpu-layers 30
```

Pi 노드만 선택한 CLI 실험은 반드시 `--n-gpu-layers 0`에 해당하는 설정을 사용한다.
실험 설정 예시는 `cluster/config/experiment_defaults.json`을 복사한다. 기존 head 이름을
사용하는 설치에서는 `node_names`를 실제 인벤토리 이름으로 바꾼다.

```bash
cp cluster/config/experiment_defaults.json .run/cluster/experiment.json
python -m cluster.benchmark.runner \
  --config .run/cluster/experiment.json \
  --inventory .run/cluster/nodes.local.csv
```

## 안전 및 재현성 원칙

- SSH는 전용 키와 `BatchMode`로만 실행한다.
- LAN 검색 범위는 head가 연결된 RFC1918 네트워크의 최대 `/24`로 제한한다.
- 브라우저에서 임의 SSH identity 파일이나 공인 IP를 등록할 수 없다.
- 대시보드 토큰 인증은 기본적으로 꺼져 있다. 필요할 때 `설정 → 대시보드 토큰 인증`에서
  켤 수 있으며, 활성화할 때 현재 `.run/cluster/dashboard.token` 값을 한 번 확인한다.
  토큰 파일은 인증을 꺼도 삭제하지 않아 나중에 같은 토큰으로 다시 보호할 수 있다.
- Worker API 인증은 기본적으로 꺼져 있어 신뢰 LAN에서 간단히 사용할 수 있다. 대시보드
  `설정 → 워커 API 토큰 인증`을 켜면 브라우저에 노출하지 않는 head/worker 공유 토큰으로
  상태, 모델 변경과 추론 요청을 보호하고 모든 활성 노드 API를 자동 재시작한다.
- 모델 분할 RPC 포트는 Worker API 토큰의 보호 대상이 아니다. 인증을 켠 상태에서는 실행을
  차단하며, 신뢰 LAN 직접 연결 모드에서만 실험 중 일시적으로 사용한다.
- apt 자동 설치는 고정된 패키지 allowlist와 `sudo -n`에서만 허용한다.
- 코드 동기화에서 `.git`, `.venv`, `models`, `outputs`, `.run`을 제외한다.
- 모델은 `rsync --partial --append-verify`로 선택 파일만 보내며 `--delete`를 사용하지 않는다.
- 실제 인벤토리, 토큰, 작업 로그, 실험 카탈로그와 원시 결과는 `.run/`에 둔다.
- 공정한 시간축 비교를 위해 모든 노드의 NTP 동기화를 권장한다.
