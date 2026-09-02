import os
import io
import json
import re
import hashlib
import socket
import secrets
import threading
import time
import requests
import qrcode
from flask import Flask, Response, request, jsonify, send_file, redirect
from flask_cors import CORS
from db import get_conn as open_database, init_db

# [추가됨] 앱 설정과 런타임 경로를 스크립트 위치 기준으로 고정.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.local.json")
DB_PATH = os.path.join(BASE_DIR, "database.db")
REPORT_DIR = os.path.join(BASE_DIR, "reports")


def _ensure_runtime_paths() -> None:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG_PATH) or ".", exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


_ensure_runtime_paths()


# [추가됨] 수동 환경변수 입력 대신 config.local.json 로딩 추가.
def load_json_config(path: str = CONFIG_PATH) -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8-sig") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise RuntimeError(f"{path} must contain a JSON object.")

    for key, value in config.items():
        if value is None:
            continue
        os.environ.setdefault(str(key), str(value))

# [추가됨] 새 refresh token을 config.local.json에 다시 저장.
def update_json_config(updates: dict[str, str], path: str = CONFIG_PATH) -> None:
    config: dict[str, object] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as config_file:
            loaded = json.load(config_file)
        if not isinstance(loaded, dict):
            raise RuntimeError(f"{path} must contain a JSON object.")
        config = loaded

    for key, value in updates.items():
        config[str(key)] = value
        os.environ[str(key)] = value

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)


load_json_config()

APP_HOST = os.environ.get("KAKAO_APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("KAKAO_APP_PORT", "5001"))

app = Flask(__name__)
CORS(app)
_PHONE_RE = re.compile(r"\D+")

OAUTH_STATE_STORE: dict[str, dict[str, object]] = {}
OAUTH_STATE_LOCK = threading.Lock()
OAUTH_STATE_TTL_SECONDS = int(os.environ.get("KAKAO_OAUTH_STATE_TTL_SECONDS", "600"))



def get_lan_ip() -> str:
    """
    폰/다른 기기에서 접근 가능한 '내 PC의 로컬(LAN) IP'를 잡는다.
    (172.x 같은 가상/WSL IP가 아니라 보통 192.168.x.x가 나옴)
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # 실제 전송 안 하고 로컬 IP만 얻음
        return s.getsockname()[0]
    finally:
        s.close()

def normalize_phone(phone: str) -> str:
    digits = _PHONE_RE.sub("", str(phone))
    if len(digits) < 9:
        raise ValueError(f"phone looks too short: {digits}")
    return digits


def issue_oauth_state(phone: str, popup: bool = False, session_id: str = "") -> str:
    now = time.time()
    token = secrets.token_urlsafe(24)
    with OAUTH_STATE_LOCK:
        expired_keys = [
            key for key, value in OAUTH_STATE_STORE.items()
            if float(value.get("expires_at", 0)) < now
        ]
        for key in expired_keys:
            OAUTH_STATE_STORE.pop(key, None)

        OAUTH_STATE_STORE[token] = {
            "phone": phone,
            "popup": bool(popup),
            "session_id": session_id,
            "expires_at": now + OAUTH_STATE_TTL_SECONDS,
        }

    return token


def consume_oauth_state(token: str) -> dict[str, object] | None:
    now = time.time()
    with OAUTH_STATE_LOCK:
        payload = OAUTH_STATE_STORE.pop(token, None)

    if not payload:
        return None

    if float(payload.get("expires_at", 0)) < now:
        return None

    return payload


def popup_result_html(ok: bool, message: str, phone: str = "") -> str:
        color = "#0f766e" if ok else "#b91c1c"
        escaped_message = str(message).replace("'", "\\'")
        escaped_phone = str(phone).replace("'", "\\'")
        return f"""
        <!doctype html>
        <html>
        <head>
            <meta charset=\"utf-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
            <title>Kakao OAuth Result</title>
            <style>
                body {{ font-family: sans-serif; padding: 24px; text-align: center; }}
                .msg {{ color: {color}; font-weight: 700; }}
            </style>
        </head>
        <body>
            <h2 class=\"msg\">{message}</h2>
            <p>창이 자동으로 닫힙니다.</p>
            <script>
                if (window.opener && !window.opener.closed) {{
                    window.opener.postMessage({{
                        type: 'kakao-link-result',
                        ok: {str(ok).lower()},
                        message: '{escaped_message}',
                        phone: '{escaped_phone}'
                    }}, '*');
                }}
                setTimeout(function () {{ window.close(); }}, 500);
            </script>
        </body>
        </html>
        """

def phone_hash_id(phone: str) -> str:
    pepper = os.environ.get("HASH_PEPPER")
    if not pepper:
        raise RuntimeError("HASH_PEPPER 환경변수 필요! PowerShell: $env:HASH_PEPPER='secret'")
    norm = normalize_phone(phone)
    return hashlib.sha256(f"{norm}|{pepper}".encode("utf-8")).hexdigest()

# [수정됨] access token 직접 사용 대신 refresh token으로 재발급.
def refresh_kakao_token(refresh_token: str | None = None) -> dict:
    refresh_token = refresh_token or os.environ.get("KAKAO_REFRESH_TOKEN")
    client_id = os.environ.get("KAKAO_CLIENT_ID")
    client_secret = os.environ.get("KAKAO_CLIENT_SECRET")

    if not refresh_token:
        raise RuntimeError("KAKAO_REFRESH_TOKEN 환경변수가 필요합니다.")
    if not client_id:
        raise RuntimeError("KAKAO_CLIENT_ID 환경변수가 필요합니다.")

    token_data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }
    if client_secret:
        token_data["client_secret"] = client_secret

    token_response = requests.post(
        "https://kauth.kakao.com/oauth/token",
        data=token_data,
        timeout=10
    )

    if token_response.status_code != 200:
        raise RuntimeError(f"Kakao token refresh failed: {token_response.status_code} {token_response.text}")

    return token_response.json()


# [추가됨] 전역 fallback 토큰은 유지하되, 새 refresh token이 오면 자동 저장.
def get_kakao_access_token(refresh_token: str | None = None) -> tuple[str, dict]:
    uses_global_refresh_token = refresh_token is None
    token_payload = refresh_kakao_token(refresh_token=refresh_token)
    if uses_global_refresh_token and token_payload.get("refresh_token"):
        update_json_config({"KAKAO_REFRESH_TOKEN": token_payload["refresh_token"]})
    access_token = token_payload.get("access_token")
    if not access_token:
        raise RuntimeError(f"Kakao token refresh response missing access_token: {token_payload}")

    return access_token, token_payload

def get_kakao_oauth_config() -> tuple[str, str, str | None]:
    client_id = os.environ.get("KAKAO_CLIENT_ID")
    redirect_uri = os.environ.get("KAKAO_REDIRECT_URI")
    client_secret = os.environ.get("KAKAO_CLIENT_SECRET")

    if not client_id:
        raise RuntimeError("KAKAO_CLIENT_ID 환경변수가 필요합니다.")
    if not redirect_uri:
        raise RuntimeError("KAKAO_REDIRECT_URI 환경변수가 필요합니다.")

    return client_id, redirect_uri, client_secret

# [수정됨] 카카오 전송 시 사용자별 refresh token을 받을 수 있게 변경.
def kakao_send_me(text: str, link_url: str, refresh_token: str | None = None):
    token, token_payload = get_kakao_access_token(refresh_token=refresh_token)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    template_object = {
        "object_type": "text",
        "text": text,
        "link": {
            "web_url": link_url,
            "mobile_web_url": link_url
        },
        "button_title": "리포트 열기",
    }

    data = {
        "template_object": json.dumps(template_object, ensure_ascii=False)
    }

    r = requests.post(
        "https://kapi.kakao.com/v2/api/talk/memo/default/send",
        headers=headers,
        data=data,
        timeout=10
    )

    if r.status_code != 200:
        raise RuntimeError(f"Kakao send failed: {r.status_code} {r.text}")

    return {
        "send_result": r.json(),
        "token_payload": token_payload,
    }

# [추가됨] 사용자별 카카오 로그인 시작 라우트 추가. 전화번호를 state로 전달.
@app.get("/kakao/login")
def kakao_login():
    phone = request.args.get("phone", "").strip()
    if not phone:
        return jsonify({"error": "phone query required"}), 400

    session_id = request.args.get("session_id", "").strip()

    normalized = normalize_phone(phone)
    popup = request.args.get("popup", "").strip().lower() in {"1", "true", "yes", "y"}
    oauth_state = issue_oauth_state(normalized, popup=popup, session_id=session_id)
    client_id, redirect_uri, _ = get_kakao_oauth_config()
    auth_url = (
        "https://kauth.kakao.com/oauth/authorize"
        f"?response_type=code&client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&state={oauth_state}"
        f"&prompt=login"
    )
    return redirect(auth_url)

# [추가됨] 카카오 로그인 완료 후 사용자별 refresh token 저장.
@app.get("/kakao/callback")
def kakao_callback():
    code = request.args.get("code")
    error = request.args.get("error")
    state = request.args.get("state", "").strip()

    state_payload = consume_oauth_state(state) if state else None
    linked_phone = str(state_payload.get("phone")) if state_payload else ""
    popup_mode = bool(state_payload.get("popup")) if state_payload else False
    linked_session_id = str(state_payload.get("session_id", "")).strip() if state_payload else ""

    if error:
        if popup_mode:
            return popup_result_html(False, "카카오 로그인 실패", linked_phone), 400
        return jsonify({
            "error": "kakao_login_failed",
            "error_description": request.args.get("error_description"),
            "details": request.args.to_dict()
        }), 400

    if not code:
        return jsonify({"error": "code query required"}), 400
    if not state:
        return jsonify({"error": "state query required"}), 400
    if not state_payload or not linked_phone:
        if request.args.get("popup") in {"1", "true"}:
            return popup_result_html(False, "세션이 만료되었습니다. 다시 시도해 주세요."), 400
        return jsonify({"error": "invalid_or_expired_state"}), 400

    client_id, redirect_uri, client_secret = get_kakao_oauth_config()
    token_data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code": code,
    }
    if client_secret:
        token_data["client_secret"] = client_secret

    token_response = requests.post(
        "https://kauth.kakao.com/oauth/token",
        data=token_data,
        timeout=10
    )

    if token_response.status_code != 200:
        if popup_mode:
            return popup_result_html(False, "카카오 토큰 발급 실패", linked_phone), 400
        return jsonify({
            "error": "token_exchange_failed",
            "status_code": token_response.status_code,
            "body": token_response.text,
        }), 400

    token_payload = token_response.json()
    issued_refresh_token = token_payload.get("refresh_token")
    refresh_token_expires_in = token_payload.get("refresh_token_expires_in")
    scope = token_payload.get("scope")

    conn = get_conn()
    try:
        user_id = upsert_user_by_phone(conn, linked_phone)
        if issued_refresh_token:
            set_user_kakao_token(
                conn,
                user_id=user_id,
                refresh_token=issued_refresh_token,
                refresh_token_expires_in=refresh_token_expires_in,
                scope=scope,
            )
        conn.commit()
    finally:
        conn.close()

        if popup_mode:
                return popup_result_html(True, "카카오 연동 완료", linked_phone)

    if linked_session_id:
        return redirect(f"/open/{linked_session_id}")

    return Response(f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>카카오 연동 완료</title>
    </head>
    <body style="font-family:sans-serif; padding:40px; text-align:center;">
        <h2>카카오 연동 완료</h2>
        <p>{linked_phone} 번호에 카카오 계정이 연결되었습니다.</p>
        <p>이제 리포트 전송이 가능합니다.</p>
    </body>
    </html>
    """, mimetype="text/html")


@app.post("/kakao/send_report")
def kakao_send_report():
    data = request.get_json(force=True, silent=True) or {}
    phone = str(data.get("phone", "")).strip()
    text = str(data.get("text", "")).strip()
    link_url = str(data.get("link_url", "")).strip()

    if not phone:
        return jsonify({"status": "error", "message": "phone is required"}), 400
    if not text:
        return jsonify({"status": "error", "message": "text is required"}), 400
    if not link_url:
        return jsonify({"status": "error", "message": "link_url is required"}), 400

    try:
        normalized_phone = normalize_phone(phone)
    except Exception:
        return jsonify({"status": "error", "message": "invalid phone format"}), 400

    conn = get_conn()
    try:
        user_id = upsert_user_by_phone(conn, normalized_phone)
        user_refresh_token = get_user_kakao_token_by_phone(conn, normalized_phone)

        global_refresh_token = os.environ.get("KAKAO_REFRESH_TOKEN", "").strip()
        if not user_refresh_token and not global_refresh_token:
            return jsonify({
                "status": "error",
                "message": "카카오 계정 연동이 필요합니다. 메인 화면에서 카카오 연동을 먼저 완료해 주세요.",
                "code": "KAKAO_LINK_REQUIRED",
                "phone": normalized_phone,
            }), 400

        send_result = kakao_send_me(text=text, link_url=link_url, refresh_token=user_refresh_token or None)

        refreshed_refresh_token = send_result["token_payload"].get("refresh_token")
        if user_refresh_token and refreshed_refresh_token:
            set_user_kakao_token(
                conn,
                user_id=user_id,
                refresh_token=refreshed_refresh_token,
                refresh_token_expires_in=send_result["token_payload"].get("refresh_token_expires_in"),
                scope=send_result["token_payload"].get("scope"),
            )

        conn.commit()
        return jsonify({
            "status": "ok",
            "phone": normalized_phone,
            "send_result": send_result["send_result"],
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"카카오 전송 실패: {e}",
            "code": "KAKAO_SEND_FAILED",
        }), 400
    finally:
        conn.close()

def get_conn():
    return open_database(DB_PATH)


def initialize_runtime_database() -> None:
    init_db(DB_PATH)


initialize_runtime_database()


def upsert_user_by_phone(conn, phone: str, display_name: str | None = None) -> int:
    ph = phone_hash_id(phone)
    cur = conn.cursor()
    cur.execute("SELECT id, display_name FROM users WHERE phone_hash=?", (ph,))
    row = cur.fetchone()
    if row:
        if display_name and display_name != row["display_name"]:
            cur.execute("UPDATE users SET display_name=? WHERE id=?", (display_name, row["id"]))
        return int(row["id"])

    cur.execute("INSERT INTO users (phone_hash, display_name) VALUES (?, ?)", (ph, display_name))
    return int(cur.lastrowid)


# [추가됨] users 테이블에 사용자 카카오 refresh token과 메타데이터 저장.
def set_user_kakao_token(conn, user_id: int, refresh_token: str, refresh_token_expires_in: int | None = None,
                         scope: str | None = None) -> None:
    conn.execute(
        """
        UPDATE users
        SET kakao_refresh_token=?, kakao_refresh_token_expires_in=?, kakao_scope=?, kakao_connected_at=datetime('now')
        WHERE id=?
        """,
        (refresh_token, refresh_token_expires_in, scope, user_id)
    )


# [추가됨] 진단 요청의 전화번호에 연결된 사용자 카카오 토큰 조회.
def get_user_kakao_token_by_phone(conn, phone: str) -> str | None:
    ph = phone_hash_id(phone)
    row = conn.execute(
        "SELECT kakao_refresh_token FROM users WHERE phone_hash=?",
        (ph,)
    ).fetchone()
    if not row:
        return None
    return row["kakao_refresh_token"]


def create_session(conn, user_id: int, ai_reading: dict, pixel_metrics: dict, survey: dict,
                   impression: str | None = None, fhir_bundle: dict | None = None,
                   status: str = "done") -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO diagnosis_sessions (
          user_id, ai_reading_json, pixel_metrics_json, survey_json,
          fhir_bundle_json, impression, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            json.dumps(ai_reading, ensure_ascii=False),
            json.dumps(pixel_metrics, ensure_ascii=False),
            json.dumps(survey, ensure_ascii=False),
            json.dumps(fhir_bundle, ensure_ascii=False) if fhir_bundle else None,
            impression,
            status,
        )
    )
    return int(cur.lastrowid)
    
def get_user_history_by_phone(conn, phone: str):
    ph = phone_hash_id(phone)

    user_row = conn.execute(
        "SELECT id, display_name FROM users WHERE phone_hash=?",
        (ph,)
    ).fetchone()

    if not user_row:
        return None

    user_id = user_row["id"]

    rows = conn.execute(
        """
        SELECT id, diagnosed_at, status, impression, ai_reading_json, pixel_metrics_json
        FROM diagnosis_sessions
        WHERE user_id=?
        ORDER BY diagnosed_at ASC
        """,
        (user_id,)
    ).fetchall()

    history = []
    for row in rows:
        ai_reading = json.loads(row["ai_reading_json"]) if row["ai_reading_json"] else {}
        pixel_metrics = json.loads(row["pixel_metrics_json"]) if row["pixel_metrics_json"] else {}

        history.append({
            "session_id": row["id"],
            "diagnosed_at": row["diagnosed_at"],
            "status": row["status"],
            "impression": row["impression"],
            "ai_label": ai_reading.get("label"),
            "ai_score": ai_reading.get("score"),
            "redness_area": pixel_metrics.get("redness_area"),
            "vessel_density": pixel_metrics.get("vessel_density"),
        })

    return {
        "user_id": user_row["id"],
        "display_name": user_row["display_name"],
        "history": history
    }

@app.get("/history")
def get_history():

    phone = request.args.get("phone")

    if not phone:
        return jsonify({"error": "phone query required"}), 400

    conn = get_conn()

    try:
        result = get_user_history_by_phone(conn, phone)

        if not result:
            return jsonify({"error": "user not found"}), 404

        return jsonify(result)

    finally:
        conn.close()


def add_asset(conn, session_id: int, asset_type: str, file_path: str, mime_type: str | None = None):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO session_assets (session_id, asset_type, file_path, mime_type)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, asset_type, file_path, mime_type)
    )
    return int(cur.lastrowid)


def log_event(conn, session_id: int | None, event_type: str, payload: dict | None = None):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO event_logs (session_id, event_type, payload_json) VALUES (?, ?, ?)",
        (session_id, event_type, json.dumps(payload, ensure_ascii=False) if payload else None)
    )
    return int(cur.lastrowid)


def generate_pdf_report(session_id: int, ai_reading: dict, pixel_metrics: dict, survey: dict, impression: str | None):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    os.makedirs(REPORT_DIR, exist_ok=True)
    pdf_path = os.path.join(REPORT_DIR, f"report_{session_id}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"Diagnosis Report (session_id={session_id})")
    y -= 40

    c.setFont("Helvetica", 11)
    c.drawString(50, y, "AI Reading:")
    y -= 18
    c.drawString(70, y, json.dumps(ai_reading, ensure_ascii=False))
    y -= 28

    c.drawString(50, y, "Pixel Metrics:")
    y -= 18
    c.drawString(70, y, json.dumps(pixel_metrics, ensure_ascii=False))
    y -= 28

    c.drawString(50, y, "Survey:")
    y -= 18
    c.drawString(70, y, json.dumps(survey, ensure_ascii=False))
    y -= 28

    if impression:
        c.drawString(50, y, "Impression:")
        y -= 18
        c.drawString(70, y, impression)
        y -= 28

    c.showPage()
    c.save()
    return pdf_path


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/diagnosis")
def post_diagnosis():
    data = request.get_json(force=True)

    phone = data.get("phone", "")
    display_name = data.get("display_name")

    ai_reading = data.get("ai_reading", {})
    pixel_metrics = data.get("pixel_metrics", {})
    survey = data.get("survey", {})
    impression = data.get("impression")
    make_pdf = bool(data.get("make_pdf", False))
    print("DEBUG make_pdf =", make_pdf)
    conn = get_conn()
    try:
        user_id = upsert_user_by_phone(conn, phone, display_name)
        session_id = create_session(conn, user_id, ai_reading, pixel_metrics, survey, impression=impression)
        log_event(conn, session_id, "DIAG_DONE", {"user_id": user_id})

        pdf_path = None
        if make_pdf:
            pdf_path = generate_pdf_report(session_id, ai_reading, pixel_metrics, survey, impression)
            add_asset(conn, session_id, "pdf_report", pdf_path, "application/pdf")
            log_event(conn, session_id, "PDF_CREATED", {"path": pdf_path})

            # (카톡) 일단 꺼두는 게 안전 -> 아래 3줄은 주석 처리해도 됨
            #server_ip = socket.gethostbyname(socket.gethostname())
            server_ip = get_lan_ip()   # <-- 이걸로 바꿔야 172.x 같은 거 안 잡힘
            open_url = f"http://{server_ip}:{APP_PORT}/open/{session_id}"

        
            try:    
                # [수정됨] 전화번호에 연결된 사용자 토큰을 우선 사용하고, 없으면 전역 설정 토큰 사용.
                user_refresh_token = get_user_kakao_token_by_phone(conn, phone)
                send_result = kakao_send_me(
                    text=f"진단 리포트가 생성되었습니다. (session_id={session_id})\n\n리포트 열기:\n{open_url}",
                    link_url=open_url,
                    refresh_token=user_refresh_token
                )
                refreshed_refresh_token = send_result["token_payload"].get("refresh_token")
                if user_refresh_token and refreshed_refresh_token:
                    set_user_kakao_token(
                        conn,
                        user_id=user_id,
                        refresh_token=refreshed_refresh_token,
                        refresh_token_expires_in=send_result["token_payload"].get("refresh_token_expires_in"),
                        scope=send_result["token_payload"].get("scope"),
                    )
                log_event(conn, session_id, "KAKAO_SENT", {"url": open_url})
            except Exception as e:
                print("DEBUG kakao error:", e)
                log_event(conn, session_id, "KAKAO_FAILED", {"error": str(e)})

        conn.commit()
        server_ip = get_lan_ip()
        kakao_login_url = f"http://{server_ip}:{APP_PORT}/kakao/login?phone={phone}&session_id={session_id}"
        open_url = f"http://{server_ip}:{APP_PORT}/open/{session_id}"

        return jsonify({
            "session_id": session_id,
            "user_id": user_id,
            "pdf_path": pdf_path,
            "open_url": open_url,
            "kakao_login_url": kakao_login_url
        })

    finally:
        conn.close()


@app.get("/report/<int:session_id>")
def get_report(session_id: int):
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT file_path FROM session_assets WHERE session_id=? AND asset_type='pdf_report' ORDER BY created_at DESC LIMIT 1",
            (session_id,)
        ).fetchone()
        if not row:
            return jsonify({"error": "pdf not found"}), 404

        path = row["file_path"]
        if not os.path.exists(path):
            return jsonify({"error": "file missing on disk", "path": path}), 500
        # download=1 이면 첨부(다운로드), 아니면 inline(브라우저에서 열기)
        download = request.args.get("download") == "1"
        return send_file(path, mimetype="application/pdf", as_attachment=download, download_name=os.path.basename(path))
    finally:
        conn.close()
@app.get("/open/<int:session_id>")
def open_report(session_id: int):
    # 카톡 인앱브라우저에서 PDF 바로 렌더링이 불안정해서,
    # HTML 페이지에서 iframe으로 보여주는 방식
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Report {session_id}</title>
      <style>
        body, html {{ margin:0; padding:0; height:100%; }}
        .bar {{ padding:12px; font-family: sans-serif; }}
        iframe {{ width:100%; height: calc(100% - 52px); border:0; }}
        a {{ display:inline-block; padding:8px 12px; border:1px solid #ccc; border-radius:10px; text-decoration:none; }}
      </style>
    </head>
    <body>
      <div class="bar">
        <a href="/report/{session_id}" target="_blank">PDF 새탭으로 열기</a>
        &nbsp;
        <a href="/report/{session_id}?download=1">다운로드</a>
      </div>
      <iframe src="/report/{session_id}"></iframe>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")

@app.get("/qr/<int:session_id>")
def get_qr(session_id: int):
    # 서버 IP 구하기
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    # PDF 다운로드 링크
    url = f"http://{local_ip}:{APP_PORT}/report/{session_id}"

    # QR 코드 생성
    img = qrcode.make(url)

    # 메모리 버퍼에 이미지 저장
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=True, use_reloader=False)
