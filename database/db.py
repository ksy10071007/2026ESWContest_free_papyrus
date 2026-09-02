import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.security_utils import phone_hash_id


DATABASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = str(DATABASE_DIR / 'database.db')
SCHEMA_PATH = DATABASE_DIR / 'schema.sql'
_HASH_RE = re.compile(r'^[0-9a-fA-F]{64}$')
_PHONE_RE = re.compile(r'^\+?[0-9() .-]{9,20}$')


def get_conn(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA synchronous=NORMAL;')
    conn.execute('PRAGMA foreign_keys=ON;')
    return conn


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row['name']) for row in conn.execute(f'PRAGMA table_info({table_name})')}


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_sql: str) -> None:
    column_name = column_sql.split()[0]
    if column_name not in _table_columns(conn, table_name):
        conn.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_sql}')


def _upgrade_schema(conn: sqlite3.Connection) -> None:
    # CREATE TABLE IF NOT EXISTS does not add columns to an existing deployment.
    for column_sql in (
        'kakao_refresh_token TEXT',
        'kakao_refresh_token_expires_in INTEGER',
        'kakao_scope TEXT',
        'kakao_connected_at TEXT',
    ):
        _ensure_column(conn, 'users', column_sql)
    _ensure_column(conn, 'session_assets', 'public_url TEXT')


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    schema_sql = SCHEMA_PATH.read_text(encoding='utf-8')
    conn = get_conn(db_path)
    try:
        conn.executescript(schema_sql)
        _upgrade_schema(conn)
        conn.commit()
    finally:
        conn.close()


def identifier_hash_id(identifier: str) -> str:
    """Return the canonical, non-reversible key used by every application service."""
    value = str(identifier or '').strip()
    if not value:
        value = 'anonymous'
    if _HASH_RE.fullmatch(value):
        return value.lower()
    if _PHONE_RE.fullmatch(value):
        return phone_hash_id(value)

    pepper = os.environ.get('HASH_PEPPER', '').strip()
    if not pepper:
        raise RuntimeError('HASH_PEPPER environment variable is required')
    normalized = value.casefold()
    return hashlib.sha256(f'{normalized}|{pepper}'.encode('utf-8')).hexdigest()


def _upsert_user_hash_conn(
    conn: sqlite3.Connection,
    phone_hash: str,
    display_name: Optional[str] = None,
) -> int:
    row = conn.execute(
        'SELECT id, display_name FROM users WHERE phone_hash=?',
        (phone_hash,),
    ).fetchone()
    if row:
        if display_name and display_name != row['display_name']:
            conn.execute('UPDATE users SET display_name=? WHERE id=?', (display_name, row['id']))
        return int(row['id'])

    cur = conn.execute(
        'INSERT INTO users (phone_hash, display_name) VALUES (?, ?)',
        (phone_hash, display_name),
    )
    return int(cur.lastrowid)


def upsert_user_by_identifier(
    db_path: str,
    identifier: str,
    display_name: Optional[str] = None,
) -> int:
    conn = get_conn(db_path)
    try:
        user_id = _upsert_user_hash_conn(conn, identifier_hash_id(identifier), display_name)
        conn.commit()
        return user_id
    finally:
        conn.close()


def upsert_user_by_phone(db_path: str, phone: str, display_name: Optional[str] = None) -> int:
    return upsert_user_by_identifier(db_path, phone, display_name)


def get_user_by_phone_hash(db_path: str, phone_hash: str) -> Optional[Dict[str, Any]]:
    conn = get_conn(db_path)
    try:
        row = conn.execute('SELECT * FROM users WHERE phone_hash=?', (phone_hash,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_diagnosis_session(
    db_path: str,
    user_id: int,
    ai_reading: Dict[str, Any],
    pixel_metrics: Optional[Dict[str, Any]] = None,
    survey: Optional[Dict[str, Any]] = None,
    impression: Optional[str] = None,
    fhir_bundle: Optional[Dict[str, Any]] = None,
    status: str = 'done',
) -> int:
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO diagnosis_sessions (
              user_id, ai_reading_json, pixel_metrics_json, survey_json,
              fhir_bundle_json, impression, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                json.dumps(ai_reading, ensure_ascii=False),
                json.dumps(pixel_metrics or {}, ensure_ascii=False),
                json.dumps(survey or {}, ensure_ascii=False),
                json.dumps(fhir_bundle, ensure_ascii=False) if fhir_bundle else None,
                impression,
                status,
            ),
        )
        session_id = int(cur.lastrowid)
        conn.execute(
            'INSERT INTO event_logs (session_id, event_type, payload_json) VALUES (?, ?, ?)',
            (session_id, 'DIAG_CREATED', json.dumps({'user_id': user_id, 'status': status})),
        )
        conn.commit()
        return session_id
    finally:
        conn.close()


def update_fhir_bundle(db_path: str, session_id: int, fhir_bundle: Dict[str, Any]) -> None:
    conn = get_conn(db_path)
    try:
        conn.execute(
            'UPDATE diagnosis_sessions SET fhir_bundle_json=? WHERE id=?',
            (json.dumps(fhir_bundle, ensure_ascii=False), session_id),
        )
        conn.execute(
            'INSERT INTO event_logs (session_id, event_type, payload_json) VALUES (?, ?, ?)',
            (session_id, 'FHIR_UPDATED', json.dumps({'session_id': session_id})),
        )
        conn.commit()
    finally:
        conn.close()


def get_session(db_path: str, session_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn(db_path)
    try:
        row = conn.execute('SELECT * FROM diagnosis_sessions WHERE id=?', (session_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_sessions_for_user(db_path: str, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            'SELECT * FROM diagnosis_sessions WHERE user_id=? ORDER BY diagnosed_at DESC LIMIT ?',
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def add_asset(
    db_path: str,
    session_id: int,
    asset_type: str,
    file_path: str,
    mime_type: Optional[str] = None,
    sha256: Optional[str] = None,
    public_url: Optional[str] = None,
) -> int:
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO session_assets
              (session_id, asset_type, file_path, public_url, mime_type, sha256)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, asset_type, file_path, public_url, mime_type, sha256),
        )
        asset_id = int(cur.lastrowid)
        conn.execute(
            'INSERT INTO event_logs (session_id, event_type, payload_json) VALUES (?, ?, ?)',
            (session_id, 'ASSET_ADDED', json.dumps({'asset_id': asset_id, 'asset_type': asset_type})),
        )
        conn.commit()
        return asset_id
    finally:
        conn.close()


def list_assets(db_path: str, session_id: int) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            'SELECT * FROM session_assets WHERE session_id=? ORDER BY created_at ASC',
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def log_event(
    db_path: str,
    session_id: Optional[int],
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> int:
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            'INSERT INTO event_logs (session_id, event_type, payload_json) VALUES (?, ?, ?)',
            (
                session_id,
                event_type,
                json.dumps(payload, ensure_ascii=False) if payload else None,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_events(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            'SELECT * FROM event_logs ORDER BY created_at DESC LIMIT ?',
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _has_migration_record(
    conn: sqlite3.Connection,
    source_name: str,
    record_type: str,
    source_id: int,
) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM migration_records
        WHERE source_name=? AND record_type=? AND source_id=?
        """,
        (source_name, record_type, source_id),
    ).fetchone()
    return row is not None


def migrate_legacy_history(
    legacy_db_path: str,
    target_db_path: str = DEFAULT_DB_PATH,
    source_name: str = 'legacy_history',
) -> Dict[str, int]:
    """Import the former history.db into the canonical schema without deleting it."""
    result = {'diagnoses': 0, 'surveys': 0, 'skipped': 0}
    if not os.path.exists(legacy_db_path):
        return result
    if os.path.abspath(legacy_db_path) == os.path.abspath(target_db_path):
        raise ValueError('legacy and target database paths must differ')

    init_db(target_db_path)
    legacy = sqlite3.connect(legacy_db_path)
    legacy.row_factory = sqlite3.Row
    target = get_conn(target_db_path)
    try:
        legacy_tables = {
            row['name']
            for row in legacy.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

        if 'diagnosis_history' in legacy_tables:
            for row in legacy.execute('SELECT * FROM diagnosis_history ORDER BY id'):
                source_id = int(row['id'])
                if _has_migration_record(target, source_name, 'diagnosis', source_id):
                    result['skipped'] += 1
                    continue

                user_id = _upsert_user_hash_conn(target, identifier_hash_id(row['user_id']))
                raw_analysis = row['analysis_json'] or '{}'
                try:
                    analysis = json.loads(raw_analysis)
                    status = str(analysis.get('status') or 'done') if isinstance(analysis, dict) else 'done'
                except (TypeError, ValueError):
                    status = 'failed'

                cur = target.execute(
                    """
                    INSERT INTO diagnosis_sessions (
                      user_id, diagnosed_at, ai_reading_json, pixel_metrics_json,
                      survey_json, status, created_at
                    ) VALUES (?, ?, ?, '{}', '{}', ?, ?)
                    """,
                    (user_id, row['created_at'], raw_analysis, status, row['created_at']),
                )
                session_id = int(cur.lastrowid)
                target.execute(
                    """
                    INSERT INTO session_assets
                      (session_id, asset_type, file_path, public_url, mime_type, created_at)
                    VALUES (?, 'image_raw', ?, ?, 'image/jpeg', ?)
                    """,
                    (session_id, row['image_path'], row['image_url'], row['created_at']),
                )
                target.execute(
                    'INSERT INTO event_logs (session_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)',
                    (session_id, 'LEGACY_DIAGNOSIS_MIGRATED', json.dumps({'source_id': source_id}), row['created_at']),
                )
                target.execute(
                    'INSERT INTO migration_records (source_name, record_type, source_id, target_id) VALUES (?, ?, ?, ?)',
                    (source_name, 'diagnosis', source_id, session_id),
                )
                result['diagnoses'] += 1

        if 'survey_history' in legacy_tables:
            for row in legacy.execute('SELECT * FROM survey_history ORDER BY id'):
                source_id = int(row['id'])
                if _has_migration_record(target, source_name, 'survey', source_id):
                    result['skipped'] += 1
                    continue

                user_id = _upsert_user_hash_conn(target, identifier_hash_id(row['user_id']))
                cur = target.execute(
                    'INSERT INTO survey_responses (user_id, survey_json, created_at) VALUES (?, ?, ?)',
                    (user_id, row['survey_json'] or '{}', row['created_at']),
                )
                survey_id = int(cur.lastrowid)
                target.execute(
                    'INSERT INTO migration_records (source_name, record_type, source_id, target_id) VALUES (?, ?, ?, ?)',
                    (source_name, 'survey', source_id, survey_id),
                )
                result['surveys'] += 1

        target.commit()
        return result
    except Exception:
        target.rollback()
        raise
    finally:
        target.close()
        legacy.close()
