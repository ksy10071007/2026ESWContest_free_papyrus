PRAGMA foreign_keys = ON;

-- Canonical application schema. Runtime code must not create tables directly.
CREATE TABLE IF NOT EXISTS users (
  id                             INTEGER PRIMARY KEY AUTOINCREMENT,
  phone_hash                     TEXT NOT NULL UNIQUE,
  display_name                   TEXT,
  kakao_refresh_token            TEXT,
  kakao_refresh_token_expires_in INTEGER,
  kakao_scope                    TEXT,
  kakao_connected_at             TEXT,
  created_at                     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_users_phone_hash ON users(phone_hash);

CREATE TABLE IF NOT EXISTS diagnosis_sessions (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id            INTEGER NOT NULL,
  diagnosed_at       TEXT NOT NULL DEFAULT (datetime('now')),
  ai_reading_json    TEXT NOT NULL,
  pixel_metrics_json TEXT NOT NULL DEFAULT '{}',
  survey_json        TEXT NOT NULL DEFAULT '{}',
  fhir_bundle_json   TEXT,
  impression         TEXT,
  status             TEXT NOT NULL DEFAULT 'done',
  created_at         TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_diag_user_time
  ON diagnosis_sessions(user_id, diagnosed_at DESC);

CREATE TABLE IF NOT EXISTS session_assets (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  INTEGER NOT NULL,
  asset_type  TEXT NOT NULL,
  file_path   TEXT NOT NULL,
  public_url  TEXT,
  mime_type   TEXT,
  sha256      TEXT,
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(session_id) REFERENCES diagnosis_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_assets_session_type
  ON session_assets(session_id, asset_type);

CREATE TABLE IF NOT EXISTS survey_responses (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id     INTEGER NOT NULL,
  survey_json TEXT NOT NULL,
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_surveys_user_time
  ON survey_responses(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS event_logs (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id   INTEGER,
  event_type   TEXT NOT NULL,
  payload_json TEXT,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(session_id) REFERENCES diagnosis_sessions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_event_time ON event_logs(created_at DESC);

-- Makes imports from the former history.db safe to run more than once.
CREATE TABLE IF NOT EXISTS migration_records (
  source_name TEXT NOT NULL,
  record_type TEXT NOT NULL,
  source_id   INTEGER NOT NULL,
  target_id   INTEGER NOT NULL,
  migrated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY(source_name, record_type, source_id)
);
