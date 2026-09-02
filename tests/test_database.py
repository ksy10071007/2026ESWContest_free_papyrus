import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('HASH_PEPPER', 'test-only-pepper')

from database.db import identifier_hash_id, init_db, migrate_legacy_history


class DatabaseSchemaTests(unittest.TestCase):
    def test_schema_is_sql_and_has_no_legacy_tables(self):
        schema = Path('database/schema.sql').read_text(encoding='utf-8')
        self.assertIn('CREATE TABLE IF NOT EXISTS diagnosis_sessions', schema)
        self.assertIn('CREATE TABLE IF NOT EXISTS survey_responses', schema)
        self.assertNotIn('def phone_hash_id', schema)
        self.assertNotIn('CREATE TABLE IF NOT EXISTS diagnosis_history', schema)

    def test_legacy_migration_is_idempotent_and_hashes_identifiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_path = os.path.join(temp_dir, 'history.db')
            target_path = os.path.join(temp_dir, 'database.db')
            legacy = sqlite3.connect(legacy_path)
            legacy.executescript(
                """
                CREATE TABLE diagnosis_history (
                  id INTEGER PRIMARY KEY,
                  user_id TEXT NOT NULL,
                  image_path TEXT NOT NULL,
                  image_url TEXT NOT NULL,
                  analysis_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE survey_history (
                  id INTEGER PRIMARY KEY,
                  user_id TEXT NOT NULL,
                  survey_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                """
            )
            legacy.execute(
                'INSERT INTO diagnosis_history VALUES (?, ?, ?, ?, ?, ?)',
                (
                    7,
                    '010-1234-5678',
                    '/tmp/source.jpg',
                    '/static/source.jpg',
                    json.dumps({'status': 'success', 'left_eye': {'disease': 'Normal'}}),
                    '2026-01-02 03:04:05',
                ),
            )
            legacy.execute(
                'INSERT INTO survey_history VALUES (?, ?, ?, ?)',
                (9, '010-1234-5678', json.dumps({'age': 20}), '2026-01-02 03:05:00'),
            )
            legacy.commit()
            legacy.close()

            init_db(target_path)
            first = migrate_legacy_history(legacy_path, target_path, source_name='test-v1')
            second = migrate_legacy_history(legacy_path, target_path, source_name='test-v1')

            self.assertEqual(first, {'diagnoses': 1, 'surveys': 1, 'skipped': 0})
            self.assertEqual(second, {'diagnoses': 0, 'surveys': 0, 'skipped': 2})

            conn = sqlite3.connect(target_path)
            conn.row_factory = sqlite3.Row
            try:
                user = conn.execute('SELECT * FROM users').fetchone()
                self.assertEqual(user['phone_hash'], identifier_hash_id('010-1234-5678'))
                self.assertNotIn('010', user['phone_hash'])
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM diagnosis_sessions').fetchone()[0], 1)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM session_assets').fetchone()[0], 1)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM survey_responses').fetchone()[0], 1)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM migration_records').fetchone()[0], 2)
                asset = conn.execute('SELECT * FROM session_assets').fetchone()
                self.assertEqual(asset['public_url'], '/static/source.jpg')
            finally:
                conn.close()


if __name__ == '__main__':
    unittest.main()
