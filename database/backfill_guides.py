import argparse
import json
import os
import sqlite3


def build_ai_guide(analysis):
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


def backfill(db_path: str, dry_run: bool = False):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f'DB not found: {db_path}')

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            "SELECT id, ai_reading_json FROM diagnosis_sessions ORDER BY id ASC"
        ).fetchall()
        updated = 0
        skipped = 0

        for row in rows:
            rec_id = row['id']
            raw = row['ai_reading_json'] or '{}'
            try:
                analysis = json.loads(raw)
            except Exception:
                skipped += 1
                continue

            if isinstance(analysis, dict) and isinstance(analysis.get('guide'), dict):
                skipped += 1
                continue

            analysis['guide'] = build_ai_guide(analysis if isinstance(analysis, dict) else {})

            if not dry_run:
                conn.execute(
                    "UPDATE diagnosis_sessions SET ai_reading_json=? WHERE id=?",
                    (json.dumps(analysis, ensure_ascii=False), rec_id)
                )
            updated += 1

        if not dry_run:
            conn.commit()

        print(f'total={len(rows)} updated={updated} skipped={skipped} dry_run={dry_run}')
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill guide field into diagnosis session readings')
    parser.add_argument('--db', default='database/database.db', help='Path to SQLite DB')
    parser.add_argument('--dry-run', action='store_true', help='Do not write updates')
    args = parser.parse_args()

    backfill(args.db, dry_run=args.dry_run)
