# security_utils.py
import os
import re
import hashlib
import hmac

PHONE_RE = re.compile(r"\D+")

HASH_PEPPER = os.environ.get("HASH_PEPPER")  # 하위 호환용 초기값

def _require_pepper() -> str:
    pepper = os.environ.get("HASH_PEPPER") or HASH_PEPPER
    if not pepper:
        raise RuntimeError("HASH_PEPPER 환경변수가 설정되어야 합니다. (코드에 하드코딩 금지)")
    return pepper

def normalize_phone(phone: str) -> str:
    """
    010-1234-5678 -> 01012345678
    """
    if phone is None:
        raise ValueError("phone is None")
    digits = PHONE_RE.sub("", phone)
    if len(digits) < 9:
        raise ValueError("phone looks too short")
    return digits

def phone_hash_id(phone: str) -> str:
    """
    사용자 고유 ID로 사용할 전화 해시.
    동일 전화 -> 동일 해시(= 유니크 키)
    보안: pepper를 섞어서 사전 공격을 어렵게 함.
    """
    p = _require_pepper()
    norm = normalize_phone(phone)
    msg = f"{norm}|{p}".encode("utf-8")
    return hashlib.sha256(msg).hexdigest()

def secure_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)
