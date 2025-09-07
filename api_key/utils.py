import secrets
import hashlib
from sqlalchemy.orm import Session
from .models import APIKey # type: ignore

def generate_api_key():
    return secrets.token_urlsafe(32)

def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def create_api_key(db: Session, owner: str):
    raw_key = generate_api_key()
    hashed = hash_api_key(raw_key)

    api_key = APIKey(owner=owner, hashed_key=hashed)
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return raw_key  # Return raw key only once!

def verify_api_key(db: Session, raw_key: str) -> bool:
    hashed = hash_api_key(raw_key)
    key_entry = db.query(APIKey).filter_by(hashed_key=hashed, is_active=True).first()
    return key_entry is not None
