from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from api_key.utils import verify_api_key
from api_key.database import get_db

router = APIRouter(
    prefix="/security_check",
    tags=['security']
)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Dependency
def get_valid_api_key(api_key: str = Depends(api_key_header), db: Session = Depends(get_db)) -> str:
    if not api_key or not verify_api_key(db, api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key

@router.get("")
def security_check_endpoint(api_key: str = Depends(get_valid_api_key)):
    """
    Input: API Key (str)
    Ouptut: message (str)
    """
    return {"message": "You have access!"}
