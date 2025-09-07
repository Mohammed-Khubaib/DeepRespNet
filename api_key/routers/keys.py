from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api_key.utils import create_api_key 
from api_key.database import get_db

router = APIRouter(
    prefix="/create-key",
    tags=['generate keys']
)

@router.post("")
def create_key(owner: str, db: Session = Depends(get_db)):
    key = create_api_key(db, owner)
    return {"api_key": key}
