from sqlalchemy import Column, String, Integer, Boolean, DateTime
from datetime import datetime
from .database import Base, engine

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    hashed_key = Column(String, unique=True, index=True)
    owner = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)
