from __future__ import annotations
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DEFAULT_DATA_DIR = Path(os.getenv("CHROMABOOK_DATA_DIR", str(Path(__file__).resolve().parents[1] / "var"))).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DATA_DIR / 'chromabook.db'}")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
