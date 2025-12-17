# db.py
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base

# Load .env from current working directory (safe to call multiple times)
load_dotenv()

def _get_database_url():
    raw = os.environ.get("DATABASE_URL", "")
    if not raw:
        raise RuntimeError(
            "Set DATABASE_URL env var before importing db. "
            "Example .env: DATABASE_URL=postgresql://user:password@localhost:5432/farmdb"
        )
    # sanitize: strip surrounding whitespace & surrounding quotes
    s = raw.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return raw, s

raw_db, DATABASE_URL = _get_database_url()

# Debug prints (remove in production)
print("DEBUG: raw DATABASE_URL repr:", repr(raw_db))
print("DEBUG: sanitized DATABASE_URL repr:", repr(DATABASE_URL))

# Basic validation: must start with a recognized scheme
if not (DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgresql+")):
    raise RuntimeError(
        "DATABASE_URL doesn't start with 'postgresql://' or 'postgresql+...'.\n"
        f"Sanitized value: {repr(DATABASE_URL)}\n"
        "Make sure your .env line is like:\n"
        "DATABASE_URL=postgresql://user:password@localhost:5432/dbname\n"
        "Remove any leading 'DATABASE_URL=' prefix inside the value and any trailing comments on the same line."
    )

# Create engine
engine = create_engine(DATABASE_URL, echo=False, future=True, pool_pre_ping=True)

# Session factory
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True))

# Base declarative class for models
Base = declarative_base()
