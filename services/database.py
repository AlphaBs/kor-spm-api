import os
from typing import Generator
from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.exc import OperationalError

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:password@localhost/spam_api')

engine: Engine | None = None
try:
    engine = create_engine(DATABASE_URL)
except OperationalError as e:
    print(f"Error connecting to the database: {e}")
    engine = None

def get_db() -> Generator[Connection, None, None]:
    if engine is None:
        raise ValueError("Database engine is not initialized")

    db = engine.connect()
    try:
        yield db
    finally:
        db.close()
