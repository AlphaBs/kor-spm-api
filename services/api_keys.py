from sqlalchemy import Connection, text
from schemas import APIKey, APIKeyCreate, APIKeyUpdate
from typing import List

def create_api_key(db: Connection, api_key: APIKeyCreate) -> APIKey:
    db.execute(
        text("INSERT INTO api_keys (api_key, description) VALUES (:api_key, :description)"),
        {"api_key": api_key.api_key, "description": api_key.description}
    )
    db.commit()
    result = get_api_key_by_key(db, api_key.api_key)
    if result is None:
        raise Exception("API Key not found")
    return result

def get_api_key(db: Connection, api_key_id: int) -> APIKey | None:
    result = db.execute(text("SELECT * FROM api_keys WHERE id = :id"), {"id": api_key_id}).fetchone()
    if result:
        return APIKey.model_validate(result)
    return None

def get_api_key_by_key(db: Connection, api_key: str) -> APIKey | None:
    result = db.execute(text("SELECT * FROM api_keys WHERE api_key = :api_key"), {"api_key": api_key}).fetchone()
    if result:
        return APIKey.model_validate(result)
    return None

def get_api_keys(db: Connection, skip: int = 0, limit: int = 100) -> List[APIKey]:
    results = db.execute(text("SELECT * FROM api_keys ORDER BY id LIMIT :limit OFFSET :skip"), {"limit": limit, "skip": skip}).fetchall()
    return [APIKey.model_validate(row) for row in results]

def update_api_key(db: Connection, api_key_id: int, api_key: APIKeyUpdate) -> None:
    result = db.execute(
        text("UPDATE api_keys SET description = :description WHERE id = :id"),
        {"id": api_key_id, "description": api_key.description}
    )
    if result.rowcount == 0:
        raise Exception("API Key not found")
    db.commit()

def delete_api_key(db: Connection, api_key_id: int) -> None:
    result = db.execute(text("DELETE FROM api_keys WHERE id = :id"), {"id": api_key_id})
    if result.rowcount == 0:
        raise Exception("API Key not found")
    db.commit()