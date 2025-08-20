from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import Connection
from services.database import get_db
from services.api_keys import create_api_key, get_api_keys, update_api_key, delete_api_key
from schemas import APIKey, APIKeyCreate, APIKeyUpdate
from typing import List
import secrets
import os

router = APIRouter()
security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    admin_user = os.environ.get("ADMIN_USERNAME")
    admin_pass = os.environ.get("ADMIN_PASSWORD")

    if not admin_user or not admin_pass:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error: Admin credentials are not configured.",
        )

    is_correct_username = secrets.compare_digest(credentials.username, admin_user)
    is_correct_password = secrets.compare_digest(credentials.password, admin_pass)

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@router.post("/api-keys/", status_code=status.HTTP_201_CREATED, tags=["api-keys"])
def create_new_api_key(api_key: APIKeyCreate, db: Connection = Depends(get_db), username: str = Depends(get_current_username)) -> APIKey:
    return create_api_key(db, api_key)

@router.get("/api-keys/", tags=["api-keys"])
def read_api_keys(skip: int = 0, limit: int = 100, db: Connection = Depends(get_db), username: str = Depends(get_current_username)) -> List[APIKey]:
    return get_api_keys(db, skip=skip, limit=limit)

@router.put("/api-keys/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["api-keys"])
def update_existing_api_key(api_key_id: int, api_key: APIKeyUpdate, db: Connection = Depends(get_db), username: str = Depends(get_current_username)) -> None:
    update_api_key(db, api_key_id, api_key)

@router.delete("/api-keys/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["api-keys"])
def delete_existing_api_key(api_key_id: int, db: Connection = Depends(get_db), username: str = Depends(get_current_username)) -> None:
    delete_api_key(db, api_key_id)
