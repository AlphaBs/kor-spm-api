from pydantic import BaseModel
import datetime

class APIKeyBase(BaseModel):
    description: str | None = None

class APIKeyCreate(APIKeyBase):
    api_key: str

class APIKeyUpdate(BaseModel):
    description: str | None = None

class APIKey(APIKeyBase):
    id: int
    api_key: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True
