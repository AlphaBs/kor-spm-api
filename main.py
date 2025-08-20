from fastapi import FastAPI
from routers import api_keys, model
from contextlib import asynccontextmanager
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
import os

@asynccontextmanager
async def lifespan(_: FastAPI):
    redis_connection = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379'), encoding="utf-8")
    await FastAPILimiter.init(redis_connection)
    yield
    await FastAPILimiter.close()

app = FastAPI(lifespan=lifespan)

app.include_router(api_keys.router)
app.include_router(model.router)
