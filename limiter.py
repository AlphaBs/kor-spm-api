from fastapi import Depends
from fastapi_limiter.depends import RateLimiter

limiter = Depends(RateLimiter(times=10, minutes=1))
