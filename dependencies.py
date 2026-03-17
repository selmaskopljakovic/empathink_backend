"""
Shared dependencies for the EmpaThink backend.
Avoids circular imports between main.py and route modules.
"""

from slowapi import Limiter
from starlette.requests import Request


def get_real_ip(request: Request) -> str:
    """Extract the real client IP, respecting X-Forwarded-For behind a load balancer."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=get_real_ip)
