"""
Long-running service layer for ctxai.

Exposes the agent via REST + WebSocket APIs with session persistence and
health/metrics endpoints. All FastAPI/uvicorn imports are lazy so that
the rest of ctxai keeps working when the `server` extras are not
installed.
"""

from ctxai.service.daemon import DaemonManager
from ctxai.service.health import HealthCheck, HealthResult, HealthStatus
from ctxai.service.rate_limiter import InMemoryBackend, RateLimiter
from ctxai.service.session_manager import Session, SessionManager, SessionState
from ctxai.service.state_store import SQLiteStateStore, StateStore

__all__ = [
    "DaemonManager",
    "HealthCheck",
    "HealthResult",
    "HealthStatus",
    "InMemoryBackend",
    "RateLimiter",
    "SQLiteStateStore",
    "Session",
    "SessionManager",
    "SessionState",
    "StateStore",
]
