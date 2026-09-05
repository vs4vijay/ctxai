"""
FastAPI REST + WebSocket server for ctxai.

FastAPI/uvicorn are optional dependencies — install with the `server`
extra (`pip install ctxai[server]` or `uv pip install -e '.[server]'`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ctxai.agent.config import AgentConfig
from ctxai.logging import RequestContext, get_logger, setup_logging
from ctxai.monitoring import REQUEST_DURATION, REQUESTS_TOTAL, get_metrics
from ctxai.service.health import HealthCheck
from ctxai.service.rate_limiter import RateLimiter
from ctxai.service.session_manager import SessionManager
from ctxai.service.state_store import SQLiteStateStore

logger = get_logger("service.api")


@dataclass
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    api_key: str | None = None
    cors_origins: list[str] | None = None
    state_db: str = "ctxai_state.db"
    max_sessions: int = 100
    request_size_limit: int = 10 * 1024 * 1024
    rate_limit_max: int = 30
    rate_limit_window: int = 60


try:
    from pydantic import BaseModel as _BaseModel  # noqa: F401

    class CreateSessionRequest(_BaseModel):
        config: dict[str, Any] | None = None
        metadata: dict[str, Any] | None = None

    class SendMessageRequest(_BaseModel):
        message: str

    class UpdateSessionRequest(_BaseModel):
        config: dict[str, Any] | None = None
        metadata: dict[str, Any] | None = None
except ImportError:
    CreateSessionRequest = SendMessageRequest = UpdateSessionRequest = None  # type: ignore[assignment]


def create_app(config: APIConfig | None = None):
    """
    Build the FastAPI application. Lazy-imports FastAPI so the rest of
    ctxai works without the `server` extra installed.
    """
    try:
        from fastapi import (
            Body,
            Depends,
            FastAPI,
            Header,
            HTTPException,
            Request,
            WebSocket,
            WebSocketDisconnect,
            status,
        )
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import PlainTextResponse
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is required for the service layer. Install with: pip install 'ctxai[server]'"
        ) from exc

    cfg = config or APIConfig()
    setup_logging()

    state_store = SQLiteStateStore(cfg.state_db)
    session_manager = SessionManager(state_store=state_store, max_sessions=cfg.max_sessions)
    rate_limiter = RateLimiter()
    health = HealthCheck(session_manager=session_manager, state_store_path=cfg.state_db)

    app = FastAPI(title="ctxai", version="1.0.0", description="ctxai REST + WS API")

    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if CreateSessionRequest is None:
        raise RuntimeError("Pydantic is required to build the API.")

    # ----- Auth -----

    def require_api_key(authorization: str | None = Header(default=None)):
        if cfg.api_key is None:
            return True
        expected = f"Bearer {cfg.api_key}"
        if authorization != expected:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
        return True

    # ----- Middleware -----

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        with RequestContext():
            metrics = get_metrics()
            with metrics.timer(REQUEST_DURATION, labels={"path": request.url.path}):
                metrics.increment(
                    REQUESTS_TOTAL,
                    labels={"method": request.method, "path": request.url.path},
                )
                response = await call_next(request)
            return response

    # ----- Lifecycle -----

    @app.on_event("startup")
    async def _startup():
        recovered = await session_manager.recover_from_store()
        logger.info(f"Recovered {recovered} sessions from store")

    # ----- Routes -----

    @app.get("/api/v1/health")
    async def get_health():
        return health.run().to_dict()

    @app.get("/api/v1/metrics", response_class=PlainTextResponse)
    async def get_metrics_route():
        return get_metrics().to_prometheus()

    @app.get("/api/v1/info")
    async def get_info():
        return {
            "name": "ctxai",
            "version": "1.0.0",
            "active_sessions": session_manager.active_count,
            "uptime_seconds": get_metrics().uptime_seconds(),
        }

    @app.post("/api/v1/sessions", status_code=201, dependencies=[Depends(require_api_key)])
    async def create_session(payload: CreateSessionRequest = Body(default_factory=CreateSessionRequest)):
        config = AgentConfig.from_dict(payload.config) if payload.config else AgentConfig()
        session = await session_manager.create_session(config=config, metadata=payload.metadata)
        return session.to_dict()

    @app.get("/api/v1/sessions", dependencies=[Depends(require_api_key)])
    async def list_sessions():
        sessions = await session_manager.list_sessions()
        return [s.to_dict() for s in sessions]

    @app.get("/api/v1/sessions/{session_id}", dependencies=[Depends(require_api_key)])
    async def get_session(session_id: str):
        session = await session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(404, "Session not found")
        return session.to_dict()

    @app.patch("/api/v1/sessions/{session_id}", dependencies=[Depends(require_api_key)])
    async def update_session(session_id: str, payload: UpdateSessionRequest = Body(...)):
        session = await session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(404, "Session not found")
        if payload.metadata is not None:
            session.metadata.update(payload.metadata)
        if payload.config is not None:
            session.config = AgentConfig.from_dict(payload.config)
        session_manager.state_store.save_session(session_id, session.to_dict())
        return session.to_dict()

    @app.delete("/api/v1/sessions/{session_id}", dependencies=[Depends(require_api_key)])
    async def delete_session(session_id: str):
        ok = await session_manager.delete_session(session_id)
        if not ok:
            raise HTTPException(404, "Session not found")
        return {"deleted": True, "session_id": session_id}

    @app.post(
        "/api/v1/sessions/{session_id}/messages",
        dependencies=[Depends(require_api_key)],
    )
    async def send_message(session_id: str, payload: SendMessageRequest = Body(...)):
        result = await rate_limiter.check(
            f"session:{session_id}",
            max_requests=cfg.rate_limit_max,
            window_seconds=cfg.rate_limit_window,
        )
        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": str(result.remaining),
                    "X-RateLimit-Reset": str(result.reset_at),
                },
            )
        try:
            response = await session_manager.send_message(session_id, payload.message)
        except KeyError:
            raise HTTPException(404, "Session not found")
        return {"response": response, "timestamp": datetime.utcnow().isoformat()}

    @app.get(
        "/api/v1/sessions/{session_id}/messages",
        dependencies=[Depends(require_api_key)],
    )
    async def list_messages(session_id: str, limit: int = 100):
        return {"messages": session_manager.get_messages(session_id, limit=limit)}

    @app.delete(
        "/api/v1/sessions/{session_id}/messages",
        dependencies=[Depends(require_api_key)],
    )
    async def clear_messages(session_id: str):
        try:
            cleared = await session_manager.clear_history(session_id)
        except Exception:
            raise HTTPException(404, "Session not found")
        return {"cleared": cleared}

    @app.get("/api/v1/sessions/{session_id}/plan", dependencies=[Depends(require_api_key)])
    async def get_plan(session_id: str):
        plan = session_manager.state_store.get_active_plan(session_id)
        if plan is None:
            return {"plan": None}
        return {"plan": plan}

    @app.websocket("/ws/sessions/{session_id}")
    async def ws_session(websocket: WebSocket, session_id: str):
        await websocket.accept()
        try:
            while True:
                payload = await websocket.receive_text()
                try:
                    msg = json.loads(payload)
                except Exception:
                    await websocket.send_json({"event": "error", "detail": "invalid json"})
                    continue
                if msg.get("type") != "message":
                    continue
                await websocket.send_json(
                    {
                        "event": "message.start",
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                try:
                    async for chunk in session_manager.stream_message(session_id, msg.get("content", "")):
                        await websocket.send_json({"event": "message.chunk", "content": chunk})
                except KeyError:
                    await websocket.send_json({"event": "error", "detail": "session not found"})
                    return
                await websocket.send_json({"event": "message.complete"})
        except WebSocketDisconnect:
            pass

    return app


def run(host: str = "0.0.0.0", port: int = 8000, workers: int = 1, **kwargs):
    """Convenience entry point: run uvicorn with the FastAPI app."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the service. Install with: pip install 'ctxai[server]'") from exc
    cfg = APIConfig(host=host, port=port, workers=workers, **kwargs)
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, workers=cfg.workers)
