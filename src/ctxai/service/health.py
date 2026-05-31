"""
Health checks for the service layer.

`HealthCheck.run()` collects provider connectivity, database accessibility,
and process resource stats into a `HealthResult` that the API serves on
`/api/v1/health`.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ctxai.monitoring import get_metrics


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthResult:
    status: HealthStatus
    version: str
    uptime_seconds: float
    active_sessions: int
    providers: dict[str, str] = field(default_factory=dict)
    database: str = "unknown"
    memory_usage_mb: float = 0.0
    disk_space_gb: float = 0.0
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class HealthCheck:
    """Service-level liveness/readiness probe."""

    def __init__(
        self,
        session_manager: Any | None = None,
        state_store_path: Path | str | None = None,
        version: str = "1.0.0",
    ):
        self.session_manager = session_manager
        self.state_store_path = Path(state_store_path) if state_store_path else None
        self.version = version

    def _check_providers(self) -> dict[str, str]:
        try:
            from ctxai.agent.llm.factory import LLMProviderFactory
        except Exception:
            return {}
        providers = ("openrouter", "anthropic", "openai", "ollama", "github-copilot")
        status: dict[str, str] = {}
        for name in providers:
            try:
                available, _ = LLMProviderFactory.check_provider_availability(name)
                status[name] = "healthy" if available else "unavailable"
            except Exception:
                status[name] = "error"
        return status

    def _check_database(self) -> str:
        if self.state_store_path is None:
            return "not_configured"
        if not self.state_store_path.exists():
            return "missing"
        try:
            return "healthy" if os.access(self.state_store_path, os.R_OK | os.W_OK) else "readonly"
        except Exception:
            return "error"

    def _memory_mb(self) -> float:
        try:
            import resource  # POSIX-only
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return usage / 1024.0  # ru_maxrss is KB on Linux
        except Exception:
            try:
                import psutil  # type: ignore

                return psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0

    def _disk_free_gb(self) -> float:
        try:
            total, used, free = shutil.disk_usage(Path.cwd())
            return free / (1024**3)
        except Exception:
            return 0.0

    def run(self) -> HealthResult:
        providers = self._check_providers()
        database = self._check_database()
        active = self.session_manager.active_count if self.session_manager else 0

        overall = HealthStatus.HEALTHY
        if database not in ("healthy", "not_configured"):
            overall = HealthStatus.DEGRADED
        if all(v not in ("healthy",) for v in providers.values()) and providers:
            overall = HealthStatus.DEGRADED
        if database == "missing":
            overall = HealthStatus.UNHEALTHY

        return HealthResult(
            status=overall,
            version=self.version,
            uptime_seconds=get_metrics().uptime_seconds(),
            active_sessions=active,
            providers=providers,
            database=database,
            memory_usage_mb=self._memory_mb(),
            disk_space_gb=self._disk_free_gb(),
            checks={"timestamp": time.time()},
        )
