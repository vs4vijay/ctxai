"""
Hook registry for ctxai_core plugins.

Lets plugins register lightweight callbacks against named events without
implementing the full PluginInterface. Useful when you only want one
hook.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class HookType(str, Enum):
    AGENT_INIT = "agent_init"
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"


@dataclass
class Hook:
    name: str
    callback: Callable[..., Any]
    priority: int = 100  # lower runs first

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.callback(*args, **kwargs)


class HookRegistry:
    """Holds callbacks for each HookType in priority order."""

    def __init__(self) -> None:
        self._hooks: dict[HookType, list[Hook]] = defaultdict(list)

    def register(
        self,
        hook_type: HookType | str,
        callback: Callable[..., Any],
        name: str | None = None,
        priority: int = 100,
    ) -> Hook:
        if isinstance(hook_type, str):
            hook_type = HookType(hook_type)
        hook = Hook(name=name or callback.__name__, callback=callback, priority=priority)
        self._hooks[hook_type].append(hook)
        self._hooks[hook_type].sort(key=lambda h: h.priority)
        return hook

    def unregister(self, hook_type: HookType, name: str) -> bool:
        before = len(self._hooks.get(hook_type, []))
        self._hooks[hook_type] = [h for h in self._hooks.get(hook_type, []) if h.name != name]
        return len(self._hooks[hook_type]) < before

    def trigger(self, hook_type: HookType | str, *args: Any, **kwargs: Any) -> list[Any]:
        if isinstance(hook_type, str):
            hook_type = HookType(hook_type)
        results: list[Any] = []
        for hook in self._hooks.get(hook_type, []):
            try:
                results.append(hook(*args, **kwargs))
            except Exception as exc:
                # Hooks must not crash the agent.
                results.append({"hook_error": str(exc), "name": hook.name})
        return results

    def pipeline(self, hook_type: HookType | str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Apply hooks as a transformation pipeline: each hook receives the
        output of the previous one and returns a new value.
        """
        if isinstance(hook_type, str):
            hook_type = HookType(hook_type)
        for hook in self._hooks.get(hook_type, []):
            try:
                value = hook(value, *args, **kwargs)
            except Exception:
                # Pipeline must continue with the previous value on failure.
                continue
        return value

    def clear(self, hook_type: HookType | None = None) -> None:
        if hook_type is None:
            self._hooks.clear()
        else:
            self._hooks.pop(hook_type, None)

    def list_hooks(self, hook_type: HookType | None = None) -> dict[HookType, list[str]]:
        if hook_type is not None:
            return {hook_type: [h.name for h in self._hooks.get(hook_type, [])]}
        return {ht: [h.name for h in hooks] for ht, hooks in self._hooks.items()}
