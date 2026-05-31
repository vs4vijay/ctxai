"""
Plugin discovery and loading.

Plugins can come from three sources:
1. Programmatic registration via PluginManager.register().
2. Python entry points under the `ctxai.plugins` group.
3. Single-file Python modules under `~/.ctxai/plugins/*.py` that expose
   a top-level `PLUGIN` instance or `get_plugin()` callable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from ctxai_core.plugins.base import PluginContext, PluginInterface
from ctxai_core.plugins.hooks import HookRegistry, HookType


class PluginManager:
    """Owns the set of active plugins and dispatches lifecycle hooks."""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginInterface] = {}
        self.hooks = HookRegistry()

    # ----- Registration -----

    def register(self, plugin: PluginInterface) -> None:
        name = plugin.metadata.name
        if name in self._plugins:
            self.unregister(name)
        self._plugins[name] = plugin
        plugin.on_register(self)

    def unregister(self, name: str) -> bool:
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            return False
        try:
            plugin.on_unregister(self)
        except Exception:
            pass
        return True

    def list_plugins(self) -> list[str]:
        return list(self._plugins.keys())

    def get(self, name: str) -> PluginInterface | None:
        return self._plugins.get(name)

    # ----- Discovery -----

    def discover_entry_points(self, group: str = "ctxai.plugins") -> int:
        """Load plugins from installed packages exposing entry points."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            return 0
        count = 0
        try:
            eps = entry_points(group=group)
        except TypeError:
            eps = entry_points().get(group, [])  # py<3.10 fallback
        for ep in eps:
            try:
                obj = ep.load()
                plugin = obj() if isinstance(obj, type) else obj
                if isinstance(plugin, PluginInterface):
                    self.register(plugin)
                    count += 1
            except Exception:
                continue
        return count

    def discover_directory(self, plugins_dir: Path | str | None = None) -> int:
        """
        Load plugins from a directory of single-file Python modules.

        Each module should expose either:
        - PLUGIN: a PluginInterface instance, OR
        - get_plugin(): a callable returning a PluginInterface
        """
        if plugins_dir is None:
            plugins_dir = Path.home() / ".ctxai" / "plugins"
        else:
            plugins_dir = Path(plugins_dir)
        if not plugins_dir.exists():
            return 0

        count = 0
        for path in sorted(plugins_dir.glob("*.py")):
            try:
                plugin = _load_plugin_file(path)
                if plugin is not None:
                    self.register(plugin)
                    count += 1
            except Exception:
                continue
        return count

    # ----- Dispatch -----

    def dispatch(self, hook_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call the named hook on every plugin and return all results."""
        method_name = f"on_{hook_name}" if not hook_name.startswith("on_") else hook_name
        results: list[Any] = []
        for plugin in self._plugins.values():
            method = getattr(plugin, method_name, None)
            if callable(method):
                try:
                    results.append(method(*args, **kwargs))
                except Exception as exc:
                    results.append({"plugin_error": str(exc), "plugin": plugin.metadata.name})
        results.extend(self.hooks.trigger(_hook_from_name(hook_name), *args, **kwargs))
        return results

    def pipeline(self, hook_name: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Run plugin hooks as a transformation pipeline on `value`."""
        method_name = f"on_{hook_name}" if not hook_name.startswith("on_") else hook_name
        for plugin in self._plugins.values():
            method = getattr(plugin, method_name, None)
            if callable(method):
                try:
                    value = method(value, *args, **kwargs)
                except Exception:
                    continue
        try:
            value = self.hooks.pipeline(_hook_from_name(hook_name), value, *args, **kwargs)
        except ValueError:
            pass
        return value


def _hook_from_name(name: str) -> HookType:
    short = name[3:] if name.startswith("on_") else name
    return HookType(short)


def _load_plugin_file(path: Path) -> PluginInterface | None:
    """Load a single .py file as an isolated module and pull the plugin out."""
    module_name = f"_ctxai_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    candidate = getattr(module, "PLUGIN", None)
    if candidate is None and hasattr(module, "get_plugin"):
        candidate = module.get_plugin()
    if candidate is None:
        return None
    if not isinstance(candidate, PluginInterface):
        return None
    return candidate


_default_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Process-wide singleton plugin manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PluginManager()
    return _default_manager
