"""Plugin system for extending ctxai_core."""

from ctxai_core.plugins.base import (
    Plugin,
    PluginContext,
    PluginInterface,
    PluginMetadata,
)
from ctxai_core.plugins.hooks import Hook, HookRegistry, HookType
from ctxai_core.plugins.loader import PluginManager, get_plugin_manager

__all__ = [
    "Hook",
    "HookRegistry",
    "HookType",
    "Plugin",
    "PluginContext",
    "PluginInterface",
    "PluginManager",
    "PluginMetadata",
    "get_plugin_manager",
]
