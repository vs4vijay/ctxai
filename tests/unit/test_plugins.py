"""Tests for ctxai_core plugin system."""

from pathlib import Path

import pytest

from ctxai_core.plugins import (
    HookRegistry,
    HookType,
    PluginContext,
    PluginInterface,
    PluginManager,
    PluginMetadata,
)


class CountingPlugin(PluginInterface):
    metadata = PluginMetadata(name="counter", version="1.0", description="counts hooks")

    def __init__(self):
        self.calls = {"agent_init": 0, "message_start": 0, "tool_call": 0}

    def on_agent_init(self, context):
        self.calls["agent_init"] += 1

    def on_message_start(self, message, context):
        self.calls["message_start"] += 1
        return message + " [intercepted]"

    def on_tool_call(self, tool_name, args, context):
        self.calls["tool_call"] += 1
        return args


def test_plugin_register_and_lookup():
    mgr = PluginManager()
    p = CountingPlugin()
    mgr.register(p)
    assert "counter" in mgr.list_plugins()
    assert mgr.get("counter") is p


def test_plugin_dispatch_calls_method():
    mgr = PluginManager()
    p = CountingPlugin()
    mgr.register(p)
    ctx = PluginContext()
    mgr.dispatch("agent_init", ctx)
    assert p.calls["agent_init"] == 1


def test_plugin_pipeline_applies_transformation():
    mgr = PluginManager()
    mgr.register(CountingPlugin())
    ctx = PluginContext()
    out = mgr.pipeline("message_start", "hello", ctx)
    assert out == "hello [intercepted]"


def test_plugin_unregister():
    mgr = PluginManager()
    mgr.register(CountingPlugin())
    assert mgr.unregister("counter") is True
    assert mgr.unregister("counter") is False


def test_hook_registry_priority():
    reg = HookRegistry()
    order = []
    reg.register(HookType.MESSAGE_START, lambda m, c=None: order.append("a") or m, name="a", priority=10)
    reg.register(HookType.MESSAGE_START, lambda m, c=None: order.append("b") or m, name="b", priority=1)
    reg.pipeline(HookType.MESSAGE_START, "msg", PluginContext())
    assert order == ["b", "a"]


def test_hook_registry_unregister():
    reg = HookRegistry()
    reg.register(HookType.AGENT_INIT, lambda c: None, name="x")
    assert reg.unregister(HookType.AGENT_INIT, "x") is True
    assert reg.unregister(HookType.AGENT_INIT, "x") is False


def test_directory_discovery_loads_file(tmp_path: Path):
    plugin_file = tmp_path / "demo_plugin.py"
    plugin_file.write_text(
        "from ctxai_core.plugins.base import PluginInterface, PluginMetadata\n"
        "class _P(PluginInterface):\n"
        "    metadata = PluginMetadata(name='demo')\n"
        "PLUGIN = _P()\n"
    )
    mgr = PluginManager()
    loaded = mgr.discover_directory(tmp_path)
    assert loaded == 1
    assert "demo" in mgr.list_plugins()


def test_directory_discovery_skips_invalid(tmp_path: Path):
    (tmp_path / "broken.py").write_text("raise RuntimeError('bad')\n")
    (tmp_path / "noplugin.py").write_text("x = 1\n")
    mgr = PluginManager()
    loaded = mgr.discover_directory(tmp_path)
    assert loaded == 0


def test_dispatch_catches_plugin_errors():
    class Boom(PluginInterface):
        metadata = PluginMetadata(name="boom")

        def on_agent_init(self, ctx):
            raise RuntimeError("fail")

    mgr = PluginManager()
    mgr.register(Boom())
    results = mgr.dispatch("agent_init", PluginContext())
    assert any(isinstance(r, dict) and "plugin_error" in r for r in results)
