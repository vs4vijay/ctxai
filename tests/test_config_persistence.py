from pathlib import Path

import pytest

from ctxai.config import ConfigManager
from ctxai.utils import get_global_ctxai_home


@pytest.fixture
def global_home(tmp_path, monkeypatch):
    """Isolate the global config layer in a temp dir (avoids the real ~/.ctxai)."""
    home = tmp_path / "global-home"
    home.mkdir()
    monkeypatch.setenv("CTXAI_HOME", str(home))
    return home


def test_toml_configuration_round_trip(tmp_path, global_home):
    manager = ConfigManager(tmp_path)
    config = manager.load()
    config.index_name = "durable-index"
    config.index_chunks_count = 12
    manager.save(config)

    saved = manager.get_config_path().read_text(encoding="utf-8")
    assert "durable-index" in saved
    assert "None" not in saved
    reloaded = ConfigManager(tmp_path).load()
    assert reloaded.index_name == "durable-index"
    assert reloaded.index_chunks_count == 12


def test_global_config_supplies_defaults(global_home, tmp_path):
    (global_home / "config.toml").write_text(
        'version = "2.0"\n'
        'default_provider = "openrouter"\n'
        "[providers.openrouter]\n"
        "enabled = true\n"
        'model = "global-model"\n'
    )

    config = ConfigManager(tmp_path).load()

    assert config.default_provider == "openrouter"
    assert config.providers["openrouter"].model == "global-model"
    # No project file is materialized when a global layer exists
    assert not (tmp_path / ".ctxai" / "config.toml").exists()


def test_project_overrides_global(global_home, tmp_path):
    (global_home / "config.toml").write_text(
        'version = "2.0"\n'
        'default_provider = "openrouter"\n'
        "[providers.openrouter]\n"
        "enabled = true\n"
        'model = "global-model"\n'
        "[providers.custom]\n"
        "enabled = true\n"
        'model = "custom-model"\n'
    )
    project_dir = tmp_path / "proj"
    (project_dir / ".ctxai").mkdir(parents=True)
    (project_dir / ".ctxai" / "config.toml").write_text(
        'default_provider = "custom"\n[providers.custom]\napi_key = "sk-project"\n'
    )

    config = ConfigManager(project_dir).load()

    # Project wins where it sets values, global fills the rest
    assert config.default_provider == "custom"
    assert config.providers["custom"].api_key == "sk-project"
    assert config.providers["custom"].model == "custom-model"
    assert config.providers["openrouter"].model == "global-model"


def test_project_save_writes_only_overrides(global_home, tmp_path):
    (global_home / "config.toml").write_text(
        'version = "2.0"\n'
        'default_provider = "openrouter"\n'
        "[providers.openrouter]\n"
        "enabled = true\n"
        'model = "global-model"\n'
        "temperature = 0.7\n"
        "[embedding]\n"
        'provider = "local"\n'
        "batch_size = 100\n"
    )

    manager = ConfigManager(tmp_path)
    config = manager.load()
    config.default_provider = "custom"
    manager.save(config)

    saved = (tmp_path / ".ctxai" / "config.toml").read_text()
    assert 'default_provider = "custom"' in saved
    # Global values are not frozen into the project layer
    assert "global-model" not in saved
    assert "batch_size" not in saved
    # No empty tables for fields the project layer does not override
    assert "[providers." not in saved

    # The merge still sees global values after reload
    reloaded = ConfigManager(tmp_path).load()
    assert reloaded.default_provider == "custom"
    assert reloaded.providers["openrouter"].model == "global-model"
    assert reloaded.embedding.batch_size == 100


def test_global_mode_writes_global_file(global_home, tmp_path):
    manager = ConfigManager(tmp_path, use_global=True)
    assert manager.config_path == global_home / "config.toml"

    config = manager.load()
    config.default_provider = "custom"
    manager.save(config)

    saved = (global_home / "config.toml").read_text()
    assert 'default_provider = "custom"' in saved

    # A project-merged load picks up the global default
    assert ConfigManager(tmp_path).load().default_provider == "custom"


def test_global_home_defaults_to_user_home(monkeypatch, tmp_path):
    monkeypatch.delenv("CTXAI_HOME", raising=False)
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    assert get_global_ctxai_home() == fake_home / ".ctxai"


def test_global_home_respects_ctxai_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CTXAI_HOME", str(tmp_path / "custom"))

    assert get_global_ctxai_home() == (tmp_path / "custom").resolve()
