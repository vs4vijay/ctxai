from ctxai.config import ConfigManager


def test_toml_configuration_round_trip(tmp_path):
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
