"""Tests for ctxai.export."""

import json
from pathlib import Path

import pytest

from ctxai.export.config import ExportConfig
from ctxai.export.formatters import detect_language, render_tree
from ctxai.export.html_codemap import HtmlCodemap
from ctxai.export.json_export import JsonExporter
from ctxai.export.repo_to_text import RepoTextExporter


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    return 1\n")
    (tmp_path / "src" / "util.py").write_text("X = 42\n")
    (tmp_path / "README.md").write_text("# Demo\n")
    (tmp_path / ".gitignore").write_text("ignored/\n*.log\n")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "skip.py").write_text("# skip me\n")
    (tmp_path / "debug.log").write_text("noise\n")
    (tmp_path / "binary.bin").write_bytes(b"\x00" * 100)
    return tmp_path


def test_export_config_validate_rejects_bad_format():
    cfg = ExportConfig(output_format="bogus")
    with pytest.raises(ValueError):
        cfg.validate()


def test_export_config_include_exclude():
    cfg = ExportConfig(include_patterns=["*.py"], exclude_patterns=["build/*"])
    assert cfg.matches_include(Path("a.py"))
    assert not cfg.matches_include(Path("a.txt"))
    assert cfg.matches_exclude(Path("build/x.py"))


def test_detect_language():
    assert detect_language(Path("a.py")) == "python"
    assert detect_language(Path("a.unknownext")) == "text"


def test_repo_to_text_markdown(sample_repo: Path):
    exporter = RepoTextExporter(sample_repo, ExportConfig(output_format="markdown"))
    summary = exporter.collect()
    paths = [f.relative for f in summary.files]
    assert "src/main.py" in paths
    assert "src/util.py" in paths
    # .gitignore excludes ignored/ and *.log
    assert all("ignored/" not in p for p in paths)
    assert all("debug.log" not in p for p in paths)
    # binary file excluded
    assert all(not p.endswith(".bin") for p in paths)

    rendered = exporter.render(summary)
    assert "# Repository Export" in rendered
    assert "main.py" in rendered


def test_repo_to_text_respects_max_files(sample_repo: Path):
    exporter = RepoTextExporter(sample_repo, ExportConfig(max_files=1, output_format="markdown"))
    summary = exporter.collect()
    assert len(summary.files) == 1
    assert summary.truncated is True


def test_repo_to_text_writes_file(sample_repo: Path, tmp_path: Path):
    out = tmp_path / "out.md"
    exporter = RepoTextExporter(sample_repo, ExportConfig(output_format="markdown"))
    written = exporter.export(out)
    assert written.exists()
    assert "main.py" in written.read_text()


def test_json_export_schema(sample_repo: Path, tmp_path: Path):
    exporter = JsonExporter(sample_repo)
    payload = exporter.to_dict()
    assert payload["repository"]["name"] == sample_repo.name
    assert "total_files" in payload["repository"]["stats"]
    assert isinstance(payload["files"], list)
    out = tmp_path / "out.json"
    exporter.export(out)
    parsed = json.loads(out.read_text())
    assert parsed["repository"]["name"] == sample_repo.name


def test_html_codemap_renders(sample_repo: Path, tmp_path: Path):
    codemap = HtmlCodemap(sample_repo)
    html = codemap.render()
    assert "<html" in html.lower()
    assert "main.py" in html
    assert "filelist" in html

    out = tmp_path / "out.html"
    codemap.export(out)
    assert out.exists()


def test_render_tree_handles_nested():
    from ctxai.export.formatters import FileEntry

    files = [
        FileEntry(Path("a.py"), "a.py", "", 0, 0, "python", "x"),
        FileEntry(Path("src/b.py"), "src/b.py", "", 0, 0, "python", "x"),
        FileEntry(Path("src/util/c.py"), "src/util/c.py", "", 0, 0, "python", "x"),
    ]
    tree = render_tree(Path("root"), files)
    assert "root/" in tree
    assert "src" in tree
    assert "a.py" in tree


def test_xml_export(sample_repo: Path):
    exporter = RepoTextExporter(sample_repo, ExportConfig(output_format="xml"))
    summary = exporter.collect()
    text = exporter.render(summary)
    assert text.startswith("<?xml")
    assert "<repository>" in text


def test_text_export_format(sample_repo: Path):
    exporter = RepoTextExporter(sample_repo, ExportConfig(output_format="text"))
    summary = exporter.collect()
    out = exporter.render(summary)
    assert "Repository:" in out
    assert "File 1/" in out
