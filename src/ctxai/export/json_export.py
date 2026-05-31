"""JSON repository export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ctxai.export.config import ExportConfig
from ctxai.export.repo_to_text import RepoTextExporter


class JsonExporter:
    """Wrap RepoTextExporter to emit a structured JSON document."""

    def __init__(self, root: Path | str, config: ExportConfig | None = None):
        cfg = config or ExportConfig()
        cfg.output_format = "json"
        self.exporter = RepoTextExporter(root, cfg)

    def to_dict(self) -> dict[str, Any]:
        summary = self.exporter.collect()
        return {
            "repository": {
                "name": summary.name,
                "path": str(summary.root),
                "generated_at": summary.generated_at,
                "stats": {
                    "total_files": len(summary.files),
                    "total_lines": summary.total_lines,
                    "total_size_bytes": summary.total_size,
                    "languages": summary.languages,
                    "truncated": summary.truncated,
                },
            },
            "files": [
                {
                    "path": f.relative,
                    "language": f.language,
                    "lines": f.lines,
                    "size_bytes": f.size,
                    "last_modified": f.last_modified,
                    "content": f.content,
                }
                for f in summary.files
            ],
        }

    def export(self, output: Path | str, indent: int = 2) -> Path:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return out
