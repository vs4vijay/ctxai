# ruff: noqa: E501
"""HTML code map generator with self-contained CSS/JS.

The template embeds a complete stylesheet inline; CSS rules are
intentionally one-per-line and may exceed the project line-length budget.
"""

from __future__ import annotations

import html as html_module
import json
from pathlib import Path
from typing import Any

from ctxai.export.config import ExportConfig
from ctxai.export.formatters import render_tree
from ctxai.export.repo_to_text import RepoTextExporter

_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
    <meta charset="UTF-8" />
    <title>Code Map – {name}</title>
    <style>
        :root[data-theme="dark"]  {{ --bg:#0d1117; --fg:#c9d1d9; --accent:#58a6ff; --muted:#8b949e; --panel:#161b22; --border:#30363d; }}
        :root[data-theme="light"] {{ --bg:#ffffff; --fg:#24292f; --accent:#0969da; --muted:#57606a; --panel:#f6f8fa; --border:#d0d7de; }}
        * {{ box-sizing: border-box; }}
        body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; background:var(--bg); color:var(--fg); height:100vh; display:flex; }}
        nav.sidebar {{ width:320px; min-width:220px; background:var(--panel); border-right:1px solid var(--border); display:flex; flex-direction:column; overflow:hidden; }}
        nav.sidebar header {{ padding:12px 16px; border-bottom:1px solid var(--border); }}
        nav.sidebar header h1 {{ font-size:14px; margin:0; }}
        nav.sidebar .stats {{ padding:12px 16px; border-bottom:1px solid var(--border); font-size:12px; color:var(--muted); }}
        nav.sidebar .search {{ padding:8px 12px; border-bottom:1px solid var(--border); }}
        nav.sidebar .search input {{ width:100%; padding:6px 8px; background:var(--bg); color:var(--fg); border:1px solid var(--border); border-radius:4px; }}
        nav.sidebar .filelist {{ flex:1; overflow:auto; padding:6px 0; font-size:13px; }}
        nav.sidebar .filelist button {{ display:block; width:100%; text-align:left; background:none; border:0; color:var(--fg); padding:4px 16px; cursor:pointer; font-family:inherit; }}
        nav.sidebar .filelist button:hover {{ background:rgba(120,120,120,0.1); }}
        nav.sidebar .filelist button.active {{ color:var(--accent); font-weight:600; }}
        main {{ flex:1; display:flex; flex-direction:column; overflow:hidden; }}
        main header {{ padding:8px 16px; border-bottom:1px solid var(--border); display:flex; gap:12px; align-items:center; }}
        main header .breadcrumb {{ flex:1; font-size:13px; color:var(--muted); font-family:monospace; }}
        main header button {{ background:none; border:1px solid var(--border); color:var(--fg); padding:4px 10px; border-radius:4px; cursor:pointer; font-size:12px; }}
        main .viewer {{ flex:1; overflow:auto; padding:16px; }}
        main pre {{ margin:0; font-family:'JetBrains Mono','Fira Code',Menlo,Consolas,monospace; font-size:13px; line-height:1.5; }}
        .languages {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }}
        .lang-badge {{ background:var(--bg); color:var(--accent); padding:2px 6px; border-radius:10px; font-size:11px; border:1px solid var(--border); }}
        .tree {{ white-space:pre; font-family:monospace; font-size:12px; padding:8px 12px; color:var(--muted); border-bottom:1px solid var(--border); max-height:200px; overflow:auto; }}
    </style>
</head>
<body>
<nav class="sidebar">
    <header>
        <h1>{name}</h1>
        <div class="languages">{language_badges}</div>
    </header>
    <div class="stats">
        Files: {total_files}<br/>
        Lines: {total_lines}<br/>
        Size: {total_size}<br/>
        Generated: {generated_at}
    </div>
    <div class="search">
        <input id="filter" type="search" placeholder="Filter files…" />
    </div>
    <div class="tree">{tree}</div>
    <div class="filelist" id="filelist">{file_buttons}</div>
</nav>
<main>
    <header>
        <div class="breadcrumb" id="breadcrumb">Select a file</div>
        <button id="copy-btn">Copy</button>
        <button id="theme-btn">Toggle theme</button>
    </header>
    <div class="viewer">
        <pre id="viewer">// Select a file from the sidebar.</pre>
    </div>
</main>
<script>
const files = {files_json};
const viewer = document.getElementById('viewer');
const breadcrumb = document.getElementById('breadcrumb');
const filelist = document.getElementById('filelist');
const filter = document.getElementById('filter');

function selectFile(path) {{
    const item = files[path];
    if (!item) return;
    viewer.textContent = item.content;
    breadcrumb.textContent = path;
    document.querySelectorAll('.filelist button').forEach(b => b.classList.toggle('active', b.dataset.path === path));
}}

filelist.addEventListener('click', e => {{
    if (e.target.matches('button')) selectFile(e.target.dataset.path);
}});

filter.addEventListener('input', () => {{
    const q = filter.value.toLowerCase();
    document.querySelectorAll('.filelist button').forEach(b => {{
        b.style.display = b.dataset.path.toLowerCase().includes(q) ? 'block' : 'none';
    }});
}});

document.getElementById('copy-btn').addEventListener('click', () => {{
    navigator.clipboard.writeText(viewer.textContent);
}});

document.getElementById('theme-btn').addEventListener('click', () => {{
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
}});
</script>
</body>
</html>
"""


class HtmlCodemap:
    """Render an interactive single-file HTML code map."""

    def __init__(self, root: Path | str, config: ExportConfig | None = None, theme: str = "dark"):
        cfg = config or ExportConfig()
        cfg.output_format = "html"
        self.exporter = RepoTextExporter(root, cfg)
        self.theme = theme if theme in ("dark", "light") else "dark"

    def render(self) -> str:
        summary = self.exporter.collect()
        files_json = json.dumps(
            {
                f.relative: {
                    "language": f.language,
                    "lines": f.lines,
                    "size": f.size,
                    "content": f.content,
                }
                for f in summary.files
            }
        )

        file_buttons = "\n".join(
            f'<button data-path="{html_module.escape(f.relative)}">'
            f"{html_module.escape(f.relative)}</button>"
            for f in summary.files
        )
        language_badges = "".join(
            f'<span class="lang-badge">{html_module.escape(lang)} '
            f'({stats["files"]})</span>'
            for lang, stats in summary.languages.items()
        )
        tree = html_module.escape(render_tree(self.exporter.root, summary.files))

        return _TEMPLATE.format(
            theme=self.theme,
            name=html_module.escape(summary.name),
            total_files=len(summary.files),
            total_lines=summary.total_lines,
            total_size=summary.total_size,
            generated_at=summary.generated_at,
            language_badges=language_badges,
            tree=tree,
            file_buttons=file_buttons,
            files_json=files_json,
        )

    def export(self, output: Path | str) -> Path:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render(), encoding="utf-8")
        return out
