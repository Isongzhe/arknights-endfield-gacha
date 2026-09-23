"""Output helpers shared by experiments: tables (CSV + Markdown) and docs/results.md sections."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def _fmt(v) -> str:
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.4g}"
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, out_dir: Path, name: str) -> list[Path]:
    tables, _ = ensure_dirs(out_dir)
    csv = tables / f"{name}.csv"
    md = tables / f"{name}.md"
    df.to_csv(csv, index=False)
    md.write_text(to_markdown(df))
    return [csv, md]


def update_results_md(path: Path, section: str, lines: list[str]) -> None:
    """Replace (or append) the ``## section`` block of a Markdown file."""
    path = Path(path)
    text = path.read_text() if path.exists() else "# Results\n"
    parts = text.split("\n## ")
    head, blocks = parts[0], parts[1:]
    kept = [b for b in blocks if b.split("\n", 1)[0].strip() != section]
    kept.append(f"{section}\n\n" + "\n".join(lines) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n## ".join(b.rstrip("\n") + "\n" for b in kept)
    path.write_text(head.rstrip("\n") + "\n\n## " + body)
