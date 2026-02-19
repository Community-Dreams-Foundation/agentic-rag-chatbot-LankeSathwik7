from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".log"}


@dataclass
class RawDocument:
    source: str
    source_path: str
    text: str
    lines: list[str]


def read_text_file(path: Path) -> RawDocument:
    text = path.read_text(encoding="utf-8")
    return RawDocument(
        source=path.name,
        source_path=path.as_posix(),
        text=text,
        lines=text.splitlines(),
    )


def discover_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(f)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    return files


def ingest_paths(paths: list[str]) -> list[RawDocument]:
    docs: list[RawDocument] = []
    for p in discover_files(paths):
        try:
            docs.append(read_text_file(p))
        except UnicodeDecodeError:
            # Skip binary/non-utf8 files instead of failing the entire batch.
            continue
    return docs
