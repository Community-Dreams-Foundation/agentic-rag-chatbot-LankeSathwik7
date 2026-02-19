from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    section: str
    start_line: int
    end_line: int
    text: str

    @property
    def locator(self) -> str:
        return f"{self.section} | lines {self.start_line}-{self.end_line} | {self.chunk_id}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


@dataclass
class Citation:
    source: str
    locator: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class QAResult:
    question: str
    answer: str
    citations: list[Citation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass
class MemoryWrite:
    target: str
    summary: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

