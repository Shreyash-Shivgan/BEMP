"""Policy version management and loader with strict candidate path protection."""

from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel

from strikeshield.errors import PolicyPathError


class PolicyVersion(BaseModel):
    version: str
    text: str
    path: Optional[Path] = None

    @classmethod
    def load(cls, path: Union[Path, str]) -> "PolicyVersion":
        """Loads and parses a policy file. Refuses to load from policies/candidates/ directory."""
        return cls.from_file(path)

    @classmethod
    def from_file(cls, path: Union[Path, str]) -> "PolicyVersion":
        """Loads and parses a policy file. Refuses to load from policies/candidates/ directory."""
        p = Path(path)
        # Check if path contains 'candidates' directory in its path parts
        resolved_parts = [part.lower() for part in p.resolve().parts]
        raw_parts = [part.lower() for part in p.parts]
        if "candidates" in resolved_parts or "candidates" in raw_parts:
            raise PolicyPathError(
                f"Cannot load candidate policies from candidates directory: {p}. Candidate artifacts are not gate-eligible.",
                context={"path": str(p)},
            )

        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Policy file {p} is empty")
        return cls(version=p.stem, text=text, path=p)
