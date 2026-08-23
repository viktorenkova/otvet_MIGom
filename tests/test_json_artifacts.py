from __future__ import annotations

from pathlib import Path

from backend.tools.validate_json_artifacts import validate_json_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_repository_json_artifacts_are_valid() -> None:
    roots = tuple(ROOT / name for name in ("configs", "knowledge", "reports", "tests/data"))
    assert validate_json_artifacts(roots) == []
