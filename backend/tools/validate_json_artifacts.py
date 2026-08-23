from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOTS = (Path("configs"), Path("knowledge"), Path("reports"), Path("tests/data"))


def validate_json_artifacts(roots: tuple[Path, ...] = DEFAULT_ROOTS) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    paths = sorted({path for root in roots if root.exists() for path in root.rglob("*.json")})
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append({"path": path.as_posix(), "error": str(exc)})
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate committed JSON configuration, knowledge, test and report artifacts.")
    parser.add_argument("roots", nargs="*", type=Path, default=list(DEFAULT_ROOTS))
    args = parser.parse_args()
    failures = validate_json_artifacts(tuple(args.roots))
    print(json.dumps({"passed": not failures, "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
