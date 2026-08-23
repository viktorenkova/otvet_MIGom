from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "tests" / "data" / "regression_corpora_manifest.json"
DEFAULT_LEAKAGE_BASELINE = ROOT / "tests" / "data" / "routing_content_leakage_baseline.json"
ROUTING_CODE = (
    ROOT / "backend" / "app" / "bot" / "routing_v3.py",
    ROOT / "backend" / "app" / "bot" / "scenario_engine.py",
    ROOT / "backend" / "app" / "bot" / "knowledge_search.py",
    ROOT / "backend" / "app" / "bot" / "intent_classifier.py",
)
ROUTING_CONFIGS = (
    ROOT / "configs" / "no_llm_matching_config.json",
    ROOT / "configs" / "intent_patterns.json",
    ROOT / "configs" / "synonym_groups.json",
    ROOT / "configs" / "typo_corrections.json",
)
ROUTING_KNOWLEDGE = (
    ROOT / "knowledge" / "v2" / "scenarios.json",
    ROOT / "knowledge" / "normalized" / "migtorg_knowledge_base.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[0-9a-zа-яё]+", value.casefold().replace("ё", "е")))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _dataset_case_count(path: Path) -> int:
    if path.suffix == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "case_count" in payload:
        return int(payload["case_count"])
    if "cases" in payload:
        return len(payload["cases"])
    if "groups" in payload:
        return sum(len(group.get("queries", [])) for group in payload["groups"])
    raise ValueError(f"Cannot determine case count for {path}")


def validate_locked_corpora(manifest_path: Path = DEFAULT_MANIFEST) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for item in manifest.get("corpora", []):
        relative = str(item["path"])
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
            continue
        actual_hash = _sha256(path)
        actual_count = _dataset_case_count(path)
        if actual_hash != item["sha256"]:
            errors.append(f"sha256:{relative}:{actual_hash}")
        if actual_count != int(item["case_count"]):
            errors.append(f"case_count:{relative}:{actual_count}")
    return errors


def _dataset_texts(path: Path) -> set[str]:
    texts: set[str] = set()
    if path.suffix == ".jsonl":
        payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "cases" in raw:
            payloads = raw["cases"]
        elif "groups" in raw:
            payloads = [query for group in raw["groups"] for query in group.get("queries", [])]
        else:
            payloads = []
    for item in payloads:
        for key in ("text", "message", "query", "q"):
            value = item.get(key)
            if isinstance(value, str):
                normalized = _normalize(value)
                if len(normalized.split()) >= 3:
                    texts.add(normalized)
                break
    return texts


def _routing_strings() -> set[str]:
    values: set[str] = set()
    for path in (*ROUTING_CONFIGS, *ROUTING_KNOWLEDGE):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.update(_normalize(item) for item in _strings(payload))
    for path in ROUTING_CODE:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        values.update(
            _normalize(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    return {value for value in values if len(value.split()) >= 3}


def leakage_snapshot(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    route_strings = _routing_strings()
    overlaps: list[dict[str, str]] = []
    for corpus in manifest.get("corpora", []):
        relative = str(corpus["path"])
        for text in sorted(_dataset_texts(ROOT / relative).intersection(route_strings)):
            overlaps.append(
                {
                    "corpus": relative,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
    canonical = json.dumps(overlaps, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "policy": "Baseline of pre-existing exact normalized overlaps. Any change requires explicit review; new audit phrases must not be copied into routing content.",
        "overlap_count": len(overlaps),
        "overlaps_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "overlaps": overlaps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate immutable regression corpora and routing leakage baseline.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--leakage-baseline", type=Path, default=DEFAULT_LEAKAGE_BASELINE)
    parser.add_argument("--write-leakage-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_locked_corpora(args.manifest)
    snapshot = leakage_snapshot(args.manifest)
    if args.write_leakage_baseline:
        args.leakage_baseline.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif not args.leakage_baseline.is_file():
        errors.append(f"missing:{args.leakage_baseline}")
    else:
        expected = json.loads(args.leakage_baseline.read_text(encoding="utf-8"))
        if snapshot != expected:
            errors.append(
                "routing_content_leakage_changed:"
                f"expected={expected.get('overlaps_sha256')}:actual={snapshot['overlaps_sha256']}"
            )
    print(json.dumps({"locked_corpora_errors": errors, "leakage": snapshot}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
