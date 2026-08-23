from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from backend.app.bot.text_processing import normalize_text


DEFAULT_KNOWLEDGE = Path("knowledge/v3_1/scenarios.json")
DEFAULT_REGRESSION_MANIFEST = Path("tests/data/regression_corpora_manifest.json")
DEFAULT_OUTPUT = Path("tests/data/retrieval_v31_development_validation.json")
WORD_RE = re.compile(r"[а-яё]{6,}", re.IGNORECASE)


def _transpose_longest_word(text: str) -> str:
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return f"подскажите {text}"
    match = max(matches, key=lambda item: len(item.group(0)))
    index = match.start() + max(1, len(match.group(0)) // 2 - 1)
    chars = list(text)
    chars[index], chars[index + 1] = chars[index + 1], chars[index]
    return "".join(chars)


def _rotate_words(text: str) -> str:
    words = text.split()
    if len(words) < 4:
        return f"нужно узнать: {text}"
    pivot = len(words) // 2
    return " ".join([*words[pivot:], *words[:pivot]])


def _taxonomy_paraphrase(record: dict[str, Any], unit: dict[str, Any]) -> str:
    terms = [
        str(group["terms"][-1])
        for group in record.get("retrieval_taxonomy_terms", [])
        if group.get("terms")
    ]
    selected = list(dict.fromkeys(terms))[:2]
    discriminators = [str(item) for item in unit.get("discriminator_terms", []) if str(item).strip()]
    base = discriminators[0] if discriminators else unit["title"]
    return _rotate_words(base) + "; " + ", ".join(selected)


def _corpus_texts(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        return [
            str(json.loads(line).get("message") or json.loads(line).get("text") or "")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") or payload.get("queries") or payload.get("single_turn_cases") or []
    return [str(item.get("text") or item.get("message") or "") for item in cases]


def build(knowledge_path: Path, regression_manifest_path: Path) -> dict[str, Any]:
    knowledge_raw = knowledge_path.read_bytes()
    knowledge = json.loads(knowledge_raw.decode("utf-8"))
    manifest_raw = regression_manifest_path.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    frozen_texts: set[str] = set()
    for corpus in manifest["corpora"]:
        frozen_texts.update(
            normalize_text(text)
            for text in _corpus_texts(Path(corpus["path"]))
            if normalize_text(text)
        )

    records = {item["scenario_id"]: item for item in knowledge["records"]}
    cases: list[dict[str, Any]] = []
    for unit in knowledge["atomic_units"]:
        scenario_id = unit["canonical_scenario_id"]
        record = records[scenario_id]
        role = "guest" if "guest" in record["roles"] else record["roles"][0]
        variants = {
            "taxonomy_paraphrase": _taxonomy_paraphrase(record, unit),
            "transposed_letters": _transpose_longest_word(unit["title"]),
            "word_order": _rotate_words(unit["title"]),
        }
        for variant, text in variants.items():
            fingerprint = hashlib.sha256(f"{unit['unit_id']}\0{variant}".encode("utf-8")).digest()
            split = "validation" if fingerprint[0] % 5 == 0 else "development"
            normalized = normalize_text(text)
            cases.append({
                "id": f"{unit['unit_id']}::{variant}",
                "split": split,
                "variant": variant,
                "text": text,
                "role": role,
                "expected_scenario_id": scenario_id,
                "source_unit_id": unit["unit_id"],
                "exact_frozen_overlap": normalized in frozen_texts,
            })

    overlaps = [item["id"] for item in cases if item["exact_frozen_overlap"]]
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "version": "2026.08.23.1",
        "purpose": "Development/validation candidate-retrieval benchmark generated from normalized atomic taxonomy, not user regression phrases.",
        "knowledge_sha256": hashlib.sha256(knowledge_raw).hexdigest(),
        "regression_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "case_count": len(cases),
        "development_count": sum(item["split"] == "development" for item in cases),
        "validation_count": sum(item["split"] == "validation" for item in cases),
        "exact_frozen_overlap_count": len(overlaps),
        "exact_frozen_overlap_case_ids": overlaps,
        "cases_sha256": hashlib.sha256(canonical).hexdigest(),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build KB v3.1 retrieval development/validation cases.")
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--regression-manifest", type=Path, default=DEFAULT_REGRESSION_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.knowledge, args.regression_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("case_count", "development_count", "validation_count", "exact_frozen_overlap_count", "cases_sha256")}, ensure_ascii=False, indent=2))
    return 0 if payload["exact_frozen_overlap_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
