from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from backend.app.bot.scenario_engine import load_scenarios


DEFAULT_SEEDS = Path("tests/data/routing_v3_acceptance_seeds.json")
DEFAULT_OUTPUT = Path("tests/data/routing_v3_independent_acceptance.json")
WORD_RE = re.compile(r"[а-яёА-ЯЁ]{5,}")

MORPHOLOGY_REPLACEMENTS = {
    "автомобиля": "автомобилю",
    "автомобиль": "автомобили",
    "возврат": "возвращение",
    "возвратить": "возвращать",
    "заявление": "заявления",
    "депозита": "депозиту",
    "ставку": "ставки",
    "торгов": "торгах",
    "фильтр": "фильтры",
    "документы": "документов",
    "письмо": "письма",
    "площадке": "площадку",
    "изображения": "изображение",
    "платёж": "платежа",
    "тарифа": "тарифу",
}


def _longest_word_span(text: str) -> tuple[int, int]:
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return (0, len(text))
    match = max(matches, key=lambda item: len(item.group(0)))
    return match.span()


def delete_letter(text: str) -> str:
    start, end = _longest_word_span(text)
    index = start + max(1, (end - start) // 2)
    return text[:index] + text[index + 1 :]


def transpose_letters(text: str) -> str:
    start, end = _longest_word_span(text)
    index = start + max(1, (end - start) // 2 - 1)
    if index + 1 >= end:
        return text
    chars = list(text)
    chars[index], chars[index + 1] = chars[index + 1], chars[index]
    return "".join(chars)


def change_morphology(text: str) -> str:
    words = text.split()
    for index, word in enumerate(words):
        clean = word.casefold().strip(".,!?;:")
        replacement = MORPHOLOGY_REPLACEMENTS.get(clean)
        if replacement:
            words[index] = replacement
            return " ".join(words)
    return "по вопросу " + text


def change_word_order(text: str) -> str:
    words = text.split()
    if len(words) < 4:
        return " ".join(reversed(words))
    pivot = len(words) // 2
    return " ".join([*words[pivot:], *words[:pivot]])


TRANSFORMS: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("typo", delete_letter),
    ("transposed_letters", transpose_letters),
    ("morphology", change_morphology),
    ("word_order", change_word_order),
)


def build(source: Path) -> dict[str, Any]:
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    scenarios = {item.scenario_id: item for item in load_scenarios()}
    cases: list[dict[str, Any]] = []
    for seed in payload["seeds"]:
        expected_intents = sorted({
            scenarios[scenario_id].intent
            for scenario_id in seed["scenario_ids"]
            if scenario_id is not None
        })
        if any(scenario_id is None for scenario_id in seed["scenario_ids"]):
            expected_intents.extend(intent for intent in ("unknown",) if intent not in expected_intents)
        for class_name, transform in TRANSFORMS:
            text = transform(seed["text"])
            if text == seed["text"]:
                raise ValueError(f"Mutation {class_name} did not change seed {seed['id']}")
            cases.append({
                "id": f"mutation-{seed['id']}-{class_name}",
                "source": "routing_v3_acceptance_seeds",
                "group": seed["group"],
                "class": class_name,
                "text": text,
                "expected": {
                    "expected_scenario_ids": seed["scenario_ids"],
                    "expected_intents": expected_intents,
                    "allowed_resolutions": ["answered", "status", "escalated", "clarified"] if seed.get("allow_clarification") else ["answered", "status", "escalated"],
                    "expect_direct": not seed.get("allow_clarification", False),
                },
            })
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "version": "routing-v3-independent-mutations-1",
        "purpose": "Deterministically generated acceptance set; do not use generated phrases as routing rules.",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "case_count": len(cases),
        "cases_sha256": hashlib.sha256(canonical).hexdigest(),
        "global_forbidden_answer_fragments": ["для демо", "демо-доступа"],
        "release_gates": {
            "transport_ok_pct": 100.0,
            "route_hit_pct": 85.0,
            "answer_hit_pct": 80.0,
            "typo_answer_hit_pct": 80.0,
            "transposed_letters_answer_hit_pct": 75.0,
            "morphology_answer_hit_pct": 85.0,
            "no_response_answer_hit_pct": 90.0,
            "ambiguous_answer_hit_pct": 90.0,
            "dialogue_completion_pct": 0.0,
            "forbidden_content_ok_pct": 100.0,
            "confident_wrong_max": 2
        },
        "dialogues": [],
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic typo and phrasing variants for Routing v3 acceptance.")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": payload["case_count"], "sha256": payload["cases_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
