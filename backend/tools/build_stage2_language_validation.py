from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from backend.app.bot.text_processing import normalize_text
from backend.tools.build_retrieval_v31_validation import _corpus_texts


DEFAULT_MANIFEST = Path("tests/data/regression_corpora_manifest.json")
DEFAULT_OUTPUT = Path("tests/data/stage2_language_validation.json")

_RU_TO_EN_LAYOUT = str.maketrans(
    "йцукенгшщзхъфывапролджэячсмитьбю",
    "qwertyuiop[]asdfghjkl;'zxcvbnm,.",
)
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

_SEEDS = (
    ("bid.not_visible", "ставка не отображается", "бид не видно"),
    ("bid.place", "как сделать ставку", "как кинуть бид"),
    ("pickup.receive_lot", "документы для получения автомобиля", "какие доки нужны чтобы забрать тачку"),
    ("commission.explained", "как устроена комиссия площадки", "что за комса на площадке"),
    ("account.blocked", "аккаунт заблокирован", "акк заблокирован"),
    ("account.registration", "как пройти регистрацию аккаунта", "как регаться на площадке"),
    ("payment.not_visible", "платеж списался но не отображается", "оплата ушла но в личке пусто"),
    ("seller.publish_lot", "как выставить автомобиль на продажу", "как выставить тачку на продажу"),
    ("contract.receive", "где получить документы по договору", "где доки по договору"),
    ("inspection.arrange", "как организовать осмотр автомобиля", "как посмотреть тачку вживую"),
)


def _translit(text: str) -> str:
    return "".join(_TRANSLIT.get(char, char) for char in text.lower())


def build(manifest_path: Path) -> dict:
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    frozen = {
        normalize_text(text)
        for corpus in manifest["corpora"]
        for text in _corpus_texts(Path(corpus["path"]))
        if normalize_text(text)
    }
    cases = []
    for scenario_id, canonical, slang in _SEEDS:
        variants = {
            "keyboard_layout": canonical.translate(_RU_TO_EN_LAYOUT),
            "transliteration": _translit(canonical),
            "slang": slang,
        }
        for variant, text in variants.items():
            cases.append({
                "id": f"{scenario_id}::{variant}",
                "class": variant,
                "text": text,
                "expected_scenario_ids": [scenario_id],
                "exact_frozen_overlap": normalize_text(text) in frozen,
            })
    canonical_rows = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    overlaps = [case["id"] for case in cases if case["exact_frozen_overlap"]]
    return {
        "schema_version": 1,
        "version": "2026.08.23.1",
        "purpose": "Independent language-slice validation for layout, Russian transliteration and slang; seeds are not copied from frozen regression queries.",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "case_count": len(cases),
        "exact_frozen_overlap_count": len(overlaps),
        "exact_frozen_overlap_case_ids": overlaps,
        "cases_sha256": hashlib.sha256(canonical_rows).hexdigest(),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("case_count", "exact_frozen_overlap_count", "cases_sha256")}, indent=2))
    return 0 if not payload["exact_frozen_overlap_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
