"""Legacy text registry. Presence here does not approve a business assertion."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def runtime_templates():
    return json.loads((Path(__file__).resolve().parents[3] / "configs/runtime_answer_templates.json").read_text(encoding="utf-8"))


def legacy_template(template_id: str) -> str:
    return runtime_templates()["exceptions"][template_id]["text"]
