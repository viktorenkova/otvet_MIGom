"""Execute published uncertainty policies; never supply a missing business fact."""
from functools import lru_cache
import json
import re
from pathlib import Path

from backend.app.config import get_settings


@lru_cache(maxsize=1)
def _policies():
    payload = json.loads((get_settings().knowledge_root / "v3_1/scenarios.json").read_text(encoding="utf-8"))
    config = Path(__file__).resolve().parents[3] / "configs/knowledge_gap_policy.json"
    rules = json.loads(config.read_text(encoding="utf-8"))["rules"]
    return {g["gap_id"]: g for g in payload["knowledge_gaps"]}, rules


def matching_gap(message: str, scenario_id: str):
    gaps, rules = _policies()
    for rule in rules:
        gap = gaps[rule["gap_id"]]
        if gap["scenario_id"] != scenario_id:
            continue
        if rule.get("unless") and re.search(rule["unless"], message, re.I):
            continue
        if all(re.search(pattern, message, re.I) for pattern in rule["all"]):
            return gap
    return None
