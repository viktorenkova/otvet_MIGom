"""Extractive answer plans: exact published fragments with scope and provenance.

Profiles are an experimental mechanism, not newly approved business knowledge.
No lexical-overlap verifier is allowed to authorize edits to these fragments.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from backend.app.bot.answer_contracts import get_answer_contract
from backend.app.bot.scenario_engine import get_scenario
from backend.app.bot.scenario_policy import scenario_allowed
from backend.app.config import get_settings


@dataclass(frozen=True)
class AnswerFragment:
    id: str
    text: str
    source_path: str
    source_pointer: str
    source_sha256: str
    attribution: str
    source_version: str
    primary_evidence_verified: bool = False


@dataclass(frozen=True)
class AnswerPlan:
    scenario_id: str
    profile: str
    fragments: tuple[AnswerFragment, ...]
    required_fact_ids: tuple[str, ...]
    reason: str
    documents: str = "keep"

    @property
    def text(self):
        return " ".join(fragment.text for fragment in self.fragments)


@lru_cache(maxsize=1)
def assembly_policy():
    return json.loads((Path(__file__).resolve().parents[3] / "configs/answer_assembly_policy.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def published_sources():
    root = get_settings().knowledge_root
    sources = {}
    for name in ("scenarios", "answer_contracts"):
        path = root / "v3_1" / (name + ".json")
        raw = path.read_bytes()
        rows = json.loads(raw)["records"]
        sources[name] = (hashlib.sha256(raw).hexdigest(),
                         {r["scenario_id"]: (i, r) for i, r in enumerate(rows)})
    return sources


def _fragment(scenario, text, fragment_id, source, pointer):
    digest = published_sources()[source][0]
    return AnswerFragment(fragment_id, text, f"knowledge/v3_1/{source}.json", pointer,
                          digest, scenario.source, scenario.source_version)


def build_answer_plan(message: str, scenario_id: str, role: str) -> AnswerPlan | None:
    scenario = get_scenario(scenario_id)
    contract = get_answer_contract(scenario_id)
    if not scenario_allowed(scenario, role) or not contract:
        return None
    sources = published_sources()
    ci, published_contract = sources["answer_contracts"][1][scenario_id]
    si, published_scenario = sources["scenarios"][1][scenario_id]
    template = _fragment(scenario, published_contract["approved_template"], scenario_id + ":approved_template",
                         "answer_contracts", f"/records/{ci}/approved_template")
    fallback = AnswerPlan(scenario_id, "published_template", (template,), (), "no_unique_scoped_profile")
    config = assembly_policy()["scenarios"].get(scenario_id)
    if not config:
        return fallback
    if config["source_version"] != scenario.source_version:
        return AnswerPlan(scenario_id, "published_template", (template,), (), "profile_source_version_changed")
    profiles = [p for p in config["profiles"] if re.search(p["pattern"], message, re.I)
                and not re.search(p["exclude"], message, re.I)]
    if len(profiles) != 1:
        return fallback
    profile = profiles[0]
    ids = tuple(f"{scenario_id}.fact.{n:03d}" for n in profile["facts"])
    records = {f["fact_id"]: (i, f) for i, f in enumerate(published_scenario["fact_records"])}
    if not ids or any(fid not in contract.allowed_fact_ids or fid not in records
                      or records[fid][1].get("status") != "approved"
                      or records[fid][1]["text"] != contract.facts.get(fid) for fid in ids):
        return AnswerPlan(scenario_id, "published_template", (template,), (), "profile_fact_contract_mismatch")
    fragments = [_fragment(scenario, records[fid][1]["text"], fid, "scenarios",
                           f"/records/{si}/fact_records/{records[fid][0]}/text") for fid in ids]
    if profile["include_next_step"] and published_scenario["next_step"]:
        fragments.append(_fragment(scenario, published_scenario["next_step"], scenario_id + ":next_step",
                                   "scenarios", f"/records/{si}/next_step"))
    return AnswerPlan(scenario_id, profile["id"], tuple(fragments), ids,
                      "scoped_exact_published_facts", profile["documents"])


def verify_plan_text(candidate: str, plan: AnswerPlan, expected: AnswerPlan) -> bool:
    """Expected is rebuilt from trusted policy, scenario and the current question.

    Exact equality preserves numbers, conditions, subjects, negations and promises
    together. A bag of words or a caller-invented fragment cannot authorize output.
    """
    return plan == expected and candidate == expected.text
