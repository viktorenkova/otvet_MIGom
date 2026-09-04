"""Read-only evidence inventory and technical review; never renew expert approval."""
import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from backend.tools.master_knowledge import load_master_bundle, DEFAULT_MASTER


def normalized(text):
    return " ".join(text.casefold().replace("ё", "е").split()).strip(" .")


def audit(as_of: date):
    master, _, _ = load_master_bundle(DEFAULT_MASTER)
    master_rows = {r["scenario_id"]: (i, r) for i,r in enumerate(master["records"])}
    path = Path("knowledge/v3_1/scenarios.json")
    raw = path.read_bytes()
    runtime = json.loads(raw)["records"]
    contracts = {r["scenario_id"]: r for r in json.loads(Path("knowledge/v3_1/answer_contracts.json").read_text(encoding="utf-8"))["records"]}
    evidence, paragraphs = [], []
    for doc in sorted(Path("docs/evidence").glob("*.docx")):
        digest = hashlib.sha256(doc.read_bytes()).hexdigest()
        with ZipFile(doc) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        texts = ["".join(n.text or "" for n in p.findall(".//w:t", ns)) for p in root.findall(".//w:p", ns)]
        evidence.append({"path": doc.as_posix(), "sha256": digest, "paragraphs": len(texts), "semantic_review": "pending"})
        paragraphs.extend({"path": doc.as_posix(), "sha256": digest,
                           "pointer": f"word/document.xml//w:p[{i+1}]", "text": text}
                          for i,text in enumerate(texts) if text.strip())
    records = []
    for si, scenario in enumerate(runtime):
        sid = scenario["scenario_id"]
        mi, canonical = master_rows[sid]
        due = date.fromisoformat(scenario["reviewed_at"]) + timedelta(days=scenario["review_interval_days"])
        issues, facts = [], []
        for fi, fact in enumerate(scenario["fact_records"]):
            fid = fact["fact_id"]
            matches = [p for p in paragraphs if len(normalized(fact["text"])) >= 30
                       and normalized(fact["text"]) in normalized(p["text"])]
            canonical_matches = [i for i,text in enumerate(canonical["facts"]) if text == fact["text"]]
            if not canonical_matches:
                issues.append("missing_canonical_fact:" + fid)
            if contracts[sid]["facts"].get(fid) != fact["text"]:
                issues.append("contract_text_mismatch:" + fid)
            facts.append({"fact_id": fid, "text": fact["text"], "status": fact["status"],
                "runtime_pointer": f"/records/{si}/fact_records/{fi}/text",
                "canonical_pointer": f"MASTER_CANONICAL_V2/records/{mi}/facts/{canonical_matches[0]}" if canonical_matches else None,
                "source_description": fact.get("source"), "primary_evidence_verified": False,
                "exact_text_candidates": matches,
                "evidence_review": "exact_text_requires_context_confirmation" if matches else "no_exact_primary_text_match"})
        contract = contracts[sid]
        if not set(contract["required_fact_ids"]).issubset(contract["allowed_fact_ids"]):
            issues.append("invalid_required_fact_scope")
        if due < as_of:
            issues.append("expert_review_overdue")
        records.append({"scenario_id": sid, "priority": "P0" if scenario.get("domain") in {"finance", "fulfillment"} else "P1",
            "reviewed_at": scenario["reviewed_at"], "review_due": due.isoformat(), "overdue": due < as_of,
            "review_owner": scenario["review_owner"], "expert": scenario["expert"], "issues": issues,
            "technical_review": "completed", "expert_reapproval": "pending", "facts": facts})
    records.sort(key=lambda r: (not r["overdue"], r["priority"], r["review_due"], r["scenario_id"]))
    registry = json.loads(Path("configs/runtime_answer_templates.json").read_text(encoding="utf-8"))
    return {"schema_version": 1, "as_of": as_of.isoformat(), "knowledge_modified": False,
        "canonical_sha256": hashlib.sha256(DEFAULT_MASTER.read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(raw).hexdigest(), "evidence_documents": evidence,
        "summary": {"scenarios": len(records), "facts": sum(len(r["facts"]) for r in records),
            "overdue_scenarios": sum(r["overdue"] for r in records),
            "structural_issues": dict(Counter(i for r in records for i in r["issues"] if i != "expert_review_overdue")),
            "facts_with_exact_primary_candidates": sum(bool(f["exact_text_candidates"]) for r in records for f in r["facts"]),
            "expert_reapprovals": 0},
        "legacy_text_review": {"defaults": list(registry["defaults"]), "exceptions": registry["exceptions"],
            "status": "pending_not_authorized_by_migration"}, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--output", type=Path, default=Path("reports/answer-source-review.json"))
    args = parser.parse_args()
    report = audit(args.as_of)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
