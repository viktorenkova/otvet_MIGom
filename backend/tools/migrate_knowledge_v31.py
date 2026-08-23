from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("knowledge/v2/scenarios.json")
DEFAULT_QUALITY_REPORT = Path("reports/quality-stage0-adjudicated-local.json")
DEFAULT_OUTPUT = Path("knowledge/v3_1/scenarios.json")
DEFAULT_CONFLICTS = Path("knowledge/v3_1/scenario_conflicts.json")
DEFAULT_MIGRATION_REPORT = Path("reports/knowledge-v31-migration.json")
DEFAULT_RETRIEVAL_TAXONOMY = Path("configs/retrieval_taxonomy_terms.json")
VERSION = "2026.08.23.2"
GENERATED_AT = "2026-08-23T00:00:00+03:00"


DOMAIN_BY_PREFIX = {
    "account": "identity_access",
    "registration": "identity_access",
    "buyer": "onboarding",
    "platform": "onboarding",
    "seller": "seller_operations",
    "bid": "auction_bidding",
    "auction": "auction_bidding",
    "lot": "catalog_lot",
    "property": "catalog_lot",
    "vehicle": "catalog_lot",
    "insurance": "catalog_lot",
    "tariff": "finance",
    "payment": "finance",
    "balance": "finance",
    "commission": "finance",
    "refund": "finance",
    "deposit": "finance",
    "penalty": "finance",
    "finance": "finance",
    "contract": "fulfillment",
    "transfer": "fulfillment",
    "pickup": "fulfillment",
    "documents": "fulfillment",
    "refusal": "fulfillment",
    "win": "fulfillment",
    "inspection": "fulfillment",
    "support": "support_feedback",
    "feedback": "support_feedback",
    "technical": "support_feedback",
    "complaint": "support_feedback",
    "partnership": "support_feedback",
    "privacy": "compliance_safety",
    "compliance": "compliance_safety",
    "safety": "compliance_safety",
    "loyalty": "finance",
}


# Mixed canonical answers stay API-compatible, while retrieval is split into
# atomic units that contain one user goal and a non-overlapping fact subset.
ATOMIC_SPLITS: dict[str, list[dict[str, Any]]] = {
    "payment.methods": [
        {"suffix": "balance_topup", "title": "Пополнение баланса картой или СБП", "facts": [0], "terms": ["пополнить баланс", "карта", "сбп"]},
        {"suffix": "tariff", "title": "Оплата тарифа", "facts": [1], "terms": ["оплатить тариф", "счёт на тариф"]},
        {"suffix": "platform_commission", "title": "Оплата комиссии площадки", "facts": [2], "terms": ["оплатить комиссию", "комиссия с баланса"]},
        {"suffix": "lot", "title": "Оплата стоимости лота продавцу", "facts": [3], "terms": ["оплатить лот", "стоимость лота", "баланс migtorg"]},
        {"suffix": "cash", "title": "Оплата наличными в офисе", "facts": [4], "terms": ["наличные", "оплата в офисе"]},
    ],
    "tariff.choose": [
        {"suffix": "vehicle_plans", "title": "Разовый или Премиум-тариф", "facts": [0, 1], "terms": ["разовый тариф", "премиум", "какой тариф"]},
        {"suffix": "property", "title": "Доступ к разделу Имущество", "facts": [2], "terms": ["имущество", "без тарифа"]},
        {"suffix": "auction_cost_policy", "title": "Платежи и ограничения автомобильного тарифа", "facts": [3, 4], "terms": ["обеспечительный платеж", "гарантия передачи"]},
    ],
    "bid.autobid_extension": [
        {"suffix": "autobid", "title": "Автоматическое повышение ставки до лимита", "facts": [0], "terms": ["автоставка", "максимальная сумма", "лимит"]},
        {"suffix": "last_minutes", "title": "Продление торгов после ставки в последние минуты", "facts": [1, 2, 3], "terms": ["последние пять минут", "продление торгов", "последняя секунда"]},
        {"suffix": "descending_price", "title": "Пошаговое снижение начальной цены", "facts": [4], "terms": ["снижение цены", "нет заявок на повышение"]},
    ],
    "pickup.receive_lot": [
        {"suffix": "readiness", "title": "Готовность лота к выдаче", "facts": [0, 1], "terms": ["когда забрать", "место выдачи", "готовность"]},
        {"suffix": "documents", "title": "Документы для получения лота", "facts": [2], "terms": ["документы на получение", "что взять с собой"]},
    ],
    "contract.receive": [
        {"suffix": "preparation", "title": "Подготовка договора продавцом", "facts": [0], "terms": ["кто готовит договор", "реквизиты"]},
        {"suffix": "delivery", "title": "Получение договора в почтовой ветке", "facts": [1], "terms": ["где договор", "получить дкп", "почта"]},
    ],
    "lot.payment.details": [
        {"suffix": "invoice_payer", "title": "Счёт и плательщик по лоту", "facts": [0, 1], "terms": ["счёт за лот", "кто платит", "реквизиты"]},
        {"suffix": "correspondence", "title": "Официальная переписка по оплате лота", "facts": [2], "terms": ["почтовая ветка", "info@migtorg.com"]},
        {"suffix": "payment_confirmation", "title": "Оплата и подтверждение стоимости лота", "facts": [3, 4, 5], "terms": ["оплатить продавцу", "платёжное поручение", "статус оплачен"]},
    ],
    "refund.application": [
        {"suffix": "one_time", "title": "Возврат Разового тарифа", "facts": [0, 1], "terms": ["возврат разового тарифа", "кнопка возврата"]},
        {"suffix": "premium", "title": "Возврат обеспечительного платежа Премиум", "facts": [2, 3], "terms": ["расторгнуть премиум", "обеспечительный платеж", "заявление"]},
        {"suffix": "commission", "title": "Возврат комиссии по несостоявшейся сделке", "facts": [4], "terms": ["возврат комиссии", "сделка не состоялась"]},
    ],
    "payment.accounting_documents": [
        {"suffix": "request", "title": "Запрос финансовых и закрывающих документов", "facts": [0, 1], "terms": ["закрывающие документы", "счёт", "акт"]},
        {"suffix": "safe_channel", "title": "Безопасный канал для бухгалтерских документов", "facts": [2, 3], "terms": ["бухгалтерские документы", "по телефону", "платёжные данные"]},
    ],
    "bid.price_terms": [
        {"suffix": "definitions", "title": "Ставка, стартовая цена, текущая цена и шаг", "facts": [0, 1, 2, 3], "terms": ["минимальная ставка", "шаг торгов", "текущая цена"]},
        {"suffix": "rules", "title": "Правила цены в открытых и закрытых торгах", "facts": [4, 5], "terms": ["открытые торги", "закрытые торги", "равные предложения"]},
    ],
}


KNOWLEDGE_GAPS = [
    {
        "gap_id": "gap.lot_photo_archive_download",
        "scenario_id": "lot.card_information",
        "question": "Можно ли скачать все фотографии автомобиля одним архивом?",
        "missing_fact": "Наличие и порядок пакетного скачивания фотографий архивом.",
        "status": "owner_confirmation_required",
        "safe_answer": "В подтверждённой БЗ нет сведений о скачивании всех фотографий архивом. Проверьте доступные действия в карточке лота или уточните функцию у поддержки.",
        "answer_policy": "disclose_gap_then_offer_manual_check",
    },
    {
        "gap_id": "gap.tariff_expired_unused",
        "scenario_id": "tariff.status",
        "question": "Что означает, что неиспользованный тариф «сгорел»?",
        "missing_fact": "Условия истечения или прекращения каждого типа тарифа и основания восстановления.",
        "status": "owner_confirmation_required",
        "safe_answer": "Единого подтверждённого правила для всех тарифов нет. Нужно проверить тип тарифа, дату оплаты, статус в кабинете и конкретную операцию.",
        "answer_policy": "disclose_gap_then_collect_operation_context",
    },
    {
        "gap_id": "gap.tariff_access_term_unspecified",
        "scenario_id": "tariff.choose",
        "question": "На какой срок действует доступ без указания типа тарифа?",
        "missing_fact": "Единый срок доступа для вопроса без выбранного типа тарифа.",
        "status": "owner_confirmation_required",
        "safe_answer": "Срок зависит от типа доступа. Уточните, речь о Разовом или Премиум-тарифе; актуальные условия нужно проверить перед оплатой.",
        "answer_policy": "clarify_tariff_type_before_term",
    },
]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _domain(scenario_id: str) -> str:
    if scenario_id.startswith("lot.payment."):
        return "finance"
    prefix = scenario_id.split(".", 1)[0]
    return DOMAIN_BY_PREFIX.get(prefix, "platform_operations")


def _source_version(record: dict[str, Any]) -> str:
    return _sha256_bytes(f"{record['source']}\0{record['reviewed_at']}".encode("utf-8"))[:16]


def _search_document(record: dict[str, Any]) -> str:
    parts = [
        record["title"],
        f"domain:{record['domain']}",
        f"stage:{record['stage']}",
        "objects:" + " ".join(record["objects"]),
        "actions:" + " ".join(record["operations"]),
        "states:" + " ".join(record["states"] or ["unspecified"]),
        *record["positive_examples"],
        *(term for group in record["retrieval_taxonomy_terms"] for term in group["terms"]),
    ]
    return "\n".join(dict.fromkeys(item.strip() for item in parts if item.strip()))


def _retrieval_taxonomy_terms(record: dict[str, Any], vocabulary: dict[str, Any]) -> list[dict[str, Any]]:
    values = {
        "objects": record["objects"],
        "actions": record["operations"],
        "states": record["states"],
        "stages": [record["stage"]],
    }
    groups: list[dict[str, Any]] = []
    for field, taxonomy_values in values.items():
        mapping = vocabulary.get(field, {})
        for value in taxonomy_values:
            terms = mapping.get(value, []) if isinstance(mapping, dict) else []
            if terms:
                groups.append({"field": field, "value": value, "terms": list(terms)})
    return groups


def _fact_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "fact_id": f"{record['scenario_id']}.fact.{index:03d}",
            "text": text,
            "source": record["source"],
            "reviewed_at": record["reviewed_at"],
            "source_version": record["source_version"],
            "status": "approved",
        }
        for index, text in enumerate(record["facts"], start=1)
    ]


def _atomic_units(record: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = ATOMIC_SPLITS.get(record["scenario_id"])
    if definitions is None:
        definitions = [{
            "suffix": "primary",
            "title": record["title"],
            "facts": list(range(len(record["facts"]))),
            "terms": list(record["positive_examples"]),
        }]
    units: list[dict[str, Any]] = []
    for definition in definitions:
        fact_ids = [record["fact_records"][index]["fact_id"] for index in definition["facts"]]
        terms = list(dict.fromkeys(str(item) for item in definition["terms"] if str(item).strip()))
        units.append({
            "unit_id": f"{record['scenario_id']}::{definition['suffix']}",
            "canonical_scenario_id": record["scenario_id"],
            "title": definition["title"],
            "domain": record["domain"],
            "stage": record["stage"],
            "objects": record["objects"],
            "actions": record["operations"],
            "states": record["states"] or ["unspecified"],
            "fact_ids": fact_ids,
            "discriminator_terms": terms,
            "search_document": "\n".join([definition["title"], *terms]),
        })
    return units


def migrate(source: dict[str, Any], retrieval_vocabulary: dict[str, Any] | None = None) -> dict[str, Any]:
    retrieval_vocabulary = retrieval_vocabulary or json.loads(
        DEFAULT_RETRIEVAL_TAXONOMY.read_text(encoding="utf-8")
    )
    records: list[dict[str, Any]] = []
    atomic_units: list[dict[str, Any]] = []
    gap_ids_by_scenario: dict[str, list[str]] = defaultdict(list)
    for gap in KNOWLEDGE_GAPS:
        gap_ids_by_scenario[gap["scenario_id"]].append(gap["gap_id"])

    for raw in source["records"]:
        record = dict(raw)
        record["domain"] = _domain(record["scenario_id"])
        record["taxonomy"] = {
            "domain": record["domain"],
            "objects": list(record["objects"]),
            "actions": list(record["operations"]),
            "states": list(record["states"] or ["unspecified"]),
            "stage": record["stage"],
        }
        signature_payload = {
            **record["taxonomy"],
            "title": record["title"],
        }
        record["discriminators"] = {
            "required": list(dict.fromkeys([*record["objects"], *record["operations"]])),
            "state": list(record["states"] or ["unspecified"]),
            "positive": list(record["positive_examples"]),
            "excluded": list(record["negative_examples"]),
            "signature": _sha256_bytes(_canonical_json(signature_payload)),
        }
        record["source_version"] = _source_version(record)
        record["fact_records"] = _fact_records(record)
        record["answer_policy"] = {
            "mode": "approved_template",
            "fact_scope": "listed_fact_ids_only",
            "llm_role": "wording_only",
            "unknown_fact": "disclose_gap_and_offer_clarification_or_manual_check",
            "first_sentence": "direct_answer_or_subject_clarification",
            "forbidden": [
                "invent_amount",
                "invent_deadline",
                "invent_contact",
                "invent_feature",
                "promise_unverified_result",
            ],
        }
        record["knowledge_gap_ids"] = gap_ids_by_scenario.get(record["scenario_id"], [])
        record["retrieval_taxonomy_terms"] = _retrieval_taxonomy_terms(record, retrieval_vocabulary)
        record["search_document"] = _search_document(record)
        units = _atomic_units(record)
        record["atomic_unit_ids"] = [item["unit_id"] for item in units]
        atomic_units.extend(units)
        records.append(record)

    return {
        "schema_version": "3.1",
        "version": VERSION,
        "generated_at": GENERATED_AT,
        "migrated_from": {
            "schema_version": source.get("schema_version"),
            "version": source.get("version"),
        },
        "retrieval_taxonomy": {
            "path": str(DEFAULT_RETRIEVAL_TAXONOMY),
            "version": retrieval_vocabulary.get("version"),
            "policy": retrieval_vocabulary.get("policy"),
            "sha256": _sha256_bytes(_canonical_json(retrieval_vocabulary)),
        },
        "publication_policy": source.get("publication_policy", "manual_review_only"),
        "knowledge_gaps": KNOWLEDGE_GAPS,
        "records": records,
        "atomic_units": atomic_units,
    }


def build_conflicts(quality_report: dict[str, Any], knowledge: dict[str, Any]) -> dict[str, Any]:
    by_id = {item["scenario_id"]: item for item in knowledge["records"]}
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    for result in quality_report["single_turn_results"]:
        if result["checks"]["route_hit"]:
            continue
        expected = result.get("expected", {}).get("expected_scenario_ids")
        if expected is None:
            continue
        actual = (result.get("response") or {}).get("scenario_id")
        actual_id = str(actual) if actual else "__clarification__"
        expected_ids = ["__no_scenario__" if item is None else str(item) for item in expected]
        for expected_id in expected_ids:
            if expected_id == actual_id:
                continue
            pair = tuple(sorted((expected_id, actual_id)))
            entry = observations.setdefault(pair, {
                "conflict_id": "conflict." + _sha256_bytes("\0".join(pair).encode("utf-8"))[:12],
                "scenario_a": pair[0],
                "scenario_b": pair[1],
                "observed_count": 0,
                "directions": defaultdict(int),
                "examples": [],
            })
            entry["observed_count"] += 1
            entry["directions"][f"{expected_id}->{actual_id}"] += 1
            if len(entry["examples"]) < 5:
                entry["examples"].append({"case_id": result["id"], "text": result["text"]})

    records: list[dict[str, Any]] = []
    for pair, entry in sorted(observations.items()):
        distinctions: dict[str, Any] = {}
        for scenario_id in pair:
            if scenario_id in by_id:
                record = by_id[scenario_id]
                distinctions[scenario_id] = {
                    "domain": record["domain"],
                    "stage": record["stage"],
                    "objects": record["objects"],
                    "actions": record["operations"],
                    "states": record["states"] or ["unspecified"],
                    "positive": record["positive_examples"][:5],
                    "excluded": record["negative_examples"][:5],
                }
            else:
                distinctions[scenario_id] = {"policy": "do_not_select_scenario_without_required_subject"}
        records.append({
            **entry,
            "directions": dict(sorted(entry["directions"].items())),
            "distinctions": distinctions,
            "decision_policy": {
                "compare": ["domain", "stage", "objects", "actions", "states"],
                "missing_discriminator": "ask_subject_specific_clarification",
                "never_use_full_answer_as_retrieval_evidence": True,
            },
        })
    return {
        "schema_version": 1,
        "knowledge_version": knowledge["version"],
        "source_report": str(DEFAULT_QUALITY_REPORT),
        "record_count": len(records),
        "records": records,
    }


def build_migration_report(
    source_path: Path,
    source: dict[str, Any],
    knowledge: dict[str, Any],
    conflicts: dict[str, Any],
) -> dict[str, Any]:
    source_facts = sum(len(item["facts"]) for item in source["records"])
    migrated_facts = sum(len(item["fact_records"]) for item in knowledge["records"])
    return {
        "schema_version": 1,
        "knowledge_version": knowledge["version"],
        "source": str(source_path),
        "source_sha256": _sha256_bytes(source_path.read_bytes()),
        "source_scenario_count": len(source["records"]),
        "migrated_scenario_count": len(knowledge["records"]),
        "source_fact_count": source_facts,
        "migrated_fact_count": migrated_facts,
        "fact_loss_count": source_facts - migrated_facts,
        "atomic_unit_count": len(knowledge["atomic_units"]),
        "split_scenario_count": len(ATOMIC_SPLITS),
        "knowledge_gap_count": len(knowledge["knowledge_gaps"]),
        "conflict_count": conflicts["record_count"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate approved scenario KB to normalized schema v3.1.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_CONFLICTS)
    parser.add_argument("--migration-report", type=Path, default=DEFAULT_MIGRATION_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    quality_report = json.loads(args.quality_report.read_text(encoding="utf-8"))
    knowledge = migrate(source)
    conflicts = build_conflicts(quality_report, knowledge)
    report = build_migration_report(args.source, source, knowledge, conflicts)
    for path, payload in (
        (args.output, knowledge),
        (args.conflicts, conflicts),
        (args.migration_report, report),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
