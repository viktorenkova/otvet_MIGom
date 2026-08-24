from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER = ROOT / "knowledge/MASTER_KNOWLEDGE.md"
DEFAULT_V2 = ROOT / "knowledge/v2/scenarios.json"
DEFAULT_V3 = ROOT / "knowledge/v3_1/scenarios.json"
DEFAULT_REVIEW_QUEUE = ROOT / "knowledge/v2/review_queue.json"
MASTER_AUDIT_DATE = date(2026, 8, 24)
V2_BEGIN = "<!-- MASTER_CANONICAL_V2_BEGIN -->"
V2_END = "<!-- MASTER_CANONICAL_V2_END -->"
GAPS_BEGIN = "<!-- MASTER_KNOWLEDGE_GAPS_BEGIN -->"
GAPS_END = "<!-- MASTER_KNOWLEDGE_GAPS_END -->"
REVIEW_BEGIN = "<!-- MASTER_REVIEW_QUEUE_BEGIN -->"
REVIEW_END = "<!-- MASTER_REVIEW_QUEUE_END -->"

DOMAIN_TITLES = {
    "onboarding": "Начало работы и роли",
    "identity_access": "Регистрация, вход и аккаунт",
    "seller_operations": "Работа продавца",
    "auction_bidding": "Торги и ставки",
    "catalog_lot": "Каталог, карточка лота и сведения об объекте",
    "finance": "Тарифы, платежи, комиссии, возвраты и штрафы",
    "fulfillment": "Победа, передача, договор, оплата и получение",
    "support_feedback": "Поддержка, обращения и технические вопросы",
    "compliance_safety": "Безопасность, право и персональные данные",
    "platform_operations": "Прочие процессы площадки",
}
DOMAIN_BY_PREFIX = {
    "account": "identity_access", "registration": "identity_access",
    "buyer": "onboarding", "platform": "onboarding",
    "seller": "seller_operations", "bid": "auction_bidding", "auction": "auction_bidding",
    "lot": "catalog_lot", "property": "catalog_lot", "vehicle": "catalog_lot", "insurance": "catalog_lot",
    "tariff": "finance", "payment": "finance", "balance": "finance", "commission": "finance",
    "refund": "finance", "deposit": "finance", "penalty": "finance", "finance": "finance", "loyalty": "finance",
    "contract": "fulfillment", "transfer": "fulfillment", "pickup": "fulfillment", "documents": "fulfillment",
    "refusal": "fulfillment", "win": "fulfillment", "inspection": "fulfillment",
    "support": "support_feedback", "feedback": "support_feedback", "technical": "support_feedback",
    "complaint": "support_feedback", "partnership": "support_feedback",
    "privacy": "compliance_safety", "compliance": "compliance_safety", "safety": "compliance_safety",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _published_v2_json(source: dict[str, Any]) -> str:
    """Keep the established compact-array formatting of the v2 compatibility file."""
    lines = ["{"]
    items = list(source.items())
    for item_index, (key, value) in enumerate(items):
        suffix = "," if item_index < len(items) - 1 else ""
        if key != "records":
            lines.append(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}{suffix}")
            continue
        lines.append(f"  {json.dumps(key)}: [")
        records = list(value)
        for record_index, record in enumerate(records):
            lines.append("    {")
            fields = list(record.items())
            for field_index, (field, field_value) in enumerate(fields):
                field_suffix = "," if field_index < len(fields) - 1 else ""
                prefix = f"      {json.dumps(field)}: "
                if isinstance(field_value, list) and field_value and all(
                    isinstance(item, dict) for item in field_value
                ):
                    lines.append(prefix + "[")
                    for nested_index, nested in enumerate(field_value):
                        nested_suffix = "," if nested_index < len(field_value) - 1 else ""
                        lines.append(
                            "        "
                            + json.dumps(nested, ensure_ascii=False, separators=(",", ": "))
                            + nested_suffix
                        )
                    lines.append("      ]" + field_suffix)
                else:
                    lines.append(prefix + json.dumps(field_value, ensure_ascii=False) + field_suffix)
            record_suffix = "," if record_index < len(records) - 1 else ""
            lines.append("    }" + record_suffix)
        lines.append("  ]" + suffix)
    lines.append("}")
    return "\n".join(lines) + "\n"


def _extract_json(text: str, begin: str, end: str) -> dict[str, Any]:
    pattern = re.compile(
        rf"{re.escape(begin)}\s*```json\s*(.*?)\s*```\s*{re.escape(end)}",
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Canonical block is missing: {begin}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError(f"Canonical block must contain an object: {begin}")
    return payload


def load_master(path: Path = DEFAULT_MASTER) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source, gaps, _ = load_master_bundle(path)
    return source, gaps


def load_master_bundle(
    path: Path = DEFAULT_MASTER,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    source = _extract_json(text, V2_BEGIN, V2_END)
    gap_payload = _extract_json(text, GAPS_BEGIN, GAPS_END)
    try:
        review_queue = _extract_json(text, REVIEW_BEGIN, REVIEW_END)
    except ValueError:
        if not DEFAULT_REVIEW_QUEUE.exists():
            raise
        review_queue = json.loads(DEFAULT_REVIEW_QUEUE.read_text(encoding="utf-8"))
    gaps = gap_payload.get("knowledge_gaps", [])
    if not isinstance(gaps, list):
        raise ValueError("knowledge_gaps must be a list")
    if not isinstance(review_queue.get("records"), list):
        raise ValueError("review queue records must be a list")
    return source, [dict(item) for item in gaps], review_queue


def _md(value: Any) -> str:
    text = str(value or "—").replace("|", "\\|").replace("\n", " ").strip()
    return text or "—"


def _list(values: list[Any] | tuple[Any, ...]) -> str:
    return ", ".join(str(item) for item in values) if values else "—"


def _source_due(record: dict[str, Any]) -> str:
    try:
        reviewed = date.fromisoformat(str(record.get("reviewed_at") or ""))
        interval = int(record.get("review_interval_days") or 30)
    except (TypeError, ValueError):
        return "не определено"
    return (reviewed + timedelta(days=interval)).isoformat()


def _action_text(action: dict[str, Any]) -> str:
    parts = [str(action.get("label") or action.get("id") or "действие")]
    if action.get("type"):
        parts.append(f"тип: {action['type']}")
    if action.get("scenario_id"):
        parts.append(f"сценарий: {action['scenario_id']}")
    payload = action.get("payload")
    if payload:
        parts.append("параметры: " + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return "; ".join(parts)


def _business_domain(record: dict[str, Any]) -> str:
    scenario_id = str(record.get("scenario_id") or "")
    if scenario_id.startswith("lot.payment."):
        return "finance"
    return DOMAIN_BY_PREFIX.get(scenario_id.split(".", 1)[0], "platform_operations")


def render_master(
    source: dict[str, Any],
    gaps: list[dict[str, Any]],
    review_queue: dict[str, Any] | None = None,
) -> str:
    review_queue = review_queue or json.loads(DEFAULT_REVIEW_QUEUE.read_text(encoding="utf-8"))
    review_records = list(review_queue.get("records", []))
    records = [dict(item) for item in source.get("records", [])]
    fact_count = sum(len(item.get("facts", [])) for item in records)
    active = [item for item in records if str(item.get("status") or "active") == "active"]
    sources = Counter(str(item.get("source") or "не указан") for item in records)
    domains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        domain = _business_domain(record)
        domains[domain].append(record)

    v2_sha = _sha(source)
    gap_sha = _sha({"knowledge_gaps": gaps})
    review_sha = _sha(review_queue)
    expired_reviews = []
    for record in records:
        try:
            due = date.fromisoformat(str(record.get("reviewed_at"))) + timedelta(
                days=int(record.get("review_interval_days") or 30)
            )
        except (TypeError, ValueError):
            continue
        if due < MASTER_AUDIT_DATE:
            expired_reviews.append((record, due))
    lines = [
        "# MIGTORG — мастер-документ знаний и бизнес-процессов",
        "",
        "> Статус: канонический первоисточник данных проекта. Из этого Markdown публикуется",
        "> `knowledge/v2/scenarios.json`, после чего воспроизводимо строятся БЗ v3.1 и answer contracts.",
        "> Утверждения, отсутствующие в канонических блоках этого документа, не считаются фактами БЗ.",
        "",
        "## 1. Назначение и правила управления",
        "",
        "Документ объединяет все утверждённые на текущий момент сценарии, факты, ответы, действия, ограничения, источники и известные пробелы знаний MIGTORG. Он не дополняет факты догадками и не заменяет юридическую или продуктовую проверку.",
        "",
        "Правила изменения:",
        "",
        "1. Изменение факта требует подтверждённого источника, даты проверки, владельца проверки и эксперта.",
        "2. Неподтверждённая информация добавляется только в раздел обязательных пробелов, а не в факты.",
        "3. Канонические данные редактируются в JSON-блоках в конце документа; затем выполняется `python -m backend.tools.master_knowledge render`.",
        "4. Производные файлы публикуются командой `python -m backend.tools.master_knowledge publish` и не редактируются вручную.",
        "5. После публикации выполняются миграция v3.1, сборка answer contracts и строгие валидаторы.",
        "",
        "## 2. Состояние знаний",
        "",
        f"- Версия канонического слоя: `{source.get('version', 'не указана')}`.",
        f"- Сценариев: {len(records)}, активных: {len(active)}.",
        f"- Подтверждённых фактов: {fact_count}.",
        f"- Уникальных текстовых описаний источников: {len(sources)}.",
        f"- Зафиксированных содержательных пробелов: {len(gaps)}.",
        f"- Кандидатов, ожидающих экспертного решения: {len(review_records)}.",
        f"- Просроченных повторных проверок на {MASTER_AUDIT_DATE.isoformat()}: {len(expired_reviews)}.",
        f"- SHA-256 канонических сценариев: `{v2_sha}`.",
        f"- SHA-256 реестра пробелов: `{gap_sha}`.",
        f"- SHA-256 очереди экспертной проверки: `{review_sha}`.",
        "",
        "## 3. Обязательные пробелы, требующие актуальной информации",
        "",
        "Эти пункты нельзя заполнять предположениями. До подтверждения действует только указанный безопасный ответ.",
        "",
    ]
    for index, gap in enumerate(gaps, start=1):
        lines.extend([
            f"### 3.{index}. {_md(gap.get('question'))} (`{gap.get('gap_id')}`)",
            "",
            f"- Связанный сценарий: `{gap.get('scenario_id')}`.",
            f"- Чего не хватает: {_md(gap.get('missing_fact'))}",
            f"- Статус: `{gap.get('status')}`.",
            f"- Что требуется: актуальное письменное подтверждение владельца продукта или назначенного эксперта с датой и версией источника.",
            f"- Безопасный ответ до заполнения: {_md(gap.get('safe_answer'))}",
            "",
        ])
    lines.extend([
        "### 3.4. Системный пробел происхождения фактов",
        "",
        "У каждого сценария есть текстовое поле `source`, эксперт и дата проверки, однако в текущей схеме отсутствуют отдельные поля `evidence_path`, `evidence_sha256` и точная ссылка источника для каждого факта. Поэтому текст источника сохранён без изменений, но часть фактов нельзя автоматически связать с конкретным файлом-доказательством.",
        "",
        f"Обязательно заполнить прямые ссылки и контрольные суммы доказательств для всех {fact_count} фактов, начиная с финансовых, договорных, штрафных, возвратных и персональных данных. До заполнения нельзя утверждать, что происхождение каждого факта подтверждено на уровне конкретного файла, хотя все факты имеют утверждённую текстовую атрибуцию.",
        "",
        "### 3.5. Проверка актуальности",
        "",
        f"По состоянию на `{MASTER_AUDIT_DATE.isoformat()}` установленный срок повторной проверки истёк у {len(expired_reviews)} сценариев. Их текущий текст сохранён без самовольного изменения, но владелец БЗ и указанный эксперт должны подтвердить актуальность или внести новую утверждённую редакцию:",
        "",
    ])
    for record, due in expired_reviews:
        lines.append(
            f"- `{record['scenario_id']}` — проверено `{record.get('reviewed_at')}`, срок повторной проверки `{due.isoformat()}`, эксперт: {_md(record.get('expert'))}."
        )
    lines.extend([
        "",
        "### 3.6. Очередь обязательных экспертных решений",
        "",
        "Следующие записи не являются опубликованными фактами. Это вопросы и блокеры, которые необходимо закрыть подтверждённой информацией до публикации соответствующих правил:",
        "",
    ])
    for candidate in review_records:
        lines.extend([
            f"#### {_md(candidate.get('title'))} (`{candidate.get('candidate_id')}`)",
            "",
            f"- Статус: `{candidate.get('status')}`; риск: `{candidate.get('risk')}`.",
            f"- Безопасный действующий сценарий: `{candidate.get('safe_fallback_scenario_id')}`.",
            f"- Требуемая роль эксперта: {_md(candidate.get('expert_role'))}.",
            f"- Блокеры публикации: {_md(_list(candidate.get('publication_blockers', [])))}.",
            "- Вопросы, требующие ответа:",
            *[f"  - {_md(question)}" for question in candidate.get("questions_for_expert", [])],
            "",
        ])
    lines.extend([
        "## 4. Карта процессов",
        "",
        "Карта ниже показывает имеющиеся сценарии по процессным областям. Она отражает только структуру подтверждённой БЗ и не добавляет новых связей между процессами.",
        "",
        "| Область | Сценариев | Этапы |",
        "|---|---:|---|",
    ])
    for domain, domain_records in sorted(domains.items(), key=lambda item: DOMAIN_TITLES.get(item[0], item[0])):
        stages = sorted({str(item.get("stage") or "не указан") for item in domain_records})
        lines.append(f"| {_md(DOMAIN_TITLES.get(domain, domain))} (`{domain}`) | {len(domain_records)} | {_md(', '.join(stages))} |")

    lines.extend(["", "## 5. Реестр сценариев, процессов и фактов", ""])
    section = 0
    for domain, domain_records in sorted(domains.items(), key=lambda item: DOMAIN_TITLES.get(item[0], item[0])):
        section += 1
        lines.extend([
            f"## 5.{section}. {DOMAIN_TITLES.get(domain, domain)} (`{domain}`)",
            "",
        ])
        for scenario_index, record in enumerate(sorted(domain_records, key=lambda item: str(item.get("scenario_id"))), start=1):
            scenario_id = str(record.get("scenario_id"))
            lines.extend([
                f"### 5.{section}.{scenario_index}. {_md(record.get('title'))} (`{scenario_id}`)",
                "",
                "| Параметр | Подтверждённое значение |",
                "|---|---|",
                f"| Назначение/intent | `{_md(record.get('intent'))}` |",
                f"| Роли | {_md(_list(record.get('roles', [])))} |",
                f"| Этап | `{_md(record.get('stage'))}` |",
                f"| Объекты | {_md(_list(record.get('objects', [])))} |",
                f"| Операции | {_md(_list(record.get('operations', [])))} |",
                f"| Состояния | {_md(_list(record.get('states', [])))} |",
                f"| Требуемый контекст | {_md(_list(record.get('required_context', [])))} |",
                f"| Допустимый контекст | {_md(_list(record.get('allowed_context', [])))} |",
                f"| Статус | `{_md(record.get('status') or 'active')}` |",
                "",
                "#### Подтверждённые факты",
                "",
            ])
            facts = list(record.get("facts", []))
            if facts:
                for fact_index, fact in enumerate(facts, start=1):
                    fact_id = f"{scenario_id}.fact.{fact_index:03d}"
                    lines.append(f"{fact_index}. **`{fact_id}`** — {str(fact).strip()}")
            else:
                lines.append("- Подтверждённых фактов нет. Требуется проверка владельца БЗ.")
            lines.extend([
                "",
                "#### Утверждённый ответ и следующий шаг",
                "",
                f"- Короткий ответ: {_md(record.get('short_answer'))}",
                f"- Подробности: {_md(record.get('detailed_answer'))}",
                f"- Следующий шаг: {_md(record.get('next_step'))}",
                "",
                "#### Действия и эскалация",
                "",
            ])
            actions = list(record.get("actions", []))
            if actions:
                lines.extend(f"- {_action_text(action)}" for action in actions)
            else:
                lines.append("- Специальные действия не определены.")
            escalation = dict(record.get("escalation", {}))
            lines.extend([
                f"- Условия ручной проверки: {_md(_list(escalation.get('when', [])))}.",
                f"- Данные для ручной проверки: {_md(_list(escalation.get('required_fields', [])))}.",
                "",
                "#### Происхождение и актуальность",
                "",
                f"- Источник: {_md(record.get('source'))}",
                f"- Проверено: `{_md(record.get('reviewed_at'))}`; следующая проверка не позднее `{_source_due(record)}`.",
                f"- Владелец проверки: {_md(record.get('review_owner'))}",
                f"- Эксперт: {_md(record.get('expert'))}",
                "- Прямая ссылка и SHA-256 файла-доказательства: **не заполнены — обязательный provenance gap**.",
                "",
            ])
            linked_gaps = [gap for gap in gaps if gap.get("scenario_id") == scenario_id]
            if linked_gaps:
                lines.extend([
                    "#### Связанные пробелы",
                    "",
                    *[f"- `{gap['gap_id']}`: {_md(gap.get('missing_fact'))}" for gap in linked_gaps],
                    "",
                ])

    lines.extend([
        "## 6. Реестр текстовых источников",
        "",
        "Ниже сохранены все значения `source`, присутствующие в утверждённых сценариях. Это атрибуция, а не замена прямой ссылки на доказательство.",
        "",
        "| Источник | Сценариев |",
        "|---|---:|",
    ])
    for source_name, count in sorted(sources.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {_md(source_name)} | {count} |")

    lines.extend([
        "",
        "## 7. Канонические машинно-читаемые данные",
        "",
        "Следующие блоки являются первоисточником для публикации производных JSON-файлов. Человекочитаемые разделы выше воспроизводимо строятся из этих блоков и не должны редактироваться отдельно.",
        "",
        V2_BEGIN,
        "```json",
        _published_v2_json(source).rstrip(),
        "```",
        V2_END,
        "",
        GAPS_BEGIN,
        "```json",
        json.dumps({"knowledge_gaps": gaps}, ensure_ascii=False, indent=2),
        "```",
        GAPS_END,
        "",
        REVIEW_BEGIN,
        "```json",
        json.dumps(review_queue, ensure_ascii=False, indent=2),
        "```",
        REVIEW_END,
    ])
    return "\n".join(lines) + "\n"


def validate_master(path: Path = DEFAULT_MASTER, published_v2: Path = DEFAULT_V2) -> dict[str, Any]:
    errors: list[str] = []
    source, gaps, review_queue = load_master_bundle(path)
    review_records = list(review_queue.get("records", []))
    records = list(source.get("records", []))
    scenario_ids = [str(item.get("scenario_id") or "") for item in records]
    facts = [str(fact) for record in records for fact in record.get("facts", [])]
    if source.get("schema_version") != 2:
        errors.append("canonical source schema_version must be 2")
    if len(records) != 141:
        errors.append(f"expected 141 scenarios, got {len(records)}")
    if len(facts) != 574:
        errors.append(f"expected 574 facts, got {len(facts)}")
    if len(scenario_ids) != len(set(scenario_ids)) or any(not item for item in scenario_ids):
        errors.append("scenario IDs are missing or duplicated")
    if len(gaps) != 3:
        errors.append(f"expected 3 knowledge gaps, got {len(gaps)}")
    if len(review_records) != 5:
        errors.append(f"expected 5 expert-review candidates, got {len(review_records)}")
    gap_ids = [str(item.get("gap_id") or "") for item in gaps]
    if len(gap_ids) != len(set(gap_ids)) or any(not item for item in gap_ids):
        errors.append("knowledge gap IDs are missing or duplicated")
    for record in records:
        for field in ("source", "reviewed_at", "review_owner", "expert"):
            if not record.get(field):
                errors.append(f"{record.get('scenario_id')}: missing {field}")
        if any(not str(fact).strip() for fact in record.get("facts", [])):
            errors.append(f"{record.get('scenario_id')}: contains an empty fact")
    for gap in gaps:
        for field in (
            "scenario_id", "question", "missing_fact", "status", "safe_answer", "answer_policy"
        ):
            if not gap.get(field):
                errors.append(f"{gap.get('gap_id')}: missing {field}")
        if gap.get("scenario_id") not in set(scenario_ids):
            errors.append(f"{gap.get('gap_id')}: references an unknown scenario")
    candidate_ids = [str(item.get("candidate_id") or "") for item in review_records]
    if len(candidate_ids) != len(set(candidate_ids)) or any(not item for item in candidate_ids):
        errors.append("review candidate IDs are missing or duplicated")
    expected_render = render_master(source, gaps, review_queue)
    if path.read_text(encoding="utf-8") != expected_render:
        errors.append("human-readable master sections are out of sync; run master_knowledge render")
    if published_v2.exists():
        published = json.loads(published_v2.read_text(encoding="utf-8"))
        if published != source:
            errors.append("knowledge/v2/scenarios.json differs from master canonical source")
    return {
        "schema_version": 1,
        "valid": not errors,
        "errors": errors,
        "metrics": {
            "scenario_count": len(records),
            "fact_count": len(facts),
            "knowledge_gap_count": len(gaps),
            "expert_review_candidate_count": len(review_records),
            "canonical_sha256": _sha(source),
            "knowledge_gaps_sha256": _sha({"knowledge_gaps": gaps}),
            "review_queue_sha256": _sha(review_queue),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the canonical MIGTORG Markdown knowledge source.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Create the initial master from approved v2/v3 sources.")
    bootstrap.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    bootstrap.add_argument("--v2", type=Path, default=DEFAULT_V2)
    bootstrap.add_argument("--v3", type=Path, default=DEFAULT_V3)
    bootstrap.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)

    render = subparsers.add_parser("render", help="Regenerate readable sections from canonical blocks.")
    render.add_argument("--master", type=Path, default=DEFAULT_MASTER)

    publish = subparsers.add_parser("publish", help="Publish v2 compatibility JSON from the master.")
    publish.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    publish.add_argument("--v2", type=Path, default=DEFAULT_V2)

    validate = subparsers.add_parser("validate", help="Validate master structure and published parity.")
    validate.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    validate.add_argument("--v2", type=Path, default=DEFAULT_V2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "bootstrap":
        if args.master.exists():
            print(json.dumps({"created": False, "error": "master already exists"}, ensure_ascii=False))
            return 2
        source = json.loads(args.v2.read_text(encoding="utf-8"))
        v3 = json.loads(args.v3.read_text(encoding="utf-8"))
        review_queue = json.loads(args.review_queue.read_text(encoding="utf-8"))
        args.master.write_text(
            render_master(source, list(v3.get("knowledge_gaps", [])), review_queue),
            encoding="utf-8",
        )
        print(json.dumps({"created": True, "path": str(args.master)}, ensure_ascii=False))
        return 0
    if args.command == "render":
        source, gaps, review_queue = load_master_bundle(args.master)
        args.master.write_text(render_master(source, gaps, review_queue), encoding="utf-8")
        print(json.dumps({"rendered": True, "path": str(args.master)}, ensure_ascii=False))
        return 0
    if args.command == "publish":
        source, gaps, review_queue = load_master_bundle(args.master)
        published_source = (
            json.loads(args.v2.read_text(encoding="utf-8")) if args.v2.exists() else None
        )
        if published_source != source:
            args.v2.parent.mkdir(parents=True, exist_ok=True)
            args.v2.write_text(_published_v2_json(source), encoding="utf-8")
        published_review_queue = (
            json.loads(DEFAULT_REVIEW_QUEUE.read_text(encoding="utf-8"))
            if DEFAULT_REVIEW_QUEUE.exists()
            else None
        )
        if published_review_queue != review_queue:
            DEFAULT_REVIEW_QUEUE.write_text(
                json.dumps(review_queue, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        from backend.tools.build_stage4_answer_contracts import build as build_answer_contracts
        from backend.tools.migrate_knowledge_v31 import (
            DEFAULT_CONFLICTS,
            DEFAULT_MIGRATION_REPORT,
            DEFAULT_OUTPUT,
            DEFAULT_QUALITY_REPORT,
            build_conflicts,
            build_migration_report,
            migrate,
        )

        quality_report = json.loads(DEFAULT_QUALITY_REPORT.read_text(encoding="utf-8"))
        knowledge = migrate(source, knowledge_gaps=gaps)
        conflicts = build_conflicts(quality_report, knowledge)
        migration_report = build_migration_report(args.master, source, knowledge, conflicts)
        for path, payload in (
            (DEFAULT_OUTPUT, knowledge),
            (DEFAULT_CONFLICTS, conflicts),
            (DEFAULT_MIGRATION_REPORT, migration_report),
        ):
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        answer_contracts_path = ROOT / "knowledge/v3_1/answer_contracts.json"
        answer_contracts = build_answer_contracts(DEFAULT_OUTPUT, DEFAULT_CONFLICTS)
        answer_contracts_path.write_text(
            json.dumps(answer_contracts, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "published": True,
            "master": str(args.master),
            "v2": str(args.v2),
            "v3": str(DEFAULT_OUTPUT),
            "answer_contracts": str(answer_contracts_path),
            "scenario_count": len(source["records"]),
            "fact_count": sum(len(item["facts"]) for item in source["records"]),
            "sha256": _sha(source),
        }, ensure_ascii=False))
        return 0
    result = validate_master(args.master, args.v2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
