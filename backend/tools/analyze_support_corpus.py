from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from backend.app.bot.scenario_engine import match_scenario


@dataclass(frozen=True)
class Theme:
    key: str
    title: str
    pattern: re.Pattern[str]
    kind: str = "topic"
    existing_scenario_id: str | None = None
    risk: str = "normal"


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


THEMES = (
    Theme("generic_lot_question", "Недостаточно конкретный вопрос по лоту", _rx(r"вопрос\w*\s+(?:по|о|насч[её]т)\s+[^.!?]{0,40}\bлот"), "quality"),
    Theme("callback_requested", "Запрос звонка или обратной связи", _rx(r"перезвон|обратн\w*\s+(?:связ|звон)|связат|жд[её]т\s+звон|ожидает\s+(?:звон|связ)"), "action", "support.callback"),
    Theme("email_no_response", "Нет ответа на письмо или запрос", _rx(r"(?:писал|отправлял|запрос|письмо).{0,30}почт|почт.{0,30}(?:не\s+получил|не\s+приход|нет\s+ответ)"), existing_scenario_id="support.email_no_response"),
    Theme("account_blocked", "Аккаунт или профиль заблокирован", _rx(r"заблок|разблок|блокировк\w*\s+(?:аккаунт|профил|уч[её]тн)"), existing_scenario_id="account.blocked", risk="account"),
    Theme("login_access", "Не удаётся войти или восстановить доступ", _rx(r"не\s+может\s+(?:войти|зайти)|логин|парол|личн\w*\s+кабинет|восстанов"), existing_scenario_id="account.login_problem", risk="account"),
    Theme("registration", "Регистрация и тип аккаунта", _rx(r"регистрац|зарегистр")),
    Theme("tariff_general", "Выбор и условия тарифа", _rx(r"тариф"), existing_scenario_id="tariff.connect", risk="financial"),
    Theme("tariff_not_activated", "Тариф оплачен, но не активирован", _rx(r"(?:оплат|подключ).{0,30}тариф|тариф.{0,30}(?:не\s+актив|не\s+подключ|доступа\s+нет)"), risk="financial"),
    Theme("payment_lot", "Оплата выигранного лота", _rx(r"оплат\w*.{0,25}\bлот|\bлот.{0,25}оплат"), risk="financial"),
    Theme("payment_missing", "Платёж списан, но не отображается", _rx(r"оплат\w*.{0,35}(?:не\s+отображ|не\s+зачис|не\s+приш|списал)|списал\w*.{0,30}(?:не\s+отображ|нет)"), risk="financial"),
    Theme("commission", "Комиссия площадки или платежа", _rx(r"комисси"), risk="financial"),
    Theme("payment_general", "Общий вопрос по оплате, счёту или балансу", _rx(r"оплат|плат[её]ж|сч[её]т|баланс|кошел"), risk="financial"),
    Theme("refund", "Возврат денежных средств", _rx(r"возврат|вернуть\s+(?:деньги|средств)"), risk="financial"),
    Theme("deposit", "Возврат или назначение депозита", _rx(r"депозит"), risk="financial"),
    Theme("bid_cancel_change", "Изменение, отмена или ошибка ставки", _rx(r"(?:отмен|измен|уменьш|ошиб).{0,25}ставк|ставк.{0,25}(?:отмен|измен|уменьш|ошиб)")),
    Theme("bid_not_visible", "Ставка не видна", _rx(r"ставк.{0,30}(?:не\s+вид|не\s+отображ|пропал)|(?:не\s+вид|не\s+отображ|пропал).{0,30}ставк"), existing_scenario_id="bid.not_visible"),
    Theme("bid_general", "Как сделать ставку и как она работает", _rx(r"ставк|автоставк|позици"), existing_scenario_id="bid.place"),
    Theme("auction_result", "Результат или завершение торгов", _rx(r"результат.{0,20}торг|торг.{0,20}(?:заверш|результат)|кто\s+побед|победил\s+ли"), existing_scenario_id="auction.status"),
    Theme("won_lot_next_step", "Что происходит после победы", _rx(r"выигр\w*.{0,30}\bлот|\bлот.{0,20}выигр|побед"), existing_scenario_id="win.next_steps"),
    Theme("documents_contract", "Договор и документы по сделке", _rx(r"документ|док-т|договор|дкп|купли.?продаж"), existing_scenario_id="contract.receive", risk="contractual"),
    Theme("lot_transfer_delay", "Лот долго не передают", _rx(r"не\s+переда[её]т|передач.{0,30}(?:жд[её]т|долго|когда|нет)|статус.{0,20}передач")),
    Theme("pickup", "Получение, выдача и вывоз лота", _rx(r"забрат|выдач|самовывоз|стоянк|пропуск|получить\s+(?:лот|авто)"), existing_scenario_id="pickup.receive_lot"),
    Theme("inspection", "Осмотр автомобиля", _rx(r"осмотр|посмотреть\s+(?:авто|машин)"), existing_scenario_id="inspection.arrange"),
    Theme("vin_vehicle_info", "VIN, фотографии и сведения об автомобиле", _rx(r"\bvin\b|\bвин\b|фото|информац\w*\s+(?:об|по)\s+авто|поврежд|описан\w*\s+лот")),
    Theme("seller_listing", "Продажа и размещение имущества", _rx(r"хочет\s+продать|продаж|выставить|разместить|стать\s+продав"), existing_scenario_id="seller.publish_lot"),
    Theme("seller_contact", "Продавец не выходит на связь", _rx(r"контакт.{0,20}продав|связ.{0,20}продав|продавец.{0,25}(?:не\s+связ|не\s+отвеч)|страхов.{0,25}(?:не\s+связ|не\s+отвеч)")),
    Theme("refusal", "Отказ от выигранного лота", _rx(r"отказ"), risk="contractual"),
    Theme("penalty", "Штраф и его оспаривание", _rx(r"штраф"), risk="financial"),
    Theme("technical_error", "Ошибка сайта или личного кабинета", _rx(r"ошибк|не\s+работает|не\s+откры|завис|слетел"), existing_scenario_id="technical.site_error"),
)

_INTENSITY_RE = _rx(r"срочно|повторн|до\s+сих\s+пор|не\s+отвеч|не\s+получ|заблок|штраф|отказ|ошиб|не\s+работ")


def load_normalized(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reference(row: dict[str, Any]) -> str:
    return f"{row.get('source', 'unknown')}#{row.get('source_message_id', 'unknown')}"


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("message_kind") == "candidate"]
    decisions = {
        _reference(row): match_scenario(str(row.get("text_redacted") or ""), "guest")
        for row in candidates
    }
    theme_results = []
    for theme in THEMES:
        matched = [row for row in candidates if theme.pattern.search(str(row.get("text_redacted") or ""))]
        conversations = {
            str(row.get("conversation_id") or _reference(row))
            for row in matched
        }
        source_files = sorted({str(row.get("source") or "unknown") for row in matched})
        v2_matches = 0
        for row in matched:
            decision = decisions[_reference(row)]
            if decision.confidence == "high" and decision.scenario:
                if theme.existing_scenario_id is None or decision.scenario.scenario_id == theme.existing_scenario_id:
                    v2_matches += 1
        if len(conversations) >= 10 and len(source_files) >= 2:
            confidence = "high"
        elif len(conversations) >= 3:
            confidence = "medium"
        else:
            confidence = "low"
        intense = sum(bool(_INTENSITY_RE.search(str(row.get("text_redacted") or ""))) for row in matched)
        intensity_rate = intense / len(matched) if matched else 0.0
        intensity = "high" if intensity_rate >= 0.3 else "medium" if intensity_rate >= 0.1 else "low"
        theme_results.append(
            {
                "theme": theme.key,
                "title": theme.title,
                "kind": theme.kind,
                "risk": theme.risk,
                "message_count": len(matched),
                "conversation_count": len(conversations),
                "candidate_share": round(len(matched) / len(candidates), 4) if candidates else 0.0,
                "source_files": source_files,
                "confidence": confidence,
                "intensity": intensity,
                "existing_scenario_id": theme.existing_scenario_id,
                "v2_high_confidence_matches": v2_matches,
                "example_refs": [_reference(row) for row in matched[:5]],
            }
        )
    ranked = sorted(theme_results, key=lambda item: item["conversation_count"], reverse=True)
    risk_multiplier = {"normal": 1.0, "account": 1.15, "financial": 1.25, "contractual": 1.25}
    intensity_multiplier = {"low": 1.0, "medium": 1.1, "high": 1.2}
    scenario_backlog = []
    for item in ranked:
        if item["kind"] != "topic" or not item["conversation_count"]:
            continue
        existing = item["existing_scenario_id"]
        match_rate = item["v2_high_confidence_matches"] / item["message_count"] if item["message_count"] else 0.0
        if item["theme"] in {"payment_general", "bid_general", "registration"}:
            recommendation = "create_clarification_tree"
        elif existing and match_rate < 0.3:
            recommendation = "expand_examples_or_split_scenario"
        elif existing:
            recommendation = "review_existing_scenario"
        else:
            recommendation = "create_scenario"
        priority_score = round(
            item["conversation_count"]
            * risk_multiplier[item["risk"]]
            * intensity_multiplier[item["intensity"]],
            1,
        )
        priority = "P0" if priority_score >= 50 else "P1" if priority_score >= 15 else "P2"
        scenario_backlog.append(
            {
                "theme": item["theme"],
                "title": item["title"],
                "priority": priority,
                "priority_score": priority_score,
                "conversation_count": item["conversation_count"],
                "risk": item["risk"],
                "requires_expert_review": item["risk"] in {"financial", "contractual"},
                "existing_scenario_id": existing,
                "recommendation": recommendation,
                "example_refs": item["example_refs"],
            }
        )
    scenario_backlog.sort(key=lambda item: item["priority_score"], reverse=True)
    covered_by_any = {
        _reference(row)
        for row in candidates
        if any(theme.pattern.search(str(row.get("text_redacted") or "")) for theme in THEMES)
    }
    return {
        "schema_version": 1,
        "publication_allowed": False,
        "source_messages": len(rows),
        "candidate_messages": len(candidates),
        "messages_covered_by_research_taxonomy": len(covered_by_any),
        "taxonomy_coverage": round(len(covered_by_any) / len(candidates), 4) if candidates else 0.0,
        "v2_high_confidence_messages": sum(
            decision.confidence == "high" and decision.scenario is not None
            for decision in decisions.values()
        ),
        "v2_high_confidence_coverage": round(
            sum(decision.confidence == "high" and decision.scenario is not None for decision in decisions.values())
            / len(candidates),
            4,
        )
        if candidates
        else 0.0,
        "methodology": {
            "themes_are_nonexclusive": True,
            "confidence": "high = 10+ conversations in 2+ export periods; medium = 3+ conversations; low = fewer than 3",
            "sample_bias": "Internal support-routing traffic overrepresents problems and callback requests; it is not a product satisfaction sample.",
            "quotes_omitted": "Only message references are exported because residual names can occur in free-form internal notes.",
        },
        "themes": ranked,
        "scenario_backlog": scenario_backlog,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MIGTORG support corpus — scenario backlog",
        "",
        f"- Source messages: {report['source_messages']}",
        f"- Candidate problem messages: {report['candidate_messages']}",
        f"- Research taxonomy coverage: {report['taxonomy_coverage']:.1%}",
        f"- Current knowledge v2 high-confidence coverage: {report['v2_high_confidence_coverage']:.1%}",
        "- Publication allowed: no",
        "",
        "The themes overlap: a won-lot request may also mention payment, documents and a callback.",
        "No verbatim quotes are included; use the local message references for controlled review.",
        "",
        "| Theme | Conversations | Share | Confidence | Intensity | Current v2 scenario | v2 matches |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for item in report["themes"]:
        scenario = item["existing_scenario_id"] or "—"
        lines.append(
            f"| {item['title']} | {item['conversation_count']} | {item['candidate_share']:.1%} | "
            f"{item['confidence']} | {item['intensity']} | {scenario} | {item['v2_high_confidence_matches']} |"
        )
    lines.extend(
        [
            "",
            "## Prioritized scenario backlog",
            "",
            "| Priority | Theme | Conversations | Recommendation | Expert review |",
            "|---|---|---:|---|---|",
        ]
    )
    for item in report["scenario_backlog"]:
        lines.append(
            f"| {item['priority']} | {item['title']} | {item['conversation_count']} | "
            f"{item['recommendation']} | {'yes' if item['requires_expert_review'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `generic_lot_question` is a data-quality signal, not an answerable scenario; the bot must ask for the lot and the exact problem.",
            "- `callback_requested` is an action signal and must be combined with the underlying topic rather than replacing it.",
            "- Financial and contractual themes require expert approval before facts or deadlines are published.",
            "- Zero occurrences do not prove that a scenario is unnecessary: this export contains routed calls, not all web-chat traffic.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze normalized MIGTORG support messages.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = analyze(load_normalized(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("source_messages", "candidate_messages", "taxonomy_coverage")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
