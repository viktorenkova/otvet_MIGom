from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Iterator

from backend.app.models.safety import SafetyEvent
from backend.app.models.llm import LLMResult
from backend.app.models.ticket import Ticket
from backend.app.bot.review_queue import build_review_queue


class DialogLogger:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self._database_lock = RLock()
        self._clarification_cache: dict[str, dict] = {}
        path = Path(database_path)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
        self._warm_clarification_cache()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._database_lock:
            conn = sqlite3.connect(self.database_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("PRAGMA synchronous = NORMAL")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS dialogue_states (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dialogue_leases (
                    session_id TEXT PRIMARY KEY,
                    token TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS response_states (
                    session_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    actions_json TEXT NOT NULL DEFAULT '[]',
                    response_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decision_traces (
                    message_id TEXT PRIMARY KEY,
                    trace_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm_budget_reservations (
                    id TEXT PRIMARY KEY,
                    amount REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dialog_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    intent TEXT,
                    answer TEXT,
                    needs_ticket INTEGER NOT NULL DEFAULT 0,
                    ticket_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    description TEXT NOT NULL,
                    contact TEXT,
                    role TEXT NOT NULL,
                    user_id TEXT,
                    lot_id TEXT,
                    payment_id TEXT,
                    session_id TEXT NOT NULL,
                    page_type TEXT,
                    dialog_history TEXT NOT NULL,
                    attachments TEXT NOT NULL,
                    category TEXT,
                    priority TEXT NOT NULL DEFAULT 'normal',
                    scenario_id TEXT,
                    source_message_id TEXT,
                    collected_fields TEXT NOT NULL DEFAULT '{}',
                    delivery_attempts INTEGER NOT NULL DEFAULT 0,
                    next_delivery_attempt_at TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS safety_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    message TEXT NOT NULL,
                    category TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    needs_review INTEGER NOT NULL,
                    ticket_created INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    rating INTEGER,
                    comment TEXT,
                    message_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    escalation_required INTEGER NOT NULL,
                    safety_flags TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error TEXT,
                    environment TEXT NOT NULL DEFAULT 'dev',
                    verification_accepted INTEGER,
                    verification_reason TEXT NOT NULL DEFAULT '',
                    fallback_used INTEGER NOT NULL DEFAULT 0,
                    correlation_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS matching_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    original_message TEXT NOT NULL,
                    normalized_message TEXT NOT NULL,
                    corrected_message TEXT NOT NULL,
                    detected_entities TEXT NOT NULL,
                    matched_intent TEXT,
                    score INTEGER NOT NULL,
                    confidence TEXT NOT NULL,
                    matched_features TEXT NOT NULL,
                    fallback_reason TEXT,
                    query_facets TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quality_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matching_event_id INTEGER,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    page_type TEXT,
                    intent TEXT NOT NULL,
                    article_id TEXT,
                    score INTEGER NOT NULL,
                    confidence TEXT NOT NULL,
                    action TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    needs_ticket INTEGER NOT NULL,
                    ticket_created INTEGER NOT NULL,
                    fallback_reason TEXT,
                    safety_categories TEXT NOT NULL,
                    scenario_id TEXT,
                    resolution TEXT NOT NULL DEFAULT 'answered',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (matching_event_id) REFERENCES matching_events(id)
                );
                CREATE TABLE IF NOT EXISTS clarification_states (
                    session_id TEXT PRIMARY KEY,
                    options TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_quality_events_created_at
                    ON quality_events(created_at);
                CREATE INDEX IF NOT EXISTS idx_quality_events_session_id
                    ON quality_events(session_id);
                CREATE INDEX IF NOT EXISTS idx_quality_events_confidence_action
                    ON quality_events(confidence, action);
                CREATE INDEX IF NOT EXISTS idx_clarification_states_expires_at
                    ON clarification_states(expires_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_message_id
                    ON feedback(message_id);
                """
            )
            ticket_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(tickets)").fetchall()
            }
            ticket_migrations = {
                "category": "TEXT",
                "priority": "TEXT NOT NULL DEFAULT 'normal'",
                "scenario_id": "TEXT",
                "source_message_id": "TEXT",
                "collected_fields": "TEXT NOT NULL DEFAULT '{}'",
                "delivery_attempts": "INTEGER NOT NULL DEFAULT 0",
                "next_delivery_attempt_at": "TEXT",
            }
            for column, definition in ticket_migrations.items():
                if column not in ticket_columns:
                    conn.execute(f"ALTER TABLE tickets ADD COLUMN {column} {definition}")
            matching_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(matching_events)").fetchall()
            }
            if "query_facets" not in matching_columns:
                conn.execute("ALTER TABLE matching_events ADD COLUMN query_facets TEXT NOT NULL DEFAULT '{}'")
            quality_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(quality_events)").fetchall()
            }
            if "scenario_id" not in quality_columns:
                conn.execute("ALTER TABLE quality_events ADD COLUMN scenario_id TEXT")
            if "resolution" not in quality_columns:
                conn.execute("ALTER TABLE quality_events ADD COLUMN resolution TEXT NOT NULL DEFAULT 'answered'")
            llm_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(llm_requests)").fetchall()
            }
            llm_migrations = {
                "environment": "TEXT NOT NULL DEFAULT 'dev'",
                "verification_accepted": "INTEGER",
                "verification_reason": "TEXT NOT NULL DEFAULT ''",
                "fallback_used": "INTEGER NOT NULL DEFAULT 0",
                "correlation_id": "TEXT NOT NULL DEFAULT ''",
            }
            for column, definition in llm_migrations.items():
                if column not in llm_columns:
                    conn.execute(f"ALTER TABLE llm_requests ADD COLUMN {column} {definition}")
            clarification_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(clarification_states)").fetchall()
            }
            if "original_message" not in clarification_columns:
                conn.execute(
                    "ALTER TABLE clarification_states ADD COLUMN original_message TEXT NOT NULL DEFAULT ''"
                )
            if "context_json" not in clarification_columns:
                conn.execute(
                    "ALTER TABLE clarification_states ADD COLUMN context_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "attempts" not in clarification_columns:
                conn.execute(
                    "ALTER TABLE clarification_states ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
                )

    @staticmethod
    def _clarification_state_from_row(row: sqlite3.Row) -> dict | None:
        try:
            options = json.loads(str(row["options"]))
            context = json.loads(str(row["context_json"] or "{}"))
        except json.JSONDecodeError:
            return None
        if not isinstance(options, list):
            return None
        return {
            "options": [dict(option) for option in options if isinstance(option, dict)],
            "original_message": str(row["original_message"] or ""),
            "context": dict(context) if isinstance(context, dict) else {},
            "attempts": int(row["attempts"] or 0),
            "expires_at": str(row["expires_at"]),
        }

    def _warm_clarification_cache(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM clarification_states WHERE expires_at < ?", (now,))
            rows = conn.execute(
                """
                SELECT session_id, options, original_message, context_json, attempts, expires_at
                FROM clarification_states
                """
            ).fetchall()
        with self._database_lock:
            self._clarification_cache = {
                str(row["session_id"]): state
                for row in rows
                if (state := self._clarification_state_from_row(row)) is not None
            }

    def log_dialog(
        self,
        session_id: str,
        role: str,
        message: str,
        intent: str,
        answer: str,
        needs_ticket: bool,
        ticket_id: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dialog_messages
                (session_id, role, message, intent, answer, needs_ticket, ticket_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    message,
                    intent,
                    answer,
                    int(needs_ticket),
                    ticket_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def log_matching_event(
        self,
        session_id: str,
        original_message: str,
        normalized_message: str,
        corrected_message: str,
        detected_entities: dict,
        matched_intent: str,
        score: int,
        confidence: str,
        matched_features: list[str],
        fallback_reason: str = "",
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO matching_events
                (session_id, original_message, normalized_message, corrected_message, detected_entities,
                 matched_intent, score, confidence, matched_features, fallback_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    original_message,
                    normalized_message,
                    corrected_message,
                    json.dumps(detected_entities, ensure_ascii=False),
                    matched_intent,
                    int(score),
                    confidence,
                    json.dumps(matched_features, ensure_ascii=False),
                    fallback_reason,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def log_turn(
        self,
        *,
        session_id: str,
        role: str,
        original_message: str,
        normalized_message: str,
        corrected_message: str,
        detected_entities: dict,
        intent: str,
        answer: str,
        article_id: str | None,
        score: int,
        confidence: str,
        matched_features: list[str],
        action: str,
        latency_ms: int,
        needs_ticket: bool,
        ticket_id: str | None,
        ticket_created: bool,
        page_type: str = "",
        fallback_reason: str = "",
        safety_categories: list[str] | None = None,
        scenario_id: str | None = None,
        resolution: str = "answered",
        query_facets: dict | None = None,
    ) -> tuple[int, int]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            matching_cursor = conn.execute(
                """
                INSERT INTO matching_events
                (session_id, original_message, normalized_message, corrected_message, detected_entities,
                 matched_intent, score, confidence, matched_features, fallback_reason, query_facets, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    original_message,
                    normalized_message,
                    corrected_message,
                    json.dumps(detected_entities, ensure_ascii=False),
                    intent,
                    int(score),
                    confidence,
                    json.dumps(matched_features, ensure_ascii=False),
                    fallback_reason,
                    json.dumps(query_facets or {}, ensure_ascii=False),
                    now,
                ),
            )
            matching_event_id = int(matching_cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO dialog_messages
                (session_id, role, message, intent, answer, needs_ticket, ticket_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    original_message,
                    intent,
                    answer,
                    int(needs_ticket),
                    ticket_id,
                    now,
                ),
            )
            quality_cursor = conn.execute(
                """
                INSERT INTO quality_events
                (matching_event_id, session_id, role, page_type, intent, article_id, score,
                 confidence, action, latency_ms, needs_ticket, ticket_created, fallback_reason,
                 safety_categories, scenario_id, resolution, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    matching_event_id,
                    session_id,
                    role,
                    page_type,
                    intent,
                    article_id,
                    int(score),
                    confidence,
                    action,
                    max(0, int(latency_ms)),
                    int(needs_ticket),
                    int(ticket_created),
                    fallback_reason,
                    json.dumps(safety_categories or [], ensure_ascii=False),
                    scenario_id,
                    resolution,
                    now,
                ),
            )
            return matching_event_id, int(quality_cursor.lastrowid)

    def get_history(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, message, intent, answer, needs_ticket, ticket_id, created_at
                FROM dialog_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_pending_clarification(
        self,
        session_id: str,
        options: list[dict[str, str]],
        ttl_minutes: int = 30,
        *,
        original_message: str = "",
        context: dict | None = None,
        attempts: int = 0,
    ) -> None:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=max(1, int(ttl_minutes)))
        state = {
            "options": [dict(option) for option in options],
            "original_message": original_message,
            "context": dict(context or {}),
            "attempts": max(0, int(attempts)),
            "expires_at": expires_at.isoformat(),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO clarification_states
                (session_id, options, original_message, context_json, attempts, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    json.dumps(options, ensure_ascii=False),
                    original_message,
                    json.dumps(state["context"], ensure_ascii=False),
                    state["attempts"],
                    now.isoformat(),
                    state["expires_at"],
                ),
            )
        with self._database_lock:
            self._clarification_cache[session_id] = state

    def get_pending_clarification_state(self, session_id: str) -> dict | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM clarification_states WHERE session_id = ? AND expires_at >= ?",
                (session_id, now),
            ).fetchone()
        if not row:
            return None
        return {"options": json.loads(row["options"]),
                "original_message": row["original_message"],
                "context": json.loads(row["context_json"]),
                "attempts": row["attempts"], "expires_at": row["expires_at"]}

    def get_pending_clarification(self, session_id: str) -> list[dict[str, str]]:
        state = self.get_pending_clarification_state(session_id)
        return list(state["options"]) if state else []

    def clear_pending_clarification(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM clarification_states WHERE session_id = ?", (session_id,))
        with self._database_lock:
            self._clarification_cache.pop(session_id, None)

    def consume_clarification_choice(self, session_id: str, message: str) -> dict[str, str] | None:
        state = self.get_pending_clarification_state(session_id)
        if not state:
            return None
        options = list(state["options"])
        normalized_message = " ".join(message.casefold().split())
        selected = next(
            (
                option
                for option in options
                if " ".join(str(option.get("label", "")).casefold().split()) == normalized_message
            ),
            None,
        )
        if selected:
            self.clear_pending_clarification(session_id)
        return selected

    @staticmethod
    def _ticket_date_part(created_at: datetime) -> str:
        return created_at.astimezone().strftime("%d.%m.%y")

    def _assign_ticket_id(self, conn: sqlite3.Connection, ticket: Ticket) -> None:
        if ticket.id.strip():
            return

        date_part = self._ticket_date_part(ticket.created_at)
        rows = conn.execute("SELECT id FROM tickets WHERE id LIKE ?", (f"{date_part}-%",)).fetchall()
        max_sequence = 0
        for row in rows:
            existing_id = str(row["id"])
            existing_date, separator, sequence_part = existing_id.partition("-")
            if separator and existing_date == date_part and sequence_part.isdigit():
                max_sequence = max(max_sequence, int(sequence_part))

        sequence = max_sequence + 1
        while True:
            candidate = f"{date_part}-{sequence:04d}"
            exists = conn.execute("SELECT 1 FROM tickets WHERE id = ?", (candidate,)).fetchone()
            if not exists:
                ticket.id = candidate
                return
            sequence += 1

    def save_ticket(self, ticket: Ticket) -> Ticket:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assign_ticket_id(conn, ticket)
            conn.execute(
                """
                INSERT OR REPLACE INTO tickets
                (id, status, topic, description, contact, role, user_id, lot_id, payment_id,
                 session_id, page_type, dialog_history, attachments, category, priority,
                 scenario_id, source_message_id, collected_fields, delivery_attempts,
                 next_delivery_attempt_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket.id,
                    ticket.status,
                    ticket.topic,
                    ticket.description,
                    ticket.contact,
                    ticket.role,
                    ticket.user_id,
                    ticket.lot_id,
                    ticket.payment_id,
                    ticket.session_id,
                    ticket.page_type,
                    json.dumps(ticket.dialog_history, ensure_ascii=False),
                    json.dumps(ticket.attachments, ensure_ascii=False),
                    ticket.category,
                    ticket.priority,
                    ticket.scenario_id,
                    ticket.source_message_id,
                    json.dumps(ticket.collected_fields, ensure_ascii=False),
                    ticket.delivery_attempts,
                    ticket.next_delivery_attempt_at.isoformat() if ticket.next_delivery_attempt_at else None,
                    ticket.created_at.isoformat(),
                ),
            )
        return ticket

    def update_ticket_status(self, ticket_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id))

    def record_ticket_delivery_failure(self, ticket_id: str) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT delivery_attempts FROM tickets WHERE id = ?",
                (ticket_id,),
            ).fetchone()
            attempts = int(row["delivery_attempts"] or 0) + 1 if row else 1
            delay_minutes = min(60, 5 ** min(attempts - 1, 2))
            next_attempt = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)
            conn.execute(
                """
                UPDATE tickets
                SET status = 'delivery_failed', delivery_attempts = ?, next_delivery_attempt_at = ?
                WHERE id = ?
                """,
                (attempts, next_attempt.isoformat(), ticket_id),
            )

    def get_due_delivery_tickets(self, limit: int = 20) -> list[dict]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tickets
                WHERE status = 'delivery_failed'
                  AND delivery_attempts < 5
                  AND COALESCE(next_delivery_attempt_at, '') <= ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (now, min(max(int(limit), 1), 100)),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_ticket(self, ticket_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return dict(row) if row else None

    def log_safety_event(self, event: SafetyEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO safety_events
                (session_id, user_id, message, category, answer, created_at, needs_review, ticket_created)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.session_id,
                    event.user_id,
                    event.message,
                    event.category,
                    event.answer,
                    event.created_at.isoformat(),
                    int(event.needs_review),
                    int(event.ticket_created),
                ),
            )

    def save_feedback(self, session_id: str, rating: int | None, comment: str | None, message_id: str | None) -> None:
        with self._connect() as conn:
            if not message_id:
                latest = conn.execute(
                    "SELECT id FROM quality_events WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                message_id = str(latest["id"]) if latest else None
            conn.execute(
                """
                INSERT INTO feedback (session_id, rating, comment, message_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, rating, comment, message_id, datetime.now(timezone.utc).isoformat()),
            )

    def log_llm_request(
        self,
        result: LLMResult,
        session_id: str,
        user_role: str,
        escalation_required: bool,
        safety_flags: list[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_requests
                (provider, model, task_type, input_tokens, output_tokens, total_tokens,
                 estimated_cost_usd, latency_ms, session_id, user_role, escalation_required,
                 safety_flags, success, error, environment, verification_accepted,
                 verification_reason, fallback_used, correlation_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.provider,
                    result.model,
                    result.task_type,
                    result.input_tokens,
                    result.output_tokens,
                    result.total_tokens,
                    result.estimated_cost_usd,
                    result.latency_ms,
                    session_id,
                    user_role,
                    int(escalation_required),
                    json.dumps(safety_flags, ensure_ascii=False),
                    int(result.success),
                    result.error,
                    result.environment,
                    None if result.verification_accepted is None else int(result.verification_accepted),
                    result.verification_reason,
                    int(result.fallback_used),
                    result.correlation_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_llm_spend(self, environment: str | None = None, days: int | None = None) -> float:
        clauses: list[str] = []
        values: list[str] = []
        if environment:
            clauses.append("environment = ?")
            values.append(environment)
        if days is not None:
            since = (datetime.now(timezone.utc) - timedelta(days=max(0, days))).isoformat()
            clauses.append("created_at >= ?")
            values.append(since)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(estimated_cost_usd), 0) AS spend FROM llm_requests" + where,
                values,
            ).fetchone()
        return float(row["spend"] or 0)

    def get_llm_metrics(self, days: int = 7) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(days=max(1, days))).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT provider, model, estimated_cost_usd, latency_ms, success,
                       verification_accepted, verification_reason, fallback_used
                FROM llm_requests
                WHERE created_at >= ?
                """,
                (since,),
            ).fetchall()
        events = [dict(row) for row in rows]
        latencies = [int(row["latency_ms"] or 0) for row in events]
        reasons = Counter(str(row["verification_reason"] or "not_recorded") for row in events)
        return {
            "requests": len(events),
            "success": sum(int(row["success"] or 0) for row in events),
            "accepted": sum(int(row["verification_accepted"] or 0) for row in events),
            "rejected": sum(row["verification_accepted"] == 0 for row in events),
            "fallback_used": sum(int(row["fallback_used"] or 0) for row in events),
            "estimated_cost_usd": round(sum(float(row["estimated_cost_usd"] or 0) for row in events), 6),
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "verification_reasons": dict(reasons.most_common(10)),
            "models": dict(Counter(f"{row['provider']}:{row['model']}" for row in events).most_common()),
        }

    def get_last_llm_request(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM llm_requests ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def log_quality_event(
        self,
        *,
        matching_event_id: int | None,
        session_id: str,
        role: str,
        page_type: str,
        intent: str,
        article_id: str | None,
        score: int,
        confidence: str,
        action: str,
        latency_ms: int,
        needs_ticket: bool,
        ticket_created: bool,
        fallback_reason: str = "",
        safety_categories: list[str] | None = None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO quality_events
                (matching_event_id, session_id, role, page_type, intent, article_id, score,
                 confidence, action, latency_ms, needs_ticket, ticket_created, fallback_reason,
                 safety_categories, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    matching_event_id,
                    session_id,
                    role,
                    page_type,
                    intent,
                    article_id,
                    int(score),
                    confidence,
                    action,
                    max(0, int(latency_ms)),
                    int(needs_ticket),
                    int(ticket_created),
                    fallback_reason,
                    json.dumps(safety_categories or [], ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def get_last_quality_event(self, session_id: str | None = None) -> dict | None:
        query = "SELECT * FROM quality_events"
        params: tuple[str, ...] = ()
        if session_id:
            query += " WHERE session_id = ?"
            params = (session_id,)
        query += " ORDER BY id DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    def mark_ticket_created(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE quality_events
                SET ticket_created = 1
                WHERE id = (
                    SELECT id
                    FROM quality_events
                    WHERE session_id = ? AND needs_ticket = 1
                    ORDER BY id DESC
                    LIMIT 1
                )
                """,
                (session_id,),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _percentile(values: list[int], percentile: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return int(ordered[index])

    @staticmethod
    def _rate(part: int, total: int) -> float:
        return round(part * 100 / total, 1) if total else 0.0

    @staticmethod
    def _redact_message(message: str) -> str:
        redacted = re.sub(
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            "[email]",
            message,
        )
        redacted = re.sub(
            r"(?:\+7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}",
            "[телефон]",
            redacted,
        )
        redacted = re.sub(r"\b\d{7,}\b", "[номер]", redacted)
        return re.sub(r"\s+", " ", redacted).strip()[:240]

    def get_review_queue(
        self,
        days: int = 30,
        *,
        include_dev_sessions: bool = False,
    ) -> list[dict]:
        days = min(max(int(days), 1), 3650)
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT q.id, q.session_id, q.intent, q.article_id, q.score, q.confidence,
                       q.action, q.fallback_reason, q.created_at, m.original_message,
                       m.matched_features
                FROM quality_events q
                LEFT JOIN matching_events m ON m.id = q.matching_event_id
                WHERE q.created_at >= ?
                ORDER BY q.id ASC
                """,
                (since,),
            ).fetchall()
            feedback_rows = conn.execute(
                "SELECT session_id, rating FROM feedback WHERE created_at >= ? AND rating IS NOT NULL",
                (since,),
            ).fetchall()

        events = []
        for row in rows:
            event = dict(row)
            event["redacted_message"] = self._redact_message(str(event.get("original_message") or ""))
            events.append(event)
        feedback_by_session: dict[str, list[int]] = {}
        for row in feedback_rows:
            feedback_by_session.setdefault(str(row["session_id"]), []).append(int(row["rating"]))
        return build_review_queue(
            events,
            feedback_by_session,
            include_dev_sessions=include_dev_sessions,
        )

    def get_quality_report(
        self,
        days: int = 30,
        *,
        include_examples: bool = False,
        example_limit: int = 10,
    ) -> dict:
        days = min(max(int(days), 1), 3650)
        example_limit = min(max(int(example_limit), 0), 100)
        generated_at = datetime.now(timezone.utc)
        since = (generated_at - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT q.*, m.matched_features
                FROM quality_events q
                LEFT JOIN matching_events m ON m.id = q.matching_event_id
                WHERE q.created_at >= ?
                ORDER BY q.id ASC
                """,
                (since,),
            ).fetchall()
            feedback_rows = conn.execute(
                "SELECT session_id, rating, message_id FROM feedback WHERE created_at >= ?",
                (since,),
            ).fetchall()
            example_rows = []
            if include_examples and example_limit:
                example_rows = conn.execute(
                    """
                    SELECT m.original_message, q.intent, q.confidence, q.fallback_reason, q.created_at
                    FROM quality_events q
                    JOIN matching_events m ON m.id = q.matching_event_id
                    WHERE q.created_at >= ?
                      AND q.action <> 'safety_refusal'
                      AND (
                        q.action = 'clarify'
                        OR q.confidence IN ('low', 'medium')
                        OR COALESCE(q.fallback_reason, '') <> ''
                      )
                    ORDER BY q.id DESC
                    LIMIT ?
                    """,
                    (since, example_limit * 3),
                ).fetchall()

        events = [dict(row) for row in rows]
        total = len(events)
        latencies = [int(event["latency_ms"]) for event in events]
        confidences = Counter(str(event["confidence"]) for event in events)
        actions = Counter(str(event["action"]) for event in events)
        intents = Counter(str(event["intent"]) for event in events)
        scenarios = Counter(
            str(event["scenario_id"])
            for event in events
            if str(event.get("scenario_id") or "")
        )
        resolutions = Counter(str(event.get("resolution") or "answered") for event in events)
        articles = Counter(
            str(event["article_id"])
            for event in events
            if str(event.get("article_id") or "")
        )
        fallbacks = Counter(
            str(event["fallback_reason"])
            for event in events
            if str(event["fallback_reason"] or "")
        )
        safety_categories: Counter[str] = Counter()
        requests_with_safety = 0
        semantic_used = 0
        semantic_overrides = 0
        semantic_clarifications = 0
        semantic_confirmations = 0
        clarification_choices = 0
        clarification_other = 0
        for event in events:
            try:
                categories = json.loads(str(event["safety_categories"] or "[]"))
            except json.JSONDecodeError:
                categories = []
            if categories:
                requests_with_safety += 1
                safety_categories.update(str(category) for category in categories)
            try:
                matching_features = json.loads(str(event["matched_features"] or "[]"))
            except json.JSONDecodeError:
                matching_features = []
            if any(str(feature).startswith(("semantic_tfidf:", "semantic_hybrid:")) for feature in matching_features):
                semantic_used += 1
            semantic_overrides += int("semantic_override" in matching_features)
            semantic_clarifications += int("semantic_clarification" in matching_features)
            semantic_confirmations += int("semantic_confirms_rule" in matching_features)
            clarification_choices += int("clarification_choice" in matching_features)
            clarification_other += int("clarification_choice:other" in matching_features)

        ticket_offers = sum(int(event["needs_ticket"]) for event in events)
        tickets_created = sum(int(event["ticket_created"]) for event in events)
        offered_sessions = {
            str(event["session_id"])
            for event in events
            if int(event["needs_ticket"])
        }
        ratings = [int(row["rating"]) for row in feedback_rows if row["rating"] is not None]
        linked_feedback = sum(1 for row in feedback_rows if str(row["message_id"] or ""))
        negative_feedback_sessions = {
            str(row["session_id"])
            for row in feedback_rows
            if row["rating"] is not None and int(row["rating"]) <= 2
        }
        clarified = actions.get("clarify", 0)
        low_or_medium = confidences.get("low", 0) + confidences.get("medium", 0)

        report = {
            "generated_at": generated_at.isoformat(),
            "period_days": days,
            "requests": {
                "total": total,
                "unique_sessions": len({str(event["session_id"]) for event in events}),
                "with_article": sum(1 for event in events if event["article_id"]),
                "with_article_rate": self._rate(sum(1 for event in events if event["article_id"]), total),
            },
            "latency": {
                "average_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
                "p50_ms": self._percentile(latencies, 0.50),
                "p95_ms": self._percentile(latencies, 0.95),
                "maximum_ms": max(latencies, default=0),
                "over_1000_ms": sum(1 for latency in latencies if latency > 1000),
                "over_1000_ms_rate": self._rate(sum(1 for latency in latencies if latency > 1000), total),
            },
            "confidence": {
                "high": confidences.get("high", 0),
                "medium": confidences.get("medium", 0),
                "low": confidences.get("low", 0),
                "high_rate": self._rate(confidences.get("high", 0), total),
                "needs_attention": low_or_medium,
                "needs_attention_rate": self._rate(low_or_medium, total),
            },
            "outcomes": {
                "actions": dict(actions.most_common()),
                "resolutions": dict(resolutions.most_common()),
                "clarifications": clarified,
                "clarification_rate": self._rate(clarified, total),
                "clarification_choices": clarification_choices,
                "clarification_other": clarification_other,
                "clarification_resolution_rate": self._rate(clarification_choices, clarified),
            },
            "tickets": {
                "offered": ticket_offers,
                "created": tickets_created,
                "creation_rate_from_offers": self._rate(tickets_created, ticket_offers),
                "negative_feedback_alerts": len(offered_sessions & negative_feedback_sessions),
            },
            "safety": {
                "requests": requests_with_safety,
                "request_rate": self._rate(requests_with_safety, total),
                "categories": dict(safety_categories.most_common()),
            },
            "semantic": {
                "used": semantic_used,
                "usage_rate": self._rate(semantic_used, total),
                "overrides": semantic_overrides,
                "clarifications": semantic_clarifications,
                "rule_confirmations": semantic_confirmations,
            },
            "feedback": {
                "ratings": len(ratings),
                "linked_to_message": linked_feedback,
                "linked_to_message_rate": self._rate(linked_feedback, len(feedback_rows)),
                "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
                "positive": sum(1 for rating in ratings if rating >= 4),
                "negative": sum(1 for rating in ratings if rating <= 2),
            },
            "top_intents": [
                {"intent": intent, "count": count}
                for intent, count in intents.most_common(10)
            ],
            "top_scenarios": [
                {"scenario_id": scenario_id, "count": count}
                for scenario_id, count in scenarios.most_common(15)
            ],
            "article_attractors": [
                {
                    "article_id": article_id,
                    "count": count,
                    "share": self._rate(count, sum(articles.values())),
                }
                for article_id, count in articles.most_common(15)
            ],
            "fallback_reasons": [
                {"reason": reason, "count": count}
                for reason, count in fallbacks.most_common(10)
            ],
            "llm": self.get_llm_metrics(days),
        }

        if include_examples:
            examples = []
            seen: set[str] = set()
            for row in example_rows:
                text = self._redact_message(str(row["original_message"]))
                if not text or text in seen:
                    continue
                seen.add(text)
                examples.append(
                    {
                        "text": text,
                        "intent": str(row["intent"]),
                        "confidence": str(row["confidence"]),
                        "fallback_reason": str(row["fallback_reason"] or ""),
                        "created_at": str(row["created_at"]),
                    }
                )
                if len(examples) >= example_limit:
                    break
            report["problem_examples"] = examples

        return report

    def get_response_state(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM response_states WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return None
        return {**dict(row), "actions": json.loads(row["actions_json"]),
                "response": json.loads(row["response_json"])}

    def save_response_state(self, response, trace: dict, dialogue_turn=None, lease_token=None) -> int:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if dialogue_turn is not None:
                lease = conn.execute("SELECT token, expires_at FROM dialogue_leases WHERE session_id = ?", (response.session_id,)).fetchone()
                if not lease or lease["token"] != lease_token or lease["expires_at"] < datetime.now(timezone.utc).isoformat():
                    raise RuntimeError("dialogue_lease_lost")
            turn = conn.execute(
                "SELECT q.created_at, m.matched_features FROM quality_events q "
                "JOIN matching_events m ON m.id = q.matching_event_id WHERE q.id = ? AND q.session_id = ?",
                (response.message_id, response.session_id)).fetchone()
            if turn:
                features = json.loads(turn["matched_features"])
                trace["answer_provenance"] = [f for f in features if f.startswith(("answer_fact:", "answer_verifier:"))]
                trace.setdefault("decision", {})["pipeline_features"] = features
                conn.execute("UPDATE dialog_messages SET answer = ? WHERE session_id = ? AND created_at = ?",
                             (response.answer, response.session_id, turn["created_at"]))
                conn.execute("UPDATE quality_events SET resolution = ?, action = ?, confidence = ?, latency_ms = ? WHERE id = ?",
                             (response.resolution, response.action, response.confidence_level,
                              int(trace.get("elapsed_ms", 0)), response.message_id))
            row = conn.execute("SELECT version FROM response_states WHERE session_id = ?", (response.session_id,)).fetchone()
            version = int(row[0]) + 1 if row else 1
            response.state_version = version
            if dialogue_turn is not None:
                from backend.app.bot.dialogue_understanding import finish_turn
                pending = conn.execute("SELECT options FROM clarification_states WHERE session_id = ?", (response.session_id,)).fetchone()
                state = finish_turn(dialogue_turn, response, {"options": json.loads(pending[0])} if pending else None, trace)
                state.version = version
                trace["dialogue_state"] = {"version": version, "status": state.status, "expected_field": state.expected_field}
                conn.execute("INSERT OR REPLACE INTO dialogue_states VALUES (?, ?)", (response.session_id, state.model_dump_json()))
            conn.execute(
                "INSERT OR REPLACE INTO response_states VALUES (?, ?, ?, ?, ?, ?)",
                (response.session_id, response.message_id, version,
                 json.dumps([a.model_dump() for a in response.actions], ensure_ascii=False),
                 response.model_dump_json(), datetime.now(timezone.utc).isoformat()))
            conn.execute("INSERT OR REPLACE INTO decision_traces VALUES (?, ?)",
                         (response.message_id, json.dumps(trace, ensure_ascii=False)))
        return version

    def reserve_llm_budget(self, amount: float, daily_limit: float, monthly_limit: float) -> str | None:
        from uuid import uuid4
        now = datetime.now(timezone.utc)
        if amount <= 0:
            return None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM llm_budget_reservations WHERE expires_at < ?", (now.isoformat(),))
            held = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM llm_budget_reservations").fetchone()[0]
            for days, limit in ((1, daily_limit), (31, monthly_limit)):
                spent = conn.execute("SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM llm_requests WHERE created_at >= ?",
                                     ((now - timedelta(days=days)).isoformat(),)).fetchone()[0]
                if spent + held + amount > limit:
                    return None
            reservation = str(uuid4())
            conn.execute("INSERT INTO llm_budget_reservations VALUES (?, ?, ?, ?)",
                         (reservation, amount, now.isoformat(), (now + timedelta(minutes=5)).isoformat()))
        return reservation

    def release_llm_budget(self, reservation: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM llm_budget_reservations WHERE id = ?", (reservation,))

    def load_dialogue_state(self, session_id: str):
        from backend.app.models.dialogue import DialogueState
        with self._connect() as conn:
            row = conn.execute("SELECT state_json FROM dialogue_states WHERE session_id = ?", (session_id,)).fetchone()
        return DialogueState.model_validate_json(row[0]) if row else DialogueState()

    def acquire_dialogue_turn(self, session_id: str) -> str | None:
        from uuid import uuid4
        now = datetime.now(timezone.utc)
        token = str(uuid4())
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT expires_at FROM dialogue_leases WHERE session_id = ?", (session_id,)).fetchone()
            if row and row[0] > now.isoformat():
                return None
            conn.execute("INSERT OR REPLACE INTO dialogue_leases VALUES (?, ?, ?)",
                         (session_id, token, (now + timedelta(seconds=120)).isoformat()))
        return token

    def release_dialogue_turn(self, session_id: str, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM dialogue_leases WHERE session_id = ? AND token = ?", (session_id, token))
