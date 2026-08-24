# MIGTORG knowledge v2

`scenarios.json` is a compatibility artifact published from
`knowledge/MASTER_KNOWLEDGE.md`. It complements the legacy normalized knowledge
base; it does not delete legacy records. Do not edit it manually.

Each active scenario must contain approved facts, separate user-facing answers,
positive and negative examples, structured actions, escalation fields, source,
owner, expert, and review metadata. Candidate clusters produced by
`backend.tools.knowledge_pipeline` are never loaded by the bot and cannot be
published without review.

Release workflow:

1. Build an anonymized backlog from support traffic.
2. Review candidate clusters and update the canonical JSON blocks in
   `knowledge/MASTER_KNOWLEDGE.md`.
3. Run `python -m backend.tools.master_knowledge render`.
4. Run `python -m backend.tools.master_knowledge publish`.
5. Run `python -m backend.tools.master_knowledge validate`.
6. Run `python -m backend.tools.audit_knowledge --strict`.
7. Run `python -m backend.tools.evaluate_scenarios tests/data/scenario_gold.jsonl --gate`.
8. Run the full test suite and deploy behind `KNOWLEDGE_V2_ENABLED` or shadow mode.

Risk-sensitive financial, contractual, refusal, penalty, and refund scenarios
must have a named expert and a non-expired review date before release.

`review_queue.json` contains support-backed candidates that are intentionally
not loaded by the bot. A record may move to `scenarios.json` only after every
publication blocker is resolved and the named expert has approved the exact
facts, wording, required fields, and escalation rules.
