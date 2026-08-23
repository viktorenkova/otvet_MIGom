# Этап 2 — candidate retrieval

Дата: 23 августа 2026 года

Контур: текущая локальная папка; LLM отключён; multilingual E5 загружен локально. Frozen-корпуса использовались только как regression-gate и не переносились в словари, профили или поисковые карточки.

## Результат

Первоочередная recovery-задача, добавленная перед этапом 2, завершена до перехода к candidate retrieval.

| Корпус | Baseline до v3.1 | После этапа 1 | Финал | Изменение к baseline |
|---|---:|---:|---:|---:|
| Live 160 | 94,38% | 92,50% | 94,38% | 0,00 п. п. |
| Adjudicated closed 270 | 80,37% | 78,52% | 80,37% | 0,00 п. п. |

Потеря этапа 1 полностью восстановлена: +1,88 п. п. на live и +1,85 п. п. на closed. Transport и forbidden-content gate равны 100% на обоих финальных прогонах; live dialogue quality — 100%, confident wrong — 1. Прежние 95,62%/81,48% были получены из dirty working tree и после унификации нормализатора не воспроизвелись. В отчёте оставлены только метрики повторного прогона текущего кода; отдельные frozen-фразы не переносились в правила для возврата прежних значений.

## Candidate retrieval API

`retrieve_knowledge_candidates()` возвращает Top-K кандидатов и не выбирает финальный сценарий. Для каждого кандидата доступны:

- итоговый candidate score;
- lexical score;
- отдельные char и word scores;
- normalized dense score и исходная dense similarity;
- intent boost;
- признак фактической доступности dense-канала.

Route, intent, knowledge search и scenario engine используют единый `normalize_matching_text`: нормализацию регистра/символов, восстановление неверной раскладки и русского транслита, исправление доменных опечаток и безопасные сленговые алиасы. Широкая синонимизация и лёгкий русский stemming применяются только как признаки retrieval/pattern matching и не подменяют канонический смысл сообщения. Для offline evaluation query embeddings кодируются пакетно и кэшируются по model/dataset fingerprint.

Candidate fusion отделён от answer fallback. Для Recall@10 используются веса lexical 0,75 / dense 0,25; прежние runtime-веса answer fallback 0,30 / 0,70 не изменены.

## Gates

| Набор | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| Development 372 | 90,86% | 96,77% | 98,66% |
| Validation 99 | 91,92% | 100,00% | 100,00% |
| Independent 116 | — | — | 100,00% |
| Language validation 30 | — | — | 100,00% |

Срезы development/validation:

- taxonomy paraphrase Recall@10 — 99,36%;
- transposed letters — 98,73%;
- word order — 98,73%;
- минимальная крупная intent-группа — tariffs, 92,59%;
- dense доступен для всех 471 запросов.

Independent 116:

- typo — 29/29;
- transposed letters — 29/29;
- morphology — 29/29;
- word order — 29/29;
- каждая тематическая группа имеет Recall@10 не ниже 90%.

Отдельный language validation без точных пересечений с frozen-корпусами:

- неверная раскладка — 10/10;
- русский транслит — 10/10;
- разговорные формы и сленг — 10/10;
- dense доступен для всех 30 запросов.

## Артефакты

- `reports/quality-recovery-v31-live-160.json`;
- `reports/quality-recovery-v31-closed-270.json`;
- `reports/semantic-retrieval-v31-development-validation.json`;
- `reports/candidate-retrieval-v31-independent-116.json`;
- `reports/stage2-language-validation.json`;
- `backend/tools/evaluate_semantic_retrieval_v31.py`;
- `backend/tools/evaluate_candidate_retrieval_blind.py`.
- `backend/tools/evaluate_stage2_gates.py`.

## Следующий этап

Этап 3 может начинать hard-negative reranker и confidence calibration. Candidate retrieval остаётся recall-oriented и не получает права принимать финальное решение.
