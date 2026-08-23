# Этап 2 — candidate retrieval

Дата: 23 августа 2026 года

Контур: текущая локальная папка; LLM отключён; multilingual E5 загружен локально. Frozen-корпуса использовались только как regression-gate и не переносились в словари, профили или поисковые карточки.

## Результат

Первоочередная recovery-задача, добавленная перед этапом 2, завершена до перехода к candidate retrieval.

| Корпус | Baseline до v3.1 | После этапа 1 | Финал | Изменение к baseline |
|---|---:|---:|---:|---:|
| Live 160 | 94,38% | 92,50% | 95,62% | +1,24 п. п. |
| Adjudicated closed 270 | 80,37% | 78,52% | 81,48% | +1,11 п. п. |

Transport и forbidden-content gate равны 100% на обоих финальных прогонах. На live confident wrong равен 0.

## Candidate retrieval API

`retrieve_knowledge_candidates()` возвращает Top-K кандидатов и не выбирает финальный сценарий. Для каждого кандидата доступны:

- итоговый candidate score;
- lexical score;
- отдельные char и word scores;
- normalized dense score и исходная dense similarity;
- intent boost;
- признак фактической доступности dense-канала.

Routing и lexical semantic index используют единый `routing_normalize`: нормализацию регистра/символов, исправление доменных опечаток и перестановок, синонимизацию и лёгкий русский stemming. Для offline evaluation query embeddings кодируются пакетно и кэшируются по model/dataset fingerprint.

Candidate fusion отделён от answer fallback. Для Recall@10 используются веса lexical 0,75 / dense 0,25; прежние runtime-веса answer fallback 0,30 / 0,70 не изменены.

## Gates

| Набор | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| Development 372 | 91,40% | 96,77% | 98,66% |
| Validation 99 | 91,92% | 100,00% | 100,00% |
| Independent 116 | — | — | 100,00% |

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

## Артефакты

- `reports/quality-recovery-v31-live-160.json`;
- `reports/quality-recovery-v31-closed-270.json`;
- `reports/semantic-retrieval-v31-development-validation.json`;
- `reports/candidate-retrieval-v31-independent-116.json`;
- `backend/tools/evaluate_semantic_retrieval_v31.py`;
- `backend/tools/evaluate_candidate_retrieval_blind.py`.

## Следующий этап

Этап 3 может начинать hard-negative reranker и confidence calibration. Candidate retrieval остаётся recall-oriented и не получает права принимать финальное решение.
