# Этап 0.5: воспроизводимый GitHub build

Дата завершения: 23 августа 2026 года

Статус текущего gate: **пройден**

## Что сделано

1. Build manifest публикуется и на `/health`, и на `/api/health`, чтобы его можно было получить через типовой reverse proxy.
2. Remote evaluator сначала проверяет `/api/health`, затем `/health`; при необходимости принимает явный `--health-endpoint`.
3. Backend, БЗ, routing/config/prompt/widget fingerprints и инструменты контроля зафиксированы commit `094d1fa41f24cbfd5a610c4ba12d8ed99763fc4e` и отправлены в `origin/main` GitHub.
4. Один и тот же commit прогнан напрямую в Python и через локальный FastAPI HTTP endpoint с `LLM_ENABLED=false` и одним closed-control корпусом.
5. GitLab и прежний внешний стенд не использовались согласно уточнённому владельцем проекта контуру. После ручного переноса тот же gate нужно повторить на целевом dev URL.

## Результат comparator

| Проверка | Результат |
|---|---:|
| Dataset | 270 кейсов |
| Dataset SHA-256 | `603982053d24941b3b96a43ac109c28f471fc3f0b63793b52d49ea865a2a7c75` |
| Manifest SHA-256 local / HTTP | `1a1f13f6a826ddc44be8f2e06b16f24cb83010d28802c358b5f68d4c81164a91` |
| Одинаковый build | да |
| Одинаковый dataset | да |
| Расхождения route/intent/resolution | 0 из 270 |
| `deterministic_gate_passed` | `true` |

Продуктовые показатели совпали полностью: route hit — 80,37%, quality pass — 79,63%, уверенно неверных ответов — 41. Это не достижение целевого KPI, а доказательство того, что дальнейшие изменения можно измерять на воспроизводимой базе.

## Проверка реализации

- 18 уникальных профильных тестов manifest, evaluator, immutable corpora, leakage guard, Routing v3, production readiness и widget delivery прошли;
- `compileall` прошёл;
- `git diff --check` прошёл перед фиксацией build;
- HTTP runtime после аудита остановлен; production и внешние стенды не изменялись.

## Артефакты

- `reports/quality-stage0_5-local.json` — direct-local прогон;
- `reports/quality-stage0_5-http.json` — прогон того же build через HTTP;
- `reports/quality-stage0_5-local-http-comparison.json` — итог comparator с 0 расхождений.

## Следующий шаг

Экспертная сверка завершена: 110 из 110 записей утверждены, 15 исправлены, решения подключены отдельным SHA-bound overlay. Следующий инженерный этап — нормализация БЗ и таксономии сценариев.
