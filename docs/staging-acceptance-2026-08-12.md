# Staging acceptance — 2026-08-12

## Scope

Проверен локальный production-like экземпляр без LLM. Это техническая staging-приёмка backend и виджета; подключение к внешнему staging-домену MIGTORG и реальная доставка обращений ещё не выполнены.

## Artifact

- commit: `772f083`
- deploy version: `staging-772f083`
- knowledge version: `2026.08.12.26`
- local URL: `http://127.0.0.1:8002/widget/`
- `widget.js` SHA-256: `f73597527a8ffbdc8f9f1d30cc44fef2c2ed6e1c7ba019e5395984db1abb4c30`
- LLM: disabled, deterministic mode (`mock`)

## Automated gates

- Knowledge audit: passed, `production_release_ready=true`, 140 scenarios, 0 errors, 0 example collisions.
- Gold dataset: 312/312, release and production gates passed.
- Full regression suite: 545 passed.
- Production-like preflight: 1 failure and 2 warnings.

The required failure is `ticket_delivery`: SMTP delivery is not configured. Warnings are the missing `QUALITY_REPORT_TOKEN` and disabled trusted personal-status integration.

## Smoke-test results

- `/health`: `status=ok`, `environment=production`, `deploy_version=staging-772f083`, `widget_ready=true`, `knowledge_mode=v2`.
- CORS allows `https://migtorg.com` and does not return an allow-origin header for an unrelated origin.
- Widget assets use `Cache-Control: no-store, max-age=0`.
- Frontend scan found no names of backend secret variables.
- Guest, typo, complaint, filter, search, and missing-image requests routed correctly (6/6).
- An untrusted browser claim of authorization remains `guest`, as required until a signed context is configured.
- A written ticket was saved in the isolated staging database and received ID `12.08.26-0001`, status `new`.
- Desktop and mobile layouts were inspected in the browser; the chat opens correctly and no console errors were recorded.
- The `В начало` action resets the visible conversation.
- The mobile ticket form opens with thematic prefill and contains no callback-time field.

During the first smoke pass, two uncovered phrases were found and fixed in commit `772f083`: a sharp complaint and `поиск не находит лот`.

## Pending before staging sign-off

1. Provide the public HTTPS staging backend URL and the staging site origin.
2. Configure SMTP sender, recipient, host and credentials in staging secret storage.
3. Configure a random `QUALITY_REPORT_TOKEN` of at least 32 characters.
4. Add the backend domain to the staging site's CSP and connect `widget.js`.
5. Create a ticket from the actual staging site and confirm receipt by support.
6. Verify real site routes, filters, lot search, images, guest context and signed authorized context.
7. Repeat desktop/mobile checks on the actual staging pages, including backend-unavailable behavior.

Staging sign-off remains blocked until item 5 succeeds. The local staging process is running on port `8002` for continued checks.
