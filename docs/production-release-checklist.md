# Production release checklist

Этот чек-лист превращает готовую базу знаний и виджет в управляемый production-релиз. Выпуск разрешается только при зелёном автоматическом preflight и пройденной ручной staging-приёмке.

## 1. Конфигурация релиза

Создать production-env вне Git и задать как минимум:

```env
APP_ENVIRONMENT=production
DEPLOY_VERSION=<image-tag-or-commit-sha>
DEBUG=false
CORS_ALLOWED_ORIGINS=https://migtorg.com,https://www.migtorg.com
DATABASE_PATH=<absolute-path-on-persistent-volume>

KNOWLEDGE_V2_ENABLED=true
KNOWLEDGE_V2_SHADOW_MODE=false

TICKET_EMAIL_ENABLED=true
TICKET_EMAIL_TO=<support-address>
TICKET_EMAIL_FROM=<sender-address>
SMTP_HOST=<smtp-host>
QUALITY_REPORT_TOKEN=<random-secret-at-least-32-characters>
```

Секреты SMTP, LLM, внутренних API и `QUALITY_REPORT_TOKEN` хранятся только в secret storage среды исполнения. `DEPLOY_VERSION` должен однозначно связывать `/health` с выпущенным артефактом.

## 2. Автоматические gates

```bash
python -m backend.tools.audit_knowledge --strict --output .work/knowledge-audit.json
python -m backend.tools.check_production_readiness --env-file .env.production --strict --output .work/production-readiness.json
python -m pytest -q
```

Ожидаемый результат:

- `production_release_ready: true` в аудите базы знаний;
- `production_ready: true` и `failures: 0` в preflight;
- весь набор тестов проходит.

Preflight блокирует открытый CORS, debug-режим, незаданную версию, shadow/legacy knowledge mode, неготовую доставку обращений и небезопасную конфигурацию включённых LLM/status-интеграций. Относительный путь SQLite, выключенные персональные статусы и короткий/отсутствующий operations token отмечаются предупреждениями.

## 3. Staging smoke-test

1. Проверить `GET /health`: `status=ok`, правильные `environment` и `deploy_version`, `widget_ready=true`, `knowledge_mode=v2`.
2. Открыть виджет на desktop и mobile, отправить обычный вопрос, резкое сообщение и запрос с опечаткой.
3. Проверить фильтры, поиск лота и ссылки/изображения на реальных staging-страницах.
4. Проверить гостевой контекст, авторизованный контекст и карточку с `lot_id`.
5. Создать обращение и подтвердить не только номер заявки, но и фактическую доставку в поддержку.
6. Убедиться, что ошибки backend отображаются понятно, а страница сайта продолжает работать.
7. Проверить отсутствие CORS/CSP-ошибок и секретов в HTML, JavaScript и сетевых ответах.

## 4. Выпуск

- Зафиксировать URL backend, `DEPLOY_VERSION`, контрольную сумму или версию `widget.js` и владельца релиза.
- Сохранить JSON-отчёты preflight и knowledge audit как артефакты релиза.
- Подключить виджет на production и повторить короткий smoke-test.
- Проверить доставку первого production-обращения и внутренний quality report.
- Наблюдать ошибки API, долю fallback/эскалаций и ошибки доставки обращений.

## 5. Откат

Быстрый пользовательский откат — убрать тег `widget.js` на сайте. Backend откатывается на предыдущий известный `DEPLOY_VERSION`; SQLite-файл на persistent volume не удаляется. После отката проверить отсутствие запросов виджета, доступность основной страницы и сохранность ранее созданных обращений.
