# MIGTORG Chatbot

Чат-бот поддержки для сайта MIGTORG с web-виджетом и backend на FastAPI.

Проект отвечает на справочные вопросы по работе площадки, использует контекст страницы и пользователя, показывает документы и при необходимости помогает создать обращение сотрудникам поддержки.

## Возможности

- отдельные сценарии для гостей и авторизованных пользователей;
- ответы на основе базы знаний MIGTORG;
- обработка вопросов по лотам, торгам, тарифам, платежам, возвратам и документам;
- safety-проверки перед формированием ответа;
- уточняющие вопросы при недостаточном контексте;
- создание обращений в поддержку;
- локальное хранение истории и обращений в SQLite;
- опциональная доставка обращений по email;
- standalone web-виджет без frontend-фреймворка;
- подготовленная архитектура для LLM, Langfuse, Telegram и Bitrix.

## Структура проекта

```text
backend/
  app/
    bot/             логика классификации, поиска и формирования ответов
    delivery/        доставка обращений
    integrations/    внешние интеграции
    models/          модели запросов и ответов
    main.py          FastAPI-приложение и API
  static/templates/  документы, доступные из чата

frontend/
  chat-widget/
    index.html       демонстрационная страница
    widget.js        standalone-виджет
    style.css        стили виджета

knowledge/
  public/            публичные статьи
  authorized/        сценарии для авторизованных пользователей
  internal_rules/    правила безопасности, тона и эскалации
  normalized/        нормализованная база знаний

configs/             правила классификации и смыслового поиска
docs/                документация проекта
```

## Требования

- Python 3.10 или новее;
- `pip`;
- современный браузер для проверки виджета.

## Локальный запуск

Создайте и активируйте виртуальное окружение удобным для вашей ОС способом, затем установите зависимости:

```bash
pip install -r requirements.txt
```

Создайте локальный `.env` на основе примера:

```bash
cp .env.example .env
```

На Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Запустите приложение из корня репозитория:

```bash
uvicorn --env-file .env backend.app.main:app --reload
```

После запуска доступны:

- demo-виджет: `http://127.0.0.1:8000/widget/`;
- health-check: `http://127.0.0.1:8000/health`;
- OpenAPI: `http://127.0.0.1:8000/docs`.

По умолчанию проект работает без внешней LLM в режиме локальных правил и базы знаний.

## Конфигурация

Основные переменные находятся в [.env.example](.env.example).

| Переменная | Назначение | Значение по умолчанию |
|---|---|---|
| `APP_NAME` | Название приложения | `migtorg-chatbot` |
| `APP_ENVIRONMENT` | Среда запуска (`development`/`production`) | `development` |
| `DEPLOY_VERSION` | Версия образа или commit SHA для диагностики | `local` |
| `DEBUG` | Режим отладки | `true` в примере |
| `CORS_ALLOWED_ORIGINS` | Разрешенные origin сайта через запятую | `*` |
| `DATABASE_PATH` | Путь к SQLite-файлу | `migtorg_chatbot.sqlite3` |
| `LLM_ENABLED` | Использование внешней LLM | `false` |
| `LLM_PROVIDER` | Провайдер LLM | `mock` |
| `LLM_ROLLOUT_PERCENTAGE` | Доля стабильной пользовательской когорты с LLM-формулировкой | `0` |
| `LLM_DAILY_BUDGET_USD` | Дневной hard stop расходов | `5.0` |
| `LLM_INPUT_COST_PER_MILLION_USD` | Актуальная цена входных токенов; обязательна при включении LLM | `0` |
| `LLM_OUTPUT_COST_PER_MILLION_USD` | Актуальная цена выходных токенов; обязательна при включении LLM | `0` |
| `QUALITY_REPORT_TOKEN` | Токен внутренних отчетов | не задан |
| `TICKET_EMAIL_ENABLED` | Доставка обращений по email | `false` |
| `TICKET_EMAIL_TO` | Получатель обращений | тестовое значение |

Для production нельзя оставлять CORS открытым. Пример:

```env
CORS_ALLOWED_ORIGINS=https://migtorg.com,https://www.migtorg.com
```

Секреты LLM, SMTP и внутренних сервисов должны находиться только в окружении backend. Их запрещено добавлять в Git или frontend-код.

## API

Основные endpoint'ы:

| Метод и путь | Назначение |
|---|---|
| `GET /health` | Состояние приложения |
| `POST /api/chat/start` | Стартовый экран и серверные кнопки гибридной навигации |
| `POST /api/chat/message` | Отправка сообщения в чат |
| `POST /api/chat/ticket` | Создание обращения |
| `GET /api/chat/history/{session_id}` | История диалога |
| `POST /api/chat/feedback` | Оценка ответа |

`POST /chat` оставлен для обратной совместимости и не должен использоваться в новой интеграции.

Пример запроса:

```bash
curl -X POST http://127.0.0.1:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session",
    "message": "Как участвовать в торгах?",
    "context": {
      "is_authorized": false,
      "page_type": "public_site"
    }
  }'
```

Интерактивное описание актуальных моделей запросов и ответов доступно в `/docs` после запуска backend.

## Подключение виджета к сайту

Минимальный пример:

```html
<script>
  window.MIGTORG_CHAT_API_BASE = "https://CHAT_BACKEND_URL";
  window.MIGTORG_CHAT_CONTEXT = {
    is_authorized: false,
    user_id: null,
    page_type: "public_site",
    lot_id: null,
    user_email: null,
    user_phone: null
  };
</script>
<script src="https://CHAT_BACKEND_URL/widget/widget.js"></script>
```

Полная инструкция, включая CORS, CSP, контекст страниц, staging-проверки, безопасность и откат: [Интеграция чат-виджета MIGTORG на migtorg.com](docs/site-widget-integration.md).

Перенос репозитория из GitHub в GitLab и безопасный запуск dev с включённой Qwen LLM: [GitHub → GitLab и dev LLM](docs/github-to-gitlab-dev-llm.md).

## Важное ограничение авторизации

Контекст `MIGTORG_CHAT_CONTEXT` формируется в браузере и может быть изменен пользователем. Значения `is_authorized`, `user_id`, `lot_id`, email и телефона нельзя считать подтвержденными сервером.

Текущая интеграция подходит для выбора справочного сценария и заполнения формы обращения, но не для раскрытия персональных данных или выполнения операций с аккаунтом. Для персонализированной функциональности требуется отдельная защищенная авторизация между сайтом и chatbot backend.

## Обращения в поддержку

Обращение сначала сохраняется локально в SQLite. Если включена email-доставка, backend дополнительно отправляет его по SMTP.

Для включения email необходимо настроить:

```env
TICKET_EMAIL_ENABLED=true
TICKET_EMAIL_TO=support@example.com
TICKET_EMAIL_FROM=bot@example.com
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
```

Если SMTP недоступен, сохраненное обращение не удаляется и получает статус ошибки доставки.

## Проверка перед production

После заполнения production-env сначала запустить автоматический preflight:

```bash
python -m backend.tools.check_production_readiness --env-file .env.production --strict --output .work/production-readiness.json
```

Команда не выводит значения секретов и завершается с кодом `1`, если не выполнено обязательное условие. Предупреждения не блокируют выпуск, но должны получить явное решение владельца релиза. Полный порядок: [production release checklist](docs/production-release-checklist.md).

- `GET /health` отвечает успешно;
- CORS ограничен доменами MIGTORG;
- backend и виджет доступны по HTTPS;
- виджет проверен на desktop и mobile;
- гостевой и авторизованный контекст передаются корректно;
- создание и доставка обращения проверены на staging;
- ссылки на документы открываются;
- в frontend и Git отсутствуют секреты;
- определены владельцы мониторинга и канала обращений;
- подготовлен способ отключения виджета без изменения backend.

## Статус интеграций

| Интеграция | Статус |
|---|---|
| SQLite | используется |
| Email/SMTP | реализовано, включается через окружение |
| LLM через LiteLLM | опциональная конфигурация |
| Langfuse | заготовка, требует production-реквизитов |
| Telegram | заготовка |
| Bitrix | запланированная интеграция |

Перед production-запуском фактический канал доставки обращений и режим LLM должны быть согласованы отдельно.

## Поиск по базе знаний и качество ответов

Поиск устроен в несколько уровней:

1. safety-проверки и точные бизнес-маршруты для критичных сценариев;
2. canonical matching, устойчивый к перестановке слов в коротких запросах;
3. гибридный retrieval: символьный TF-IDF для опечаток и идентификаторов плюс локальные multilingual embeddings `intfloat/multilingual-e5-small` для разговорных формулировок и перефразировок;
4. уточнение между несколькими близкими сценариями вместо уверенного случайного ответа;
5. ответ только из активной статьи БЗ. Опциональная LLM переформулирует найденный ответ, но не является источником фактов.

### Сценарная база знаний v2

Поверх legacy-БЗ подключён версионированный слой `knowledge/v2/scenarios.json`.
Он различает этап, объект, действие и состояние, хранит отрицательные
примеры, утверждённые факты, следующие шаги и структурированные действия.
Сильное сценарное совпадение имеет приоритет; при близких вариантах бот задаёт
предметное уточнение. Слой можно независимо отключить через
`KNOWLEDGE_V2_ENABLED=false`.

Ответ `/api/chat/message` обратно совместим и дополнительно возвращает
`message_id`, `scenario_id`, `resolution`, `actions`, `used_context` и
`data_freshness`. Виджет передаёт `selected_action_id` и связывает feedback с
`message_id`.

### Гибридная навигация

Виджет получает стартовое меню через `POST /api/chat/start`, при этом поле
свободного ввода остаётся доступным на каждом шаге. Конечная кнопка связана с
конкретным активным сценарием на сервере и не запускает повторный нечёткий
поиск. Сервер принимает выбор только среди действий, выданных этой сессии в
последнем ответе. Конфигурация пилота из 30 сценариев находится в
`configs/guided_navigation.v1.json`.

```env
GUIDED_NAVIGATION_ENABLED=false
GUIDED_NAVIGATION_ROLLOUT_PERCENTAGE=0
GUIDED_NAVIGATION_CONFIG_PATH=configs/guided_navigation.v1.json
GUIDED_NAVIGATION_MAX_DEPTH=2
```

Допустимые доли включения: `0`, `5`, `25`, `50`, `100`. Распределение стабильно
для `session_id`; возврат к контрольному интерфейсу выполняется установкой
процента в `0` без отката данных.

### Защищённые персональные статусы

Браузерный `is_authorized` не даёт права читать персональные статусы. Backend
сайта должен передать `MIGTORG_CHAT_TRUSTED_CONTEXT_TOKEN`: HMAC-SHA256 токен
в формате `base64url(JSON).base64url(signature)`. Payload содержит `iss`,
`sub`, `exp` и scopes, например `status:bid:read` или общий `status:read`.

Read-only адаптер вызывает `${INTERNAL_STATUS_API_URL}/v1/status/{kind}` и
поддерживает `lot`, `bid`, `auction`, `payment`, `tariff`, `documents` и
`transfer`. Пока API не настроен, бот не предполагает статус и предлагает
обращение.

### Наполнение и контроль качества

```bash
python -m backend.tools.knowledge_pipeline migtorg_chatbot.sqlite3 .work/knowledge-candidates.json
python -m backend.tools.audit_knowledge --output .work/knowledge-audit.json
python -m backend.tools.evaluate_scenarios tests/data/scenario_gold.jsonl --gate
```

Word-saved Telegram support exports (`.html`/`.htm`, including UTF-16 files)
can be processed together. The normalized JSONL excludes Telegram participant
names, replaces contacts, URLs and identifiers, and marks system/contact-only
messages so they cannot become knowledge automatically:

```powershell
python -m backend.tools.knowledge_pipeline `
  C:\path\messages.html .work\support-candidates.json `
  --include C:\path\messages2.html `
  --include C:\path\messages3.html `
  --normalized-output .work\support-normalized.jsonl
```

Both outputs are review artifacts only: `publication_allowed` remains `false`.

Build a privacy-safe research summary and prioritized scenario backlog without
exporting verbatim messages:

```powershell
python -m backend.tools.analyze_support_corpus `
  .work\support-normalized.jsonl `
  .work\support-research.json `
  --markdown-output .work\support-research.md
```

The report separates topical scenarios from action signals such as callback
requests, labels recurring themes with confidence and intensity, and requires
expert review for financial and contractual candidates.

Pipeline обезличивает обращения, кластеризует формулировки и создаёт только
очередь на ручную проверку. Автопубликация кандидатов запрещена. Аудит проверяет
источники, владельцев, сроки ревью, пересечения примеров и формирует карту
совместимости legacy ID.

Модель embeddings загружается через `sentence-transformers`. Матрица статей кэшируется по отпечатку модели и содержимого БЗ. Для production рекомендуется указать постоянный каталог и прогреть индекс на этапе сборки или перед переключением трафика:

```env
SEMANTIC_MODEL_ALLOW_DOWNLOAD=true
SEMANTIC_CACHE_DIR=/var/cache/migtorg-semantic
```

```bash
python -c "from backend.app.bot.knowledge_search import warm_knowledge_indexes; warm_knowledge_indexes()"
```

После первичной загрузки модели в изолированной среде установите `SEMANTIC_MODEL_ALLOW_DOWNLOAD=false`. Если модель или `sentence-transformers` недоступны, бот продолжит работу на TF-IDF и точных маршрутах, записав предупреждение в лог.

Для опционального переформулирования через LiteLLM используйте GPT-5.6 Luna как основной высокочастотный маршрут и Terra как резервный. В Chat Completions reasoning явно зафиксирован в `none`, чтобы не менять стоимость и задержку неявным default-поведением модели:

```env
LLM_ENABLED=true
LLM_PROVIDER=litellm
LLM_PRIMARY_MODEL=openai/gpt-5.6-luna
LLM_FALLBACK_MODEL=openai/gpt-5.6-terra
LLM_REASONING_EFFORT=none
LITELLM_PROXY_URL=https://YOUR_LITELLM_PROXY
LITELLM_API_KEY=...
```

Для прямого подключения Alibaba Cloud Model Studio (Qwen) без LiteLLM:

```env
LLM_ENABLED=true
LLM_PROVIDER=qwen
LLM_PRIMARY_MODEL=qwen-plus
LLM_FALLBACK_MODEL=qwen-flash
LLM_ROLLOUT_PERCENTAGE=0
LLM_INPUT_COST_PER_MILLION_USD=<current-provider-price>
LLM_OUTPUT_COST_PER_MILLION_USD=<current-provider-price>
QWEN_BASE_URL=https://YOUR_WORKSPACE_ID.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=...
```

Qwen используется только для переформулирования уже найденного ответа БЗ. При ошибке API,
пустом ответе или создании обращения бот возвращает безопасный ответ из БЗ без участия внешней
модели. Ключ хранится только в окружении backend. Значение rollout `0` обязательно для shadow:
кандидат LLM сохраняется для проверки, но пользователю не показывается. После успешной экспертной
проверки допустимы только ступени `5`, `25`, `50`, `100`.

Redacted shadow на 200–500 запросах запускается отдельно от пользовательского rollout:

```bash
python -m backend.tools.run_llm_shadow <source.json> --target-count 200
```

Агрегированный отчёт создаётся в `reports/stage6_1-llm-shadow.json`, а рабочий пакет для
независимой оценки — в игнорируемом Git каталоге `.work/`. Сырые prompt и немаскированные
персональные данные в эти файлы не записываются.

Проверка регрессий и живых перефразировок:

```bash
python -m pytest -q
```
