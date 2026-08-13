# Обновление MIGTORG Chatbot в GitLab и запуск dev с LLM

Инструкция рассчитана на точечное обновление существующего GitLab-проекта из очищенного GitHub-репозитория. В GitLab сохраняются собственные CI/CD-, Docker-, Compose- и nginx-файлы, а из GitHub переносятся актуальные исходники, БЗ, тесты, безопасные шаблоны конфигурации, утверждённые материалы и эксплуатационная документация. Секреты и runtime-артефакты в репозиторий не входят.

## 1. Что уже проверено

На 13 августа 2026 года перед публикацией очищенного `main`:

- `.env`, `.env.staging`, `.venv`, `.work`, `.run`, SQLite и логи игнорируются;
- среди tracked-файлов нет env-файлов с секретами, баз, логов, ключей и сертификатов;
- в достижимой истории не найдены сигнатуры действующих API-токенов или приватных ключей;
- значения ключей в README и `.env.example` являются пустыми значениями или placeholders;
- из истории удалён старый коммит с `otvet_migom/migtorg_chatbot.sqlite3`, в котором находились сообщения и обращения;
- очищенная история начинается с одного корневого коммита и содержит только проверенное актуальное дерево проекта.

Тесты, БЗ, документация и утверждённые шаблоны документов остаются в репозитории намеренно. Это исходные и проверочные материалы проекта, а не runtime-данные или секреты.

Старые локальные клоны GitHub, созданные до очистки, использовать как источник нельзя: в их объектах и reflog может сохраняться удалённая история. Источником служит только свежий архив актуального GitHub `main`.

## 2. Правила обновления существующего GitLab

- Все изменения выполняются в отдельной ветке от актуального GitLab `main`.
- Прямой push и загрузка файлов в `main` запрещены: действующий pipeline разворачивает `main` на production.
- `.gitlab-ci.yml`, `.dockerignore`, `Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml` и `frontend/deploy/nginx.conf` сохраняются из GitLab.
- Актуальные `frontend/chat-widget/index.html`, `style.css` и `widget.js` размещаются также в корне `frontend`, чтобы существующий frontend-контейнер получил новый UI.
- Незаполненный `backend/static/templates/Шаблон_заявления_на_возврат_депозита.docx` сохраняется из GitLab. Заполненные пользовательские документы переносить запрещено.
- Перед merge обязательны автоматические проверки и ручная dev/staging-приёмка.

## 3. Обновление через GitLab Web IDE

1. В GitLab откройте **Code → Branches** и создайте ветку от `main`, например `update/kb-v2-ux-ui-YYYYMMDD`.
2. Откройте её через **Code → Open in Web IDE**.
3. Скачайте и распакуйте свежий архив GitHub `main`.
4. Перенесите код пакетами, проверяя Source Control перед каждым коммитом:
   - корневые безопасные файлы и документацию;
   - `backend/app`, `backend/tools` и `configs`;
   - `knowledge`, `tests` и `reports`;
   - файлы виджета в `frontend/chat-widget` и их совместимую копию в `frontend`.
5. Не загружайте GitLab-only инфраструктуру из другого источника и не удаляйте её.
6. Создавайте обычные коммиты в рабочую ветку; не используйте force push.

Web IDE позволяет загрузить несколько файлов одновременно, но не запускает тесты. Поэтому зелёный commit/pipeline в GitLab и внешняя приёмка обязательны.

## 4. Проверка ветки и Merge Request

Перед созданием Merge Request проверьте в GitLab **Code → Compare revisions** или на вкладке **Changes** MR:

- target branch — `main`, source branch — рабочая `update/...`;
- GitLab-only инфраструктурные файлы не удалены и не заменены;
- нет `.env`, SQLite, логов, PID-файлов, заполненных форм и секретов;
- в `frontend` и `frontend/chat-widget` находятся одинаковые актуальные `index.html`, `style.css`, `widget.js`;
- пустой шаблон заявления доступен по имени, указанному в БЗ;
- pipeline и security jobs завершились успешно.

Не выполняйте merge в `main`, пока не подготовлен dev-деплой: текущий production pipeline может автоматически собрать образы и выполнить SSH-деплой.

## 5. Что должно остаться после обновления

GitLab становится основным рабочим репозиторием и сохраняет собственную историю и инфраструктуру. GitHub остаётся очищенным проверенным источником актуального пакета и резервной точкой сверки, но его независимую историю не следует принудительно накладывать на GitLab.

## 6. Dev-конфигурация с включённой LLM

В репозитории находится безопасный шаблон `configs/dev.env.example`. В нём LLM уже включена, но ключ отсутствует.

Для локального dev:

```powershell
Copy-Item configs/dev.env.example .env.dev
```

В `.env.dev` задайте:

```env
LLM_ENABLED=true
LLM_PROVIDER=qwen
LLM_ENVIRONMENT=dev
LLM_PRIMARY_MODEL=qwen-plus
LLM_FALLBACK_MODEL=qwen-flash
QWEN_BASE_URL=<OpenAI-compatible URL dev workspace>
QWEN_API_KEY=<dev key>
```

`.env.dev` подходит под правило `.env.*` и не попадёт в Git. Не используйте production-ключ в dev.

Для GitLab откройте **Settings → CI/CD → Variables** и создайте переменные с environment scope `dev`:

| Variable | Value / policy |
|---|---|
| `LLM_ENABLED` | `true` |
| `LLM_PROVIDER` | `qwen` |
| `LLM_ENVIRONMENT` | `dev` |
| `LLM_PRIMARY_MODEL` | `qwen-plus` |
| `LLM_FALLBACK_MODEL` | `qwen-flash` |
| `QWEN_BASE_URL` | URL dev workspace |
| `QWEN_API_KEY` | реальный dev-ключ; Masked and hidden |
| `LLM_DEV_BUDGET_USD` | согласованный dev-лимит |
| `DEPLOY_VERSION` | `$CI_COMMIT_SHA` в deploy job |
| `CORS_ALLOWED_ORIGINS` | точный HTTPS origin dev-сайта |
| `DATABASE_PATH` | абсолютный путь на persistent volume |

Если `QWEN_API_KEY` отмечен как Protected, ветка или GitLab environment `dev` также должны быть защищены, иначе deploy job не получит ключ.

## 7. Запуск dev

Локально:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn --env-file .env.dev backend.app.main:app --host 0.0.0.0 --port 8000
```

Для серверного dev-деплоя env-переменные должен передать GitLab Runner или secret storage. Не генерируйте `.env.dev` как CI artifact и не печатайте переменные командой `env` в job log.

## 8. Проверка, что LLM действительно подключена

Сначала:

```powershell
Invoke-RestMethod https://DEV_CHAT_BACKEND/health
```

Ожидаются правильный `deploy_version`, `knowledge_mode=v2` и `widget_ready=true`.

Затем отправьте обычный справочный вопрос, не требующий обращения:

```powershell
$body = @{
  message = "Что такое MIGTORG?"
  session_id = "dev-llm-smoke"
  context = @{ is_authorized = $false; page_type = "public_site" }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "https://DEV_CHAT_BACKEND/api/chat/message" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

В ответе `model_used` должен быть `qwen-plus` или `qwen-flash`, а не `mock`. Сценарии, требующие эскалации или создания обращения, намеренно не отправляются в LLM и могут вернуть `mock` — это штатное безопасное поведение.

Если Qwen недоступен, приложение возвращает безопасный исходный ответ БЗ. Поэтому дополнительно проверяйте backend-логи и метрики LLM, а не только наличие текста ответа.

## 9. Финальная проверка перед dev-деплоем

```powershell
python -m backend.tools.audit_knowledge --strict
python -m pytest -q
git status -sb
```

В `git status` не должно быть неожиданных файлов. Никогда не добавляйте с `git add -f` env-файлы, БД, логи или `.work`.

## 10. Что ещё потребуется для автоматического deploy job

В текущем репозитории нет Dockerfile, `.gitlab-ci.yml` и описания целевого dev-сервера. Их нельзя корректно создать без выбранного способа размещения. Для следующего шага нужны:

- тип dev-хостинга или адрес сервера;
- способ доступа GitLab Runner к серверу;
- домен backend и origin dev-сайта;
- путь persistent volume;
- способ перезапуска процесса: systemd, Docker Compose, Kubernetes или другой.
