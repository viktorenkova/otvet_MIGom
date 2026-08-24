# MIGTORG knowledge schema v3.1

`scenarios.json` — нормализованная runtime-БЗ, воспроизводимо построенная из канонического `knowledge/MASTER_KNOWLEDGE.md`. Файл `knowledge/v2/scenarios.json` публикуется из того же источника как слой совместимости.

Основные правила:

- канонические `scenario_id` и утверждённые ответы сохраняются для обратной совместимости API;
- каждый факт имеет стабильный `fact_id`, источник, дату проверки и source version;
- `answer_policy` разрешает использовать только перечисленные факты, а LLM — только редактировать формулировку;
- `search_document` содержит только тему, таксономию и пользовательские формулировки — факты и готовые ответы в retrieval не индексируются;
- смешанные канонические сценарии разделены на `atomic_units`, которые покрывают факты ровно один раз и возвращают canonical scenario ID;
- `scenario_conflicts.json` фиксирует все наблюдавшиеся пары путаницы и признаки, которыми их следует различать;
- `knowledge_gaps` не заполняются догадками: до подтверждения владельцем продукта используется безопасная gap policy.

Пересборка и проверка:

```powershell
python -m backend.tools.master_knowledge publish
python -m backend.tools.master_knowledge validate
python -m backend.tools.validate_knowledge_v31 --output reports/knowledge-v31-validation.json
```

Файлы v2 и v3.1 изменяются через публикационный инструмент; ручное расхождение с мастер-документом блокируется тестами.
