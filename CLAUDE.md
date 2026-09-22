Работаем над RAG — универсальным ЛОКАЛЬНЫМ Python dev-тулом (llama-index 0.14.x)
для семантического поиска по кодовым базам на C#. НЕ часть индексируемого
приложения — отдельный tooling в ОТДЕЛЬНОМ репозитории (github.com/nakitmn/RAG,
этот каталог), `.venv` в корне репо.

Первый источник знаний про систему — `.memory/` (локальный каталог, вне git;
см. ниже), а не код.

ПЕРВЫЙ ШАГ — прочитай память системы, прежде чем что-либо делать:
  .memory/README.md        — индекс + текущее состояние (начни отсюда)
  .memory/architecture.md  — конвейер, компоненты, метаданные чанка, языки
  .memory/usage.md         — запуск, env-переменные, CLI, примеры
  .memory/roadmap.md       — что готово, история решений, открытые шаги

КОНТЕКСТ (коротко):
- Структура: ast/ (домен AST-чанкинга).
- Главный скрипт: ast/scripts/ast_llamaindex.py — IngestionPipeline с
  transformations [CodeSplitter, SummaryExtractor, KeywordExtractor]. Enrichment
  идёт ВНУТРИ pipeline.run(), не отдельным шагом.
- LLM по доменам (enrichment/query): ast/scripts/llm_provider.py
  (OpenAILike-клиент); модели/эндпоинты задаются в .env (сейчас z.ai
  glm-5.3-flash через coding-эндпоинт).
- КОНФИГУРАЦИЯ: единственная точка — корневой .env репо (+ коммитимый
  шаблон .env.example). Читает/валидирует env ТОЛЬКО ast/scripts/config.py;
  дефолтов tunable-параметров в коде нет. Самопроверка:
  .venv/bin/python ast/scripts/config.py.
- Поддержка языков: ТОЛЬКО C# (.cs). Реестр SUPPORTED_LANGUAGES в ast_llamaindex.py
  готов к добавлению kotlin/java/js одной строкой (+ pip tree-sitter-пакет).
- Запуск:
    .venv/bin/python ast/scripts/ast_llamaindex.py --path <папка .cs> [--no-enrich] [--max-files N] [--output <jsonl>]
  --no-enrich = оффлайн-чанкинг без LLM (быстро, без ключа).

СОГЛАШЕНИЯ:
- Источник кода — любой C#-проект, путь задаётся RAG_CODE_PATH в .env
  (относительные пути — от корня репо). Python-тул индексирует целевой
  проект, не себя.
- Зависимости: requirements.txt (прямые импорты, pinned).
- Метаданные чанка: file_path/file_name (от SimpleDirectoryReader/CodeSplitter),
  section_summary (SummaryExtractor), excerpt_keywords (KeywordExtractor).
- Любые новые знания/решения/статусы фиксируй в .memory/ (обнови нужный файл).
- Не коммить без явного запроса.
