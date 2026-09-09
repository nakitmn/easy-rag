Работаем над RAG-системой проекта launcher (VR Club Launcher, C#/Unity).
Это ЛОКАЛЬНЫЙ Python dev-тул (llama-index 0.14.x) для семантического поиска по
коду лаунчера. НЕ часть Unity-приложения — отдельный tooling в ОТДЕЛЬНОМ
репозитории (github.com/nakitmn/RAG, этот каталог), `.venv` в корне репо.

RAG-система НЕ описана в основной wiki лаунчера (проверено) — её нет в index.md.
Первый источник знаний про неё — `.memory/` (см. ниже), а не вики и не код.

ПЕРВЫЙ ШАГ — прочитай память системы, прежде чем что-либо делать:
  .memory/README.md        — индекс + текущее состояние (начни отсюда)
  .memory/architecture.md  — конвейер, компоненты, метаданные чанка, языки
  .memory/usage.md         — запуск, env-переменные, CLI, примеры
  .memory/roadmap.md       — что готово, история решений, открытые шаги

КОНТЕКСТ (коротко):
- Структура: ast/ (домен AST-чанкинга), repomix/ (домен repomix).
- Главный скрипт: ast/scripts/ast_llamaindex.py — IngestionPipeline с
  transformations [CodeSplitter, SummaryExtractor, KeywordExtractor]. Enrichment
  идёт ВНУТРИ pipeline.run(), не отдельным шагом.
- LLM по доменам (enrichment/query): ast/scripts/llm_provider.py
  (OpenAILike-клиент), оба — z.ai glm-5.3-flash через coding-эндпоинт.
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
- Источник кода — репозиторий launcher (/Users/admin/Documents/Sotbifay/Projects/launcher),
  путь src/Launcher/Assets/Scripts (задаётся RAG_CODE_PATH в .env). Python-тул
  индексирует код лаунчера, не себя.
- Зависимости: requirements.txt (прямые импорты, pinned).
- Метаданные чанка: file_path/file_name (от SimpleDirectoryReader/CodeSplitter),
  section_summary (SummaryExtractor), excerpt_keywords (KeywordExtractor).
- Любые новые знания/решения/статусы фиксируй в .memory/ (обнови нужный файл).
- Не коммить без явного запроса.
