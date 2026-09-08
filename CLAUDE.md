
Работаем над RAG-системой проекта launcher (VR Club Launcher, C#/Unity).
Проект: /Users/admin/Documents/Sotbifay/Projects/launcher
Это ЛОКАЛЬНЫЙ Python dev-тул (llama-index 0.14.x) для семантического поиска по
коду лаунчера. НЕ часть Unity-приложения — отдельный tooling. Python-код лежит
в RAG/, использует .venv в корне репо.

RAG-система НЕ описана в основной wiki лаунчера (проверено) — её нет в index.md.
Первый источник знаний про неё — RAG/.memory/ (см. ниже), а не вики и не код.

ПЕРВЫЙ ШАГ — прочитай память системы, прежде чем что-либо делать:
  RAG/.memory/README.md        — индекс + текущее состояние (начни отсюда)
  RAG/.memory/architecture.md  — конвейер, компоненты, метаданные чанка, языки
  RAG/.memory/usage.md         — запуск, env-переменные, CLI, примеры
  RAG/.memory/roadmap.md       — что готово, что заблокировано, план шага 2

КОНТЕКСТ (коротко):
- Структура: RAG/ast/ (домен AST-чанкинга), RAG/repomix/ (домен repomix).
- Главный скрипт: RAG/ast/scripts/ast_llamaindex.py — IngestionPipeline с
  transformations [CodeSplitter, SummaryExtractor, KeywordExtractor]. Enrichment
  идёт ВНУТРИ pipeline.run(), не отдельным шагом.
- LLM по доменам (enrichment/query): RAG/ast/scripts/llm_provider.py
  (OpenAILike-клиент), оба — z.ai glm-5.3-flash через coding-эндпоинт.
- КОНФИГУРАЦИЯ: единственная точка — корневой .env workspace (+ коммитимый
  шаблон .env.example). Читает/валидирует env ТОЛЬКО RAG/ast/scripts/config.py;
  дефолтов tunable-параметров в коде нет, legacy GLM_* и scripts/.env удалены.
  Самопроверка: .venv/bin/python RAG/ast/scripts/config.py.
- Поддержка языков: ТОЛЬКО C# (.cs). Реестр SUPPORTED_LANGUAGES в ast_llamaindex.py
  готов к добавлению kotlin/java/js одной строкой (+ pip tree-sitter-пакет).
- Запуск:
    .venv/bin/python RAG/ast/scripts/ast_llamaindex.py --path <папка .cs> [--no-enrich] [--max-files N] [--output <jsonl>]
  --no-enrich = оффлайн-чанкинг без LLM (быстро, без ключа).

СОГЛАШЕНИЯ:
- Источник кода: src/Launcher/Assets/Scripts (C# лаунчера). Python-тул индексирует
  код лаунчера, не себя.
- Зависимости: RAG/requirements.txt (прямые импорты, pinned).
- Метаданные чанка: file_path/file_name (от SimpleDirectoryReader/CodeSplitter),
  section_summary (SummaryExtractor), excerpt_keywords (KeywordExtractor).
- Любые новые знания/решения/статусы фиксируй в RAG/.memory/ (обнови нужный файл).
- Не коммить без явного запроса. git: staged renames src/Launcher→RAG/scripts
  (от старого переноса) + untracked RAG/ast/ — при коммите git разрулит сам.

