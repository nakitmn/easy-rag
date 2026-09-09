# Архитектура RAG (домен `ast/`)

## Конвейер

```
SimpleDirectoryReader(.cs)          загрузка файлов (рекурсивно, exclude по glob)
        │ documents
        ▼
IngestionPipeline.run(documents=…)
  transformations = [
    CodeSplitter(c_sharp, tree-sitter)   ← нарезка по AST (RAG_CHUNK_MAX_CHARS=400000 — файл ≤10K строк = 1 цельный чанк)
    SummaryExtractor (LLM, промпт под код)   ← metadata["section_summary"]
    KeywordExtractor (LLM, промпт под код)   ← metadata["excerpt_keywords"]
  ]
        │ nodes (конкурентно, num_workers=4)
        ▼
_inject_context_headers()           пост-факту: '// File: … / Chunk i/n' в начало текста
        │
        ▼
save_chunks() → ast_chunks.jsonl    {id, text, metadata} на строку
```

Ключевое отличие от предыдущей версии: enrichment — это **transformation в pipeline**,
а не отдельный ручной цикл. Конкурентность встроена в extractors (`num_workers`).

**Важно:** `CodeSplitter` не имеет параметра `metadata_extractor` (проверено по
сигнатуре в 0.14.23), поэтому enrichment нельзя прикрепить к сплиттеру напрямую —
только через список `transformations` в `IngestionPipeline`.

### Почему два прохода по расширениям

`CodeSplitter` создаётся под один язык (`language` фиксируется в конструкторе), а
`IngestionPipeline.transformations` — линейный список без условной логики. Поэтому
для каждого расширения из `SUPPORTED_LANGUAGES` строится свой pipeline со своим
CodeSplitter (extractors общие). Сейчас расширение одно (.cs) — один проход.

## Компоненты

### `config.py` (единственная точка конфигурации, с 2026-09-07)
Все tunable-параметры системы живут в корневом `.env` репо (шаблон —
коммитимый `.env.example`); `config.py` — единственный модуль, который читает
и валидирует env. Дефолтов tunable-параметров в коде нет.
- Группы-акцессоры → frozen-датаклассы: `paths()` / `chunking()` / `embed()` /
  `retrieval()` / `enrichment_llm()` / `query_llm()`. env читается **в момент
  вызова** (валидация контекстная: `--no-enrich` не трогает LLM-группы,
  `--no-llm` — query-группу), ничего не кэшируется.
- Ошибки накапливаются по группе и падают одним `ConfigError` (имена переменных
  + путь к `.env`). Пустая строка = отсутствует; единственное исключение —
  `RAG_MAX_FILES` (пусто = без лимита). Парсинг строгий (int ≥ min, float
  isfinite/>порог) — мусорные значения больше не игнорируются молча.
- При импорте: `WORKSPACE_ROOT` — первый каталог вверх от config.py с `.git`
  (корень репо; переживает изменение вложенности — фикс-глубина `parents[3]`
  сломалась при миграции в отдельный репо), guard «не внутри git-репо»,
  `load_dotenv(корневой .env, override=False)` — переменные процесса
  приоритетнее .env. Загрузка `scripts/.env` и legacy `GLM_*` удалены.
- `resolve_path()` — относительные пути от корня репо.
- `python config.py` — самопроверка всех групп (ключи маскируются), exit 1 при
  провале. Импортирует только stdlib+dotenv (нет циклов с остальными модулями).

### `ast_llamaindex.py`
- `SUPPORTED_LANGUAGES` — реестр: расширение → `{language, grammar}` (только
  структурные поля; `max_chars` берётся из `config.chunking().chunk_max_chars`).
- `_build_parser(grammar_module)` — лениво импортирует pip-пакет tree-sitter-грамматики,
  собирает `Parser`. `tree-sitter-language-pack 1.x` не отдаёт c_sharp через скачивание,
  поэтому грамматики — отдельные пакеты (`tree-sitter-c-sharp`).
- `run(code_path=None, llm, enrich, max_files=None)` — главный вход;
  None-параметры берутся из `config.paths()`/`config.chunking()`. При
  `enrich=False` — только AST-чанкинг (LLM-группы конфига не читаются).
- `_build_extractors(llm, num_workers, num_keywords)` — параметры передаются
  явно (из `config.chunking()`), модульных глобалов нет.
- `_inject_context_headers(nodes)` — группировка по `file_path`, нумерация чанков,
  заголовок в начало текста. Выполняется **после** extractors, поэтому summary/keywords
  строятся на чистом коде (без заголовка).
- `save_chunks(nodes, output_path=None)` — JSONL (None → `config.paths().output_path`).

### `llm_provider.py`
Фабрика LLM по доменам назначения (`enrichment`/`query`; `DOMAINS` — кортеж
имён). Конфигурация домена полностью из `.env` через `config.enrichment_llm()` /
`config.query_llm()` (модель, base_url, ключ, температура, контекстное окно).
- `build_llm(domain, *, model, api_key, base_url, temperature, context_window,
  is_chat_model=True)` — OpenAILike-клиент; kwargs — точечный override
  отдельных полей поверх ПОЛНОГО конфига домена (недостроенный .env override'ом
  не обойти). + алиасы `build_enrichment_llm()` / `build_query_llm()`.
- `configure_global_llm(domain)` — ставит `Settings.llm`.
- `LLMConfigError` — неизвестный домен; ошибки конфигурации — `config.ConfigError`.
- Смоук: `python llm_provider.py --domain enrichment|query` (реальный ping).
Потребители зависят только от абстракции `LLM`; смена модели = правка `.env`,
пайплайны не трогаются.

### `build_index.py` (шаг 2)
Ингест кода → персистентный векторный индекс. Переиспользует из
`ast_llamaindex` только импорты (`EXCLUDE_DIRS`, `SUPPORTED_LANGUAGES`,
`_build_extractors`, `_build_splitter`, `_inject_context_headers`).
- `build_embed_model(model_name, device, max_length, batch_size)` — все
  параметры = None → из `config.embed()` (`BAAI/bge-m3`, max_length 8192,
  batch 16); device: `RAG_EMBED_DEVICE=auto` → авторедетект mps/cpu.
- `_normalize_documents()` — стабильные `doc.id_ = sha256("rag-doc:<rel_path>")`,
  `file_path` → repo-relative (иначе абсолютные пути попадают в doc.hash),
  выбрасывание `creation_date`/`last_modified_date` (mtime дрейфует при
  checkout/переносе корпуса — иначе docstore-дедуп видит «изменения» во всех
  файлах), маркер `metadata["rag_profile"] = "enrich|plain:<embed_model>"`.
- `HeaderMetadataTransform(TransformComponent)` — заголовки `// File: …` ПОСЛЕ
  extractors и ДО embedding + `excluded_embed_metadata_keys` (в эмбеддинг идут
  `file_path` + `section_summary` + `excerpt_keywords` + код с заголовком; не
  идут file_name/размеры/даты/rag_profile). Plain-callable не подходит —
  pipeline 0.14.x pydantic-валидирует transformations.
- Pipeline по расширению: `transformations=[splitter, *extractors, header,
  embed_model]` + `vector_store=` + `docstore=`, стратегия всегда **UPSERTS**.
- **Батч-персист** (2026-09-08): документы режутся на батчи по
  `RAG_BATCH_SIZE` (`config.chunking()`, CLI `--batch-size`); после каждого
  батча — `insert_nodes` + `sc.persist()` (no-op батч не персистится).
  Упавший батч деградирует до per-file-прогона (битые файлы → `failed`-список
  со сводкой и `files_failed` в статистике; exit 0; ретрай — повторным
  прогоном). Guard `_MAX_CONSECUTIVE_FILE_FAILURES = 10`: подряд упавшие
  файлы останавливают прогон (RuntimeError) — защита от полного outage
  эндпоинта, прогресс уже персистнут. Финальный безусловный `sc.persist()`
  флашит prune/деградационные удаления при all-no-op батчах. Падение батча
  происходит в трансформациях — до записей в сторы (проверено по исходникам
  0.14.23), частичных записей не остаётся; `files_processed` считается по
  смене hash исключая failed (изменившийся-и-упавший док теряет hash
  in-memory ещё в `_handle_upserts`).
- Инкрементальность: docstore-дедуп llama-index по doc.hash (текст+metadata);
  неизменённые файлы пропускаются до сплиттера — 0 LLM, 0 эмбеддингов. Смена
  `rag_profile` (enrich↔plain, другая embed-модель) меняет hash → полная
  переиндексация (защита от смешивания несовместимых векторов).
- `--prune` — **ручная** реализация (не `UPSERTS_AND_DELETE`): встроенная
  стратегия в 0.14.x сносит чанки из docstore как «пропавшие документы»
  (`get_all_document_hashes()` возвращает id и чанков тоже). Ручной prune
  отбирает только `isinstance(n, Document)` и делает `docstore.delete_ref_doc()`
  + `vector_store.delete()` — ровно то же, что pipeline делает для изменённых.
- Storage: `index_store.json` есть → `load_index_from_storage` (реюз
  index_struct), иначе `VectorStoreIndex(nodes=[])`; после run —
  `index.insert_nodes(all_nodes)` (чанки → docstore: их тексты нужны ретриверу
  при `stores_text=False`; идемпотентно, эмбеддинги уже на узлах) →
  `sc.persist()`.
- `_ensure_enrichment_llm_alive()` — префлайт GET `{base}/models` перед
  LLM-прогоном (понятная ошибка вместо httpx-трейсбека; таймаут
  `RAG_ENRICHMENT_PREFLIGHT_TIMEOUT`). Шлёт `Authorization: Bearer <key>` —
  z.ai без заголовка отвечает 401 (LM Studio ключ игнорировала, из-за чего
  баг долго не всплывал). Контекстный триггер LLM-группы конфига: вызывается
  только при `enrich=True`.
- `--max-files` — дефолт из `RAG_MAX_FILES` (раньше build_index игнорировал
  env, в отличие от ast_llamaindex — асимметрия закрыта).

### `query.py` (шаг 2)
- `load_index(index_dir=None)` — `Settings.embed_model` ставится ДО загрузки;
  SystemExit с подсказкой, если индекса нет; None → `config.paths().index_dir`.
- Retriever `index.as_retriever(similarity_top_k=RAG_TOP_K)`; запрос
  эмбеддится той же моделью, что индексировала.
- Синтез: `build_query_llm().chat([system, user])`; контекст — блоки
  `[i] Файл / Описание(section_summary) / Ключевые слова / ```csharp код```
  (обрезка `RAG_SOURCE_MAX_CHARS`); вывод ответа + «Источники»
  (`file_path`, score, summary).
- `--no-llm` — retrieval-дебаг (чанки+score) без LLM; `RAG_QUERY_*` не
  требуются (контекстная валидация).

## Метаданные чанка

| ключ | источник | когда |
|------|----------|-------|
| `file_path`, `file_name` | SimpleDirectoryReader / CodeSplitter | всегда (`file_path` нормализован к repo-relative) |
| `file_size` | SimpleDirectoryReader | всегда (в эмбеддинг не идёт); `creation_date`/`last_modified_date` выбрасываются нормализацией шага 2 (с 2026-09-08 не входят в doc-hash) |
| `section_summary` | SummaryExtractor | только при `enrich=True` |
| `excerpt_keywords` | KeywordExtractor | только при `enrich=True` |
| `rag_profile` | build_index.py | шаг 2; `"enrich\|plain:<embed_model>"` — маркер профиля индексации |

`node.text` дополнительно получает контекстный заголовок `// File: … / Chunk i/n`.

## Storage шага 2

```
ast/data/vector_index/        ← persist-директория (runtime → .gitignore)
  docstore.json                   ← документы + чанки (тексты для ретривера)
  index_store.json                ← index_struct (VectorStoreIndex)
  default__vector_store.json      ← векторы (SimpleVectorStore, именаовано по коллекции)
```

Ключевой паттерн 0.14.23 (проверено): `IngestionPipeline(vector_store=…)` НЕ
регистрирует index_struct → `load_index_from_storage` падает. Решение: релоад
существующего индекса либо `VectorStoreIndex(nodes=[])` + `insert_nodes()` после
прогона. `from_vector_store(SimpleVectorStore)` не работает (stores_text=False).

Персист полный перезаписью 3 JSON и неатомарен (`docstore.json` можно
повредить крэшем в момент записи — окно мало, принято; батчи лишь чаще
пишут). Повторные `sc.persist()`/`insert_nodes` идемпотентны.

## Расширение языков (kotlin/java/js)

1. `pip install tree-sitter-kotlin` (и т.п.) — пакет грамматики.
2. Добавить запись в `SUPPORTED_LANGUAGES` в `ast_llamaindex.py`:
   ```python
   ".kt": {"language": "kotlin", "grammar": "tree_sitter_kotlin",
           "max_chars": 400_000},
   ```
   `_build_parser` и цикл по расширениям сделают остальное. Заглушки для kt/java/js
   уже лежат закомментированными в реестре.

## Зависимости

Прямые (см. `requirements.txt`): `llama-index-core`, `llama-index-llms-openai-like`,
`llama-index-embeddings-huggingface` (+ транзитивно torch/sentence-transformers),
`tree-sitter`, `tree-sitter-c-sharp`, `python-dotenv`. Остальные (pydantic, openai,
httpx, numpy) — транзитивные.
