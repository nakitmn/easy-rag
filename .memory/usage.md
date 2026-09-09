# Запуск RAG (домен `ast/`)

## Установка

```bash
# из корня репо, в существующий .venv
.venv/bin/pip install -r requirements.txt
```

## Конфигурация: единая точка — корневой `.env` (2026-09-07)

Вся конфигурация RAG-тула — переменные корневого `.env` репо RAG (шаблон —
коммитимый `.env.example`). Читает и валидирует их **только**
`ast/scripts/config.py`; **дефолтов tunable-параметров в коде нет**.

- Группы-акцессоры (frozen-датаклассы): `config.paths()` / `config.chunking()` /
  `config.embed()` / `config.retrieval()` / `config.enrichment_llm()` /
  `config.query_llm()`. Читают env в момент вызова, ничего не кэшируют.
- Валидация **контекстная**: переменная проверяется, когда реально нужна —
  `--no-enrich` не требует LLM-переменных, `--no-llm` не требует `RAG_QUERY_*`.
- Отсутствующая или пустая переменная → `ConfigError` со списком всех проблем
  группы и путём к `.env`. Единственное исключение: `RAG_MAX_FILES` — пусто
  легально (= без лимита).
- Приоритет: переменные процесса > `.env` (`load_dotenv(override=False)`, грузит
  только корневой `.env`; загрузка `scripts/.env` и legacy `GLM_*` **удалены**).
- Относительные пути (`RAG_CODE_PATH` и др.) резолвятся от корня репо RAG.
- Самопроверка без сети (ключи маскируются):

```bash
.venv/bin/python ast/scripts/config.py
```

### Полная таблица переменных (все обязательны, кроме RAG_MAX_FILES)

| переменная | значение | назначение |
|------------|---------|------------|
| `RAG_ENRICHMENT_BASE_URL` | `https://api.z.ai/api/coding/paas/v4` | enrichment-домен: z.ai coding-эндпоинт |
| `RAG_ENRICHMENT_MODEL` | `glm-5.3-flash` | модель enrichment |
| `RAG_ENRICHMENT_API_KEY` | *(секрет)* | ключ enrichment (Coding Plan) |
| `RAG_ENRICHMENT_TEMPERATURE` | `0.1` | температура enrichment-LLM |
| `RAG_ENRICHMENT_CONTEXT_WINDOW` | `32768` | контекстное окно (OpenAILike не знает реестр GLM) |
| `RAG_ENRICHMENT_PREFLIGHT_TIMEOUT` | `10.0` | таймаут GET /models префлайта в build_index |
| `RAG_QUERY_BASE_URL` | `https://api.z.ai/api/coding/paas/v4` | query-домен: тот же coding-эндпоинт |
| `RAG_QUERY_MODEL` | `glm-5.3-flash` | модель query |
| `RAG_QUERY_API_KEY` | *(секрет)* | ключ query |
| `RAG_QUERY_TEMPERATURE` | `0.1` | температура query-LLM |
| `RAG_QUERY_CONTEXT_WINDOW` | `128000` | контекстное окно query-LLM |
| `RAG_CODE_PATH` | `../launcher/src/Launcher/Assets/Scripts` | папка с исходниками |
| `RAG_OUTPUT_PATH` | `ast/data/ast_chunks.jsonl` | выходной JSONL (шаг 1) |
| `RAG_INDEX_DIR` | `ast/data/vector_index` | каталог векторного индекса (шаг 2) |
| `RAG_NUM_WORKERS` | `4` | конкурентность extractors |
| `RAG_MAX_FILES` | *(пусто)* | лимит файлов; пусто = без лимита |
| `RAG_CHUNK_MAX_CHARS` | `400000` | whole-file чанкинг (файл ≤10K строк = 1 чанк) |
| `RAG_NUM_KEYWORDS` | `5` | keywords на чанк (KeywordExtractor) |
| `RAG_BATCH_SIZE` | `50` | файлов на батч ингеста в build_index (персист после каждого батча); НЕ путать с `RAG_EMBED_BATCH_SIZE` |
| `RAG_EMBED_MODEL` | `BAAI/bge-m3` | embedding-модель HF; **входит в `rag_profile` — смена = полная переиндексация** |
| `RAG_EMBED_DEVICE` | `auto` | `auto` = авторедетект mps/cpu, либо явно `mps`/`cpu`/`cuda` |
| `RAG_EMBED_MAX_LENGTH` | `8192` | max_length embedding-токенизации |
| `RAG_EMBED_BATCH_SIZE` | `16` | batch эмбеддинга |
| `RAG_TOP_K` | `20` | чанков на запрос (query.py; поднят с 6 после полного прогона — шире выдача на корпусе 1634 файлов) |
| `RAG_SOURCE_MAX_CHARS` | `6000` | обрезка кода чанка в контексте LLM |

Модели `glm-4.7-flash` у z.ai **не существует** (линейка: glm-4.5,
glm-4.5-air, glm-4.6, glm-4.7, glm-5, glm-5-turbo, glm-5.1, glm-5.2,
glm-5.3, glm-5.3-flash — проверено GET /models).

## CLI-аргументы

CLI-аргумент перекрывает env; дефолты CLI-флагов берутся из конфига
(то есть из `.env`). У всех скриптов `ConfigError` печатается одной
понятной ошибкой (exit 1), без трейсбека.

### `ast_llamaindex.py` (шаг 1 — дебаг-нарезка в JSONL)

| флаг | default | назначение |
|------|---------|------------|
| `--path` | `$RAG_CODE_PATH` | папка с исходниками |
| `--output` | `$RAG_OUTPUT_PATH` | выходной JSONL |
| `--no-enrich` | выкл | только AST-чанкинг, без LLM (оффлайн) |
| `--max-files` | `$RAG_MAX_FILES` | лимит загружаемых файлов |

### `build_index.py` (шаг 2 — построение/обновление векторного индекса)

| флаг | default | назначение |
|------|---------|------------|
| `--path` | `$RAG_CODE_PATH` | папка с исходниками |
| `--index-dir` | `$RAG_INDEX_DIR` | каталог индекса |
| `--no-enrich` | выкл | без LLM-enrichment (оффлайн; профиль меняется → полная переиндексация) |
| `--max-files` | `$RAG_MAX_FILES` | лимит загружаемых файлов (теперь учитывает env — раньше игнорировал) |
| `--batch-size` | `$RAG_BATCH_SIZE` (50) | файлов на батч: персист индекса после каждого батча |
| `--prune` | выкл | удалить из индекса документы, отсутствующие в текущем прогоне |
| `--reset` | выкл | снести каталог индекса перед прогоном |

Требования: с `enrich=True` нужен доступный enrichment-эндпоинт (префлайт
проверяет его до прогона); с `--no-enrich` — полностью оффлайн. Повторные
прогоны инкрементальны (неизменённые файлы пропускаются до LLM/эмбеддинга).

### Батч-персист и resume (build_index, с 2026-09-08)

Документы идут батчами по `RAG_BATCH_SIZE`, индекс персистится после каждого
батча → падение длинного прогона теряет максимум текущий батч; перезапуск
продолжает с места остановки (docstore-дедуп). Упавший батч деградирует до
per-file: битые файлы попадают в `files_failed` (сводка в конце, exit 0) и
ретраятся повторным прогоном; 10 файлов подряд с ошибками останавливают
прогон (guard от недоступного эндпоинта). Полностью no-op батч не персистится
(быстрый повторный прогон).

### `query.py` (шаг 2 — вопрос по коду)

| флаг | default | назначение |
|------|---------|------------|
| `question` / `-q` | — | вопрос |
| `--top-k` | `$RAG_TOP_K` (20) | сколько чанков доставать |
| `--index-dir` | `$RAG_INDEX_DIR` | каталог индекса |
| `--no-llm` | выкл | retrieval-дебаг: чанки + score, без LLM (не требует `RAG_QUERY_*`) |

## Примеры

### Проверить конфигурацию (без сети)

```bash
.venv/bin/python ast/scripts/config.py
```

### Построить индекс (оффлайн, без LLM — быстро)

```bash
.venv/bin/python ast/scripts/build_index.py \
  --path ../launcher/src/Launcher/Assets/Scripts/Chat --no-enrich
```

### Построить enriched-индекс

```bash
.venv/bin/python ast/scripts/build_index.py --path ../launcher/src/Launcher/Assets/Scripts/Chat
# пересоздать с нуля: … --reset
# синхронизировать с удалёнными файлами: … --prune
```

⚠️ Не запускать `--no-enrich` в боевой `RAG_INDEX_DIR` — профиль
`rag_profile` перевернётся в `plain:…` и enriched-состояние будет утрачено
(для оффлайн-тестов указывай `--index-dir <scratch>`).

### Запрос по коду (retrieval + LLM-синтез)

```bash
.venv/bin/python ast/scripts/query.py "Где регистрируются сервисы чата в VContainer?"
.venv/bin/python ast/scripts/query.py "как отправить сообщение" --no-llm --top-k 5
```

### Тест нарезки шага 1 (оффлайн, быстро и бесплатно)

```bash
.venv/bin/python ast/scripts/ast_llamaindex.py \
  --path ../launcher/src/Launcher/Assets/Scripts/Chat --no-enrich
```
Проверено: 48 файлов → 48 чанков (whole-file, `RAG_CHUNK_MAX_CHARS=400000`).
`metadata` = file_path/file_name/…, **без** section_summary/excerpt_keywords
(они появятся только при enrichment).

### Смоук LLM-домена (реальный ping)

```bash
.venv/bin/python ast/scripts/llm_provider.py --domain enrichment
.venv/bin/python ast/scripts/llm_provider.py --domain query
```

### Дев-прогон с лимитом файлов

```bash
.venv/bin/python ast/scripts/ast_llamaindex.py \
  --path ../launcher/src/Launcher/Assets/Scripts/Chat --max-files 5
```

### Программный вызов (из другого скрипта)

```python
import sys; sys.path.insert(0, "ast/scripts")
from ast_llamaindex import run, save_chunks

nodes = run(code_path="../launcher/src/Launcher/Assets/Scripts/Chat", enrich=False)  # оффлайн
# nodes = run(enrich=True)  # полный (нужен RAG_ENRICHMENT_* в .env)
save_chunks(nodes, output_path="out.jsonl")
```

## Где результат

- `ast/data/ast_chunks.jsonl` — JSONL шага 1 (`{id, text, metadata}` на строку),
  дебаг-артефакт.
- `ast/data/vector_index/` — векторный индекс шага 2 (`docstore.json`,
  `index_store.json`, `default__vector_store.json`). **Runtime-артефакт**
  (regenerируется), в `.gitignore`.

## Правило workspace

Работаем в отдельном репо RAG и его локальном `.venv`; репозиторий
`~/Documents/Sotbifay/Projects/launcher` — источник кода для индексации
(читаем, не меняем).
