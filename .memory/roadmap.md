# Roadmap RAG

## Что готово (шаг 1 — AST + enrichment)

- `IngestionPipeline` с `transformations=[CodeSplitter, SummaryExtractor, KeywordExtractor]`.
- Промпты под код, оба на английском: summary — назначение/входы-выходы/побочные
  эффекты, «strictly English»; keywords — термины из кода, явный запрет
  генерических категорий («entities», «patterns», …).
- Конкурентность (`num_workers=4`).
- Контекстные заголовки в тексте чанка.
- CLI (`--path/--output/--no-enrich/--max-files`) + env-override.
- Реестр языков `SUPPORTED_LANGUAGES` (сейчас .cs; kt/java/js закомментированы).
- `requirements.txt`, доменная структура `RAG/ast/`, `RAG/repomix/`.
- Локальный `.venv` в корне ворктри (Python 3.13).

## Нарезка + enrichment: финальная конфигурация (верифицировано 2026-09-04)

Решения (пользователь):
- `max_chars=400_000` (≈10 000 строк × 40 симв.): файл ≤10K строк не режется
  вовсе → **1 чанк = 1 файл**, сироты невозможны по построению. В проекте
  максимум 609 строк / 21 КБ (VideoPlayerFacade.cs).
- Enrichment **строго на английском** — русский вывод отменён как усложняющий
  систему; попутно это убирает CJK-дрейф qwen. На шаге 2 bge-m3 мультиязычная,
  RU-запросы сматчатся с EN-summary.

Почему сироты вообще возникали: `CodeSplitter._chunk_node` рекурсивно спускается
в AST, пока узел больше `max_chars`, и выбрасывает фрагменты («namespace …»,
«{», «}»). `chunk_lines`/`chunk_lines_overlap` — мёртвые параметры 0.14.x
(в реестре отсутствуют). Эмпирика (5 файлов Chat): `max_chars` 2000 → 13 чанков
(4 мусорных), 4000 → 11 (4 мусорных), 6000 → 5 (0 мусора, но файлы >6 КБ режутся
с сиротами), 400_000 → все файлы цельные.

Верификация 2026-09-04 (папка Chat, 48 файлов, qwen2.5-7b-instruct через LM
Studio):
- Оффлайн: 48 файлов → 48 чанков; проверка «тело чанка ↔ metadata file_size» —
  0 расхождений (каждый чанк — целый файл; MessagesListPresenter.cs 15 КБ —
  бывший генератор сирот — теперь 1 цельный чанк).
- Enrichment: 48/48 с `section_summary`+`excerpt_keywords`, ~3.3 мин,
  0 CJK, 0 кириллицы, 0 дублей summary; keywords — реальные идентификаторы
  (R3, VContainer.Unity, UniTask, Cysharp.Threading.Tasks, …).
- Найден и починен регресс английского промпта: модель копировала
  слова-категории из текста промпта («entities, technologies, patterns, APIs» —
  было 24/48 чанков). Фикс — явный запрет в промпте → 0/48.

Риск: файл >~120K символов переполнит контекст enrichment-LLM (qwen 32K) —
в проекте таких нет. Появятся гиганты — вернуть нарезку или фильтрацию.

## Не проверено

- ~~**Устойчивость на полном прогоне**: API-fail (429/timeout) внутри
  extractors валит весь `pipeline.run()`~~ — **закрыто батч-персистом**
  (2026-09-08): per-file фолбэк изолирует битые файлы, персист после каждого
  батча + docstore-дедуп дают resume, guard останавливает прогон при полном
  outage (см. соответствующую секцию).

## Шаг 2 — embedding + векторная БД + query CLI (реализован, проверен на Chat 2026-09-04)

Железо: **Apple M1 Max, 32 GB unified memory**, `device="mps"`.

Что сделано (`build_index.py` + `query.py`, см. architecture.md):
- embedding: локальная `BAAI/bge-m3` (mps, max_length 8192, dim 1024, ~2.3 GB
  в HF-кэше) — мультиязычная RU↔EN (RU-запрос ↔ EN-summary работает).
- vector store: `SimpleVectorStore`, persist в `RAG/ast/data/vector_index/`.
- Эмбеддится summary + keywords + file_path + код с заголовком (не голый код).
- LLM по доменам: `enrichment` = LM Studio qwen2.5-7b-instruct (legacy
  `GLM_*` работает), `query` = GLM-4.7 z.ai (`RAG_QUERY_API_KEY`).
- Инкрементальность: стабильные doc-id (sha256 от repo-relative пути) +
  docstore-дедуп + маркер `rag_profile`.
- `--prune` ручной (см. architecture.md — почему не UPSERTS_AND_DELETE),
  `--reset`, `--no-enrich`.

Верификация (папка Chat, 48 файлов):
- Оффлайн-билд: 48 → 48 чанков/векторов, ~30 сек (setup модели ~9 с).
- Enriched-билд: ~4 мин (из них LLM ~2.3 мин, summaries 2:09 + keywords 1:14).
- Инкрементальность: no-op прогон — 0 processed / ~9.6 сек; правка одного
  файла — ровно 1 processed / 1 chunk; откат — снова 1 processed.
  Утечек векторов нет (upsert по ref_doc_id).
- `--prune` после `git rm` файла: docstore 96→94 (doc+чанк), векторы 48→47,
  предупреждение с точным числом.
- Retrieval (enriched): «Где регистрируются сервисы чата в VContainer?» →
  top-1 `ChatProjectInstaller.cs` (score 0.571, summary про VContainer DI);
  «как отправить сообщение» → top-5: ChatSendMessagePresenter/Chat/
  ChatSendMessageView/ChatMessageSender/ChatMessagesCollection.
- `query.py --no-llm` (дебаг-режим) работает без ключей.

Отклонения от ранних планов (решения пользователя):
- **LLM по доменам, не по провайдерам** (`enrichment`/`query`), query —
  обязательно GLM z.ai. Ollama не устанавливали (LM Studio уже работала).
- Дефолт `UPSERTS` + opt-in `--prune` (не UPSERTS_AND_DELETE) — частичные
  прогоны по подпапке безопасны; позже выяснилось, что встроенный
  UPSERTS_AND_DELETE вообще нельзя использовать с чанками в docstore.

## Смена LLM доменов → z.ai glm-5.3-flash: enrichment (было LM Studio), query (было glm-4.7) (2026-09-07)

Пользователь просил `glm-4.7-flash` — **такой модели у z.ai нет** (GET /models
на обоих эндпоинтах: glm-4.5, glm-4.5-air, glm-4.6, glm-4.7, glm-5,
glm-5-turbo, glm-5.1, glm-5.2, glm-5.3, glm-5.3-flash; запрос несуществующего
имени даёт ошибку 1305 «overloaded» на обоих эндпоинтах). Из flash-линейки
актуальна `glm-5.3-flash`; выбрана она.

Эмпирическое сравнение на MessagesListPresenter.cs (15 КБ, реальные промпты
пайплайна, temperature 0.1):

| модель | summary | keywords | вывод |
|---|---|---|---|
| `glm-4.5-air` | 28.6 с (1312 completion, из них 1043 reasoning) | 13.8 с (636 reasoning) | summary развёрнутый, местами многословный |
| `glm-5.3-flash` | 10.4 с (434 completion, 196 reasoning) | 9.0 с (411 reasoning) | компактнее, точнее по фактам |

Обе: строгий English, реальные идентификаторы, запрет «эха» категорий
соблюдён без дополнительных правок промптов. Выбор **glm-5.3-flash**: в
~2.5-3 раза быстрее на enrichment-задаче; verbose-вывод 4.5-air хуже ещё и
тем, что `section_summary` входит в embedding-текст — длинный summary
разбавляет сигнал. Обе reasoning-модели (первые токены уходят в
reasoning_content) — для «ping»-проверок задавать max_tokens с запаском.

Конфигурация: `RAG_ENRICHMENT_{MODEL,BASE_URL,API_KEY}` в корневом `.env`
(coding-эндпоинт, тот же ключ Coding Plan, что `RAG_QUERY_API_KEY`). Смоук
`llm_provider.py --domain enrichment` пройден. Позже в тот же день
пользователь перевёл и query-домен на `glm-5.3-flash`
(`RAG_QUERY_MODEL=glm-5.3-flash` в корневом `.env`; код-дефолт glm-4.7).
Смоук query-домена пройден.

**Важно про инкрементальность:** doc.hash не зависит от enrichment-модели →
смена модели НЕ переобогащает уже проиндексированные файлы (summary в
docstore останутся от qwen). Для полного переобогащения — `build_index.py
--reset` (Chat: ~4 мин при LM Studio, с 5.3-flash ожидаемо сопоставимо или
быстрее). Полный корпус (1634 файла): при ~10 с/файл serial и 4 воркерах —
ориентировочно ~70-90 мин, но не проверялось.

## Полный тест Chat на glm-5.3-flash (2026-09-07, --reset)

Пересборка enriched-индекса Chat после смены модели (48 файлов):

- **Сборка**: 48/48 processed, 48 чанков, 48 векторов, docstore 96;
  t_total 215 с (~3.6 мин) — быстрее qwen-прогона (~4 мин).
- **Батарея**: 48/48 enriched; 0 CJK; 0 дублей summary; 0 «эха» категорий;
  1 чанк с кириллицей в summary — MessageTimeExtensions.cs цитирует реальные
  литералы кода («Сегодня»/«Вчера»), сам summary на английском → это НЕ
  дрейв языка, а ложное срабатывание regex-проверки.
- **Тело↔файл**: 4 файла (OpenChatButton/*, TemplateMessagePresenterFactory)
  — чанк теряет ровно один завершающий `\n` (артефакт CodeSplitter, не
  модели; проверка 2026-09-04 сравнивала с `file_size` и этого не ловила).
  Контент целый, на retrieval не влияет.
- **Retrieval vs эталон qwen (2026-09-04)**:
  - «Где регистрируются сервисы чата в VContainer?» — ChatProjectInstaller
    #2 (0.599) вместо #1; top-1 OpenChatButtonInstaller (0.600) — разрыв
    0.001, фактически ничья; все top-6 — релевантные инсталлеры. Ок.
  - «как отправить сообщение» — **регресс top-1**: MessageTimeExtensions
    (0.580) обогнал ChatSendMessagePresenter (0.556); эталонные файлы
    (ChatSendMessagePresenter/Sender/View) все в top-6, но мусорный файл
    поднялся на #1.
  - Гипотеза: verbose-summary 5.3-flash (2-4 предложения с failure modes)
    повторяют generic-термины («chat», «message») → раздувание совпадений
    по частотности. Кандидат на фикс: ужать промпт summary (1-2
    предложения, лимит слов), пересборка ~3.6 мин.
- **Попутный баг-фикс**: `_ensure_enrichment_llm_alive` (build_index.py) слал
  GET /models **без Authorization** — LM Studio игнорирует, z.ai отвечает
  401. Теперь шлёт `Bearer <key>` (таймаут 3→10 с).

### Фикс verbose-summary: промпт ужат, retrieval восстановлен (тот же день)

После регресса top-1 в запросе 2 промпт `CODE_SUMMARY_PROMPT` ужат:
«1–2 предложения, максимум 40 слов; только назначение + ключевые
типы/фреймворки; НЕ перечислять входы/выходы/side effects/failure modes».
Пересборка Chat `--reset`: t_total 173 с (~2.9 мин; было 215 с) — короткие
summary дешевле и по LLM-токенам.

Результат (48 файлов):
- Длина summary: 18–39 слов (avg 30), лимит 40 соблюдён всеми чанками.
- Батарея: без изменений (48/48 enriched, 0 CJK, 0 дублей, 0 «эха»; 5
  известных ложных срабатываний: 4× завершающий `\n`, 1× цитаты литералов).
- **Retrieval = эталон 2026-09-04**:
  - «Где регистрируются сервисы чата в VContainer?» → top-1
    **ChatProjectInstaller.cs** (0.621; было #2, 0.599) — совпало с эталоном.
  - «как отправить сообщение» → top-1 **ChatSendMessagePresenter** (0.588),
    далее MessagesListPresenter/ChatMessageSender/ChatSendMessageView;
    MessageTimeExtensions упал с #1 (0.580) до #5 (0.563). Регресс закрыт.
- Вывод: для whole-file чанков короткий fact-dense summary ретривит лучше
  длинного — длинные повторы generic-терминов («chat», «message») топят
  сигнал. Keywords-промпт не менялся.

## Консолидация конфигурации в .env: единый config.py (2026-09-07)

Из кода убраны все дефолты tunable-параметров и вся fallback-логика; единственная
точка конфигурации — корневой `.env` workspace (+ коммитимый шаблон
`.env.example`; в `.gitignore` добавлен `!.env.example`). Чтение/валидация env —
только новый `RAG/ast/scripts/config.py` (6 групп-акцессоров, frozen-датаклассы,
накопление ошибок по группе, контекстная валидация, самопроверка
`python RAG/ast/scripts/config.py`).

Что изменилось:
- llm_provider: удалены DOMAINS-дефолты (qwen/LM Studio, glm-4.7), legacy
  `GLM_*`, загрузка scripts/.env; `build_llm` берёт конфиг из config.py;
  `DOMAINS` — кортеж имён доменов.
- Новые env: `RAG_*_TEMPERATURE`, `RAG_*_CONTEXT_WINDOW`,
  `RAG_ENRICHMENT_PREFLIGHT_TIMEOUT`, `RAG_CHUNK_MAX_CHARS` (был max_chars в
  реестре языков), `RAG_NUM_KEYWORDS` (был keywords=5), `RAG_EMBED_DEVICE=auto`
  (sentinel вместо «пусто = авторедетект»), `RAG_EMBED_BATCH_SIZE` (был
  инлайн-литерал 16), `RAG_QUERY_BASE_URL` (был только код-дефолт).
- `build_index --max-files` теперь учитывает `RAG_MAX_FILES` (асимметрия против
  ast_llamaindex закрыта); парсинг `RAG_MAX_FILES` строгий (мусор — ошибка,
  раньше молча игнорировался).
- Пустая env-строка = ошибка конфигурации (раньше молча проваливалась в дефолт
  по or-цепочкам llm_provider).

Верификация (2026-09-07): самопроверка config.py 6/6 групп (ключи маскированы);
смоук обоих LLM-доменов реальными ping; оффлайн-нарезка Chat 48→48; оффлайн-сборка
в scratch-каталог 48 чанков/48 векторов; оба базлайн-запроса retrieval
байт-в-байт идентичны до/после рефактора; инкрементальный no-op enriched-прогон —
**0 processed / 48 skipped** (входы doc.hash не сдвинулись, индекс НЕ
переиндексировался — главная проверка чистоты рефактора); контекстная валидация:
пустой `RAG_QUERY_API_KEY` → понятная ошибка; пустой `RAG_EMBED_MODEL` при
`--no-llm` → ошибка embed-группы (не query); полностью пустые `RAG_ENRICHMENT_*`
при `--no-enrich` → успех. Поведение системы не изменилось: все прежние
эффективные дефолты перенесены в `.env` значениями.

## Батч-персист в build_index.py (2026-09-08)

Реализован и верифицирован (план из записи 2026-09-07). Механика:
- Документы режутся на батчи по `RAG_BATCH_SIZE` (новая env, дефолт 50 в
  корневом `.env`; группа `chunking()` в config.py, CLI `--batch-size`).
  После каждого батча — `insert_nodes` + `sc.persist()`; no-op батч (0 чанков
  и 0 изменённых hash) не персистится.
- Упавший батч деградирует до per-file-прогона: битые файлы идут в
  `failed`-список (сводка в конце, exit 0), остальные обрабатываются
  (неизменённые доки дедуп пропускает бесплатно). Guard
  `_MAX_CONSECUTIVE_FILE_FAILURES = 10` (модульная константа): 10 файлов
  подряд с ошибками → abort с сообщением (защита от полного outage — ретраи
  OpenAI-клиента иначе жгут время на каждом файле). Прогресс до аборта
  персистнут, перезапуск продолжает с места остановки.
- Финальный **безусловный** `sc.persist()` после всех батчей: флешит
  `--prune`-удаления и удаления упавших доков, когда все батчи no-op (дыра,
  найденная при проектировании: prune удаляет до батчей, а персист в цикле —
  условный).
- Статистика: `files_failed`; `files_processed` по смене hash **исключая
  failed** (изменившийся-и-упавший док теряет hash-запись in-memory ещё в
  `_handle_upserts` — без исключения считался бы processed ложно).

Проверено по исходникам llama-index 0.14.23: падение батча происходит в
трансформациях — до любых записей (упавший батч не оставляет частичных
записей); повторные `run()` на одном пайплайне безопасны; `insert_nodes` и
`persist` идемпотентны. Отвергнутая альтернатива: `raise_on_error=False` у
BaseExtractor — тихо индексирует чанки без summary, без отчёта.

Верификация (2026-09-08, оффлайн — scratch-каталоги, Chat 48 файлов):
- config: самопроверка показывает batch_size; пустая RAG_BATCH_SIZE →
  понятная ConfigError.
- Смоук `--no-enrich --reset --batch-size 7`: 7 батчей, 48 processed /
  48 чанков / docstore 96 / vectors 48 (= бейзлайн).
- No-op повтор: 0 processed / 48 skipped, md5 всех 3 JSON идентичны
  (no-op батчи индекс не трогают).
- Изоляция отказа (monkeypatch «отравленного» файла): батч деградировал,
  1 файл в failed, 47 обработаны, exit 0; повторный прогон без патча —
  ровно 1 processed (бесплатный ретрай failed). Попутно подтверждён abort:
  сломанный патч дал 10 неудач подряд → RuntimeError-останов.
- Prune-флаш: `--prune --max-files 47` при полностью no-op батчах →
  docstore 96→94, vectors 48→47 (финальный персист работает).
- Kill процесса на середине (SIGTERM на батче 5): персистнуты батчи 1–4
  (56 записей), рестарт добрал ровно 20 оставшихся файлов, итог 96/48.
- Enriched-прогон Chat `--reset --batch-size 10` (z.ai): 48/48, 0 failed,
  5 батчей с персистом (38→71→114→153→182 с), t_total 192.5 с (~3.2 мин —
  накладные на батчи пренебрежимы). Retrieval: «Где регистрируются сервисы
  чата в VContainer?» → top-1 ChatProjectInstaller (0.603) = эталон;
  «как отправить сообщение» → все эталонные файлы в top-6
  (MessagesListPresenter 0.590 → ChatSendMessageView/ChatMessageSender/
  ChatSendMessagePresenter 0.561 → ChatMessagesCollection), дрейф позиций
  внутри top-6 — недетерминизм переобогащения при --reset (temp 0.1), не
  эффект батчирования (границы батчей на содержимое чанков не влияют).

Нюанс для полного прогона: `docstore.json` пишется неатомарно (как и раньше),
окно повредить его крэшем в момент персиста мало; батчи только сокращают
интервалы между записями.

## Полный прогон по корпусу — СДЕЛАН (2026-09-08)

Боевой индекс корпуса: **1634 файла → 1634 чанка → 1634 вектора** (1:1:1,
0 потерь), enriched (z.ai glm-5.3-flash, bge-m3). Прогон шёл батчами по 50
(33 батча, ~3 мин/батч); был остановлен извне на батче 16 — перезапуск
продолжил с места остановки (батч-персист отработал на полном масштабе:
персистнуто 15 батчей, потерян только in-flight). Полный no-op повтор —
33 батча / 1634 skipped / 22 с.

**Нюанс путей (пойман и закрыт)**: doc id = sha256(repo-relative file_path)
зависит от corpus_root. Прогон по подпапке (`--path …/Scripts/Chat`) и по
корпусу (`RAG_CODE_PATH=…/Scripts`) дают РАЗНЫЕ id одних и тех же файлов →
Chat-файлы при полном прогоне не дедупнулись и переобогатились заново, а 48
старых записей (пути без префикса `Chat/`) остались дублями. Вычищено
`--prune` (48 удалено, итог ровно 1634×2 записей docstore). Правило:
боевой индекс строим только от corpus root; эксперименты с другим `--path`
— в scratch `--index-dir`.

Sanity-retrieval на доменах вне Chat (top-5, `--no-llm`):
- «как обнаруживается обрыв VPN» → VpnService, VpnDisconnectedDialog/View/
  Trigger, ConnectToVpnWindowView (0.544-0.568) — все релевантные.
- «как touch события от VR контроллера попадают в WebView» → Page, PagePresenter,
  PageView (WebBrowser input pipeline) + TimelineView (0.558-0.580).
- «как устроена T9 клавиатура и установка текста» → KeyboardT9, T9CoreInstaller,
  KeyboardLifetimeScope, T9SuggestionItemView, T9ViewInstaller (0.604-0.659).

Система в рабочем состоянии: `query.py` готов к использованию. Пользователь
поднял `RAG_TOP_K` 6→20 (шире выдача на полном корпусе).

## Открыто / следующие шаги

- **Использование и улучшение по запросу**: ретривер/синтез подстраиваем по
  факту реальных запросов. Кандидаты: реранкер поверх top-k, гибрид с
  лексическим поиском, MCP-сервер к индексу (доступ из Claude Code),
  фильтры по доменам. Операционно: после рефакторинга — обычный прогон
  (тронутые файлы), после удалений/переименований — `--prune`.
- ~~`RAG_QUERY_API_KEY` (z.ai)~~ — **закрыто**: ключ — GLM Coding Plan, работает
  только через выделенный эндпоинт `https://api.z.ai/api/coding/paas/v4`
  (стандартный `paas/v4` давал 429/1113 «Insufficient balance» — это не
  отсутствие баланса, а неверный эндпоинт для ключа плана). Дефолт
  query-домена переключён на coding-эндпоинт; полный смоук пройден на реальном
  GLM-4.7 (структурированный ответ со всеми сервисами VContainer и цитатами).
- Опционально: `IngestionCache` для resume на длинных прогонах.

## Замечания / риски

- `ast_chunks.jsonl` — дебаг-артефакт шага 1; шаг 2 кладёт чанки прямо в
  vector store (+ docstore для текстов).
- git: staged renames `src/Launcher → RAG/scripts` (старый перенос) +
  untracked `RAG/ast/`; при коммите git разрулит сам.
- Whole-file чанки: эмбеддинг bge-m3 (8K токенов) покрывает крупнейший файл
  (~21 КБ ≈ 5–6K токенов); файлы >~30K символов будут обрезаться на
  эмбеддинге — пока таких нет.
- Вся работа — только в orca-workspace (ветка RAG) и его `.venv`; оригинальный
  чекаут не трогаем. Не коммитим без явного запроса пользователя.
