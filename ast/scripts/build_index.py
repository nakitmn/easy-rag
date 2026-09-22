"""Шаг 2: AST-чанкинг + enrichment + эмбеддинг → персистентный векторный индекс.

Нарезку, extractors и заголовки переиспользует из ``ast_llamaindex`` (файл не
редактируется). Enrichment — LLM домена ``enrichment`` (конфиг в корневом
``.env``, см. ``config.py``), эмбеддинг — локальная модель (``RAG_EMBED_*``).

Инкрементальность: docstore-дедуп llama-index по стабильным id документов
(sha256 от repo-relative пути) — неизменённые файлы пропускаются до сплиттера
(0 LLM-вызовов, 0 эмбеддингов). Маркер ``rag_profile`` в metadata входит в
hash документа: смена режима (enrich↔plain) или embedding-модели переиндексирует
всё, а не смешивает несовместимые векторы.

Каталог индекса (``--index-dir``) принадлежит клиенту — дефолтов нет. Индекс:
docstore.json, index_store.json, default__vector_store.json (+ стабы) и
``rag_meta.json`` с корнем корпуса и профилем. Root-guard: повторный прогон
с другим ``--path`` — ошибка без ``--switch-root``; смена профиля (enrich↔plain
или embedding-модель) — предупреждение с подсказкой ``--reset``.

Дефолт docstore_strategy=UPSERTS: повторные прогоны только обновляют, ничего
не удаляют (частичный прогон по подпапке безопасен). ``--prune`` удаляет
документы, отсутствующие в текущем прогоне, — вручную (``delete_ref_doc`` +
``vector_store.delete``): встроенный UPSERTS_AND_DELETE в 0.14.x несовместим с
чанками в docstore — сносит их как «пропавшие документы», потому что
``get_all_document_hashes()`` возвращает id всех записей, включая чанки.

Батч-персист: документы нарезаются на батчи по ``RAG_BATCH_SIZE`` из .env,
индекс персистится после каждого батча — падение длинного прогона (429/
таймаут LLM) теряет только текущий батч, а docstore-дедуп даёт бесплатный
resume при перезапуске. Упавший батч деградирует до per-file-прогона: битые
файлы фиксируются и пропускаются (сводка в конце), остальные обрабатываются;
несколько файлов подряд с ошибками останавливают прогон (guard от полного
outage эндпоинта).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.schema import BaseNode, Document, TransformComponent

import config
from ast_llamaindex import (
    EXCLUDE_DIRS,
    SUPPORTED_LANGUAGES,
    _build_extractors,
    _build_splitter,
    _inject_context_headers,
)
from llm_provider import LLMConfigError, build_enrichment_llm

# Metadata-ключи, НЕ попадающие в текст эмбеддинга и LLM-контекста. Всё
# остальное (file_path, section_summary, excerpt_keywords) — попадает.
EXCLUDED_METADATA_KEYS = [
    "file_name",
    "file_size",
    "creation_date",
    "last_modified_date",
    "rag_profile",
]

# Guard от полного outage эндпоинта: per-file-фолбэк с ретраями OpenAI-клиента
# может подолгу жечь время на каждом файле подряд; стоп после N неудач
# (прогресс уже персистнут, перезапуск продолжит с места остановки).
_MAX_CONSECUTIVE_FILE_FAILURES = 10


# --- rag_meta.json ---------------------------------------------------------------

def _read_meta(index_dir: Path) -> Optional[dict]:
    meta_path = index_dir / "rag_meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(index_dir: Path, corpus_root: Path, profile: str) -> None:
    meta = {
        "schema_version": 1,
        "corpus_root": str(corpus_root),
        "profile": profile,
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "rag_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")


# --- Embedding ----------------------------------------------------------------

def _resolve_embed_device() -> str:
    device = config.embed().device
    if device != "auto":
        return device
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_embed_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    """Локальная embedding-модель (BAAI/bge-m3 — мультиязычная RU↔EN).

    Параметры = None → значения из ``config.embed()`` (RAG_EMBED_* в .env).
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    ec = config.embed()
    return HuggingFaceEmbedding(
        model_name=model_name or ec.model,
        device=device or _resolve_embed_device(),
        max_length=max_length or ec.max_length,
        embed_batch_size=batch_size or ec.batch_size,
    )


def _ensure_enrichment_llm_alive(llm) -> None:
    """Префлайт enrichment-эндпоинта: понятная ошибка вместо httpx-трейсбека."""
    import httpx

    base = str(getattr(llm, "api_base", "")).rstrip("/")
    # Cloud-эндпоинты (z.ai) требуют Authorization — LM Studio игнорирует ключ.
    headers = {}
    if getattr(llm, "api_key", None):
        headers["Authorization"] = f"Bearer {llm.api_key}"
    try:
        r = httpx.get(
            f"{base}/models",
            timeout=config.enrichment_llm().preflight_timeout_s,
            headers=headers,
        )
        r.raise_for_status()
    except Exception as e:
        raise LLMConfigError(
            f"LLM enrichment-домена недоступна: {base} ({e})\n"
            "Проверь, что эндпоинт запущен/доступен и ключ верен "
            "(RAG_ENRICHMENT_BASE_URL / RAG_ENRICHMENT_API_KEY)."
        ) from e


# --- Стабильные id и нормализация документов -----------------------------------

def _stable_doc_id(rel_path: str) -> str:
    return hashlib.sha256(f"rag-doc:{rel_path}".encode()).hexdigest()


def _normalize_documents(
    documents: List[Document],
    corpus_root: Path,
    enrich: bool,
    embed_model_name: str,
) -> None:
    """In-place: абсолютный file_path → repo-relative, маркер rag_profile,
    стабильный ``id_`` (наследуется чанками как ref_doc_id)."""
    profile = ("enrich" if enrich else "plain") + f":{embed_model_name}"
    root = corpus_root.resolve()
    for d in documents:
        fp = Path(d.metadata["file_path"])
        rel = fp.resolve().relative_to(root).as_posix()
        d.metadata["file_path"] = rel
        d.metadata["rag_profile"] = profile
        # Даты SimpleDirectoryReader входят в Document.hash: без выбрасывания
        # любой git checkout/перенос корпуса меняет mtime и ломает docstore-дедуп.
        d.metadata.pop("creation_date", None)
        d.metadata.pop("last_modified_date", None)
        d.id_ = _stable_doc_id(rel)


# --- Transformation: заголовки + excluded-ключи metadata -----------------------

class HeaderMetadataTransform(TransformComponent):
    """Заголовки ``// File: … / Chunk i/n`` ПОСЛЕ extractors (те видят чистый
    код) и ДО embed_model (заголовок участвует в эмбеддинге), плюс
    excluded-ключи metadata.

    Обязателен именно TransformComponent: IngestionPipeline 0.14.x — pydantic-
    модель и plain-callable в transformations не пропускает.
    """

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        _inject_context_headers(list(nodes))
        for n in nodes:
            n.excluded_embed_metadata_keys = EXCLUDED_METADATA_KEYS[:]
            n.excluded_llm_metadata_keys = EXCLUDED_METADATA_KEYS[:]
        return nodes


# --- Storage -------------------------------------------------------------------

def _load_or_create_index(
    sc: StorageContext, persist_dir: Path, embed_model
) -> VectorStoreIndex:
    """Релоад переиспользует уже созданный index_struct; пустой/битый — новый.

    IngestionPipeline с vector_store= не регистрирует index_struct в
    index_store, поэтому ``load_index_from_storage`` без этого паттерна падает.
    """
    if (persist_dir / "index_store.json").exists():
        try:
            return load_index_from_storage(sc)
        except (ValueError, FileNotFoundError):
            pass
    return VectorStoreIndex(nodes=[], storage_context=sc, embed_model=embed_model)


def _vector_count(vector_store) -> int:
    try:
        data = vector_store.to_dict()
        d = data.get("data", data)
        return len(d.get("embedding_dict", {}))
    except Exception:
        return -1


# --- Главный вход ----------------------------------------------------------------

def build_index(
    code_path: Path,
    index_dir: Path,
    *,
    enrich: bool = True,
    max_files: Optional[int] = None,
    batch_size: Optional[int] = None,
    prune: bool = False,
    reset: bool = False,
    switch_root: bool = False,
) -> Dict:
    """Построить/обновить векторный индекс по папке с исходниками.

    ``max_files``/``batch_size`` = None → из конфига (``config.chunking()``).
    Каталог индекса принадлежит клиенту; корень корпуса фиксируется в
    ``rag_meta.json`` — повторный прогон с другим корнем без ``--switch-root``
    падает root-guard'ом.

    Возвращает статистику: files_loaded/processed/skipped/failed,
    chunks_indexed, docstore_docs, vectors, тайминги.
    """
    ec = config.embed()
    if max_files is None:
        max_files = config.chunking().max_files
    if batch_size is None:
        batch_size = config.chunking().batch_size
    if batch_size < 1:
        raise ValueError(f"batch_size должен быть ≥ 1, получено {batch_size}")

    profile = ("enrich" if enrich else "plain") + f":{ec.model}"
    persist_dir = Path(index_dir)
    meta = _read_meta(persist_dir)
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    if meta:
        saved_root = Path(meta.get("corpus_root", "")).resolve()
        current_root = Path(code_path).resolve()
        if saved_root != current_root:
            if not switch_root:
                raise SystemExit(
                    "❌ Corpus root индекса не совпадает с --path:\n"
                    f"   в индексе: {saved_root}\n"
                    f"   передано:  {current_root}\n"
                    "   Варианты: указать --path сохранённого корня; собрать "
                    "отдельный индекс (--index-dir <новый каталог>); сменить "
                    "корень осознанно — --switch-root (совет: добить смену "
                    "прогоном с --prune — doc-id старых путей станут сиротами)."
                )
            print(
                f"⚠️  Смена corpus root: {saved_root} → {current_root}. "
                "Doc-id старых путей останутся в индексе сиротами — добей "
                "смену прогоном с --prune."
            )
        elif meta.get("profile") != profile:
            print(
                f"⚠️  Профиль индекса {meta.get('profile')} ≠ текущий {profile}. "
                "Неизменённые файлы не будут пересчитаны — для полной "
                "пересборки запусти с --reset. Прогон продолжается."
            )

    t0 = time.perf_counter()
    print(f"🧠 Embedding: {ec.model} (device={_resolve_embed_device()}, max_length={ec.max_length})")
    embed_model = build_embed_model()
    Settings.embed_model = embed_model  # для insert_nodes и ретривера

    llm = None
    if enrich:
        llm = build_enrichment_llm()
        _ensure_enrichment_llm_alive(llm)
    t_setup = time.perf_counter() - t0

    sc = (
        StorageContext.from_defaults(persist_dir=str(persist_dir))
        if (persist_dir / "docstore.json").exists()
        else StorageContext.from_defaults()
    )
    index = _load_or_create_index(sc, persist_dir, embed_model)

    print(f"📂 Загрузка файлов из {code_path}...")
    reader = SimpleDirectoryReader(
        input_dir=str(code_path),
        recursive=True,
        exclude=EXCLUDE_DIRS,
        required_exts=list(SUPPORTED_LANGUAGES.keys()),
    )
    documents = reader.load_data()
    if max_files is not None:
        documents = documents[:max_files]
    _normalize_documents(documents, Path(code_path), enrich, ec.model)
    print(f"📄 Загружено документов: {len(documents)}")

    hashes_before = {
        d.id_: sc.docstore.get_document_hash(d.id_) for d in documents
    }

    if prune:
        doc_ids_now = {d.id_ for d in documents}
        stored_doc_ids = {
            n.id_ for n in sc.docstore.docs.values() if isinstance(n, Document)
        }
        doomed = stored_doc_ids - doc_ids_now
        if doomed:
            print(
                f"⚠️  --prune: из индекса будет УДАЛЕНО {len(doomed)} документов, "
                f"отсутствующих в текущем прогоне ({len(documents)} загружено)."
            )
            for doc_id in sorted(doomed):
                sc.docstore.delete_ref_doc(doc_id, raise_error=False)
                sc.vector_store.delete(doc_id)

    header_transform = HeaderMetadataTransform()
    splitters = {ext: _build_splitter(ext) for ext in SUPPORTED_LANGUAGES}
    if enrich:
        cc = config.chunking()
        extractors = _build_extractors(llm, cc.num_workers, cc.num_keywords)
    else:
        extractors = ()

    failed: Dict[str, Tuple[str, str]] = {}  # id_ → (file_path, error)
    consecutive_failures = 0
    all_nodes: List[BaseNode] = []
    t_run = time.perf_counter()
    for ext, splitter in splitters.items():
        docs = [
            d
            for d in documents
            if Path(d.metadata.get("file_path", "")).suffix.lower() == ext
        ]
        if not docs:
            continue
        n_batches = (len(docs) + batch_size - 1) // batch_size
        mode = "AST + enrichment + embedding" if enrich else "AST + embedding"
        print(
            f"🔧 [{ext}] {mode}: {len(docs)} файлов, "
            f"{n_batches} батчей по ≤{batch_size}..."
        )
        pipeline = IngestionPipeline(
            transformations=[splitter, *extractors, header_transform, embed_model],
            vector_store=sc.vector_store,
            docstore=sc.docstore,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )
        for bi in range(n_batches):
            batch = docs[bi * batch_size : (bi + 1) * batch_size]
            nodes: List[BaseNode] = []
            try:
                nodes = list(pipeline.run(documents=batch))
                consecutive_failures = 0
            except Exception:
                # Битый файл или сбой эндпоинта не должен валить батч:
                # изолируем per-file (неизменённые доки дедуп пропустит
                # бесплатно, без LLM).
                for d in batch:
                    try:
                        nodes.extend(pipeline.run(documents=[d]))
                        consecutive_failures = 0
                    except Exception as e:
                        consecutive_failures += 1
                        failed[d.id_] = (
                            d.metadata.get("file_path", d.id_),
                            str(e),
                        )
                        print(
                            f"   ⚠️  Файл не проиндексирован: "
                            f"{failed[d.id_][0]} ({str(e)[:160]})"
                        )
                        if consecutive_failures >= _MAX_CONSECUTIVE_FILE_FAILURES:
                            print(
                                f"\n❌ {_MAX_CONSECUTIVE_FILE_FAILURES} файлов "
                                "подряд не проиндексированы — эндпоинт "
                                "недоступен или файлы битые. Останавливаю "
                                f"прогон; прогресс персистнут ({persist_dir}). "
                                "Перезапусти позже — продолжит с места остановки."
                            )
                            raise RuntimeError(
                                f"{_MAX_CONSECUTIVE_FILE_FAILURES} файлов "
                                "подряд не проиндексированы"
                            ) from e
            all_nodes.extend(nodes)
            # Изменившийся-и-упавший док теряет hash-запись in-memory ещё в
            # _handle_upserts — исключаем failed, иначе посчитается processed.
            batch_changed = [
                d
                for d in batch
                if d.id_ not in failed
                and sc.docstore.get_document_hash(d.id_) != hashes_before[d.id_]
            ]
            # Пайплайн кладёт в docstore только исходные документы (чанки —
            # лишь в vector store). insert_nodes: чанки → docstore (тексты для
            # ретривера) + регистрация index_struct; идемпотентно, эмбеддинги
            # уже на узлах.
            if nodes:
                index.insert_nodes(nodes)
            if nodes or batch_changed:
                sc.persist(persist_dir=str(persist_dir))
            failed_here = sum(1 for d in batch if d.id_ in failed)
            print(
                f"   📦 [{ext}] батч {bi + 1}/{n_batches}: "
                f"обработано {len(batch_changed)}, "
                f"пропущено {len(batch) - len(batch_changed) - failed_here}, "
                f"не удалось {failed_here} "
                f"({time.perf_counter() - t_run:.0f} с)"
            )
    t_pipeline = time.perf_counter() - t_run

    # Финальный безусловный персист: флашит --prune-удаления и удаления
    # упавших документов, когда все батчи оказались no-op (персист по условию).
    sc.persist(persist_dir=str(persist_dir))
    _write_meta(persist_dir, Path(code_path).resolve(), profile)

    # Doc с 0 чанков (например, пустой файл) не виден в all_nodes — считаем
    # processed по смене hash-записи в docstore, а не по узлам.
    processed_ids = {
        d.id_
        for d in documents
        if d.id_ not in failed
        and sc.docstore.get_document_hash(d.id_) != hashes_before[d.id_]
    }
    stats = {
        "files_loaded": len(documents),
        "files_processed": len(processed_ids),
        "files_skipped": len(documents) - len(processed_ids) - len(failed),
        "files_failed": len(failed),
        "chunks_indexed": len(all_nodes),
        "docstore_docs": len(sc.docstore.docs),
        "vectors": _vector_count(sc.vector_store),
        "t_setup_s": round(t_setup, 1),
        "t_pipeline_s": round(t_pipeline, 1),
        "t_total_s": round(time.perf_counter() - t0, 1),
    }
    if failed:
        print(
            f"\n⚠️  Не проиндексировано файлов: {len(failed)} "
            "(повтори прогон — они будут ретраены):"
        )
        for file_path, err in failed.values():
            print(f"   • {file_path}: {err[:160]}")
    print(f"\n✅ Индекс: {persist_dir}")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    return stats


if __name__ == "__main__":
    import textwrap

    cc = config.chunking()

    parser = argparse.ArgumentParser(
        prog="build_index.py",
        description=(
            "Построить/обновить векторный индекс: AST-чанкинг + enrichment "
            "(LLM-домен enrichment) + embedding → SimpleVectorStore."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Примеры (относительные пути — от текущего каталога):
              # оффлайн (без LLM — быстро, для смоука):
              python build_index.py --path ./src --index-dir /tmp/idx --no-enrich

              # полный прогон с enrichment (ключи в корневом .env, см. config.py):
              python build_index.py --path ./src --index-dir ~/.rag/myproject

              # пересоздать индекс с нуля:
              python build_index.py --path ./src --index-dir ~/.rag/myproject --reset

              # удалить из индекса документы, которых нет в прогоне:
              python build_index.py --path ./src --index-dir ~/.rag/myproject --prune
        """),
    )
    parser.add_argument("--path", type=Path, required=True,
                        help="папка с исходниками (обязательный)")
    parser.add_argument("--index-dir", type=Path, required=True,
                        help="каталог индекса (обязательный)")
    parser.add_argument("--no-enrich", action="store_true",
                        help="без LLM-enrichment (оффлайн)")
    parser.add_argument("--max-files", type=int, default=cc.max_files,
                        help="лимит загружаемых файлов (дев-прогон)")
    parser.add_argument("--batch-size", type=int, default=cc.batch_size,
                        help=f"файлов на батч: персист индекса после каждого "
                             f"батча (по умолчанию: {cc.batch_size})")
    parser.add_argument("--prune", action="store_true",
                        help="удалить из индекса документы, отсутствующие в текущем прогоне")
    parser.add_argument("--reset", action="store_true",
                        help="снести каталог индекса перед прогоном")
    parser.add_argument("--switch-root", action="store_true",
                        help="осознанно сменить corpus root существующего индекса")
    args = parser.parse_args()

    try:
        build_index(
            code_path=args.path,
            index_dir=args.index_dir,
            enrich=not args.no_enrich,
            max_files=args.max_files,
            batch_size=args.batch_size,
            prune=args.prune,
            reset=args.reset,
            switch_root=args.switch_root,
        )
    except config.ConfigError as e:
        sys.exit(str(e))
