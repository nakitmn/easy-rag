"""Шаг 2: AST-чанкинг + enrichment + эмбеддинг → персистентный векторный индекс.

Нарезку, extractors и заголовки переиспользует из ``ast_llamaindex`` (файл не
редактируется). Enrichment — LLM домена ``enrichment`` (конфиг в корневом
``.env``, см. ``config.py``), эмбеддинг — локальная модель (``RAG_EMBED_*``).

Инкрементальность: docstore-дедуп llama-index по стабильным id документов
(sha256 от repo-relative пути) — неизменённые файлы пропускаются до сплиттера
(0 LLM-вызовов, 0 эмбеддингов). Маркер ``rag_profile`` в metadata входит в
hash документа: смена режима (enrich↔plain) или embedding-модели переиндексирует
всё, а не смешивает несовместимые векторы.

Индекс: SimpleVectorStore персистится через StorageContext в каталог
``RAG_INDEX_DIR`` из .env (по умолчанию ``ast/data/vector_index/``:
docstore.json, index_store.json, default__vector_store.json).

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
    code_path: Optional[Path] = None,
    index_dir: Optional[Path] = None,
    *,
    enrich: bool = True,
    max_files: Optional[int] = None,
    batch_size: Optional[int] = None,
    prune: bool = False,
    reset: bool = False,
) -> Dict:
    """Построить/обновить векторный индекс по папке с исходниками.

    ``code_path``/``index_dir``/``max_files``/``batch_size`` = None → из
    конфига (``config.paths()`` / ``config.chunking()``).

    Возвращает статистику: files_loaded/processed/skipped/failed,
    chunks_indexed, docstore_docs, vectors, тайминги.
    """
    ec = config.embed()
    if code_path is None:
        code_path = config.paths().code_path
    if index_dir is None:
        index_dir = config.paths().index_dir
    if max_files is None:
        max_files = config.chunking().max_files
    if batch_size is None:
        batch_size = config.chunking().batch_size
    if batch_size < 1:
        raise ValueError(f"batch_size должен быть ≥ 1, получено {batch_size}")

    persist_dir = Path(index_dir)
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

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

    pc = config.paths()
    cc = config.chunking()

    parser = argparse.ArgumentParser(
        prog="build_index.py",
        description=(
            "Построить/обновить векторный индекс: AST-чанкинг + enrichment "
            "(LLM-домен enrichment) + embedding → SimpleVectorStore."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Примеры (пути --path — относительно текущего каталога):
              # оффлайн (без LLM — быстро, для смоука):
              python build_index.py --path ../launcher/src/Launcher/Assets/Scripts/Chat --no-enrich

              # полный прогон с enrichment (ключи в корневом .env, см. config.py):
              python build_index.py --path ../launcher/src/Launcher/Assets/Scripts/Chat

              # пересоздать индекс с нуля:
              python build_index.py --path ../launcher/src/Launcher/Assets/Scripts/Chat --reset

              # удалить из индекса документы, которых нет в прогоне:
              python build_index.py --path ../launcher/src/Launcher/Assets/Scripts/Chat --prune
        """),
    )
    parser.add_argument("--path", type=Path, default=pc.code_path,
                        help=f"папка с исходниками (по умолчанию: {pc.code_path})")
    parser.add_argument("--index-dir", type=Path, default=pc.index_dir,
                        help=f"каталог индекса (по умолчанию: {pc.index_dir})")
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
        )
    except config.ConfigError as e:
        sys.exit(str(e))
