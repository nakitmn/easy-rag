"""Шаг 2 (query): вопрос → top-k чанков из векторного индекса → ответ LLM.

Retrieval — SimpleVectorStore через ``load_index_from_storage``; эмбеддинг
запроса — та же модель, что индексировала (``RAG_EMBED_*`` в корневом .env).
Синтез — LLM домена ``query`` (``RAG_QUERY_*``; см. ``config.py`` /
``llm_provider``). ``--no-llm`` — retrieval-дебаг без LLM (query-переменные
не требуются). Гард профиля: embed-модель индекса (``rag_meta.json``; у legacy
индексов — из docstore) сверяется с ``RAG_EMBED_MODEL`` до загрузки модели.

Примеры:
    python query.py --index-dir ~/.rag/myproject "Где настраивается внедрение зависимостей?"
    python query.py --index-dir ~/.rag/myproject "как отправить сообщение" --no-llm --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import MetadataMode, NodeWithScore

import config
from build_index import build_embed_model
from llm_provider import build_query_llm

ANSWER_SYSTEM_PROMPT = (
    "Ты — ассистент по кодовой базе C#.\n"
    "Отвечай на вопрос, опираясь ТОЛЬКО на приведённые фрагменты кода.\n"
    "Правила:\n"
    "1. Отвечай по-русски, кратко и по делу.\n"
    "2. Ссылайся на использованные фрагменты в тексте ответа как [1], [2] и т.д.\n"
    "3. Называй конкретные классы, методы и файлы из фрагментов.\n"
    "4. Если во фрагментах нет ответа — напиши «В индексе нет релевантного "
    "кода для этого вопроса» и не выдумывай."
)


def _find_rag_profile(value) -> Optional[str]:
    if isinstance(value, dict):
        if isinstance(value.get("rag_profile"), str):
            return value["rag_profile"]
        for v in value.values():
            found = _find_rag_profile(v)
            if found is not None:
                return found
    elif isinstance(value, list):
        for v in value:
            found = _find_rag_profile(v)
            if found is not None:
                return found
    return None


def _read_index_profile(index_dir: Path) -> Optional[str]:
    """Профиль индекса: из rag_meta.json, у legacy — из docstore (rag_profile)."""
    meta_path = index_dir / "rag_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f).get("profile")
    docstore_path = index_dir / "docstore.json"
    if not docstore_path.exists():
        return None
    with docstore_path.open("r", encoding="utf-8") as f:
        return _find_rag_profile(json.load(f))


def load_index(index_dir: Path) -> VectorStoreIndex:
    """Загрузить персистентный индекс (гард профиля — до загрузки модели)."""
    if not (index_dir / "index_store.json").exists():
        raise SystemExit(
            f"❌ Индекс не найден: {index_dir}\n"
            "   Сначала построй его: python build_index.py --path <папка с .cs> "
            "--index-dir <каталог индекса>"
        )
    index_profile = _read_index_profile(index_dir)
    expected_model = config.embed().model
    if index_profile is not None and index_profile.split(":", 1)[-1] != expected_model:
        raise SystemExit(
            "❌ Embed-модель индекса не совпадает с текущей конфигурацией:\n"
            f"   профиль индекса: {index_profile}\n"
            f"   текущая модель:  {expected_model} (RAG_EMBED_MODEL)\n"
            f"   Запусти запрос с RAG_EMBED_MODEL={index_profile.split(':', 1)[-1]} "
            "или пересобери индекс этой моделью "
            "(build_index.py --path <корень кода> --index-dir <каталог> --reset)."
        )
    print(f"🧠 Embedding: {expected_model} (загрузка модели...)")
    Settings.embed_model = build_embed_model()
    sc = StorageContext.from_defaults(persist_dir=str(index_dir))
    return load_index_from_storage(sc)


def retrieve(index: VectorStoreIndex, question: str, top_k: int) -> List[NodeWithScore]:
    return index.as_retriever(similarity_top_k=top_k).retrieve(question)


def build_context(chunks: List[NodeWithScore]) -> str:
    """Блоки [i] Файл / Описание / Ключевые слова / код — для user-сообщения."""
    limit = config.retrieval().source_max_chars
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c.metadata
        code = c.get_content(metadata_mode=MetadataMode.NONE)
        if len(code) > limit:
            code = code[:limit] + "\n… (обрезано)"
        blocks.append(
            f"[{i}] Файл: {m.get('file_path', '?')}\n"
            f"Описание: {m.get('section_summary', '(без enrichment)')}\n"
            f"Ключевые слова: {m.get('excerpt_keywords', '-')}\n"
            f"```csharp\n{code}\n```"
        )
    return "\n\n".join(blocks)


def synthesize(llm, question: str, chunks: List[NodeWithScore]) -> str:
    from llama_index.core.llms import ChatMessage, MessageRole

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=ANSWER_SYSTEM_PROMPT),
        ChatMessage(
            role=MessageRole.USER,
            content=f"Вопрос: {question}\n\nФрагменты кода:\n\n{build_context(chunks)}",
        ),
    ]
    try:
        # str(ChatResponse) добавляет префикс «assistant: » — берём чистый текст
        return str(llm.chat(messages).message.content).strip()
    except Exception as e:
        raise SystemExit(
            f"❌ Ошибка LLM query-домена ({llm.model} @ {llm.api_base}): {e}\n"
            "   Проверь RAG_QUERY_API_KEY / баланс z.ai / сеть."
        ) from e


def print_sources(chunks: List[NodeWithScore]) -> None:
    print("\n📚 Источники:")
    for i, c in enumerate(chunks, 1):
        m = c.metadata
        score = f"{c.score:.3f}" if c.score is not None else "-"
        print(f"  [{i}] {m.get('file_path', '?')}  (score={score})")
        summary = (m.get("section_summary") or "")[:110]
        if summary:
            print(f"      {summary}")


def print_debug(chunks: List[NodeWithScore]) -> None:
    for i, c in enumerate(chunks, 1):
        m = c.metadata
        score = f"{c.score:.3f}" if c.score is not None else "-"
        print(f"\n[{i}] {m.get('file_path', '?')}  (score={score})")
        print(f"    summary: {(m.get('section_summary') or '(без enrichment)')[:200]}")
        code = c.get_content(metadata_mode=MetadataMode.NONE)
        tail = code[:400].replace("\n", "\n    ")
        print(f"    {tail}{'…' if len(code) > 400 else ''}")


def main() -> int:
    try:
        rc = config.retrieval()

        parser = argparse.ArgumentParser(
            prog="query.py",
            description="Вопрос по кодовой базе → ответ LLM с цитатами на чанки индекса.",
        )
        parser.add_argument("question", nargs="?", help="вопрос")
        parser.add_argument("-q", "--question-opt", dest="question_opt",
                            help="вопрос (альтернатива позиционному аргументу)")
        parser.add_argument("--top-k", type=int, default=rc.top_k,
                            help=f"сколько чанков доставать (default {rc.top_k})")
        parser.add_argument("--index-dir", type=Path, required=True,
                            help="каталог индекса (обязательный)")
        parser.add_argument("--no-llm", action="store_true",
                            help="только retrieval-дебаг: чанки + score, без LLM")
        args = parser.parse_args()

        question = args.question or args.question_opt
        if not question:
            parser.error("укажи вопрос позиционно или через -q")

        index = load_index(args.index_dir)
        print(f"🔍 Поиск: top-{args.top_k} …")
        chunks = retrieve(index, question, args.top_k)
        if not chunks:
            print("⚠️ Ничего не найдено (индекс пуст?). Запусти build_index.py.")
            return 1

        if args.no_llm:
            print_debug(chunks)
            return 0

        llm = build_query_llm()
        print(f"🤖 Ответ ({llm.model} @ {llm.api_base}):\n")
        print(synthesize(llm, question, chunks))
        print_sources(chunks)
        return 0
    except config.ConfigError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    sys.exit(main())
