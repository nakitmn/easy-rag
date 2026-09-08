"""Шаг 2 (query): вопрос → top-k чанков из векторного индекса → ответ LLM.

Retrieval — SimpleVectorStore через ``load_index_from_storage``; эмбеддинг
запроса — та же модель, что индексировала (``RAG_EMBED_*`` в корневом .env).
Синтез — LLM домена ``query`` (``RAG_QUERY_*``; см. ``config.py`` /
``llm_provider``). ``--no-llm`` — retrieval-дебаг без LLM (query-переменные
не требуются).

Примеры:
    python query.py "Где регистрируются сервисы чата?"
    python query.py "как отправить сообщение" --no-llm --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import MetadataMode, NodeWithScore

import config
from build_index import build_embed_model
from llm_provider import build_query_llm

ANSWER_SYSTEM_PROMPT = (
    "Ты — ассистент по кодовой базе C# лаунчера (Unity, VContainer).\n"
    "Отвечай на вопрос, опираясь ТОЛЬКО на приведённые фрагменты кода.\n"
    "Правила:\n"
    "1. Отвечай по-русски, кратко и по делу.\n"
    "2. Ссылайся на использованные фрагменты в тексте ответа как [1], [2] и т.д.\n"
    "3. Называй конкретные классы, методы и файлы из фрагментов.\n"
    "4. Если во фрагментах нет ответа — напиши «В индексе нет релевантного "
    "кода для этого вопроса» и не выдумывай."
)


def load_index(index_dir: Optional[Path] = None) -> VectorStoreIndex:
    """Загрузить персистентный индекс (embed_model в Settings — до загрузки)."""
    if index_dir is None:
        index_dir = config.paths().index_dir
    if not (index_dir / "index_store.json").exists():
        raise SystemExit(
            f"❌ Индекс не найден: {index_dir}\n"
            "   Сначала построй его: python build_index.py --path <папка с .cs>"
        )
    print(f"🧠 Embedding: {config.embed().model} (загрузка модели...)")
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
        pc = config.paths()

        parser = argparse.ArgumentParser(
            prog="query.py",
            description="Вопрос по кодовой базе → ответ LLM с цитатами на чанки индекса.",
        )
        parser.add_argument("question", nargs="?", help="вопрос")
        parser.add_argument("-q", "--question-opt", dest="question_opt",
                            help="вопрос (альтернатива позиционному аргументу)")
        parser.add_argument("--top-k", type=int, default=rc.top_k,
                            help=f"сколько чанков доставать (default {rc.top_k})")
        parser.add_argument("--index-dir", type=Path, default=pc.index_dir,
                            help=f"каталог индекса (default {pc.index_dir})")
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
