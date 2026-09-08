"""AST-чанкинг кода + LLM-enrichment (summary, keywords) одним IngestionPipeline.

CodeSplitter нарезает исходники по AST; SummaryExtractor и KeywordExtractor
обогащают каждый чанк через GLM прямо во время ``pipeline.run()`` — без отдельного
ручного шага суммаризации. Конкурентность встроена в extractors (``num_workers``).

Поддерживаемые языки задаются реестром ``SUPPORTED_LANGUAGES``. Сейчас — только
C# (.cs). Чтобы добавить kotlin/java/js: установить pip-пакет tree-sitter-грамматики
и дописать запись в реестр — parser и сплиттер соберутся автоматически.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.extractors import SummaryExtractor, KeywordExtractor
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode

import config
from llm_provider import build_llm

from tree_sitter import Language, Parser


EXCLUDE_DIRS = [
    "**/.*",
    "**/bin/**",
    "**/obj/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/build/**",
]


# --- Реестр языков (расширяемый) ---------------------------------------------
# Ключ — расширение; grammar — имя pip-пакета tree-sitter (импортируется лениво).

SUPPORTED_LANGUAGES: Dict[str, dict] = {
    ".cs": {"language": "c_sharp", "grammar": "tree_sitter_c_sharp"},
    # Будущие языки — поставить pip-пакет и раскомментировать:
    # ".kt":   {"language": "kotlin",     "grammar": "tree_sitter_kotlin"},
    # ".java": {"language": "java",       "grammar": "tree_sitter_java"},
    # ".js":   {"language": "javascript", "grammar": "tree_sitter_javascript"},
}
# Внимание: CodeSplitter (llama-index 0.14.x) игнорирует chunk_lines /
# chunk_lines_overlap — реально работает только max_chars (или max_tokens при
# count_mode="token"). Поэтому их тут нет и в _build_splitter они не передаются.


def _build_parser(grammar_module: str) -> Parser:
    # tree-sitter-language-pack 1.x не отдаёт c_sharp через скачивание, поэтому
    # парсеры собираются из отдельных grammar-пакетов.
    mod = importlib.import_module(grammar_module)
    return Parser(Language(mod.language()))


def _build_splitter(ext: str) -> CodeSplitter:
    cfg = SUPPORTED_LANGUAGES[ext]
    # RAG_CHUNK_MAX_CHARS=400000 ≈ файл до 10 000 строк целиком: CodeSplitter
    # не режет файл ≤ max_chars вовсе, поэтому не порождает сиротские чанки
    # («{», «}» и т.п.). Риск: файл >~120K символов переполнит контекст
    # enrichment-LLM — в проекте таких нет (максимум ~21K).
    return CodeSplitter(
        language=cfg["language"],
        max_chars=config.chunking().chunk_max_chars,
        parser=_build_parser(cfg["grammar"]),
    )


# --- Кастомные промпты под код -----------------------------------------------

CODE_SUMMARY_PROMPT = """\
You are an expert source code analyst. Summarize the code fragment in 1-2 \
sentences, 40 words maximum.
State only what the code does (its purpose) and the key types/frameworks it \
works with. Do not enumerate inputs, outputs, side effects or failure modes; \
do not retell the code verbatim.
Respond strictly in English. Do not use any other language.

Code:
{context_str}

Summary: """

CODE_KEYWORDS_PROMPT = """\
List up to {keywords} concrete technical terms found in the code fragment: \
class, interface or method names, frameworks, protocols, or design patterns \
that the code actually uses. Take identifiers from the code itself; never \
output generic category words such as "entities", "technologies", "patterns", \
"APIs", "classes". Format: strictly comma-separated, no numbering, no \
explanations.

Code:
{context_str}

Keywords: """


def _build_extractors(
    llm: LLM, num_workers: int, num_keywords: int
) -> Tuple[SummaryExtractor, KeywordExtractor]:
    return (
        SummaryExtractor(
            llm=llm,
            summaries=["self"],
            prompt_template=CODE_SUMMARY_PROMPT,
            num_workers=num_workers,
        ),
        KeywordExtractor(
            llm=llm,
            keywords=num_keywords,
            prompt_template=CODE_KEYWORDS_PROMPT,
            num_workers=num_workers,
        ),
    )


# --- Пайплайн ----------------------------------------------------------------

def run(
    code_path: Optional[Path] = None,
    llm: Optional[LLM] = None,
    enrich: bool = True,
    max_files: Optional[int] = None,
) -> List[BaseNode]:
    """Загрузить код, нарезать по AST и (опционально) обогатить описанием.

    ``code_path``/``max_files`` = None → значение из конфига (``config.paths()``
    / ``config.chunking()``). При ``enrich=True`` summary и keywords генерируются
    LLM-доменом enrichment прямо внутри ``pipeline.run()``. При
    ``enrich=False`` — только AST-чанкинг, без LLM-вызовов (LLM-переменные .env
    не требуются).

    Метаданные каждого узла: ``file_path`` (от CodeSplitter), ``section_summary`` и
    ``excerpt_keywords`` (от extractors). Тексту узла добавляется контекстный
    заголовок (``// File: ... / Chunk i/n``) пост-факту — extractors работают с
    чистым кодом, без заголовка.
    """
    cc = config.chunking()
    if code_path is None:
        code_path = config.paths().code_path
    if max_files is None:
        max_files = cc.max_files
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
    print(f"📄 Загружено документов: {len(documents)}")

    splitters = {ext: _build_splitter(ext) for ext in SUPPORTED_LANGUAGES}
    extractors: Tuple = ()
    if enrich:
        resolved_llm = llm if llm is not None else build_llm()
        extractors = _build_extractors(resolved_llm, cc.num_workers, cc.num_keywords)

    all_nodes: List[BaseNode] = []
    for ext, splitter in splitters.items():
        docs = [
            d for d in documents
            if Path(d.metadata.get("file_path", "")).suffix.lower() == ext
        ]
        if not docs:
            continue

        mode = "AST + enrichment" if enrich else "AST only"
        print(f"🔧 [{ext}] {mode}: {len(docs)} файлов...")

        pipeline = IngestionPipeline(transformations=[splitter, *extractors])
        nodes = pipeline.run(documents=docs)
        all_nodes.extend(nodes)

    _inject_context_headers(all_nodes)
    print(f"✅ Создано чанков: {len(all_nodes)}")
    return all_nodes


def _inject_context_headers(nodes: List[BaseNode]) -> None:
    """Добавить заголовок ``// File: ... / Chunk i/n`` в начало текста узлов.

    Группировка по ``file_path``; чанки нумеруются в порядке следования. Заголовок
    улучшает семантику при эмбеддинге (шаг 2). Выполняется после extractors.
    """
    count_by_file: Dict[str, int] = {}
    for n in nodes:
        fp = n.metadata.get("file_path", "")
        count_by_file[fp] = count_by_file.get(fp, 0) + 1

    index_by_file: Dict[str, int] = {}
    for n in nodes:
        fp = n.metadata.get("file_path", "")
        i = index_by_file.get(fp, 0)
        total = count_by_file.get(fp, 1)
        header = f"// File: {Path(fp).as_posix()}\n// Chunk Part: {i + 1}/{total}\n\n"
        n.text = header + n.get_content()
        index_by_file[fp] = i + 1


def save_chunks(nodes: List[BaseNode], output_path: Optional[Path] = None) -> Path:
    if output_path is None:
        output_path = config.paths().output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for node in nodes:
            record = {
                "id": node.node_id,
                "text": node.get_content(),
                "metadata": node.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


if __name__ == "__main__":
    import argparse
    import sys
    import textwrap

    pc = config.paths()
    cc = config.chunking()

    parser = argparse.ArgumentParser(
        prog="ast_llamaindex.py",
        description="AST-чанкинг кода + LLM-enrichment (summary, keywords) через IngestionPipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Примеры:
              # только чанкинг (оффлайн, LLM-ключи не нужны) — для быстрого теста:
              python ast_llamaindex.py --path src/Launcher/Assets/Scripts/Chat --no-enrich

              # полный прогон с enrichment (ключи в корневом .env, см. config.py):
              python ast_llamaindex.py --path src/Launcher/Assets/Scripts/Chat

              # лимит файлов для дев-прогона:
              python ast_llamaindex.py --path src/Launcher/Assets/Scripts/Chat --no-enrich --max-files 5
        """),
    )
    parser.add_argument("--path", type=Path, default=pc.code_path,
                        help=f"папка с исходниками (по умолчанию: {pc.code_path})")
    parser.add_argument("--output", type=Path, default=pc.output_path,
                        help=f"выходной JSONL (по умолчанию: {pc.output_path})")
    parser.add_argument("--no-enrich", action="store_true",
                        help="только AST-чанкинг, без LLM (оффлайн)")
    parser.add_argument("--max-files", type=int, default=cc.max_files,
                        help="лимит загружаемых файлов (по умолчанию: без лимита)")
    args = parser.parse_args()

    try:
        nodes = run(code_path=args.path, enrich=not args.no_enrich, max_files=args.max_files)
        output = save_chunks(nodes, output_path=args.output)
    except config.ConfigError as e:
        sys.exit(str(e))
    print(f"\n💾 Чанки сохранены: {len(nodes)} шт. → {output}")
    print(f"   Размер файла: {output.stat().st_size / 1024:.1f} KB")
    if nodes:
        sample = nodes[0]
        print("\n--- Пример (первый чанк) ---")
        print("metadata keys:", list(sample.metadata.keys()))
        print("text (первые 200):", repr(sample.get_content()[:200]))
