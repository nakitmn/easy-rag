"""Единственная точка конфигурации RAG-тула: чтение и валидация env-переменных.

Все tunable-параметры задаются корневым ``.env`` workspace (или переменными
процесса — они приоритетнее). Дефолтов в коде нет: отсутствующая или пустая
переменная — ошибка конфигурации. Единственное исключение — ``RAG_MAX_FILES``,
где пустое значение легально и означает «без лимита».

Валидация контекстная: акцессор группы (``paths()`` / ``chunking()`` /
``embed()`` / ``retrieval()`` / ``enrichment_llm()`` / ``query_llm()``) читает
env в момент вызова и падает одним ``ConfigError`` со списком всех проблем
группы. Оффлайн-режимы (``--no-enrich``, ``--no-llm``) просто не вызывают
LLM-акцессоры и не требуют ключей.

Самопроверка без сети::

    python config.py

Относительные значения путей (``RAG_CODE_PATH`` и т.п.) резолвятся от
``WORKSPACE_ROOT`` — корня workspace, где лежит ``.env``.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Конфигурация неполна или некорректна (нет переменной / мусорное значение)."""


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = WORKSPACE_ROOT / ".env"

# config.py обязан лежать в <root>/RAG/ast/scripts/ — иначе parents[3] укажет
# не на workspace root и относительные пути из .env разъедутся молча.
if not (WORKSPACE_ROOT / ".git").exists():
    raise ConfigError(
        f"WORKSPACE_ROOT вычислен как {WORKSPACE_ROOT}, но .git там нет — "
        "config.py должен лежать в <workspace-root>/RAG/ast/scripts/."
    )

load_dotenv(ENV_FILE, override=False)


# --- Группы конфигурации --------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key: str
    base_url: str
    temperature: float
    context_window: int
    preflight_timeout_s: Optional[float] = None  # только enrichment-домен


@dataclass(frozen=True)
class PathsConfig:
    code_path: Path
    output_path: Path
    index_dir: Path


@dataclass(frozen=True)
class ChunkingConfig:
    num_workers: int
    max_files: Optional[int]  # None = без лимита
    chunk_max_chars: int
    num_keywords: int
    batch_size: int  # файлов на батч ингеста (персист после каждого батча)


@dataclass(frozen=True)
class EmbedConfig:
    model: str
    device: str  # "auto" = авторедетект mps/cpu в build_index
    max_length: int
    batch_size: int


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    source_max_chars: int


# --- Парсеры значений (накапливают проблемы, а не падают на первой) ---------------

_MISSING = "не задано (нет в окружении и пусто в .env)"


def _raw(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _str(name: str, problems: List[str]) -> Optional[str]:
    raw = _raw(name)
    if raw is None:
        problems.append(f"{name}: {_MISSING}")
    return raw


def _int(name: str, problems: List[str], minimum: int = 1) -> Optional[int]:
    raw = _raw(name)
    if raw is None:
        problems.append(f"{name}: {_MISSING}")
        return None
    try:
        value = int(raw)
    except ValueError:
        value = None
    if value is None or value < minimum:
        problems.append(f"{name}: {raw!r} — ожидается целое число ≥ {minimum}")
        return None
    return value


def _float(
    name: str, problems: List[str], minimum: float, allow_equal: bool
) -> Optional[float]:
    raw = _raw(name)
    if raw is None:
        problems.append(f"{name}: {_MISSING}")
        return None
    op = "≥" if allow_equal else ">"
    try:
        value = float(raw)
    except ValueError:
        value = None
    ok = value is not None and math.isfinite(value) and (
        value >= minimum if allow_equal else value > minimum
    )
    if not ok:
        problems.append(f"{name}: {raw!r} — ожидается число {op} {minimum}")
        return None
    return value


def _path(name: str, problems: List[str]) -> Optional[Path]:
    raw = _raw(name)
    if raw is None:
        problems.append(f"{name}: {_MISSING}")
        return None
    return resolve_path(raw)


def _raise(title: str, problems: List[str]) -> None:
    if not ENV_FILE.exists():
        header = (
            f"❌ .env не найден: {ENV_FILE} — скопируй .env.example в .env "
            f"и заполни. Проблемы ({title}):"
        )
    else:
        header = f"❌ RAG config ({title}) неполна:"
    lines = "\n".join(f"   • {p}" for p in problems)
    raise ConfigError(
        f"{header}\n{lines}\n"
        f"   Заполни: {ENV_FILE}\n"
        "   (переменные процесса имеют приоритет над .env; шаблон: .env.example)"
    )


# --- Утилиты ----------------------------------------------------------------------

def resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else WORKSPACE_ROOT / p


def mask(secret: Optional[str]) -> str:
    return "****" + secret[-4:] if secret else "****"


# --- Акцессоры групп ----------------------------------------------------------------

def enrichment_llm() -> LLMConfig:
    problems: List[str] = []
    cfg = LLMConfig(
        model=_str("RAG_ENRICHMENT_MODEL", problems),
        api_key=_str("RAG_ENRICHMENT_API_KEY", problems),
        base_url=_str("RAG_ENRICHMENT_BASE_URL", problems),
        temperature=_float("RAG_ENRICHMENT_TEMPERATURE", problems, 0.0, allow_equal=True),
        context_window=_int("RAG_ENRICHMENT_CONTEXT_WINDOW", problems),
        preflight_timeout_s=_float(
            "RAG_ENRICHMENT_PREFLIGHT_TIMEOUT", problems, 0.0, allow_equal=False
        ),
    )
    if problems:
        _raise("LLM «enrichment»", problems)
    return cfg


def query_llm() -> LLMConfig:
    problems: List[str] = []
    cfg = LLMConfig(
        model=_str("RAG_QUERY_MODEL", problems),
        api_key=_str("RAG_QUERY_API_KEY", problems),
        base_url=_str("RAG_QUERY_BASE_URL", problems),
        temperature=_float("RAG_QUERY_TEMPERATURE", problems, 0.0, allow_equal=True),
        context_window=_int("RAG_QUERY_CONTEXT_WINDOW", problems),
    )
    if problems:
        _raise("LLM «query»", problems)
    return cfg


def paths() -> PathsConfig:
    problems: List[str] = []
    cfg = PathsConfig(
        code_path=_path("RAG_CODE_PATH", problems),
        output_path=_path("RAG_OUTPUT_PATH", problems),
        index_dir=_path("RAG_INDEX_DIR", problems),
    )
    if problems:
        _raise("пути", problems)
    return cfg


def chunking() -> ChunkingConfig:
    problems: List[str] = []
    num_workers = _int("RAG_NUM_WORKERS", problems)
    # Единственная переменная, где пусто легально: пусто/отсутствие = без лимита.
    max_files = _int("RAG_MAX_FILES", problems) if _raw("RAG_MAX_FILES") else None
    chunk_max_chars = _int("RAG_CHUNK_MAX_CHARS", problems)
    num_keywords = _int("RAG_NUM_KEYWORDS", problems)
    batch_size = _int("RAG_BATCH_SIZE", problems)
    if problems:
        _raise("чанкинг", problems)
    return ChunkingConfig(
        num_workers, max_files, chunk_max_chars, num_keywords, batch_size=batch_size
    )


def embed() -> EmbedConfig:
    problems: List[str] = []
    cfg = EmbedConfig(
        model=_str("RAG_EMBED_MODEL", problems),
        device=_str("RAG_EMBED_DEVICE", problems),
        max_length=_int("RAG_EMBED_MAX_LENGTH", problems),
        batch_size=_int("RAG_EMBED_BATCH_SIZE", problems),
    )
    if problems:
        _raise("embedding", problems)
    return cfg


def retrieval() -> RetrievalConfig:
    problems: List[str] = []
    cfg = RetrievalConfig(
        top_k=_int("RAG_TOP_K", problems),
        source_max_chars=_int("RAG_SOURCE_MAX_CHARS", problems),
    )
    if problems:
        _raise("retrieval", problems)
    return cfg


if __name__ == "__main__":
    groups = [
        ("LLM enrichment", enrichment_llm),
        ("LLM query", query_llm),
        ("Пути", paths),
        ("Чанкинг", chunking),
        ("Embedding", embed),
        ("Retrieval", retrieval),
    ]
    print(f"Файл конфигурации: {ENV_FILE} "
          f"({'найден' if ENV_FILE.exists() else 'НЕ НАЙДЕН'})")

    failed = False
    for title, accessor in groups:
        try:
            cfg = accessor()
        except ConfigError as e:
            failed = True
            print(f"\n[FAIL] {title}\n{e}")
            continue
        fields = []
        for key, value in asdict(cfg).items():
            if "api_key" in key:
                value = mask(value)
            fields.append(f"{key}={value}")
        print(f"[OK] {title}: " + ", ".join(fields))

    sys.exit(1 if failed else 0)
