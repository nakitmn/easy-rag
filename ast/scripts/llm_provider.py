"""LLM-провайдер RAG-тула: домены ``enrichment`` и ``query``.

Домен задаёт назначение модели, а не провайдера:

    enrichment  summary/keywords по чанкам кода
    query       синтез ответов по найденным чанкам

Вся конфигурация (модель, эндпоинт, ключ, температура, контекстное окно) —
в корневом ``.env`` workspace; читает и валидирует её ``config.py``
(``config.enrichment_llm()`` / ``config.query_llm()``). Дефолтов и
legacy-override'ов в коде нет.

Фабрика возвращает обычный LlamaIndex ``LLM`` (OpenAILike-клиент), потребители
(например ``ast_llamaindex``) зависят только от абстракции LLM. Явные kwargs
``build_llm`` — точечный override отдельных полей поверх ПОЛНОГО конфига
домена из .env (недостроенный .env override'ом не обойти: .env — единственная
точка конфигурации).

Typical usage::

    from llm_provider import build_llm, build_query_llm

    llm = build_llm()            # enrichment-домен
    answer_llm = build_query_llm()
"""

from __future__ import annotations

from typing import Optional

from llama_index.core.llms import LLM
from llama_index.core.settings import Settings
from llama_index.llms.openai_like import OpenAILike

import config

DOMAINS = ("enrichment", "query")
DEFAULT_DOMAIN = "enrichment"


class LLMConfigError(ValueError):
    """Неизвестный домен LLM (ошибки конфигурации — config.ConfigError)."""


# --- Factory ----------------------------------------------------------------

def build_llm(
    domain: str = DEFAULT_DOMAIN,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    context_window: Optional[int] = None,
    is_chat_model: bool = True,
    **kwargs,
) -> LLM:
    """Build a LlamaIndex ``LLM`` for a usage domain (``enrichment`` | ``query``).

    Значения по умолчанию — из корневого .env (``config.enrichment_llm()`` /
    ``config.query_llm()``); kwargs точечно перекрывают отдельные поля.

    Raises:
        LLMConfigError: unknown domain.
        config.ConfigError: доменная группа .env неполна/некорректна.
    """
    resolved_domain = (domain or DEFAULT_DOMAIN).strip().lower()
    if resolved_domain not in DOMAINS:
        raise LLMConfigError(
            f"Неизвестный домен LLM: {domain!r}. "
            f"Доступные домены: {', '.join(sorted(DOMAINS))}."
        )
    cfg = (
        config.enrichment_llm()
        if resolved_domain == "enrichment"
        else config.query_llm()
    )

    return OpenAILike(
        model=model or cfg.model,
        api_key=api_key or cfg.api_key,
        api_base=base_url or cfg.base_url,
        temperature=temperature if temperature is not None else cfg.temperature,
        context_window=context_window or cfg.context_window,
        # qwen/GLM отвечают только через /chat/completions — маршрутизируем
        # .complete() через chat API.
        is_chat_model=is_chat_model,
        **kwargs,
    )


def build_enrichment_llm(**kwargs) -> LLM:
    """LLM домена ``enrichment``: summary/keywords по чанкам кода."""
    return build_llm("enrichment", **kwargs)


def build_query_llm(**kwargs) -> LLM:
    """LLM домена ``query``: синтез ответов."""
    return build_llm("query", **kwargs)


def configure_global_llm(
    llm: Optional[LLM] = None, domain: str = DEFAULT_DOMAIN
) -> LLM:
    """Set ``Settings.llm`` to a domain client and return it.

    Pass a pre-built ``llm`` to use a custom instance; otherwise a fresh one
    is built via :func:`build_llm` for the given ``domain``.
    """
    if llm is None:
        llm = build_llm(domain)
    Settings.llm = llm
    return llm


if __name__ == "__main__":
    # Manual smoke test: python llm_provider.py [--domain enrichment|query]
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Смоук LLM-домена")
    parser.add_argument("--domain", choices=DOMAINS, default=DEFAULT_DOMAIN)
    args = parser.parse_args()

    try:
        llm = build_llm(args.domain)
    except config.ConfigError as e:
        sys.exit(str(e))
    print(f"[{args.domain}] model={llm.model} base_url={llm.api_base}")
    resp = llm.complete("ping")
    print(f"[{args.domain}] response:", str(resp).strip())
