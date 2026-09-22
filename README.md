# RAG — Semantic Code Search for C# Codebases

Local Python tool for semantic search over any C# codebase.
Ask questions like "how does X work?" and get answers grounded in the actual code:
retrieval over whole-file chunks enriched with LLM-generated summaries and keywords,
followed by LLM synthesis of the answer.

Standalone developer tooling, kept in its own repository — it indexes the target
project, never itself.

## How it works

```
SimpleDirectoryReader (.cs files)
        │
IngestionPipeline
  ├── CodeSplitter (tree-sitter)      AST chunking (whole-file chunks by default)
  ├── SummaryExtractor (LLM)          section_summary metadata
  ├── KeywordExtractor (LLM)          excerpt_keywords metadata
        │
build_index.py → persistent vector index (BAAI/bge-m3 embeddings, JSON storage)
        │
query.py → top-k retrieval → LLM-synthesized answer + sources
```

- Incremental indexing: unchanged files are skipped via doc-hash dedup; documents are
  processed in batches with a persist after every batch, so interrupted runs resume for free.
- Embedding: local BAAI/bge-m3 (multilingual — Russian queries match English summaries).
- LLMs: any OpenAI-compatible endpoint, with separate configuration domains for
  enrichment and query (defaults are set up for z.ai GLM via the coding endpoint).

## Repository layout

```
ast/scripts/            AST chunking + indexing + query domain
  config.py             single point of configuration (reads/validates the root .env)
  ast_llamaindex.py     step 1: AST chunking + enrichment
  build_index.py        step 2: ingest → persistent vector index (incremental)
  query.py              step 2: question → top-k chunks → LLM answer
  llm_provider.py       per-domain LLM factory (enrichment / query)
requirements.txt
```

Vector indexes are **owned by the client project**: you pass `--index-dir` on every
call, and the RAG repository hosts no indexes itself. Each index directory carries a
`rag_meta.json` with its corpus root and profile (`enrich|plain:<embed_model>`).

## Installation

### Prerequisites

- **Python 3.13**
- **Git**
- The C# codebase you want to index — its sources folder is passed via `--path` on
  every invocation, and the index location via `--index-dir` (a `.rag/` folder inside
  the client project is a convenient default).

### 1. Clone and set up a virtual environment

macOS / Linux:

```bash
git clone https://github.com/nakitmn/RAG.git
cd RAG
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows:

```bat
git clone https://github.com/nakitmn/RAG.git
cd RAG
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Throughout this README, substitute `.venv\Scripts\python` for `.venv/bin/python` on Windows.

### 2. Configure

Copy the template into the root `.env` and fill in your API keys:

```bash
cp .env.example .env
```

The root `.env` is the single configuration point — it is read and validated exclusively
by `ast/scripts/config.py`; there are no tunable defaults in the code. The only values
you must provide are the API keys `RAG_ENRICHMENT_API_KEY` / `RAG_QUERY_API_KEY` (a z.ai
GLM Coding Plan key by default; the coding endpoint `https://api.z.ai/api/coding/paas/v4`
is already the default base URL). Everything else ships with working defaults — see
`.env.example` for the full list of variables and their meaning.

Notes:

- Paths are never configured in `.env`: `--path`, `--output` and `--index-dir` are
  required CLI arguments (relative paths resolve against the current working directory;
  absolute paths are recommended).
- `RAG_EMBED_DEVICE=auto` autodetects Apple MPS and otherwise falls back to CPU.
  With an NVIDIA GPU, install a CUDA-enabled torch build (see pytorch.org) and set
  `RAG_EMBED_DEVICE=cuda` for faster embedding.

### 3. Verify

```bash
# config self-check (offline; keys are masked)
.venv/bin/python ast/scripts/config.py

# LLM smoke test (real ping to the configured endpoint)
.venv/bin/python ast/scripts/llm_provider.py --domain enrichment
.venv/bin/python ast/scripts/llm_provider.py --domain query
```

## Building the index

```bash
# full corpus, enriched (production build; the index lives in the client project)
.venv/bin/python ast/scripts/build_index.py \
  --path /path/to/your/csharp/project --index-dir /path/to/your/project/.rag

# quick offline chunking test, no LLM required
.venv/bin/python ast/scripts/build_index.py \
  --path /path/to/your/csharp/project --index-dir <scratch> --no-enrich
```

- Runs are **incremental**: unchanged files are skipped (no LLM, no embeddings).
  After code changes, just rerun the same command. After deleting/renaming files, add `--prune`.
- An interrupted run resumes from where it stopped — rerun the same command.
- ⚠️ Do not run `--no-enrich` against a production index — the build warns about the
  profile flip (`enrich` → `plain`). Use a separate `--index-dir` for offline experiments,
  or `--reset` for a deliberate full rebuild.
- **Root guard**: rebuilding a directory with a different `--path` fails; pass
  `--switch-root` for a deliberate corpus-root change (follow up with `--prune`).
- The first run downloads BAAI/bge-m3 (~2.3 GB) from Hugging Face.
- Rough speed with enrichment: a corpus of ~1,750 files takes about 2–3 hours
  (measured on an Apple M1 Max).

## Querying

```bash
.venv/bin/python ast/scripts/query.py \
  --index-dir /path/to/your/project/.rag "Where is dependency injection configured?"

# retrieval-only debug mode (no LLM, no query keys required)
.venv/bin/python ast/scripts/query.py --index-dir /path/to/your/project/.rag "how does X work" --no-llm --top-k 5
```

If the current `RAG_EMBED_MODEL` differs from the index profile, the query exits with an
error before the embedding model loads — follow the hint in the message (override the
variable or rebuild the index).

## Language support

C# (`.cs`) only at the moment. The `SUPPORTED_LANGUAGES` registry in `ast_llamaindex.py`
is designed for extension: install a tree-sitter grammar package (e.g. `tree-sitter-kotlin`)
and add a single registry entry.
