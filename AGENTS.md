# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Architecture & Corpus Pipeline

- **Corpus directory**: `corpus/` contains the ingestion, chunking, indexing, and evaluation pipeline for regulatory standards (NRs) targeted at Android offline RAG.
- **Data location**: Source PDFs reside in `/home/rios/projetos/cemig-mobile-llm/firstmate/data/NRs/` and must never be committed to git.
- **Core 5 NRs**: NR-10 (Electricity), NR-06 (PPE), NR-35 (Height), NR-12 (Machinery/LOTO), NR-18 (Construction).
- **Retrieval engine**: Embedded SQLite FTS5 virtual table with `unicode61 remove_diacritics 2` and `bm25(chunks_fts)`.
- **Pipeline command**: `make all` executes extract -> chunk -> build_index -> eval_retrieval.
- **Python interpreter**: Uses `python3` (or `.venv` if configured) with stdlib `sqlite3` and `pypdf`. FTS5 query terms with hyphens or dots must be quoted (e.g. `"nr-10"`, `"10.5.1"`) to prevent FTS5 syntax errors.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
