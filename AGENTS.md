# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Architecture & Corpus Pipeline (`corpus/`)
- **Corpus directory**: `corpus/` contains the ingestion, chunking, indexing, and evaluation pipeline for regulatory standards (NRs) targeted at Android offline RAG.
- **Data location**: Source PDFs reside in `/home/rios/projetos/cemig-mobile-llm/firstmate/data/NRs/` and must never be committed to git.
- **Core 5 NRs**: NR-10 (Electricity), NR-06 (PPE), NR-35 (Height), NR-12 (Machinery/LOTO), NR-18 (Construction).
- **Retrieval engine**: Embedded SQLite FTS5 virtual table with `unicode61 remove_diacritics 2` and `bm25(chunks_fts)`.
- **Pipeline command**: `make all` executes extract -> chunk -> build_index -> eval_retrieval.
- **Python interpreter**: Uses `python3` (or `.venv` if configured) with stdlib `sqlite3` and `pypdf`. FTS5 query terms with hyphens or dots must be quoted (e.g. `"nr-10"`, `"10.5.1"`) to prevent FTS5 syntax errors.

## Android Subproject (`android/`)
- **Headless Toolchain**: Run `./android/setup-sdk.sh` to install OpenJDK 17 and Android SDK 34 (with build-tools, NDK, CMake) into `~/android-sdk`. Idempotent.
- **Environment**: Set `JAVA_HOME="$HOME/android-sdk/jdk-17"` and `ANDROID_HOME="$HOME/android-sdk"` before building.
- **Build & Test**: `./gradlew test` (runs JVM unit tests) and `./gradlew assembleDebug` (outputs debug APK to `android/app/build/outputs/apk/debug/app-debug.apk`).
- **Tool-calling RAG Architecture**: `AskPipeline` does not inject chunks automatically. It prompts the LLM with `retriever(query: String)`; the model formulates BM25 technical search terms; on tool call, pipeline executes `Fts5Retriever` on `index.db` and returns chunks as a `TOOL` message for final synthesis with mandatory citation.
- **Corpus override**: `Fts5Retriever` checks `getExternalFilesDir(null)/index.db` before internal `filesDir/index.db`, allowing instant corpus updates via `adb push index.db /sdcard/Android/data/br.org.ceia.cemigpoc.debug/files/index.db`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
