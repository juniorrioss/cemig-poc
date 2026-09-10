# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Architecture & Corpus Pipeline (`corpus/`)
- **Corpus directory**: `corpus/` contains the ingestion, chunking, indexing, and evaluation pipeline for regulatory standards (NRs) targeted at Android offline RAG.
- **Data locations**: Source PDFs reside in `/home/rios/projetos/cemig-mobile-llm/firstmate/data/NRs/` (v1); clean 36-NR Parquet dataset with MTE manuals resides in `/home/rios/projetos/cemig-mobile-llm/firstmate/data/nrs_hf/nrs.parquet` (v2). Never commit external data to git.
- **Standards & Datasets**: v1 covers 5 core NRs (`qa_pairs.jsonl`, 101 pairs); v2 expands to all 36 NRs with 151 pairs in `qa_pairs_v2.jsonl` (adding NR-01 GRO/recusa, NR-16 periculosidade, NR-26 sinalização/GHS, NR-33 espaços confinados).
- **Retrieval engine & Indices**: Embedded SQLite FTS5 virtual table with `unicode61 remove_diacritics 2` and `bm25(chunks_fts)`. v2 indices: `index_hf_5nr.db` (1.64 MB), `index_hf_5nr_manual.db` (3.35 MB), and `index_hf_36nr.db` (8.28 MB).
- **Core Corpus v2 Findings**:
  - Clean Markdown source improves Recall@1 (+3.9 p.p.) and Recall@5 (+1.0 p.p.) over fragile PDF extraction.
  - Adding commented manuals directly to BM25 severely pollutes retrieval (drops R@5 by >23 p.p., displacing the binding norm chunk from top-3 in 20.8% of queries). Recommendation: embed **`index_hf_36nr.db` (norm-only, 8.28 MB)** and isolate manuals in a separate table/UI tab.
  - Expanding 5 to 36 NRs has negligible dilution (-1.9 p.p. R@5) under tool-calling with norm filter, achieving **85.4% Recall@5 and 0.6881 MRR** across all 151 questions.
- **Pipeline commands**: `make all` runs v1 pipeline. `make -C corpus v2-all` runs v2 ingest -> chunk -> index -> eval. `make -C corpus v2-eval` runs the 4-way comparative evaluation.
- **Python interpreter**: Uses `python3` with stdlib `sqlite3`, `pypdf`, and `pyarrow`. FTS5 query terms with hyphens or dots must be quoted (e.g. `"nr-10"`, `"10.5.1"`) to prevent FTS5 syntax errors.

## Architecture & SLM Benchmark (`bench/`)
- **Benchmark directory**: `bench/` contains the harness, setup, judge, and performance benchmarking for mobile-oriented SLMs (≤1.2B) in GGUF Q4_K_M via `llama.cpp`.
- **Model storage**: Downloaded GGUF weights reside outside the repo at `~/models-poc/` (never committed). llama.cpp builds in `~/llama.cpp/`.
- **Tier 1 Models**: Qwen 3.5 0.8B (Gated DeltaNet hybrid), Qwen 3 0.6B (Baseline), Gemma 3 1B IT, Liquid LFM2 1.2B RAG, Liquid LFM2.5 350M, Meta Llama 3.2 1B (Control).
- **Core Findings on Tool-Calling vs Classic RAG**:
  - Direct SLM-driven tool-calling in models ≤1B is NOT viable for production offline voice on Galaxy S21/S24+ (double inference round-trip exceeds 10s budget, 4 of 6 models do not support/ignore native tool schemas in PT-BR, grammar parsing fails on multi-turn).
  - Recommended mobile architecture: **Classic RAG with deterministic external heuristic trigger** (keyword/intent classifier before SLM) + single inference turn.
  - Recommended model: **Liquid LFM2 1.2B RAG** (highest factual accuracy 3.30/5, 40% gate pass, 5.1s latency) or **Gemma 3 1B IT** (highest PT fluency 4.50/5, 3.4s latency).
- **Commands**: `make -C bench all` runs setup, perf, bench, and judge. Quick smoke: `python3 bench/harness.py --models qwen3.5-0.8b --mode both --limit 2`.

## Architecture & ASR Offline Benchmark (`asr/`)
- **ASR directory**: `asr/` contains test data generation, evaluation harness, and benchmark results for offline Brazilian Portuguese ASR targeting electrical utility field workers.
- **Model storage**: External models reside at `~/models-asr/` (never committed). Engine builds reside at `~/whisper.cpp/` and `~/sherpa-onnx/`.
- **Engines evaluated**: Whisper.cpp (Tiny, Base, Small in Q5_1), Sherpa-ONNX (Nemotron 3.5 Streaming INT8 Official and Ottema PT-BR fine-tune), Android Native SpeechRecognizer.
- **Core Findings on Mobile ASR**:
  - **Whisper Base Q5_1 is the recommended production engine**: 196.8 MB peak RSS, 56.9 MB package, RTF 0.84 (<1.0), 9.5% clean WER / 11.8% noisy WER (10 dB SNR), 85.5% domain term accuracy. Crucially, 197 MB ASR + 700 MB SLM = ~897 MB total RAM, safely under Android LMK limits for 6 GB RAM phones.
  - **Nemotron 3.5 INT8 (Official & Ottema PT-BR)**: Fastest streaming (RTF 0.36) and highest domain accuracy (91.9%), but heavy memory footprint (795 MB RSS) and package (651 MB). Reserved for dedicated 12 GB RAM devices where real-time streaming partials are required.
  - **Whisper Small Q5_1**: Unviable for interactive mobile CPU (RTF 2.81, ~16s latency).
  - **Android Native ASR**: Unreliable for offline distribution without manual user intervention to download language packs in Google Settings; zero support for technical domain biasing.
- **Commands**: `make asr-data` generates audio sets; `make asr-bench` executes benchmark on connected Android device. Reference report at `asr/README.md`.

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
