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
- **Model storage**: Downloaded GGUF weights reside outside the repo at `~/models-poc/` (never committed). llama.cpp builds in `~/llama.cpp/` (CPU) and `~/llama.cpp/build-cuda` (GPU sm_120).
- **Evaluated Models**: Qwen 3.5 0.8B (Gated DeltaNet hybrid), Qwen 3 0.6B (Baseline), Gemma 3 1B IT, Liquid LFM2 1.2B RAG, Liquid LFM2.5 1.2B Instruct, Liquid LFM2.5 1.2B Thinking, Liquid LFM2.5 350M, Meta Llama 3.2 1B (Control).
- **Modes**: Classic-Raw (BM25 com pergunta bruta), Mode C (`query_rewrite_inject`: Turno 1 reescreve query curta + Turno 2 sintetiza com chunks injetados), Tool-Calling (JSON/Pythonic), topk2 (variante top-2 chunks).
- **Core Findings (v2 Benchmark)**:
  - **Abismo Lexical**: Pergunta bruta de operário recupera o chunk correto em apenas 20.8% das buscas; Modo C eleva para 28.7%; teto oracle com termos técnicos curados é 72.3%.
  - **Fine-Tuning Cirúrgico no Turno 1**: O modelo já faz boa síntese se o chunk estiver presente; o fine-tuning (LoRA 500-1000 pares) é estritamente justificado no Turno 1 (converter fala de campo em termos técnicos de NR) para saltar de 28.7% para ~70% de recall.
  - **Variante Top-2 Chunks**: Reduz o prompt de ~1420 para ~820 tokens, acelerando a inferência em ~30% (5.0s no LFM2.5 Instruct) sem perda de acerto factual, viabilizando o teto de 10s de voz no Galaxy S24+/S21.
  - **Vencedores v2**: **Liquid LFM2.5 1.2B Instruct** (mais rápido, 5.0s no topk2, maior recall de reescrita 28.7%) e **Liquid LFM2 1.2B RAG** (maior acerto factual). LFM2.5 Thinking é inviável (15-16s de latência, reflexão em inglês destrói a reescrita).
- **Judge Automatizado**: vLLM em `http://10.100.0.111:8005/v1` (`Qwen/Qwen3.8-27B-FP8`) com 8 workers. Amostra de calibração de 50 itens contra `claude -p` com 98% de concordância $\pm 1$.
- **Commands**: `make -C bench all-v2` runs harness v2 and judge v2. Quick smoke: `python3 bench/harness.py --models qwen3.5-0.8b --mode all_three --limit 2`.
- **On-Device Benchmark (`bench/device/`)**:
  - Build script: `./bench/device/build-android.sh` builds static `llama-bench` and `llama-cli` with `-march=armv8.4-a+dotprod+i8mm+fp16 -DGGML_CPU_KLEIDIAI=ON -DBUILD_SHARED_LIBS=OFF`.
  - Runner: `./bench/device/run-device-bench.sh` runs benchmarks via ADB on Galaxy S24+ (`SM-S926B`, Exynos 2400).
  - Thread tuning: Use 6 threads (`-t 6`) to pin to the 6 Cortex-X4 / A720 performance cores; spilling into the 4 Cortex-A520 efficiency cores degrades prefill by >30%.
  - CLI non-interactive flag: Always run `llama-cli` with `--single-turn < /dev/null` on ADB shell to avoid infinite prompt loops.
  - S24+ latency findings: With 1500 tokens prefill + 100 tokens decode, only Liquid LFM2.5 350M (5.07s) meets the ≤10s voice latency budget on S24+; models ≥0.6B require restricting RAG context to ≤800 tokens to fit. On S21, context must be ≤300–500 tokens.
- **Liquid On-Device Ecosystem Benchmark (`bench/device-liquid/`)**:
  - Liquid officially deprecated LEAP SDK and `.bundle` formats; official Android recommendation is direct C++/NDK `llama.cpp` embedding.
  - Checkpoint `LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf` (Quantization-Aware Distillation) outperforms vanilla `Q4_K_M`: +22.6% prefill in lean RAG pp800 (219 vs 179 tok/s), TTFT 3.65s vs 4.48s, -67.5 MiB RAM (1420.8 MiB), cold load 6.63s vs 9.05s (-26.8%), and lower end-to-end response latency (6.84s vs 7.49s). Recommended weights for the production app. Report at `bench/device-liquid/README.md`.

## Architecture & ASR Offline Benchmark (`asr/`)
- **ASR directory**: `asr/` contains test data generation, evaluation harness, and benchmark results for offline Brazilian Portuguese ASR targeting electrical utility field workers.
- **Model storage**: External models reside at `~/models-asr/` (never committed). Engine builds reside at `~/whisper.cpp/` and `~/sherpa-onnx/`.
- **Engines evaluated**: Whisper.cpp (Tiny, Base, Small in Q5_1), Sherpa-ONNX (Nemotron 3.5 Streaming INT8 Official and Ottema PT-BR fine-tune), transcribe.cpp (Nemotron 3.5 Streaming GGUF Q4_K_M, Q5_K_M, Q6_K, Q8_0), Android Native SpeechRecognizer.
- **Core Findings on Mobile ASR**:
  - **Whisper Base Q5_1 is the recommended production engine**: 196.8 MB peak RSS, 56.9 MB package, RTF 0.84 (<1.0), 9.5% clean WER / 11.8% noisy WER (10 dB SNR), 85.5% domain term accuracy. Crucially, 197 MB ASR + 700 MB SLM = ~897 MB total RAM, safely under Android LMK limits for 6 GB RAM phones (Galaxy S21).
  - **Nemotron 3.5 INT8 & GGUF (Q4_K_M a Q8_0)**: High quality (Q5_K_M achieves 5.8% WER in noise, 95.2% domain accuracy), but peak RAM remains between 792 MB (sherpa-onnx) and 943–1186 MB (transcribe.cpp/ggml due to ~470 MB runtime graph overhead and F16->F32 conv unpacking on CPU). Quantization of 600M models does NOT reduce RAM below the 550 MB target needed for S21 coexistence.
  - **Nemotron ONNX INT4 (`onnx-community`)**: Incompatible with sherpa-onnx (requires onnxruntime-genai, lacks embedded metadata); package is 751 MB (> 651 MB INT8).
  - **Whisper Small Q5_1**: Unviable for interactive mobile CPU (RTF 2.81, ~16s latency).
  - **Android Native ASR**: Unreliable for offline distribution without manual user intervention to download language packs in Google Settings; zero support for technical domain biasing.
- **Commands**: `make asr-data` generates audio sets; `make asr-bench` executes benchmark on connected Android device. Reference report at `asr/README.md`.

## Android Subproject (`android/`)
- **Headless Toolchain**: Run `./android/setup-sdk.sh` to install OpenJDK 17 and Android SDK 34 (with build-tools, NDK, CMake) into `~/android-sdk`. Idempotent.
- **Environment**: Set `JAVA_HOME="$HOME/android-sdk/jdk-17"` and `ANDROID_HOME="$HOME/android-sdk"` before building.
- **Build & Test**: `./gradlew test` (runs JVM unit tests) and `./gradlew assembleDebug` (outputs debug APK to `android/app/build/outputs/apk/debug/app-debug.apk`).
- **Tool-calling RAG Architecture**: `AskPipeline` does not inject chunks automatically. It prompts the LLM with `retriever(query: String)`; the model formulates BM25 technical search terms; on tool call, pipeline executes `Fts5Retriever` on `index.db` and returns chunks as a `TOOL` message for final synthesis with mandatory citation.
- **Corpus override**: `Fts5Retriever` checks `getExternalFilesDir(null)/index.db` before internal `filesDir/index.db`, allowing instant corpus updates via `adb push index.db /sdcard/Android/data/br.org.ceia.cemigpoc.debug/files/index.db`.

## Fine-Tuning & Query Rewriter (`finetune/`)
- **Rewriter directory**: `finetune/` contains inverted dataset generation, BM25 execution filter, PEFT LoRA training on `LiquidAI/LFM2.5-1.2B-Instruct`, and evaluation of Turn 1 query rewriting for Mode C.
- **Data & Model storage**: `finetune/data/` stores `synthetic_raw.jsonl` (2400 pairs), `train.jsonl` (1767), `val.jsonl` (279), `test.jsonl` (294 unseen NRs). LoRA adapter (3.7 MB GGUF) in `finetune/adapter/` and merged model in `finetune/merged_model/` (never commit large weights to git).
- **Core Findings & Captain's Skepticism Resolution**:
  - **Recall@2 in Unseen NRs**: LoRA rewriter reaches **80.0% Recall@2** (and 86.0% Recall@5) on holdout NRs, far exceeding the 55% target.
  - **Synthesis Regression Gate**: Protocol v2 (20 gold + 30 colloquial questions) passed with zero regression (Accuracy +0.42, Citation +1.58, Fidelity +0.44, Fluency 4.5/5.0) anchored by 10% synthesis maintenance pairs.
  - **Captain's Skepticism on LoRA Switching Overhead**: Empirically measured in `llama.cpp`: dynamic hot-swap via `POST /lora-adapters` takes **0.71 ms** (or 0.00 ms via per-request `lora` field), consuming only 1.4% of the mobile BM25 retrieval window (~50 ms). Adapter always-active (merged model) has 0.00 ms overhead and causes zero synthesis degradation.
  - **Production Recommendation**: Ship **merged model** (~700 MB Q4_K_M) for initial release (zero runtime orchestration), or base GGUF + standalone LoRA (3.7 MB) for agile OTA continuous updates during the BM25 window.
- **Pipeline commands**: `python3 finetune/gen_dataset.py`, `python3 finetune/validate_dataset.py`, `python3 finetune/train_lora.py`, `python3 finetune/eval_rewriter.py`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
