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

## Answer-Quality E2E Judge (`bench/judge151/`)
- **Purpose**: mede a **qualidade da RESPOSTA** (não só retrieval) do pipeline híbrido do main (classificador gated + BM25 boost + FTS5-fix + síntese) nas **151 reais**, com o juiz vLLM 27B (4 eixos). Síntese na RTX 5070 (`~/llama.cpp/build-cuda`, `-ngl 99`, sampling Liquid 0.1/50/1.05); latência GPU NÃO vale p/ aparelho (`engine:cuda`, projeta-se pela tabela do device-bench).
- **Reuso**: `retrieval.py` espelha `HybridRetriever.kt` (paridade R@2 top-2 = 29.1% ≈ classifier 29.8%); síntese usa `AskPipeline.SYNTHESIS_SYSTEM_PROMPT`; juiz é `bench/judge.py`.
- **Achados**: (1) **gargalo é DUPLO, retrieval domina** — só 29.1% recebem o chunk-ouro no top-2; (2) **onde o retrieval acertou (44/151), o 1.2B embarcado converte só 11.4% em resposta aprovada vs 38.6% do 2.6B (3.4×)** — confirma o Q10 do capitão (LM é gargalo real dado o chunk certo); (3) 2.6B melhora global 1.72→2.20 e gate-pass 3.3%→14.6%, e alucina menos nas de retrieval ruim (26.2% vs 35.5%); (4) **antes/agora no 1.2B** (v2 pré-fix → híbrido): global 1.42→1.72, gate 1.0%→3.3%.
- **2.6B como sintetizador (não rewriter)**: thinking-ON é verboso (média 1430 tok, trunca 29/151 mesmo a 2048; TTFT proj. ~8.4 s S24+) — inviável p/ voz; thinking-OFF via `--reasoning-budget 0` (não há flag `enable_thinking` no template LFM2.5) gera 498 tok mas ainda verboso (markdown). Decode projetado ~18 tok/s no S24+ **fica acima da leitura humana (~4-7 tok/s)**, mas **RAM ~3.0 GB** inviabiliza S21 e aperta S24+. Recomendação: maior ganho por real ainda é **rewriter de termos técnicos** (destravar retrieval); subir p/ 2.6B só no S24+/≥8 GB com prompt curto. Relatório: `bench/judge151/README.md`.

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

## Architecture & NR Classifier / Hybrid Retrieval (`classifier/`)
- **Classifier directory**: `classifier/` contains the Opção-B 2-stage hybrid: um classificador leve de NR (estágio 1) que dá **boost suave** ao BM25 (estágio 2). Ambiente: venv dedicado `classifier/.venv` (uv, py3.12, scikit-learn 1.9; NÃO usar `.venv`/`.venv-train` da raiz — não têm sklearn). `nr_taxonomy.py` é a fonte única das 36 NRs (títulos oficiais do `nrs.parquet`).
- **Holdout FIXO (nunca em treino)**: 151 reais de `corpus/qa_pairs_v2.jsonl` + 20 de `bench/data/smoke_qa_20.jsonl` (`data_utils.load_holdout`). Dataset de treino sintético: `classifier/data/labels.jsonl` (1970 falas via vLLM, 0 contaminação, máx Jaccard 0.30).
- **Vencedor & achados**:
  - **LogisticRegression + TF-IDF (char_wb 2-5 + word 1-2)** venceu o shootout: top-1 48.5% / **top-2 69.0%** no holdout 171, <1 ms, deploy Kotlin puro. Critério: top-2 (driver do boost) > top-1 > simplicidade; probs calibradas p/ o gate.
  - **Prompt-classify no LFM2.5 confirmou o ceticismo do capitão**: melhor formato (JSON) só 25.7% top-1; "responda só o número" = 5.8%. Descartado (custo ~4.5 s TTFT/consulta no S24+, prefill de ~900 tok do catálogo).
  - **Híbrido CONFIANÇA-GATED** (filtro-duro se prob top-1 ≥0.5, senão boost suave 5x nas top-2; override de NR explícita citada na fala; sem boost se 'nenhuma'): **R@2 29.8% / R@5 41.1%** nas 151 vs baseline 13.9% (**+15.9 p.p., 2.1×**). Boost calibrado em **5x** (joelho da curva). **Reescrita PIORA** (usar fala/keywords brutas).
  - **Meta 35% R@2 não atingida por teto do desenho**: top-2 do classificador (72%) + BM25 intra-norma com fala bruta (44.4%) limitam; o gap restante depende de **termos técnicos** (tarefa do rewriter, enterrado), não de classificação.
- **Deploy Android**: `skl2onnx` NÃO converte char n-grams -> deploy é **Kotlin puro** (`NrClassifier.kt`, formato binário NRC2 `nr_classifier.bin` 4.03 MB, força-adicionado aos assets apesar do `*.bin` no .gitignore). Paridade validada vs sklearn **Δprob 2.3e-8** (crítico: L2 POR BLOCO char/word, e char_wb com break em palavra curta). `HybridRetriever.kt` decora `Fts5Retriever` (novo `searchBoosted`); ligado em `MainViewModel` e `AcceptanceRunner`.
- **E2E S24+ (release, modo avião)**: **8/10 fundamentadas** (mantém critério M3) e **latência 10.8 s < 12.5 s baseline** (estágio 1+2 = 8 ms mediana). Falhas restantes são limites honestos: Q09 (ambiguidade genuína) e Q10 (LLM diz 'não sei' apesar de chunk correto — síntese, não retrieval).
- **Commands**: `make -C classifier all` (classic+compare+hybrid+export). `make -C classifier labels` regenera o dataset via vLLM. Relatório: `classifier/README.md`.

## Architecture & Retrieval v3 (`retrieval3/`)
- **Objetivo**: destravar o gargalo do retriever (main: R@2 29.8% nas 151 coloquiais). Meta do capitão: R@2 ≥ 55%. Holdout INTOCÁVEL (151 `qa_pairs_v2` + 20 smoke); calibração só no dev-set sintético próprio (`data/dev_set.jsonl`, `gen_devset.py` seed 13, exclui chunk-ouro do holdout). Harness honesto = caminho do app (`eval_common.py` reusa `corpus.eval_*`: `app_fts_query`, pesos 1.5/3/2/1, `check_hit`).
- **Ambientes (crítico)**: busca/fusão/classificador em `classifier/.venv` (sklearn 1.9.1, py3.12); encode denso/rerank em `.venv-train` (torch cu128, sentence-transformers, sqlite-vec, py3.10). O `classic_winner.pkl` foi picklado com sklearn 1.9.1 → **NÃO carrega no .venv-train**; por isso o encode denso grava rankings em cache (`dump_dense.py`) e a fusão avalia no venv do classificador. Nenhum venv tem `pip` — use `VIRTUAL_ENV=... uv pip install`.
- **Achados (gate no holdout 151, ganhos que ACUMULAM)**:
  - **Expansão de documento** (Etapa 1, `expand_docs.py` via vLLM 27B: 4-6 perguntas coloquiais + sinônimos leigos por chunk, campo FTS5 `expansion` peso=1.0) é o maior ganho isolado: híbrido gated 29.8→**37.7% R@2**, 41.1→**52.3% R@5**. Anti-contaminação: prompt genérico por chunk, nunca vê as 151. Índice `indices/index_hf_36nr_exp.db` (14 MB, 5 campos).
  - **RRF multi-query só de BM25** (Etapa 2) NÃO supera o gated (filtro-duro confiante já é forte).
  - **Busca densa mobile-first** (Etapa 3): vencedor **EmbeddingGemma-300M** (prompts query/document, sqlite-vec). e5/MiniLM ficaram MUITO atrás dense-puro (e5-small 10.6%, MiniLM-paraphrase 4.0% — simétrico, impróprio p/ retrieval) → **confirma a ordem do capitão de não usar bge/e5 como default**. Dois índices densos: texto+exp (query↔passagem) e **expansão-only (query↔query coloquial)**.
  - **VENCEDOR MOBILE**: RRF 3-sinais (BM25-gated-exp + Gemma-texto + Gemma-exp-only, k=30) = **R@2 52.3% / R@5 63.6%** (+22.5 p.p. R@2, 1.75× vs main).
  - **Reranker** (Etapa 4): só o **27B listwise (cloud) cruza a meta (56.3% R@2)**. Cross-encoders mobile (mMiniLM/bge/jina) e o **LFM2.5-1.2B embarcado (listwise e pointwise sim/não) REGRIDEM** — modelo pequeno não ranqueia. Logo, 55% mobile-only ainda não foi atingido; é decisão do capitão (fusão 52.3% on-device vs +reranking na nuvem p/ 56.3%).
- **Custo mobile do denso**: EmbeddingGemma-300M (≤350M), dim 768 (truncável 256 via Matryoshka), índices ~31 MB, encode 1 query ~37 ms na 5070 (projeção S24+ int8 <300 ms). Motor **sqlite-vec** (compila arm64, família do `libsqliteX` já embarcado). Valida RAM no S21 antes de embarcar.
- **Entregue como biblioteca + relatório; NÃO integrado ao app** (consolidação do app corre em paralelo). Relatório: `retrieval3/README.md`; triagem de modelos: `retrieval3/MODEL_SELECTION.md`; tabela final: `results/consolidation.json`. Comandos: `make -C retrieval3 all` (a partir dos artefatos) ou alvos `baseline`/`expand`/`dense`/`rerank`/`consolidate`. Índices `.db` e caches `dense_rank_*.json` são regeneráveis e ficam no `.gitignore`.

## Android Subproject (`android/`)
- **CRÍTICO — FTS5 no Android**: O SQLite do sistema Android **não inclui o módulo FTS5**; `chunks_fts MATCH`/`bm25()` lançam `no such module: fts5` e o `Fts5Retriever` cai no `fallbackSearch` (LIKE ingênuo), retornando lixo e fazendo o app responder "Não sei". Correção (M3): dependência `mil.nga:sqlite-android:3500400` (fork do requery, SQLite 3.50, `libsqliteX.so`) + `import org.sqlite.database.sqlite.SQLiteDatabase` + `System.loadLibrary("sqliteX")` no `Fts5Retriever`. Nunca usar `android.database.sqlite` para queries FTS5. Detalhes: `android/README_M3.md`.
- **Headless Toolchain**: Run `./android/setup-sdk.sh` to install OpenJDK 17 and Android SDK 34 (with build-tools, NDK, CMake) into `~/android-sdk`. Idempotent.
- **Environment**: Set `JAVA_HOME="$HOME/android-sdk/jdk-17"` and `ANDROID_HOME="$HOME/android-sdk"` before building.
- **Build & Test**: `./gradlew test` (runs JVM unit tests), `./gradlew assembleDebug` (debug APK), `./gradlew assembleRelease` (outputs signed 868 MB APK to `android/app/build/outputs/apk/release/app-release.apk` with all weights embedded).
- **Native Architecture & Engines**: Independent library modules `:llama` (`libllama_engine.so`, commit `434ddbb`, KleidiAI, ARM64 dotprod/i8mm/fp16) and `:whisper` (`libwhisper_engine.so`, v1.9.4, 4 threads, `ggml-base-q5_1.bin`).
- **Modo C Top-2 Multiturno (`AskPipeline`)**:
  - Turn 1 (Rewrite): Raw history up to 6 turns + question -> 3-6 keywords. Jaccard similarity > 0.7 reuses previous chunks without new search.
  - BM25 Top-2: SQLite FTS5 on `index_hf_36nr.db` (pesos 1.5, 3.0, 2.0, 1.0; latency ~2-15 ms).
  - Turn 2 (Synthesis): Raw history up to 3 turns + 2 chunks + question -> streaming synthesis with mandatory citation. Prunes oldest turn if prompt > 1000 tokens.
- **Overrides**: `ModelFileManager` and `Fts5Retriever` check `getExternalFilesDir(null)/` before `filesDir/` or APK assets for `index.db`, `ggml-base-q5_1.bin`, and `LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf`.
- **UI**: CeiaTheme, Push-to-Talk, editable transcription, expandable sources per answer, and prominent Modo Engenharia / Debug toggle (with proportional timeline breakdown bar).

## Achados M3 (retrieval, rewriter, porte) — leia antes de repetir experimentos
- **Sintoma "Não sei" no S24+ era o FTS5 ausente** (ver seção Android acima), não chunking nem rewriter. O fix elevou o roteiro de aceitação de ~2/10 para **8/10 fundamentadas com NR correta** (release, modo avião; `android/acceptance_results_m3.json`). Latência 10–15 s (acima do teto de 10 s de voz); alavanca = contexto topk2 lean (~820 tok), não porte.
- **Chunking fino/médio PIORA o recall** (grosso R@2 teto 77.5% vs fino 51%, termos+filtro). Mantido `index_hf_36nr.db` grosso. Chunker de item disponível em `corpus/chunk.py --fine` (diagnóstico). Harness honesto (caminho de busca idêntico ao app): `corpus/eval_fino.py`. Relatório: `corpus/README_M3.md`.
- **Avaliar SEMPRE nas 151 perguntas reais** (`corpus/qa_pairs_v2.jsonl`) com o caminho do app (`app_fts_query`: stem-6 + prefixo* + OR, pesos 1.5/3/2/1). O "85.4% R@5" do v2 usava match-exato + filtro-de-norma-ouro (artifícios que o app não faz); zero-shot real ≈ 14.6% R@2.
- **LoRA do rewriter REGREDIU no holdout limpo** (r=8: −8.6pp; r=16: −4.6pp vs zero-shot 14.6% R@2). O "92%" anterior era zero-shot em dataset sintético invertido contaminado (falas copiavam o vocabulário do chunk). Trilha limpa e descontaminada: `finetune/gen_dataset_clean.py`, `validate_dataset_clean.py`, `eval_rewriter_clean.py`; relatório `finetune/README_M3.md`. Ambos gates falharam → treino parado por ordem.
- **LFM2.5-2.6B descartado como rewriter**: é modelo de raciocínio (sempre "pensa", ~1010 tokens/~4 s por rewrite, inviável p/ voz) e pior (7.9% R@2). O GGUF oficial `Q4_K_M` gera lixo no build `434ddbb`; usar `Q4_0`. Ambiente de treino: `.venv-train` (uv, py3.10, torch cu128 p/ RTX 5070 Blackwell); adapters preservados em `finetune/adapter_r8/`, `finetune/adapter_r16/`.

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
