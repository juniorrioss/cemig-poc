# cemig-poc — Assistente por voz 100% offline (POC)

Q&A sobre Normas Regulamentadoras (domínio elétrico) rodando inteiramente no dispositivo Android (Galaxy S21/S24+), sem internet na execução.

Pipeline: Voz (PT-BR) → ASR → SLM com tool-calling → `retriever(query)` BM25/SQLite-FTS5 → resposta com citação de fonte obrigatória.

## Estrutura
- `corpus/` — pipeline desktop (Python): PDF das NRs → chunks → index.db (FTS5) → avaliação (recall@k, P&R sintético)
- `bench/` — benchmark de SLMs (llama.cpp GGUF, protocolo único de qualidade PT + tool-calling)
- `asr/` — bateria de experimentos ASR (nativo Android, whisper.cpp, Nemotron/sherpa-onnx)
- `android/` — app Kotlin/Compose (tema CEIA, motores fake → reais, tool-calling RAG, SQLite FTS5)

## App Mobile Android (`android/`)

O aplicativo é 100% nativo em Kotlin + Jetpack Compose com o tema oficial do CEIA Design System.

### Como rodar no WSL2:
```bash
# 1. Configurar o SDK e OpenJDK 17 headless (idempotente):
./android/setup-sdk.sh

# 2. Carregar variáveis de ambiente na sessão atual:
export JAVA_HOME="$HOME/android-sdk/jdk-17"
export ANDROID_HOME="$HOME/android-sdk"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH"

# 3. Executar os testes unitários JVM:
./gradlew test

# 4. Gerar o APK de Debug:
./gradlew assembleDebug
```
O APK gerado fica em: `android/app/build/outputs/apk/debug/app-debug.apk`.
Consulte [android/README.md](android/README.md) para detalhes da arquitetura, tokens de design e telemetria.


Plano completo e decisões: artefato Lavish do firstmate (`.lavish/poc-slm-mobile.html`).

Requisitos-chave: latência total ≤ 10 s no S21 · fonte obrigatória · só texto · RAG disparado via tool pelo SLM (nem toda pergunta consulta; o modelo pode reformular a query).
