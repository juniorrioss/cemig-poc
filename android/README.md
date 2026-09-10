# CEMIG POC — Aplicativo Android de Assistente por Voz 100% Offline

Aplicativo Android nativo (Kotlin + Jetpack Compose) de assistência técnica e conformidade de segurança para operários de campo do setor elétrico (Galaxy S21 e S24+).

Funciona com **zero conectividade à internet** (100% em modo avião), resposta apenas em texto (sem TTS), citação obrigatória de fontes das Normas Regulamentadoras (NR-10, NR-35) e latência total $\le 10$ segundos.

---

## 1. Instalação do Toolchain Headless no WSL2

O script `setup-sdk.sh` configura o ambiente de compilação de ponta a ponta em ambiente WSL2 sem necessitar de interface gráfica ou Android Studio:

```bash
# A partir da raiz do repositório:
./android/setup-sdk.sh
```

### O que o script instala automaticamente:
- **OpenJDK 17 (Temurin)** em `~/android-sdk/jdk-17`
- **Android Commandline Tools** em `~/android-sdk/cmdline-tools/latest`
- Aceite automático de todas as licenças do Android SDK
- Pacotes via `sdkmanager`:
  - `platform-tools`
  - `platforms;android-34`
  - `build-tools;34.0.0`
  - `ndk;26.1.10909125`
  - `cmake;3.22.1`

O script é **idempotente**: verificações internas ignoram downloads se os componentes já existirem.

### Carregando as variáveis de ambiente:
```bash
export JAVA_HOME="$HOME/android-sdk/jdk-17"
export ANDROID_HOME="$HOME/android-sdk"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH"
```

---

## 2. Compilação e Testes via Terminal

Sem depender de Android Studio, o Gradle Wrapper já está commitado:

```bash
# Rodar os testes unitários JVM do pipeline e fakes:
./gradlew test

# Gerar o APK de Debug:
./gradlew assembleDebug
```

O APK gerado fica disponível em:
`android/app/build/outputs/apk/debug/app-debug.apk`

---

## 3. Design System CEIA (`CeiaTheme`)

A interface foi portada diretamente dos tokens e regras oficiais de `~/projetos/sebrae-portal-design`:

- **Cores Oficiais**:
  - Marinho do logotipo: `--ceia-navy-950` (`#001a5b`) e `--ceia-navy-800` (`#1b3983`)
  - Ação primária: `--ceia-blue-500` (`#2866e5`)
  - Destaque da marca: `--ceia-teal-500` (`#0cd4aa`)
  - Fundo padrão: `--ceia-cream` (`#f2f1ee`)
  - Semânticas: Sucesso (`#0a9d7f`), Atenção (`#b45309`), Falha (`#d92626`), Destaque (`#7c3aed`)
- **Tipografia Oficial**:
  - Títulos e Destaques: **Actay Wide** (`actay_wide_bold.otf`, `actay_regular.otf`)
  - Textos de corpo e botões: **Inter** (`inter_variable.ttf`)
- **Regras Estritas do DS**:
  - **Estado nunca apenas por cor**: Todo indicador de status (`StatusBadge`) possui rótulo textual explícito e dot colorido.
  - **UI 100% em PT-BR**.
  - **Logotipo CEIA**: Mantido oficial e sem recolorir (`ceia_logo_navy.png`).
  - **Formas**: Cantos generosos (`6dp`, `10dp`, `16dp`, `999dp`).

---

## 4. Arquitetura do RAG por Tool-Calling (`AskPipeline`)

Requisito central do Capitão:
> *"gostaria que o RAG do modelo fosse disparado via tool. Ou seja, nem sempre é feito alguma requisição de retriever. E quando for feito, para alguma tool default - retriver( XXXX ), podemos ensinar como pesquisar via BM25. Ou seja dar mais liberdade para o LLM melhorar a busca com termos mais assertivos" — evoluindo primeiro via system prompt.*

### Contrato de Turnos do Pipeline:
```
[Operário: Voz/Texto]
        │
        ▼
[Turno 1: LLM recebe Pergunta + System Prompt com especificação da tool]
        │
   ┌────┴───────────────────────────────┐
   │                                    │
   ▼ (Pergunta geral/saudação)          ▼ (Dúvida técnica sobre normas)
[LLM emite TextDelta]              [LLM emite ToolCall("retriever", query)]
   │                                    │
   ▼                                    ▼
[Done (Sem fontes)]                [AskPipeline executa FTS5Retriever (BM25)]
                                        │
                                        ▼
                                   [AskPipeline emite ToolResult(chunks)]
                                        │
                                        ▼
                                   [Turno 2: LLM recebe retorno da tool]
                                        │
                                        ▼
                                   [LLM emite TextDelta (streaming)]
                                        │
                                        ▼
                                   [Done (com citação de normas obrigatória)]
```

### Tipos Selados (`TurnEvent`):
- `TurnEvent.TextDelta(text: String)`: Streaming incremental de tokens.
- `TurnEvent.ToolCallEvent(call: ToolCall)`: Decisão do modelo de pesquisar termos assertivos.
- `TurnEvent.ToolResultEvent(chunks: List<Chunk>)`: Trechos encontrados no FTS5.
- `TurnEvent.Done(finalAnswer, chunksUsed, totalDurationMs, toolCallsMade)`: Conclusão do ciclo.
- `TurnEvent.Error(message, cause)`: Tratamento de exceções.

---

## 5. System Prompt (`system_prompt_pt.md`)

Localizado em `android/app/src/main/assets/system_prompt_pt.md`.
Instrui o modelo a:
1. Responder **apenas** com base nos trechos devolvidos pela ferramenta `retriever`.
2. Reformular a busca com palavras-chave técnicas assertivas para o BM25.
3. Citar obrigatoriamente Norma e Seção/Item (ex: `[NR-10, Item 10.4.1]`).
4. Responder claramente "Não encontrei informações suficientes..." caso a busca não retorne base.
5. Manter respostas concisas em 2 a 4 frases para leitura rápida em campo.

---

## 6. Índice FTS5 e Iteração sem Recompilação (`adb push`)

O `Fts5Retriever` lê o arquivo `index.db` com o seguinte ciclo de vida:
1. No primeiro boot, copia o `index.db` embutido em assets para `filesDir/index.db`.
2. **Override Externo**: se houver um arquivo em `getExternalFilesDir(null)/index.db`, ele tem prioridade imediata.

Para atualizar o corpus gerado pela trilha T1 sem reinstalar o aplicativo:
```bash
adb push index.db /sdcard/Android/data/br.org.ceia.cemigpoc.debug/files/index.db
```

---

## 7. Telemetria Local (`telemetry.jsonl`)

Gravada em `context.filesDir/telemetry.jsonl` em formato JSONL a cada interação:
```json
{
  "timestamp": "2026-09-10T01:50:25.123+0000",
  "question": "Quais são as etapas para desenergização de acordo com a NR-10?",
  "transcription": "Quais são as etapas para desenergização de acordo com a NR-10?",
  "tool_calls": [{"tool": "retriever", "query": "NR-10 desenergizacao etapas seccionamento aterramento"}],
  "chunks_used": [{"id": 1, "doc": "NR-10", "section": "Item 10.4.1 - Procedimentos de Desenergização", "score": -1.25, "preview": "São consideradas desenergizadas as instalações..."}],
  "response": "Conforme a [NR-10, Item 10.4.1], as etapas obrigatórias são...",
  "total_duration_ms": 142
}
```
Para inspecionar os logs de campo:
```bash
adb exec-out run-as br.org.ceia.cemigpoc.debug cat files/telemetry.jsonl
```
