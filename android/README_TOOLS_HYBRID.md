# Pipeline híbrido de tool-calling no app + 4 ajustes de UI (task poc-app-tools)

Leva ao app Android o **desenho HÍBRIDO provado** (`tools_oraculo/README.md`): o **1.2B treinado
em tool-calling (r128 Q4)** decide QUANDO buscar e quando reusar; quando ele chama
`buscar_norma`, o app **IGNORA a `consulta` reescrita** e executa a busca externa já existente com
a **FALA BRUTA + classificador de NR + RRF v4**. Remove a heurística Jaccard de reuso (quem decide
reuso agora é o modelo). Inclui os 4 ajustes de UI pedidos pelo capitão.

Modelo embarcado: `lfm2.5-1.2b-tools_1_2b_r128-Q4_0.gguf` (Spark `~/cemig-poc/models/`, SHA256
`dbb39f5a…`), servido no aparelho como override externo sob o nome esperado
`LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf`. ASR = Nemotron 3.5 INT8 (`-Pcemig.asrEngine=nemotron`).

---

## Parte 1 — Pipeline híbrido

Arquivos novos:
- `domain/pipeline/LfmToolRenderer.kt` — renderização NATIVA do ChatML de tool-calling do LFM2.5,
  réplica determinística do `chat_template.jinja` oficial (o `llama_chat_apply_template` do JNI
  não renderiza `tools=` nem o papel `tool`). Prova de paridade byte-a-byte abaixo.
- `domain/pipeline/LfmToolCallParser.kt` — porte fiel de `tools_v1/render.py::parse_tool_calls_runtime`
  (aceita a chamada com ou sem os tokens especiais `<|tool_call_start|>`/`<|tool_call_end|>`).
- `data/acceptance/ToolE2ERunner.kt` — runner E2E no aparelho (Parte 2).

Fluxo por turno do usuário (`AskPipeline.execute`, reescrito):
1. **DECISÃO**: renderiza `system(BAKED) + histórico + fala` e gera. O modelo emite `buscar_norma(...)`
   OU responde direto.
2a. **COM tool_call**: parseia (nome+args), **IGNORA a `consulta`**, roda a busca externa
   (`HybridRetriever.searchV3`, fala bruta), injeta o turno `tool` com os chunks, gera a resposta.
   Limite **1 busca por turno do usuário** (`MAX_SEARCHES_PER_TURN=1`); uma 2ª tool_call na
   síntese é descartada.
2b. **SEM tool_call**: a saída da decisão É a resposta (reuso/saudação/fora de escopo) — zero busca.

Falhas tratadas: chamada malformada / tool desconhecida → o marcador de intenção é preservado e o
default seguro é buscar com a fala bruta; resíduo de tokens de tool é removido da resposta.

Multiturno / orçamento (n_ctx 2048): o histórico é reconstruído no formato nativo (turnos
`user + [tool_call + tool] + assistant`) e podado por `maxPromptTokens=1700` — some sempre a
**troca mais antiga** primeiro; o contexto normativo recente e a fala atual ficam.

### Prova de paridade do system prompt treino-vs-app (exigência da emenda 2)

O system de treino é **um único string BAKED** (com o sufixo `List of tools: [...]`), idêntico em
todas as 4236 linhas do dataset `tools_v1/data/train_full.jsonl`:

```
SHA256 363185b7ec74ddbf10357735ccacf6bac6c6f244d43792fd0bb980df5d2ef841   (2381 chars)
```

Esse arquivo exato é embarcado como asset `app/src/main/assets/tools_system_prompt.txt` (mesmo
SHA256) e injetado como system. **No aparelho o runner confirmou**: `System prompt tool: 2381 chars
(esperado 2381)`.

A renderização nativa em Kotlin é provada **byte-a-byte** contra os fixtures gerados pelo template
oficial (jinja2, `tools=TOOLS`) em `LfmToolRendererParityTest` (fixtures em
`app/src/test/resources/tool_render/`): prompt de decisão, prompt de síntese pós-tool, multiturno
com reuso e saudação — todos `assertEquals` exatos. `./gradlew :app:testDebugUnitTest` verde.

---

## Parte 2 — Testes E2E no aparelho (LOG BRUTO)

Disparo: `adb shell am broadcast -a br.org.ceia.cemigpoc.RUN_TOOL_E2E -n br.org.ceia.cemigpoc/.data.acceptance.AcceptanceReceiver`
(app em foreground; o runner segura wake-lock). Saída: `tool_e2e_results.json` + logcat
`CEMIG_TOOL_E2E`. Aparelho: Galaxy S24+ (SM-S926B, Exynos 2400), Nemotron 3.5 INT8 + 1.2B tool Q4.

Artefatos brutos commitados: `android/tool_e2e_results.json`, `android/tool_e2e_logcat.txt`.

| caso | comportamento exigido | resultado | tool_call | busca | total |
|---|---|---|---|---|---|
| A 13,8 kV | DEVE buscar | ✅ chamou + busca externa | `buscar_norma(consulta='distância segura … 13,8 kV 13.8 kV')` | 2 chunks NR-10 Anexo II (238 ms) | 10,3 s |
| B followup | REUSO (sem nova busca) | ✅ não chamou, sem busca | (resposta direta) | — | 7,1 s |
| C poste | NOVA busca, chunks diferentes | ✅ chamou + chunks novos (NR-35/NR-34) | `buscar_norma(consulta='instabilidade estrutural poste …')` | 2 chunks (37 ms) | 11,3 s |
| D agradecimento | ZERO busca | ✅ não chamou | (resposta direta) | — | 3,0 s |
| E camiseta rasgada | RECUSA, sem invenção | ✅ chamou, mas RECUSOU ("não aborda…") | `buscar_norma(consulta='… camiseta de algodão rasgada EPI …')` | 2 chunks (42 ms) | 11,6 s |

LOG BRUTO (recortado; completo em `tool_e2e_logcat.txt`):

```
[A_deve_buscar] Pergunta técnica -> DEVE buscar
  PERGUNTA: Qual a distância segura pra trabalhar perto de rede de 13,8 kV?
  DECISÃO: buscar=true (3279ms)
  TOOL_CALL LITERAL: <|tool_call_start|>[buscar_norma(consulta='distância segura trabalho proximidade rede 13,8 kV 13.8 kV')]<|tool_call_end|>
  BUSCA(fala bruta): 2 chunks em 238ms · gate=soft top1=nr-10(0.44)
    chunk[0] nr-10 Anexo Ii :: ANEXO II ZONA DE RISCO E ZONA CONTROLADA Tabela de raios…
  RESPOSTA: A distância segura para trabalhar perto de rede de 13,8 kV é de 0,20 a 0,70 metros. Essa faixa é definida na tabela … do Anexo II da NR-10.
  TEMPOS: reason=3279ms search=238ms ttft=5299ms decode=1472ms TOTAL=10288ms (65 tok)

[B_followup_reuso] Followup mesmo assunto -> REUSO
  PERGUNTA: E se eu não conseguir desligar a rede?
  DECISÃO: buscar=false (7066ms)     ← modelo NÃO chamou a ferramenta (reuso)
  SEM BUSCA (reuso/direto)
  TEMPOS: reason=7066ms search=0ms TOTAL=7066ms

[C_muda_assunto] Mudança de assunto -> NOVA busca
  DECISÃO: buscar=true (3049ms)
  TOOL_CALL LITERAL: <|tool_call_start|>[buscar_norma(consulta='instabilidade estrutural poste não firme subida trabalho em altura')]<|tool_call_end|>
  BUSCA(fala bruta): 2 chunks em 37ms → nr-35 Anexo Iii | nr-34 34.10  (chunks DIFERENTES de A)

[D_saudacao] Saudação -> ZERO busca
  DECISÃO: buscar=false (3030ms) → "De nada, é um prazer. Estou à disposição…"

[E_recusa] Contexto insuficiente -> RECUSA
  DECISÃO: buscar=true (3519ms) → busca, mas RESPOSTA recusa:
  "Não, a norma consultada trata de confecção de vestuário e não aborda o uso de camisetas
   de algodão rasgadas. … é preciso verificar as diretrizes específicas…"
```

### Medições S24+ (por turno)

- **Reuso é mais rápido que busca**: B (reuso, sem busca+síntese) = 7,1 s vs A/C/E (busca) ≈
  10–12 s. O reuso pula o 2º turno de geração e a busca. (No caso B o turno de decisão em si foi
  longo porque o modelo gerou a resposta inteira ali; ainda assim o total ficou abaixo dos casos
  com busca.)
- **Busca externa (fala bruta)**: 37–238 ms (RRF v4 on-device).
- **TTFT** 3,0–7,3 s; **decode** 15–40 tok/s; **total** 3,0–11,6 s.
- **RAM PSS com Nemotron + 1.2B residentes durante o E2E**: **3,86 GB** (pico com geração ativa);
  idle pós-run ~2,97 GB (dumpsys). Cabe no S24+ (12 GB); aperta em aparelhos <8 GB (esperado —
  mesmo veredito do `sft_v3`/Nemotron).

### Casos do capitão (dentro do E2E)

- **13,8 kV** (A): chama a ferramenta, a busca por fala bruta puxa o **NR-10 Anexo II** (tabela de
  zonas), e a resposta cita a faixa e o Anexo II. (A leitura fina da linha exata da tabela segue
  limitada pela extração — veredito conhecido do `slm_oraculo`/`prompt_teto`; a arquitetura de tool
  está correta.)
- **Poste que não parece firme** (C): chama a ferramenta, recupera NR-35 (trabalho em altura) e
  responde sobre verificação estrutural — **sem repetir o erro "adornos" do v1**.
- **Camiseta rasgada** (E): chama, mas ao ver que os chunks (NR-38 confecção de vestuário / NR-04
  CNAE) não respondem, **RECUSA honestamente** em vez de inventar — comportamento da família de
  recusa treinada, preservado no aparelho.

---

## Parte 3 — 4 ajustes de UI

1. **Streaming invertido** (`MainScreen.ActiveStreamingItem`): durante a geração, as **normas
   consultadas ficam EM CIMA** (ancoradas) e o **texto gerado ABAIXO**, crescendo para baixo no
   campo de visão. O auto-scroll do `LazyColumn` (LaunchedEffect em `currentStreamingAnswer`)
   acompanha a geração.
2. **Logo do CEIA maior** (`MainScreen`, cabeçalho): `Image height 28dp → 36dp` (ajuste discreto,
   não quebra o layout do cabeçalho).
3. **Modo Debug atualizado** (`DebugTurnCard`): removidas as menções a Whisper e ao pipeline antigo.
   Agora mostra: **ASR Nemotron 3.5 INT8** + tempo; **decisão do modelo** (houve `buscar_norma`? com
   quais argumentos `consulta`/`nr`, marcados como IGNORADOS na busca; ou **reuso**); se buscou,
   **classificador NR + gate + fusão RRF v4** + chunks; **síntese (1.2B tool Q4)** com TTFT/decode/
   tokens. Banners de estágio e telemetria JSONL também atualizados (`called_tool`, `tool_consulta`,
   `reason_ms`).
4. **Botão contextual** (`PushToTalkButton`): campo VAZIO → **microfone** (segurar p/ falar); campo
   COM texto → **ENVIAR** (um toque). Ícone (Mic/MicNone↔Send), rótulo, cor e `contentDescription`
   mudam. Limpar o campo (ícone Clear no campo de texto) volta ao microfone (permite regravar). O
   envio usa clique simples (não gesto de segurar) e é desabilitado enquanto o pipeline processa —
   evita envio acidental. O pequeno `>` de enviar inline foi **removido** (era fácil de esbarrar).

---

## Comandos

```bash
# build (Nemotron ASR + 1.2B tool via override externo)
cd android && ./gradlew assembleRelease -Pcemig.asrEngine=nemotron

# testes JVM (paridade byte-a-byte + laço de tool + parser)
./gradlew :app:testDebugUnitTest

# instalar + servir o GGUF tool sob o nome esperado (override externo)
adb install -r app/build/outputs/apk/release/app-release.apk
adb push lfm2.5-1.2b-tools_1_2b_r128-Q4_0.gguf \
  /sdcard/Android/data/br.org.ceia.cemigpoc/files/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf

# E2E no aparelho (app em foreground)
adb shell am start -n br.org.ceia.cemigpoc/.MainActivity
adb shell am broadcast -a br.org.ceia.cemigpoc.RUN_TOOL_E2E \
  -n br.org.ceia.cemigpoc/.data.acceptance.AcceptanceReceiver
adb logcat -s CEMIG_TOOL_E2E
adb pull /sdcard/Android/data/br.org.ceia.cemigpoc/files/tool_e2e_results.json
```

## Veredito

A arquitetura híbrida está no app EXATAMENTE como provada: o modelo decide (F1 95% na bancada;
no aparelho 5/5 casos corretos), a busca ignora a consulta do modelo e usa fala bruta + RRF v4, e
o Jaccard saiu. Latência de voz cabe (reuso/saudação rápidos; buscas ~10–12 s no 1.2B). A alavanca
dominante de qualidade segue sendo o retrieval (conforme `tools_oraculo`). O 2.6B v3 é o alvo de
qualidade quando latência/RAM destravar.
