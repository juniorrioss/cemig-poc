# M3 — Correção End-to-End no S24+ (Fase 3)

## Causa-raiz do sintoma "Não sei com base nas normas consultadas"

O teste real do app no Galaxy S24+ revelava que quase toda pergunta retornava
"Não sei". A investigação end-to-end (log do aparelho) encontrou a causa única:

```
E Fts5Retriever: android.database.sqlite.SQLiteException: no such module: fts5
    ... while compiling: SELECT ... bm25(chunks_fts, 1.5, 3.0, 2.0, 1.0) ...
```

**O SQLite embutido no Android NÃO inclui o módulo FTS5.** A cada consulta, o
`Fts5Retriever` caía no `fallbackSearch` (um `LIKE '%primeiro_termo%'` ingênuo), que
retornava chunks irrelevantes (tipicamente `nr-01`), e o Turno 2 corretamente declarava
"Não sei". **O BM25 nunca executou no aparelho** — o gargalo não era chunking nem
rewriter (Fases 1-2), era a ausência do FTS5.

## Correção

Empacotamento do SQLite com FTS5 via `mil.nga:sqlite-android:3500400` (fork mantido do
`requery/sqlite-android`, SQLite 3.50.4, `libsqliteX.so` para arm64-v8a):

- `android/app/build.gradle.kts`: adiciona a dependência.
- `data/retriever/Fts5Retriever.kt`: importa `org.sqlite.database.sqlite.SQLiteDatabase`,
  carrega a nativa (`System.loadLibrary("sqliteX")`) uma vez e valida a versão/FTS5 na
  abertura. O restante da lógica (pesos BM25 1.5/3/2/1, sanitização, fallback) é preservado.

Log de confirmação no aparelho:
```
I Fts5Retriever: Biblioteca nativa sqliteX (FTS5) carregada com sucesso
I Fts5Retriever: SQLite empacotado carregado (versao 3.50.4) com suporte a FTS5
```

## Resultado do roteiro de aceitação (release, modo avião, S24+)

Roteiro Q01-Q10 (10 perguntas faladas, 2 multiturno), APK release reinstalado, 100%
offline. Critério: **≥ 7/10 respostas fundamentadas com citação correta** (vs ~1-2/10
antes). Resultado bruto em `android/acceptance_results_m3.json`.

| ID | NR-ouro | NR correta no top-2 | Fundamentada | Total (ms) |
|---|---|:---:|:---:|---:|
| Q01 | nr-10 | ✔ (10.5) | SIM | 10.672 |
| Q02 | nr-10 | ✔ (10.2.8) | SIM | 12.116 |
| Q03 | nr-10 | ✔ (Anexo II) | SIM | 10.307 |
| Q04 | nr-35 | ✔ (Anexo II) | SIM | 12.934 |
| Q05 | nr-35 | ✘ (nr-15) | não | 13.572 |
| Q06 | nr-10 | ✔ (10.7/Gloss.) | SIM | 10.756 |
| Q07 | nr-06 | ✔ (Anexo I) | SIM | 13.859 |
| Q08 | nr-10 | ✔ (10.3) | SIM | 11.355 |
| Q09 | nr-10 | ✔ (10.2.8) | SIM | 15.020 |
| Q10 | nr-10 | ✘ (nr-15) | não | 14.566 |

**8/10 respostas fundamentadas com a NR correta — critério ATINGIDO.**

- **Antes do fix (mesma release, FTS5 ausente)**: BM25 retornava `nr-01`/lixo; ~2/10.
- Q05 é continuação multiturno ("fator de queda do talabarte") que perde o contexto da
  NR-35; Q10 ("chuva forte") é semanticamente difícil e cai em nr-15. Ambas conhecidas.

## Latência (trade-off registrado para o capitão)

Latência total 10,3–15,0 s (mediana 12,5 s) — **acima do teto de 10 s de voz**. Composição
típica por pergunta: ASR ~2,5 s + rewrite ~1,5 s + TTFT/prefill ~6,5 s + decode ~4,5 s.

- O prefill domina: 2 chunks grossos (~500 tokens cada) + system prompt ≈ 1.420 tokens.
- Alavanca conhecida (AGENTS.md `bench/`): a variante **topk2 lean (~820 tokens)** reduz o
  prefill e a resposta para a faixa 5–10 s no S24+, sem perda de acerto factual. Não foi
  aplicada neste PR para não arriscar a regressão do número de acerto agora validado; fica
  como próximo passo de tuning (reduzir tamanho de chunk no contexto e/ou cap de decode).
- O checkpoint QAD (`LFM2.5-1.2B-Instruct-QAD-Q4_0`, já embarcado) tem TTFT/RAM melhores
  que o Q4_K_M; a folga restante vem do orçamento de contexto, não do porte.

## Como reproduzir o teste no aparelho
```bash
export JAVA_HOME=$HOME/android-sdk/jdk-17 ANDROID_HOME=$HOME/android-sdk
cd android && ./gradlew :app:assembleRelease -x lint
adb -s <device> install -r app/build/outputs/apk/release/app-release.apk
adb -s <device> shell am broadcast -a br.org.ceia.cemigpoc.RUN_ACCEPTANCE \
  -n br.org.ceia.cemigpoc/.data.acceptance.AcceptanceReceiver
adb -s <device> shell cat /sdcard/Android/data/br.org.ceia.cemigpoc/files/acceptance_results.json
```

## Escopo NÃO alterado (decisão firstmate — opção A)
Conforme Fases 1-2 (`corpus/README_M3.md`, `finetune/README_M3.md`): o índice **grosso**
`index_hf_36nr.db` foi mantido (chunking fino piora o recall), o rewriter **zero-shot** foi
mantido (LoRA r=8/r=16 regrediram no holdout real; 2.6B inviável por ser modelo de
raciocínio). O ganho de M3 veio da correção do FTS5, não de mexer nos pesos.
