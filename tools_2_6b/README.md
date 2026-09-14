# tools_2_6b — Tool-calling + multiturno no LFM2.5-2.6B (a 2ª opção de embarque)

Repete no **2.6B** EXATAMENTE a bancada de tool-calling do 1.2B (`tools_v1/` + `tools_oraculo/`)
para o capitão ter **DUAS opções comparáveis de embarque**: o **1.2B tool** (único viável em
latência de voz) e o **2.6B tool** (mais capaz, mas caro no aparelho). O valor da task é a
**comparabilidade maçã-com-maçã**: MESMOS 4.236 diálogos (mesmas 5 famílias, mesmo split), MESMO
motor de treino (`sft_v2/train_v2`), MESMA régua honesta (`bench/regua`), MESMAS métricas.

> Ambientes: geração/score/eval no `classifier/.venv` (jinja2, requests, sklearn); treino + export
> GGUF na DGX Spark (`walcyrios@spark-b431`, `~/jupyterlab/.venv`). Juiz vLLM 27B
> `10.100.0.111:8005` (roteável deste host). Holdout 151/20 **INTOCÁVEL** (só medição). 3 runs
> para média±desvio em tudo que embasa recomendação. Tabelas com **bf16 E Q4 lado a lado**
> (preferência do capitão). Higiene: paralelo, checkpoint, não-interativo, procedência.
> Comentários PT-BR; código em inglês.

---

## PASSO 0 — verificação prévia (exigência permanente), 100% confirmada

**A pergunta central do PASSO 0: o template do 2.6B é o mesmo do 1.2B?** SIM, no que importa.
`verify_2_6b.py` (local) + `verify_modules_2_6b.py` (Spark) provam:

1. **Corpo dos turnos byte-a-byte idêntico** ao 1.2B — system, `List of tools: [...]`, turno
   `tool`, e a tool_call `<|tool_call_start|>[buscar_norma(consulta='...', nr='NR-10')]<|tool_call_end|>`.
   A **ÚNICA** diferença é o **generation prompt**: o 2.6B é modelo de **raciocínio** e o template
   termina em `<|im_start|>assistant\n<think>` (o 1.2B termina em `<|im_start|>assistant\n`).
   Com **thinking-OFF** (removendo o sufixo `<think>`, réplica do `enable_thinking=False` do
   `sft_v3`) o generation prompt também fica **idêntico ao 1.2B** → a bateria roda maçã-com-maçã.
   `render_jinja_2_6b.py` renderiza o template do 2.6B em jinja2 puro; `patch_2_6b.py` injeta esse
   renderer (thinking-OFF) nos harness do `tools_v1`/`tools_oraculo` sem tocá-los (`data/passo0_render.txt`).
2. **Round-trip 100%** da tool_call (consulta/nr voltam idênticos, incl. `13,8 kV`).
3. **Módulos confirmados** no `model.safetensors.index.json`: `self_attn.{q,k,v,out}_proj` (8),
   `feed_forward.{w1,w2,w3}` (30), `conv.{in_proj,conv,out_proj}` (22). LoRA por REGEX do
   `sft_v2/train_v2` → **122 alvos (32 attn + 90 MLP, 0 conv), 2.90% trainable**; os 3 asserts
   (mlp✓ attn✓ conv✗) passam inline no treino (`data/passo0_modules.json`, `data/premises_r*.json`).
4. **Tokenizer do 2.6B ≠ 1.2B** (vocab 124.893 vs 64.400, tokeniza ~18% menos). O truncamento do
   brief foi **RE-MEDIDO** com o tokenizer do 2.6B: **5.36% a 2048 / 0% a 3072** (máx 2421 tok) →
   **max_length 3072** (= tools_v1, comparabilidade; `data/truncation.json`).

---

## PASSO 1 — treino (1 época, curva de saturação + varredura de rank)

Base `LiquidAI/LFM2.5-2.6B` (bf16, público não-QAD). Motor `sft_v2/train_v2` via
`train_2_6b_tools.py` (o train_v2 já grava GGUF `lfm2.5-2.6b-*` — sem monkeypatch). Receita
idêntica ao 1.2B: bs8×ga4, lr 1e-4, cosine, bf16, grad ckpt, alpha=2r, `assistant_only_loss=True`,
val interno, **max_length 3072**. wandb online `cemig-tools-2.6b` (run `42il0vxi`).

**FASE A — curva de saturação** (rank fixo r32, degraus aninhados 750/1500/3000/4236).
**FASE B — varredura de rank** (r32/r64/r128 no degrau completo 4236).

| checkpoint | eval_loss interno |
|---|--:|
| step750 r32  | 0.7652 |
| step1500 r32 | 0.7419 |
| step3000 r32 | 0.6993 |
| r32 (4236)   | 0.6670 |
| r64 (4236)   | 0.6484 |
| r128 (4236)  | 0.6238 |

eval_loss cai monotônica com dados e com rank (não satura na loss — igual ao 1.2B), mas a
**régua não acompanha** (ver PASSO 4).

---

## PASSO 2 — a bateria (as 7 medições, idênticas às do 1.2B)

Harness reusa `tools_v1`/`tools_oraculo` via os wrappers `run_eval_2_6b.py`, `score_e2e_2_6b.py`,
`harness_oraculo_2_6b.py` (patch do renderer 2.6B thinking-OFF). Régua = `sft_v2/score_v2.py`
(cobertura de fatos + alucinação + tautologia; citação FORA do gate); recusa = `sft_v3/refusal_metrics.py`.

### a) ORÁCULO (chunk-ouro no turno `tool`) — a pergunta central do capitão

| candidato (oráculo, 151) | aprovação % | alucinação % | cobertura |
|---|--:|--:|--:|
| 1.2B QAD base (sem tool) — Q4 | 2.7 | 2.0 | — |
| 1.2B sft sem tool r64 — Q4 / bf16 | 22.3 / 24.3 | 23.0 / 18.9 | — |
| 1.2B **tool** r128 — Q4 / bf16 | 31.4 / 34.9 | 28.2 / 24.3 | — |
| 2.6B sft_v3 sem tool r64 — Q4 / bf16 | 46.4 / 47.0 | 21.2 / 21.9 | — |
| **2.6B tool r64 — Q4** | **46.1 ± 1.1** | 21.9 ± 1.4 | 0.550 |
| **2.6B tool r64 — bf16** | **53.8 ± 1.4** | 21.4 ± 2.3 | 0.602 |
| Qwen 27B (teto) | 84.3 | 2.6 | 0.753 |

**Resposta com número:** o treino de tool **AJUDA (ou pelo menos não atrapalha) a síntese do 2.6B**
— em bf16 o 2.6B tool (53.8%) **SUPERA** o 2.6B sft_v3 sem tool (47.0%) em +6.8 p.p.; em Q4
empata (46.1 vs 46.4). A arquitetura de tool NÃO é a causa da alucinação; a alucinação do oráculo
(~21%) é a mesma do sft_v3 e do 1.2B tool — é **capacidade do LM** (o 27B fica em 2.6%). O salto
sobre o **1.2B tool** é grande: **+19 p.p. (Q4) / +19 p.p. (bf16)**.

### b) DECISÃO / SINTAXE / REUSO / NOVO-TÓPICO (r64)

| métrica | 2.6B tool r64 | 1.2B tool r128 (ref) |
|---|--:|--:|
| decisão F1 | **94.4** (TP93 FN0 FP11 TN49; recall 100%) | 94.9 |
| chamou-à-toa % | 18.3 | 15-17 |
| sintaxe válida % | **100.0** | 99.3 |
| reuso-correto % | 72.9–84.3 | 84.8 |
| novo-tópico % | **100.0** | 100.0 |

Comportamento multiturno **funciona e é aprendido** — igual ao 1.2B. Diferença dentro do ruído.

### c) QUALIDADE DO ARGUMENTO (recall@2 no v4) — o achado que o capitão pediu isolado

**A reescrita do 2.6B NÃO supera a fala crua — igual ao 1.2B.** delta model−crua = **−7.3 p.p.
R@2** (r64: model 23.2 vs crua 30.5; 27B rewrite 29.8). O capitão perguntou se a maior capacidade
do 2.6B faria a consulta reescrita **funcionar** (o que mudaria a arquitetura para busca). **NÃO
muda:** em NENHUM rank ou volume o rewrite do 2.6B supera a fala crua. O ponto **mais próximo de
paridade** apareceu em step3000 r32 (model R@2 **29.1** vs crua 30.5, delta −1.4, e R@1 20.5 > 19.9)
— ou seja, o 2.6B chega **quase** à fala crua com o volume certo, mas **nunca a supera** e a curva
oscila (r128 volta a −8.0). **Veredito: manter a busca por fala crua** (o mesmo do 1.2B); a
reescrita do modelo pequeno segue sem valor para o retrieval.

### d) RECUSA (pack sft_v2, 85; classe positiva = "deveria recusar")

| candidato | recall(correta) % | recusa-indevida(FP) % | precisão % | F1 % |
|---|--:|--:|--:|--:|
| **2.6B tool r64 — bf16** | 91.1 | **0.0** | 100.0 | **95.3** |
| **2.6B tool r64 — Q4** | 84.4 | **0.0** | 100.0 | 91.6 |
| 2.6B sft_v3 sem tool r64 — Q4 | 100.0 | 13.3 | 91.8 | 95.7 |
| 1.2B tool r128 — bf16 / Q4 | 84.4 / 77.8 | 3.3 / 6.7 | 97.4 / 94.6 | 90.5 / 85.4 |
| Qwen 27B | 95.6 | 3.3 | 97.7 | 96.6 |

**A recusa foi PRESERVADA** — bf16 F1 95.3 (≈ sft_v3 95.7) e **recusa-indevida 0%** (melhor que o
sft_v3, que over-recusa 13.3%). A família (5) transferiu bem. O Q4 perde um pouco (F1 91.6),
sensível ao 4-bit como sempre.

### e) HÍBRIDO ponta a ponta (modelo decide + busca fala crua + classificador + RRF v4)

| config | quant | aprovação % | alucinação % | chamou % | got_gold(chamou) % |
|---|---|--:|--:|--:|--:|
| **(A) híbrido** | bf16 | **28.9 ± 1.4** | 35.1 ± 1.4 | ~100 | ~51 |
| **(A) híbrido** | Q4 | **24.0 ± 1.4** | 38.4 ± 3.5 | ~99 | ~51 |
| (B) tools puro | bf16 | 14.8 ± 1.7 | 45.0 ± 1.1 | ~100 | ~19-25 |
| (B) tools puro | Q4 | 14.8 ± 1.6 | 42.6 ± 3.3 | ~99 | ~19-25 |
| 1.2B tool híbrido (ref) | bf16/Q4 | 19.2 / 19.2 | 31.8 / 34.2 | — | 51 |

O **híbrido (A) bate o tools puro (B)** por ~10-14 p.p. (a fala crua+v4 entrega o ouro em ~51% vs
~19-25% da consulta reescrita), e **supera o 1.2B híbrido** (28.9 vs 19.2 em bf16; 24.0 vs 19.2 em
Q4). A regra de embarque continua: **o modelo decide chamar/reusar; a busca usa SEMPRE fala crua +
classificador + RRF v4, NUNCA a consulta reescrita.**

### f) DECOMPOSIÇÃO da alucinação por condição

- **Tools puro (B):** 75-81% da alucinação vem de **contexto errado** (a busca reescrita erra) —
  o retrieval é o teto, confirmado.
- **Híbrido (A):** com a busca corrigida, a alucinação se redistribui (~50% got_gold ✓ / ~50%
  contexto errado); o resíduo com chunk-ouro presente (~35-42% hall_rate) é a **capacidade de
  sintetizar sem inventar** — o mesmo teto do oráculo (~21%). Menor que no 1.2B pela maior capacidade.

### g) CAPACIDADE GERAL / COMPORTAMENTAL (catastrophic forgetting) — o único ponto negativo

| camada | métrica | base 2.6B | 2.6B tool r64 Q4 | Δ | sft_v3 (sem tool) Δ |
|---|---|--:|--:|--:|--:|
| general | qualidade | 3.93 | 3.79 | **−0.14** | −0.08 |
| general | PT-BR | 4.80 | 4.52 | −0.28 | +0.02 |
| behavioral | qualidade | 3.80 | 3.37 | **−0.43** | +0.37 |
| behavioral | PT-BR | 4.80 | 4.07 | −0.73 | −0.07 |

**HÁ forgetting no 2.6B tool — ao contrário do sft_v3 (que não teve).** A regressão geral é leve
(−0.14) mas a **comportamental é relevante** (−0.43 qualidade, −0.73 PT-BR). O treino de
tool-calling (decisão/multiturno/recusa numa distribuição estreita) custa capacidade geral que o
SFT de síntese pura do sft_v3 não custava. É o preço da capacidade nova (decidir/chamar/reusar).

### Varredura de rank (Q4, régua) — ranks ~indiferentes na barra, com trade-off

| rank | oráculo aprov % | oráculo aluc % | híbrido aprov % | recusa F1 % | recusa-indevida % |
|---|--:|--:|--:|--:|--:|
| r32  | **49.6 ± 1.9** | 20.3 | 22.5 | 88.9 | 0.0 |
| r64  | 46.1 ± 1.1 | 21.9 | **24.0** | 91.6 | 0.0 |
| r128 | 41.9 ± 3.1 | 21.9 | 22.7 | **94.3** | 3.3 |

Oráculo cai levemente com o rank; recusa F1 sobe com o rank. As diferenças estão perto da barra de
erro. **r64 é o melhor equilíbrio** (oráculo alto, recusa F1 91.6, indevida 0%, híbrido melhor) e
casa com a escolha do `sft_v3` (comparabilidade). r32 é a opção barata equivalente; r128 favorece
recusa ao custo do oráculo.

---

## PASSO 3 — custo no aparelho (o que decide) — S24+ (SM-S926B, Exynos 2400)

`llama-bench` 6 threads (cluster de performance), pp1500/tg100, build `434ddbb`. O aparelho
**throttla termicamente** sob bench sustentado (frio no topo da faixa):

| modelo | tamanho | pp1500 (t/s) | tg100 (t/s) | TTFT ~1450 tok | decode ~150 tok | E2E est. | RAM (PSS) |
|---|--:|--:|--:|--:|--:|--:|--:|
| **2.6B tool r64 Q4** | 1.48 GiB | 84–131 | 12–22 | 11–17 s | 7–12 s | **~18–29 s** | ~4.3–4.75 GB* |
| 1.2B tool (embarcado) | 661 MiB | 285 | 45 | ~5 s | ~3 s | **~6–8 s** | 3.86 GB* |

\* RAM já medida na POC (mesma arch/quant): 2.6B PSS 4.32 GB (engine-upgrade) / ~4.75 GB com
Nemotron (sft_v3); 1.2B tool + Nemotron 3.86 GB (`android/tool_e2e_results.json`). O 2.6B tool tem
**arch/tamanho/quant idênticos ao 2.6B base** → latência/RAM idênticas (diferenças = ruído térmico).

**Veredito de aparelho:** o 2.6B tool **NÃO cabe no orçamento de voz (10 s) HOJE** — TTFT+decode
~18–29 s e RAM ~4.3–4.75 GB (inviável S21, aperta S24+). O **1.2B tool (~6–8 s, 3.86 GB) segue o
único viável para voz**. A troca é decisão de custo do capitão.

**Flag de build (default INALTERADO — as DUAS opções):** `-Pcemig.toolModel=2.6b` seleciona o 2.6B
tool r64 Q4 (+ thinking-OFF via `llama_jni`, `SUPPRESS_REASONING=true`); omitido = **1.2B tool
(default)**. O GGUF do 2.6B tool entra por override externo sob o nome `LFM2.5-2.6B-tools-Q4_0.gguf`.

---

## PASSO 4 — TABELA MESTRE e recomendação

Régua honesta `bench/regua` (limiar 0.5, citação FORA do gate) em TODOS. Oráculo = chunk-ouro de
bandeja; retrieval real = pipeline v4 (híbrido no tool; busca fixa nos demais). bf16 E Q4 lado a lado.

| modelo / config | oráculo aprov % | oráculo aluc % | retrieval-real aprov % | recusa F1 % | E2E S24+ | RAM |
|---|--:|--:|--:|--:|--:|--:|
| 1.2B base QAD — Q4 | 2.7 | 2.0 | 1.4 | 73.8 | ~6 s | 0.66 GB |
| 1.2B sft sem tool r64 — Q4 / bf16 | 22.3 / 24.3 | 23.0 / 18.9 | 10.1 / 16.2 | 90.2 / 90.7 | ~6 s | 0.66 GB |
| **1.2B tool r128 — Q4 / bf16** | 31.4 / 34.9 | 28.2 / 24.3 | 19.2 / 19.2 (híbr.) | 85.4 / 90.5 | **~6–8 s** | **3.86 GB** |
| 2.6B sft_v3 sem tool r64 — Q4 / bf16 | 46.4 / 47.0 | 21.2 / 21.9 | 20.0 / 26.7 | 95.7 / 94.5 | ~18–23 s | ~4.3 GB |
| **2.6B tool r64 — Q4 / bf16** | **46.1 / 53.8** | 21.9 / 21.4 | **24.0 / 28.9 (híbr.)** | 91.6 / 95.3 | **~18–29 s** | **~4.3–4.75 GB** |
| 27B (teto) | 84.3 | 2.6 | 45.7 | 96.6 | (nuvem) | — |

### O salto de qualidade do 2.6B justifica o custo? — número honesto

**Qualidade: SIM, o salto é real e grande.** Oráculo 46–54% vs 31–35% do 1.2B tool (**+15–19 p.p.**);
híbrido 24–29% vs 19% (**+5–10 p.p.**); recusa F1 até 95.3 (≈ 27B) com recusa-indevida 0%; respostas
literais visivelmente mais completas e corretas (ver amostras). O 2.6B tool iguala/supera o 2.6B
sft_v3 sem tool E ganha o multiturno/decisão.

**Custo: proibitivo para VOZ hoje.** ~18–29 s por resposta (TTFT 11–17 s) e ~4.3–4.75 GB de RAM no
S24+ — **acima do teto de 10 s** e **inviável no S21**. Some-se o **forgetting** comportamental
(−0.43), que o sft_v3 não tinha.

**RECOMENDAÇÃO:** **embarcar o 1.2B tool (default, ~6–8 s)** como modelo de campo — é o único que
cabe na voz. **Deixar o 2.6B tool r64 Q4 selecionável por build** (`-Pcemig.toolModel=2.6b`) como a
2ª opção do capitão: é a melhor qualidade tool-calling que temos on-device, alvo de embarque quando
**latência/RAM destravarem** (S24+/≥8 GB dedicado, ou aceleração/decode mais rápido) — ou destilar o
2.6B tool → 1.2B. A busca continua por **fala crua + v4** nas duas opções (a reescrita não paga em
nenhum dos dois modelos). Se subir para o 2.6B, prefira **bf16** onde houver folga (oráculo 53.8 vs
46.1 do Q4; recusa F1 95.3 vs 91.6) — o Q4 custa ~8 p.p. no oráculo neste modelo.

### Amostras literais dos casos do capitão (2.6B vs 1.2B, `data/samples_captain.txt`)

- **13,8 kV sozinho (qa-013):** o **2.6B** (oráculo/híbrido) chama, recupera NR-10 (10.5/10.6/10.7)
  e dá a ação certa e detalhada ("Não, não pode subir sozinho… NR-10 item 10.6.3 suspender na
  iminência de perigo; 10.7.4 exige OS assinada por superior"). O 1.2B acerta o rumo mas é mais raso.
- **camiseta rasgada (qa-003):** o **2.6B NÃO comete o erro de "adornos"** (responde corretamente
  sobre EPI/vestimenta obrigatória) — o **1.2B r128 ainda cita "vedado adornos"** no oráculo. O 2.6B
  é mais robusto ao distrator.
- **poste podre (qa-025):** ambos recusam/redirecionam; o 2.6B é mais completo na justificativa.
- **13,8 kV distância (qa-029):** nenhum 2.6B lê a faixa exata da tabela do Anexo II (limite de
  leitura fina de tabela, o mesmo do `slm_oraculo`) — não é resolvido por tool nem por tamanho.

---

## Curva de saturação — o 2.6B satura no mesmo ponto do 1.2B?

| exemplos (r32) | dec F1 | model R@2 (arg) | reuso % | novo-tópico % | sintaxe % |
|---|--:|--:|--:|--:|--:|
| 750  | 94.4 | 23.8 | 75.7 | 96.0 | 96.2 |
| 1500 | 96.4 | 23.2 | 84.3 | 92.0 | 94.0 |
| 3000 | 94.9 | 29.1 | 64.3 | 100.0 | 92.2 |
| 4236 | 95.4 | 25.2 | 58.6 | 100.0 | 95.1 |

- **Decisão: satura AINDA MAIS CEDO no 2.6B** — F1 já 94.4 em **750** (o 1.2B só chegou a 94.8 em
  1500; em 750 era 87.6). Mais capacidade aprende a decidir com menos dados → **~750 exemplos bastam**
  para o comportamento no 2.6B (treino futuro mais barato ainda que no 1.2B).
- **Sintaxe: satura instantâneo** (96% em 750). Confirmado.
- **Argumento: NÃO satura porque NÃO supera a fala crua** — oscila 23–29 (melhor 29.1 em step3000,
  quase paridade, mas nunca acima de 30.5 da fala crua). A maior capacidade do 2.6B aproxima a
  reescrita da fala crua, mas **não destrava o retrieval** — a alavanca segue sendo a busca, não o modelo.
- **Reuso/novo-tópico:** melhora até ~1500 e depois flutua (igual ao 1.2B).

---

## Reprodução

```bash
PY=../classifier/.venv/bin/python
export REGUA_JUDGE_URL=http://10.100.0.111:8005/v1
# PASSO 0 (render local + módulos/trunc na Spark)
make verify ; make passo0
# PASSO 1 (treino detached na Spark; NÃO deixar llama-server residente — memória unificada)
make train
# PASSO 2 (bateria de um candidato; sobe llama-server na Spark bind 0.0.0.0)
make evals GGUF='~/cemig-poc/models/lfm2.5-2.6b-tools_2_6b_r64-Q4_0.gguf' LABEL=tools_2_6b_r64 QUANT=q4
make score LABEL=tools_2_6b_r64 QUANT=q4
# curva/rank/painel/device: run_all_curve.sh, run_rank_regua.sh, run_panel_2_6b.sh, llama-bench (device)
make consolidate LABEL=tools_2_6b_r64
```

### Artefatos
- **Pesos na Spark** (`~/cemig-poc/models/`, gitignored):
  `lfm2.5-2.6b-tools_2_6b_r{32,64,128}-{bf16,Q4_0}.gguf` + `tools_2_6b_step{750,1500,3000}_r32-*`.
  wandb `cemig-tools-2.6b` (run `42il0vxi`). Dataset: `~/cemig-poc/tools_v1/data/train_*.jsonl`
  (os MESMOS do tools_v1). Candidato de embarque (2ª opção): `lfm2.5-2.6b-tools_2_6b_r64-Q4_0.gguf`.
- **Leves commitados** (`data/`): `passo0_render.txt`, `passo0_modules.json`, `truncation.json`,
  `premises_r*.json`, `metrics_r*.json`, `eval_*.json`, `e2e_*.json`, `refusal_confusion_*.json`,
  `forgetting_panel.json`, `device_bench_s24.json`, `consolidation.json`, `samples_captain.txt`,
  `chat_template_2_6b.jinja`. Gerações/scores (`gen_*`/`score_*`) gitignored (reprodutíveis).

### Arquivos
```
render_jinja_2_6b.py    render do template do 2.6B em jinja2 puro (thinking-OFF por default)
patch_2_6b.py           injeta o renderer 2.6B nos harness do tools_v1/tools_oraculo (sem tocá-los)
verify_2_6b.py          PASSO 0 (local): byte-identity + round-trip + 1 exemplo/família
verify_modules_2_6b.py  PASSO 0 (Spark): confirma módulos + 122 alvos LoRA + 3 asserts
measure_truncation_2_6b.py  trunc com o tokenizer do 2.6B (2048 vs 3072)
train_2_6b_tools.py / run_train_tools_2_6b.sh  treino (motor sft_v2/train_v2; curva + varredura)
run_eval_2_6b.py score_e2e_2_6b.py harness_oraculo_2_6b.py  wrappers (patch + delega ao tools_v1/oraculo)
run_all_evals_2_6b.sh score_gens_2_6b.sh  bateria completa de um candidato + score pela régua
run_curve_eval.sh run_all_curve.sh run_rank_regua.sh score_rank.sh  curva de saturação + varredura de rank
run_panel_2_6b.sh start_srv_net_jinja.sh  painel general/behavioral (forgetting) c/ jinja+reasoning-budget 0
consolidate_2_6b.py     tabela mestre + 7 medições + curva; extract_samples_2_6b.py: casos do capitão
```
Reuso (não alterados): `tools_v1/*`, `tools_oraculo/harness_oraculo.py`, `sft_v2/{train_v2,score_v2}.py`,
`sft_v3/refusal_metrics.py`, `finetune2/eval_panel.py`, `bench/regua`, `bench/prompt_teto/data/chunks_v4.json`.
```
