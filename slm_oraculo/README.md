# A prova decisiva do SLM — teto em oráculo + destilação do 27B (`slm_oraculo/`)

Responde à ordem do capitão: _"quero tirar a prova o quanto conseguimos melhorar o SLM
com o contexto certo. O ponto de maior risco: se mesmo com o oráculo o SLM falhar, é
simplesmente proibitivo a POC."_ A busca fica de lado (decisão dele): **tudo em ORÁCULO**
(chunk-ouro entregue de bandeja), medindo o **gap 27B − 2.6B** com a **régua honesta**
(`bench/regua`, limiar 0,5, citação fora do gate) sobre as **151 reais** e o **corpus
CORRIGIDO** (reparo cirúrgico do Anexo II da NR-10 via DocAI).

> **Infra**: o juiz vLLM 27B (`10.100.0.111:8005`) é o **gerador do teto** E o **juiz da
> régua** — roda tudo neste host (o juiz é roteável daqui). O 2.6B (bf16 e Q4_0) e o
> treino rodam na **DGX Spark** (`walcyrios@spark-b431`, GB10 121 GB) via `llama-server`
> tunelado. Nada pesado na 5070. Paralelo (≥16 workers no juiz, 4 no llama-server),
> checkpoint incremental, não-interativo. Comentários PT-BR, código em inglês.

---

## FASE 1 — TETO REFEITO com o corpus corrigido (oráculo, prompt `numeros`, n=151)

| gerador · contexto | aprov% | cobertura | **alucinação%** | conv chunk-ouro% | tok |
|---|--:|--:|--:|--:|--:|
| **27B · oráculo** (TETO) | **82.1** | 0.761 | **2.6** | 82.1 | 107 |
| 2.6B **bf16** · oráculo | 49.0 | 0.590 | 25.2 | 49.0 | 259 |
| 2.6B **Q4_0** · oráculo (embarque) | 45.0 | 0.562 | 25.8 | 45.0 | 225 |

**Comparação com os números antigos** (brief: 27B 84,1 / 2.6B 46,4):
- **O teto praticamente NÃO mudou com o corpus corrigido: 84,1 → 82,1%** (−2,0 p.p.,
  dentro do ruído de amostragem/juiz). **Razão honesta**, verificada em `corpus_fix.py`:
  **nenhum dos 151 chunks-ouro do holdout resolve para a tabela corrompida do Anexo II**
  (chunks 274/275/276). As perguntas de zona/distância da NR-10 (qa-013/029/030) resolvem
  para seções **definitórias** (10.1/10.2/10.6/Glossário), que já estavam legíveis. O
  reparo do Anexo II foi **injetado** em 3 dessas perguntas (a tabela limpa 13,8 kV →
  0,38/1,38 m passa a estar disponível), mas elas cobram a **definição** das zonas, não o
  raio numérico — por isso o teto quase não se move nas 151.
- O 2.6B base bate o número antigo: **46,4 → 45,0% (Q4_0) / 49,0% (bf16)**.

**Custo da quantização no teto**: **bf16 49,0% → Q4_0 45,0% = −4,0 p.p.** O 4-bit custa
pouco (≈ metade de um fato médio); **o gargalo é o TREINO/CAPACIDADE, não a quantização**.

**O GAP DA POC (a pergunta que decide)**: **27B 82,1% − 2.6B bf16 49,0% = 33,1 p.p.** de
capacidade pura, com o MESMO contexto perfeito. Mesmo de bandeja, o 2.6B **converte só ~49%
do que o 27B converte a 82%**.

### Alucinação (critério de rejeição, ordem do capitão — não é nota de rodapé)
- **27B em oráculo alucina 2,6%** — quando tem o trecho certo, o teto quase não inventa.
- **2.6B em oráculo alucina ~25%** (bf16 25,2%, Q4 25,8%): **1 em cada 4 respostas do 2.6B,
  mesmo com o chunk certo, contradiz a norma.** Em segurança do trabalho isso é
  inaceitável e é o segundo eixo do gap (o primeiro é cobertura).

### Caso do capitão — 13,8 kV (oráculo com o Anexo II limpo do corpus corrigido)
Com a tabela reconstruída (DocAI) na frente do modelo:
- **27B**: _"Para 13,8 kV, mantenha distância mínima de 0,38 metros da zona de risco e
  1,38 metros da zona controlada…"_ — **lê a linha certa (10–15 kV → 0,38/1,38)**.
- **2.6B (bf16 e Q4)**: leem a **linha errada** (_"0,25 metros"_, faixa 3–6 kV) **mesmo com
  a tabela limpa e completa no contexto**. Confirma o gap de capacidade: o dado corrigido é
  **necessário mas não suficiente** — o 2.6B não localiza a faixa correta na tabela.

**Leitura da FASE 1**: a régua **não está severa** (o 27B faz 82% nela, alucina 2,6%). O
corpus corrigido destrava o dado do 13,8 kV, mas **o 2.6B não o lê certo**. O gap de 33 p.p.
em oráculo é **capacidade do modelo**, exatamente o risco que o capitão quer medir.

---

## FASE 2 — Destilação SFT do estilo acionável do 27B (Spark)

**Alvo** (ordem do capitão): ensinar o 2.6B o **estilo ACIONÁVEL do 27B** (ação concreta +
número + citação inline), **não** imitar o comportamento atual (conversão ~9% não merece
ser preservada).

**Dados** (`gen_train.py` → `data/train_sft.jsonl`): o 27B, em oráculo, gera pergunta
coloquial + fatos + resposta acionável por chunk de treino; **cada par é filtrado pela
PRÓPRIA RÉGUA** (só entra o que o 27B produziu **E** a régua honesta aprovou). Muralhas
anti-contaminação herdadas do `finetune2/`: holdout 151/20 intocável, NR-33/16/26
reservadas (OOD estrutural), Jaccard<0,4 da pergunta vs holdout. **Resultado: 3.000 pares
régua-aprovados, 33 NRs, cobertura média 0,929, 0 descartes lexicais** (taxa de aprovação
do 27B ≈ 94%).

**Treino** (`train_sft.py` na Spark, jupyterlab venv torch 2.12+cu130): **LoRA r=16**
(atenção + MLP; shortconv fora — recorrente, instável a LoRA), 3 épocas, lr 1e-4, bs
efetivo 32, checkpoint por época. **Por que LoRA e não full**: a GB10 caberia full-FT, mas
o `finetune2/` provou que treino agressivo sobre distribuição estreita **regride** um base
forte; LoRA conservador limita o dano e preserva a capacidade geral. Export por época em
GGUF bf16+Q4_0 (o mesmo engine do teto). **Gate de não-regressão**: cada checkpoint é
avaliado em oráculo pela régua; **aborta se regredir** vs base (a lição do fracasso
anterior: cobertura 0,40→0,28).

> **Status**: treino iniciado na Spark (282 passos, ~70 min). **A Spark ficou
> inacessível (100% packet loss no ssh/ping) durante a época 1** — provável thrash de
> memória (bf16+Q4 servers + LoRA residentes). O treino foi lançado `setsid nohup` com
> checkpoint por época; retomável quando a Spark voltar. Ver seção "Retomada" abaixo.

---

## FASE 3 — Veredito (preenchido após o treino)

_(pendente do fim do treino na Spark)_

Baselines de busca real v4 já medidos (para o "o que sobra na prática"):
| gerador · retrieved v4 (R@2 51%) | aprov% | alucinação% |
|---|--:|--:|
| 27B · retrieved | 46.4 | 25.2 |
| 2.6B Q4_0 · retrieved | 23.8 | 51.0 |

---

## Arquivos e reprodução

```
slm_oraculo/
  corpus_fix.py          # trecho-ouro do corpus CORRIGIDO (reparo/injeção Anexo II NR-10)
  build_oracle_pack.py   # empacota o oráculo das 151 + 2 casos do capitão -> data/oracle_pack.json
  build_retrieved_pack.py# empacota o contexto v4 embarcado -> data/retrieved_pack.json
  generate.py            # síntese sobre um pack (27B via juiz OU 2.6B via llama-server)
  score.py               # régua honesta (ruler + judge_regua 27B) sobre um arquivo de respostas
  gen_train.py           # dataset SFT: 27B em oráculo, FILTRADO pela própria régua (3 muralhas)
  train_sft.py           # SFT LoRA do 2.6B na Spark (checkpoint por época + export GGUF)
  eval_checkpoint.sh     # sobe checkpoint na Spark, gera oráculo+retrieved, pontua pela régua
  consolidate.py         # tabela mestre + curva do gap + veredito -> data/consolidation.json
  data/                  # packs, respostas geradas, scores, dataset de treino (leves; commitados)
```

Comandos (juiz via `REGUA_JUDGE_URL`, default `10.100.0.111:8005`):
```bash
PY=../classifier/.venv/bin/python
# FASE 1
$PY build_oracle_pack.py
$PY generate.py --pack data/oracle_pack.json --url http://10.100.0.111:8005/v1 \
    --model Qwen/Qwen3.8-27B-FP8 --label 27b --variant numeros --deterministic --think-off \
    --out data/gen_27b_oracle.json
$PY score.py --resp data/gen_27b_oracle.json --out data/score_27b_oracle.json
# 2.6B (Spark): scripts/start_srv.sh <gguf> <port>; túnel; generate.py --url 127.0.0.1:port/v1
# FASE 2 (Spark): slm_scripts/run_train.sh  (LoRA 3 ep, checkpoint por época + GGUF)
# FASE 3
$PY consolidate.py
```

### Retomada do treino (Spark)
Artefatos de treino ficam em `~/cemig-poc/` na Spark (gitignored: reprodutíveis):
- base HF patchado: `~/cemig-poc/base_hf/` (tokenizer_class TokenizersBackend→PreTrainedTokenizerFast)
- GGUFs base: `~/cemig-poc/models/LFM2.5-2.6B-{bf16,Q4_0}.gguf`
- dataset: `~/cemig-poc/train/train_sft.jsonl` (== `data/train_sft.jsonl`)
- checkpoints por época: `~/cemig-poc/train/sft_ep{1,2,3}/` + `models/lfm2.5-2.6b-sft_ep{N}-Q4_0.gguf`
- log: `~/cemig-poc/logs/train_sft.log`
Relançar: `setsid nohup bash ~/cemig-poc/slm_scripts/run_train.sh > ~/cemig-poc/logs/train_sft.log 2>&1 &`
Avaliar checkpoint: `./eval_checkpoint.sh ~/cemig-poc/models/lfm2.5-2.6b-sft_ep{N}-Q4_0.gguf sft_ep{N}_q4 8471`
