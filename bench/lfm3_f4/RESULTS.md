# RESULTS — F4: tool-calling multiturno sobre o data2x-ep4

Pergunta da fase: o 1.7B aprende a DECIDIR buscar, sintetizar do retorno,
recusar com contexto errado e re-buscar (MULTIHOP) — sem regredir a robustez
estática do campeão F3b? Base = merged data2x-ep4 (md5 `ae5f304d…`), LoRA r32
MLP, dataset 4.624 diálogos (6 famílias), régua honesta (juiz 27B), simulador
de tool = **retriever v4 REAL** respondendo ao vivo. Tabela mestre:
`data/consolidation.json`; parquets canônicos (35): `data/eval/`.

## 1. Estáticas (1536; o modo SEM ferramenta não pode regredir ≤2pp)

| métrica | data2x-ep4 (ref) | f4-ep1 | f4-ep2 | f4-ep3 | f4-mhx3-ep2 | gate |
|---|--:|--:|--:|--:|--:|---|
| ranking aprov. | 77,0 | **80,5** | 79,6 | 78,3 | 80,1 | ≥75 ✓ |
| rec-indevida (ranking) | 8,3 | 6,2 | 6,6 | 7,4 | 6,8 | não pior ✓ |
| distrator recusa-correta | 55,6 | 63,3 | 66,6 | 64,9 | 64,0 | ✓ |
| distrator alucinação | 24,9 | 25,4 | 24,2 | 24,0 | 25,5 | ≈ ✓ |
| oráculo | 85,4 | 87,9 | 87,8 | 88,4 | — | ≥83 ✓ |
| closed-book | 7,4 | 7,3 | 6,7 | 6,8 | — | ≈ ✓ |
| probe geral | 3,17 | — | 3,38 | 3,45 | 3,38 | ≥3,17 ✓ |

**ZERO regressão** — o treino de tool até MELHOROU o uso estático de contexto
(+2,6 a +3,5pp no ranking, +2,5pp no oráculo, −1,7pp na rec-indevida) e o
probe subiu (3,38-3,45 vs 3,17): mais dado de síntese ancorada regulariza.

## 2. Métricas de tool (suites disjuntas; retriever real ao vivo)

| métrica | base (sem treino) | f4-ep2 | f4-ep3 | mhx3-ep1 | mhx3-ep2 | alvo |
|---|--:|--:|--:|--:|--:|---|
| decisão F1 | 0,0 | 97,2 | **97,6** | 98,4 | 98,0 | ≥90 ✓ |
| sintaxe XML válida | — | **100** | 100 | 100 | 100 | ≥90 ✓ |
| reuso correto (não chamar) | 100* | **100** | 100 | — | 100 | ✓ |
| resposta do reuso cobre gabarito | 90,0* | 90,0 | 86,7 | — | 88,3 | |
| troca: nova chamada | 0 | **100** | 100 | — | 98,2 | ✓ |
| troca: busca acha a NR nova | 0 | 96,4 | 96,4 | — | 96,4 | |
| recusa pós-busca correta | 60* | 50,0 | 51,7 | 45,0 | 38,3 | trade-off ↓ |
| multihop: 2ª chamada | 0 | 2,4 | 2,4 | 7,1 | **33,3** | |
| multihop: cadeia completa | 0 | 0 | 0 | 0 | **7,1** | ≥60 ✗ |

\* o base nunca chama a tool: "reuso 100%" é vacuoso (não chama nada) e a
"recusa 60%" vem do estilo de recusa herdado do F3b sem busca.

**Decisão, sintaxe, reuso e troca CRUZARAM a meta de 90% já na 1ª época**
(saturação instantânea, consistente com tools_v1: sintaxe ~750 ex). A emissão
XML nativa é perfeita (0 falhas em 311 chamadas) e o argumento fala-crua
preserva o retriever (troca acha a NR nova em 96,4%).

## 3. MULTIHOP não decolou — curva dado×métrica (ordem do brief: reportar antes de dobrar o dataset)

| ponto | dose multihop | dose recusa | 2ª chamada | arg≠ | ouro no hop2 | cadeia completa | recusa pós-busca |
|---|--:|--:|--:|--:|--:|--:|--:|
| f4-ep2/ep3 (9%) | 411 (9%) | 1048 (23%) | 2,4 | 2,4 | 2,4 | 0,0 | 50-52 |
| mhx3-ep1 (1 época) | 1233 (24%) | 700 (14%) | 7,1 | 7,1 | 2,4 | 0,0 | 45,0 |
| mhx3-ep2 (2 épocas) | idem | idem | 33,3 | 33,3 | 11,9 | 7,1 | 38,3 |

Diagnóstico (traces em `data/eval/trace_*_multihop.jsonl`): diante do hop1
PARCIAL o modelo ou **recusa** (28/41 no ep2 — o prior da família recusa
vence) ou **responde com o parcial** (13/41). Tripler a dose e treinar 2
épocas move 2,4→33,3% na decisão de re-buscar, mas o fim-a-fim fica em 7,1%
por dois gargalos em série: (a) a 2ª consulta que o 1.7B escreve recupera o
ouro bem menos que a consulta-2 validada do 27B (11,9% vs 100% no dataset —
mesmo teto de argumento do tools_v1: reescrita de modelo pequeno < consulta
curada); (b) o ganho de 2ª chamada é pago em recusa pós-busca (50→38,3%:
re-busca e então responde com o que veio, certo ou não).

### Exemplos (mhx3-ep2, raw)

Bem-sucedido (`mh_cc9bc448e6314d95`): hop1 parcial sobre plataformas →
2ª chamada `NR-18 18.12.12.1 plataformas acesso` → síntese correta com o
registro formal de liberação (NR-18 18.12.4).

Malsucedido (`mh_18fc58251f5792c6`): hop1 parcial sobre capacitação NR-34 →
NÃO re-busca; responde delimitando com o parcial ("não dá para afirmar…").

## 4. Veredito

- **Entregue e cruzando meta**: decisão (F1 97-98), sintaxe (100%), reuso
  (100%), troca (98-100%) — ≥90% em tudo; **regressão estática ZERO** (na
  verdade ganho: ranking +2,6pp, oráculo +2,5pp, probe +0,2).
- **Multihop ≥60% NÃO atingido** (7,1% fim-a-fim no melhor ponto). A curva
  dado×métrica mostra resposta forte a dose+épocas na DECISÃO de re-buscar
  (2,4→33,3%), mas o fim-a-fim esbarra no teto de ARGUMENTO do modelo pequeno
  (já visto no tools_v1) e no trade-off com a recusa. Dobrar o dataset
  multihop provavelmente sobe a decisão, não o fim-a-fim — o gargalo é a
  qualidade da 2ª consulta, não a vontade de chamar.
- **Candidato a seguir em frente: `f4-r32-ep2`** (equilíbrio: F1 97,2,
  recusa 50, estáticas no topo, probe 3,38). O mhx3-ep2 é o ponto de
  referência da curva multihop, mas paga 12pp de recusa por 7% de cadeia.
- Caminhos p/ destravar o multihop (decisão de produto, não executados):
  (i) 2ª consulta vinda do app/27B (híbrido, como tools_oraculo da POC);
  (ii) família multihop com consultas-2 MÚLTIPLAS por item (ensinar
  variância de argumento); (iii) relaxar o gate do hop2 p/ top-8.

## Artefatos

- Checkpoints H100 (`/raid/aluno_walcy/cemig_mobile/checkpoints/`):
  `lfm3-f4-tools-r32/checkpoint-{142,284,426}` (ep1-3) e
  `lfm3-f4-tools-mhx3-r32/checkpoint-{157,314}`; wandb `lfm3-1.7b-cemig`
  runs `lfm3-f4-tools-r32` (r22hqe1h) e `lfm3-f4-tools-mhx3-r32`.
- Merged base: `/workspace/cemig/f4/merged-rebuilt` (container) e
  `/tmp/f4_base/lfm3-1.7b-data2x-ep4-merged` (local), md5 agregado
  `ae5f304d91bb47402d9125d96e766a15` (`data/merged_base_md5.json`).
- Traces: `data/eval/trace_*.jsonl`; métricas: `data/scored/tools_*.json`;
  estáticas: `data/scored/scored_*.jsonl`; parquets: `data/eval/*.parquet`.
- Pipeline autônomo do container (chuva/intermitência):
  `eval/container_runner.sh` + `eval/score_f4_container.py` (juiz via
  gateway docker `172.17.0.1:8005`).
