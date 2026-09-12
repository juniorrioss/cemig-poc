# Régua honesta — a resposta CONTÉM a informação pedida? (`bench/regua/`)

O capitão pegou o erro central da nossa avaliação: a resposta
_"O EPI para trabalho em altura acima de 2 metros é obrigatório"_ **não responde nada**
(só reescreve a pergunta) e **passava** nas réguas antigas. Palavras dele:
_"Se isso for considerado uma boa resposta no nosso teste, tudo está equivocado"_.

Esta pasta reconstrói a régua a partir de uma pergunta só: **a resposta contém a
informação pedida?** Nada além disso. Duas decisões do capitão foram obedecidas:

1. **Citação sai do gate.** O classificador+BM25 já traz a norma e o usuário **vê a
   fonte na tela**. `citacao_fonte` continua como **coluna informativa**, nunca como
   critério de aprovação. Todo o gate histórico (1,3%, 21%, …) era dominado por citação.
2. **Tudo na DGX Spark** (`ssh walcyrios@spark-b431`, GB10). O juiz continua sendo o
   vLLM 27B (`Qwen/Qwen3.8-27B-FP8`) em `http://10.100.0.111:8005/v1`.

> Nota de infra: o juiz (`10.100.0.111`) não é roteável da Spark; usamos um **túnel
> reverso** (`ssh -R 18005:10.100.0.111:8005`, mantido num tmux neste host) e a Spark
> fala com o juiz por `http://127.0.0.1:18005/v1` (`REGUA_JUDGE_URL`). Nenhum token novo
> foi gerado por nós: toda a inferência 27B roda no servidor do juiz.

---

## Como a régua funciona

### Passo 1 — Gabarito de fatos (`build_gabarito.py` → `data/gabarito_151.jsonl`)
Para cada uma das **151 do holdout** (`corpus/qa_pairs_v2.jsonl`), o juiz 27B extrai do
**trecho-ouro** uma lista curta de **fatos obrigatórios** — itens concretos que qualquer
resposta correta precisa conter (ex.: EPI de altura → _cinturão paraquedista, talabarte,
trava-quedas, ponto de ancoragem_). Passada **paralela** (16 workers), com revisão
automática de formato (JSON estrito, 2–6 fatos atômicos) e **procedência** por item.

- Fonte primária dos fatos: **texto do trecho-ouro** do índice canônico
  `corpus/index_hf_36nr.db` (127/151). O `relevant_chunk_ids` do JSONL está obsoleto;
  resolvemos por `doc`+seção com fallback textual (`common.GoldResolver`).
- Fallback (24/151): quando o índice grosso de 36 NRs **não cobre a subseção** cobrada
  (ex.: `10.14.1`, `18.14.1`), a fonte passa a ser a **resposta-ouro oficial** do dataset
  (referência derivada da norma, **nunca** saída de modelo). Procedência registrada.
- Resultado: **151/151 com fatos, média 4,74 fatos/item**, zero vazios.

### Passo 2 — Verificador binário + antitautologia (`ruler.py`, `judge_regua.py`)
- **cobertura_fatos** = fração dos fatos obrigatórios presentes na resposta. Casamento
  barato por termo/radical/sinônimo (dicionário de sinônimos NR em `ruler.SYNONYMS`);
  os fatos **ambíguos** (nem claramente presentes nem ausentes) vão a **um único**
  desempate do 27B por resposta, que também checa **alucinação** (contradição ao trecho).
- **APROVADO** = `cobertura ≥ limiar (0,5)` **E** não-tautológico **E** sem alucinação
  **E** não-vazio. **Citação não entra** (decisão do capitão).
- **Detector de tautologia (determinístico, obrigatório)**: resposta que só reordena os
  termos da pergunta (sobreposição de conteúdo com a pergunta ≥ 0,60) **e** não traz
  **nenhum** fato do gabarito → **REPROVA automaticamente**. Validado nos exemplos do
  capitão em `test_tautology.py` (5 casos, todos passam), incluindo a frase exata
  _"O EPI … é obrigatório"_ (detectada como tautologia, cobertura 0,0).

### Passo 3 — Rejulgamento retroativo (`loaders.py`, `rejudge.py`)
**Nenhuma resposta nova foi gerada** — varremos o disco. `loaders.discover_candidates()`
achou **66 candidatos** com respostas salvas: `bench/judge151`, `bench/sintese2`,
`bench/ctx_topk`, `finetune2` (base/SFT/DPO, f16 e Q4_0) e o **shootout v2 completo**
(qwen3.5/qwen3, gemma-3-1b, llama-3.2-1b, lfm2/lfm2.5 em classic/rewrite/tool/topk2).
Cada um foi rejulgado pela régua honesta (paralelo, checkpoint incremental).

Comandos:
```bash
make -C bench/regua all          # gabarito -> rejudge -> analyze (na Spark, via túnel)
python3 bench/regua/test_tautology.py     # valida o detector (offline)
```

---

## Tabela única comparável (topo; régua honesta, limiar 0,5)

`aprov%` = aprovação honesta · `cov` = cobertura média de fatos · `taut%` = tautologia ·
`halluc%` = alucinação · `cit%` = citação (**informativa, fora do gate**) ·
`hit%` = retrieval do chunk-ouro.

| candidate | aprov% | cov | taut% | halluc% | cit%(info) | hit% |
|---|--:|--:|--:|--:|--:|--:|
| sintese2/lfm2.6b_noth_baseline | **25.8** | 0.43 | 0.0 | 41.1 | 100.0 | 52.3 |
| sintese2/lfm2.6b_noth_ordering | 25.2 | 0.43 | 0.0 | 45.7 | 100.0 | 52.3 |
| finetune2/base (2.6B) | 24.5 | 0.42 | 0.0 | 44.4 | 100.0 | 51.0 |
| judge151/v3_lfm2.6b_noth | 24.5 | 0.41 | 0.0 | 39.1 | 100.0 | 52.3 |
| sintese2/lfm2.6b_noth_full | 23.2 | 0.43 | 0.0 | 45.0 | 100.0 | 52.3 |
| finetune2/dpo_f16 | 21.9 | 0.29 | 0.0 | 36.4 | 98.0 | 51.0 |
| finetune2/base_f16 | 19.9 | 0.40 | 0.0 | 43.0 | 100.0 | 51.0 |
| sintese2/minicpm2b_full | 17.2 | 0.27 | 0.0 | 23.8 | 76.2 | 52.3 |
| finetune2/dpo (Q4_0) | 15.9 | 0.29 | 1.3 | 36.4 | 80.8 | 51.0 |
| finetune2/sft | 13.9 | 0.27 | 0.0 | 33.8 | 78.1 | 51.0 |
| **judge151/lfm1.2b (embarcado, hit 29%)** | 9.9 | 0.19 | 0.0 | 23.8 | 84.1 | 29.1 |
| sintese2/lfm1.2b_baseline (1.2B, hit 52%) | 2.6 | 0.10 | 0.0 | 19.2 | 77.5 | 52.3 |
| gemma-3-1b/classic_rag | 0.0 | 0.01 | 0.0 | 2.0 | 7.9 | 20.8 |
| qwen3-0.6b/* · lfm2.5-350m/* | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | — |

Tabela completa das 66 células em `data/rejudge_table.json`; item-a-item em
`data/rejudge_items.jsonl`; consolidação em `data/consolidation.json`.

---

## Respostas diretas ao capitão

### (a) Quanto da nossa "qualidade" histórica era tautologia?
**Tautologia PURA é rara: 0,1% em média** (pico 2,5% no llama-3.2-1b). O problema **não
era tautologia literal** — era **qualidade oca dominada por citação**. Cruzando as
aprovações do gate antigo com a régua honesta: **108 de 334 aprovações antigas (32,3%)
são reprovadas pela régua honesta**, e **100% dessas "aprovações ocas" carregavam
citação**. Ou seja: **~1 em cada 3 "boas respostas" históricas passava por citar a norma
sem conter a informação pedida** — exatamente o caso _"…é obrigatório"_ do capitão.
A citação-no-gate, e não a tautologia, é que mediu nada.

### (b) O ranking muda com a régua honesta? Quem sobe, quem cai?
**Muda, mas moderadamente** — a ordem das famílias se mantém (2.6B > minicpm2b > 1.2B),
porém posições internas viram:
- **Sobe**: `sintese2/lfm2.6b_noth_baseline` (old #4 → **new #1**) e as variantes 2.6B
  "enxutas de citação"; `dpo_f16` sobe (#9 → #7). Modelos que **respondem com o fato**
  ganham.
- **Cai**: `finetune2/base_f16` **despenca (old #1 → new #9)** e `lfm2.6b_noth_fewshot`
  (#3 → #6). Eram campeões por **citar bem**, não por **cobrir o fato** — a régua honesta
  os expõe. O `few-shot` incha citação/estilo sem elevar cobertura.
- O **1.2B embarcado** continua no fundo (9,9% aprov, cobertura 0,19): a régua honesta
  confirma que ele **não converte o chunk em resposta**, o que o gate de citação mascarava.

### (c) O treino SFT/DPO regrediu na métrica ÚTIL, ou a régua velha mentiu?
**Pergunta direta, resposta direta: regrediu de verdade na métrica útil — a régua velha
não salva o treino.** Comparando f16 (isola quantização) na régua honesta:

| variante | aprov honesta | cobertura | old-gate | old-gate SEM citação | cit_avg |
|---|--:|--:|--:|--:|--:|
| base_f16 (2.6B base) | **19.9%** | **0.400** | 23.2% | 24.5% | 2.17 |
| dpo_f16 | 21.9% | 0.289 | 11.9% | 13.9% | 1.74 |
| base (2.6B base, out.) | **24.5%** | **0.425** | 21.2% | 23.2% | 2.09 |
| sft | 13.9% | 0.267 | 7.9% | 10.6% | 1.56 |
| dpo (Q4_0) | 15.9% | 0.287 | 9.3% | 10.6% | 1.61 |

- **Cobertura de fatos cai com o treino**: base 0,40–0,43 → SFT/DPO 0,27–0,29. Isso é
  regressão na informação, não em citação — a mesma conclusão do `finetune2/README.md`,
  agora **confirmada pela métrica útil**, não pela citação.
- O `dpo_f16` **parece** subir em aprovação (21,9%) porque **encurta e evita alucinar**
  (halluc 36% vs 43% do base_f16), mas com **cobertura menor** (0,29 vs 0,40): ele
  aprova mais itens fáceis e perde os que exigem vários fatos. Não é ganho de capacidade.
- Removendo a trava de citação do gate antigo (`old-gate SEM citação`), o SFT/DPO
  **continuam abaixo do base** (+2–3 p.p. simétricos só). **A régua velha não mentiu a
  favor do treino** — pelo contrário, a citação até inflava o base. **Veredito mantido:
  não embarcar o SFT/DPO.**

### (d) Qual o gargalo REAL agora?
**Gargalo DUPLO, e a régua honesta reordena as prioridades:**

1. **Retrieval ainda domina o teto**: agregando as 7.717 respostas rejulgadas,
   aprovação com chunk-ouro no top = **17,4%** vs **4,2%** sem o chunk — **4,1× mais**
   quando a busca acerta. O melhor caso (v4) entrega o chunk em ~52% das perguntas.
2. **Dado o chunk certo, o LM é o gargalo no aparelho**: conversão _chunk-certo →
   resposta-aprovada_:
   - **2.6B thinking-OFF: ~40%** (`sintese2/lfm2.6b_noth_baseline` 40,5%, `finetune2/base` 40,3%)
   - **1.2B QAD embarcado: 3,8%** (`sintese2/lfm1.2b_baseline`) a 20,5% (config `judge151/lfm1.2b`)
   - **minicpm2b: 24,1%** — pior que o 2.6B no nosso domínio, como já sabíamos.

   Ou seja: **mesmo com o trecho certo na mão, o 1.2B embarcado converte ~10× menos que
   o 2.6B.** É o mesmo achado do Q10 do capitão, agora **sem o disfarce da citação**.
3. **Causa das reprovações** (7.017 falhas): **69,6% cobertura baixa** (não trouxe os
   fatos), **28,7% alucinação** (afirma item/norma errada, típico quando o chunk falta),
   **1,6% vazio**, **0,1% tautologia**. A tautologia é resíduo; o inimigo é **cobertura**.

**Conclusão operacional**: as alavancas de qualidade são (1) **retrieval intra-norma**
(entregar o chunk certo) e (2) **capacidade do sintetizador** (destravar o 2.6B
thinking-OFF no engine — já viável, mas fora do orçamento de voz hoje: ver
`android/README_ENGINE_UPGRADE.md`). **Citação e nº de trechos não são gargalo.**

---

## Higiene e reprodução
- **Não-interativo, paralelo** (16 workers vLLM), **checkpoint incremental** por item
  (gabarito) e por candidato (rejudge), **procedência** gravada em cada artefato.
- **Holdout intocado**: nada foi treinado; o gabarito extrai fatos da norma, não do
  modelo. As 151 são as reais de `corpus/qa_pairs_v2.jsonl`.
- **Arquivos**: `common.py` (resolvedor de trecho-ouro + cliente do juiz + paralelismo);
  `build_gabarito.py`; `ruler.py` (verificador + antitautologia determinística);
  `judge_regua.py` (desempate único do 27B); `loaders.py` (descoberta das respostas
  salvas); `rejudge.py`; `analyze.py`; `test_tautology.py`; `Makefile`.
- **Ambiente**: qualquer Python com `requests`+`sqlite3` (localmente `classifier/.venv`;
  na Spark o Python do sistema). Juiz via `REGUA_JUDGE_URL`.
- **Dados gerados** (commitados por serem leves e a régua ser a entrega):
  `data/gabarito_151.jsonl`, `data/rejudge_table.json`, `data/rejudge_items.jsonl`,
  `data/consolidation.json`, `data/rejudge_run.log`.
