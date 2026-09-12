# Teto & evasão — otimização de prompt, teto do 27B, desenho do próximo treino (`bench/prompt_teto/`)

O capitão está indignado com respostas evasivas REAIS do app:

> P: _"o poste que preciso subir não me parece firme, o que eu devo fazer?"_
> R: _"Conforme a NR-07 … (genérico) … Seguir as normas é essencial."_ ← não disse NADA
>
> P: _"Qual a distância segura para média tensão de 13,8 kV?"_
> R: _"conforme a norma NR-10, item 10.6.2, as distâncias seguras devem ser respeitadas"_ ← não deu o número

Palavras dele: _"não faz sentido o cara ir ler a norma, isso é o trabalho do SLM!"_.

Esta pasta responde as **4 perguntas do capitão** com número, **antes de gastar GPU em
treino**. Régua única: `bench/regua` (`ruler.py` + desempate 27B + `gabarito_151.jsonl`),
limiar 0,5, **citação fora do gate** (decisão dele). Retrieval: **v4 embarcada** (RRF
k=10 pesos 2,1,2, R@2 = 51,0%). Juiz e teto: vLLM 27B (`Qwen/Qwen3.8-27B-FP8`,
`http://10.100.0.111:8005/v1`). Síntese do 2.6B/1.2B: `llama-server` na RTX 5070
(`-ngl 99`, thinking-OFF via `--reasoning-budget 0`). **Tudo paralelo, n=151 reais.**

> Regra de validade (brief): qualidade em GPU vale; **latência de GPU NÃO vale para o
> aparelho** — projeta-se pela tabela do `bench/device-liquid`.

---

## Resposta direta às 4 perguntas do capitão

**1. Quanto uma otimização de prompt no 2.6B muda a aprovação honesta? — QUASE NADA.**
Seis variantes de prompt no 2.6B (retrieved, n=151): a aprovação honesta varia de **15,2%
a 23,8%**, e a melhor (`numeros`, 23,8%) fica **empatada com o baseline (23,2%)** dentro
do ruído. Prompt **não é a alavanca**: o modelo já entrega o número quando ele está
legível no trecho; o que falta é o **trecho certo** e a **capacidade de ler a tabela**.

**2. Qual o TETO? A régua está severa? — RÉGUA OK; o gargalo é SLM + retrieval.**
O próprio 27B, com os MESMOS chunks e o MESMO gabarito:
- chunks da v4 (retrieved): **37,7% (baseline) → 44,4% (numeros)**.
- **chunk-ouro (oráculo): 76,2% (baseline) → 84,1% (numeros)**.
O teto oracle **> 60%** ⇒ **a régua NÃO está severa**: ela mede capacidade real. Onde o
27B "falha" no oráculo, é majoritariamente **tabela mal extraída** (Anexo II da NR-10) ou
subseção fora do índice — problema de dado/retrieval, não da métrica.

**3. A resposta do 27B serve como material de SFT? — SIM, e o sinal é forte.**
Em 151 (numeros, retrieved): **27B aprova e 2.6B reprova em 40 casos** (30 já com o
chunk-ouro presente = destiláveis hoje). No oráculo: **61 casos** onde o 27B converte e o
2.6B não. São pares alvo prontos para SFT de destilação (ver Parte 3).

**4. Faz sentido manter o estilo de resposta atual? — NÃO.** O 1.2B embarcado com o
estilo atual aprova **4,6%** (conversão chunk-certo→aprovada **6,5%**) — bate a "conversão
de 9%" que o capitão citou. O alvo do treino deve ser o **estilo acionável do 27B**
(ação concreta + número), **não a imitação do comportamento atual**.

---

## PARTE 1 — Otimização de prompt no 2.6B (retrieved, régua honesta, n=151)

| variante | aprov% | cobertura | halluc% | conv_chunk_hit% | tok médios |
|---|--:|--:|--:|--:|--:|
| `numeros` (exige valores explícitos) | **23.8** | 0.429 | 45.7 | 35.1 | 222 |
| `baseline` (prompt atual do app) | 23.2 | 0.446 | 41.7 | 36.4 | 172 |
| `acao_primeiro` (1ª frase imperativa) | 19.2 | 0.441 | 55.0 | 31.2 | 182 |
| `combinado_6f` (tudo + 6 frases) | 19.2 | 0.433 | 53.0 | 27.3 | 257 |
| `anti_evasao` (proíbe "consulte a norma") | 17.2 | 0.337 | 39.7 | 26.0 | 123 |
| `combinado_4f` (tudo + 4 frases) | 15.2 | 0.422 | 55.6 | 26.0 | 217 |

**Leitura honesta:**
- **Nenhuma variante supera o baseline de forma significativa.** `numeros` ganha +0,6 p.p.
  (ruído); todas as outras **pioram**.
- **Anti-evasão e ação-primeiro pioram** porque forçam o modelo a "prometer a ação" mesmo
  quando o trecho não a contém: ele **inventa o procedimento** (halluc sobe para 55%) ou
  encurta demais e perde cobertura (anti_evasao cov 0,337). O 2.6B obedece o prompt e
  passa a alucinar a ação que exigimos.
- **Teto de frases (4 vs 6) NÃO é o gargalo**: `combinado_4f` (15,2%) é pior que
  `combinado_6f` (19,2%), mas ambos abaixo do baseline. Dar mais espaço não resolve.
- **Conclusão da Parte 1**: prompt-engineering tem **retorno ~zero** aqui. A resposta
  evasiva do app **não é (principalmente) culpa do prompt** — é falta do trecho certo e
  legível. Isso reorienta o esforço para retrieval/dado e treino de capacidade.

### Respostas literais das 2 perguntas do capitão (por variante)

**P1 — "o poste que preciso subir não me parece firme, o que eu devo fazer?"**
Retrieval traz **NR-10 Anexo II** (tabela de zonas de risco) — **norma errada**: deveria
ser NR-35 (trabalho em altura) / NR-01 (recusa). Nenhum prompt corrige isso:
- `baseline` (2.6B): _"Conforme a NR-10 Anexo II, item 10.5.1, a interposição de superfície
  de separação física… Caso o poste não seja firme, deve-se interromper as atividades e
  corrigir a instalação…"_ — **coerente mas ancorado na norma errada**.
- `anti_evasao` (2.6B): _"Eu devo desligar e aterrar o circuito antes de tocar no poste,
  conforme a NR-10, item 10.5.1…"_ — **ação inventada** (desligar circuito não resolve
  poste bambo): a proibição de evasão empurra para a alucinação.
- `numeros` (2.6B): _"Não posso responder… a NR-10 Anexo II apresenta apenas… tabelas de
  distâncias, não fornece procedimentos operacionais para postes."_ — **honesto**: admite
  que o trecho não serve.
- **1.2B (estilo atual do app)**: _"Conforme a norma NR-10, item 5.20, verifique a
  estabilidade do poste… Não viole nenhuma regra."_ — **exatamente a evasão do capitão**.
- **27B (numeros)**: _"Não sei com base nas normas consultadas."_ — o teto também não
  inventa: com a norma errada, a resposta correta é admitir.

**P2 — "Qual a distância segura para média tensão de 13,8 kV?"**
Retrieval traz **NR-10 item 10.6 + Anexo II**, mas a **tabela do Anexo II está corrompida
na extração** (linhas/colunas embaralhadas: `≥30 e <36 → 0,58`), sem a linha `1 a 15 kV →
0,70 m` legível. Resultado: **todos os modelos leem a linha errada**:
- `baseline` (2.6B): _"…distância segura é de **0,56 m** conforme o Anexo II"_ — número
  errado (leu faixa 20–30 kV).
- `combinado_6f` (2.6B): _"…é de **0,58 m**, conforme o Anexo II da NR-10, item 10.6.2."_
  — número errado (faixa 30–36 kV), **mas no formato acionável desejado**.
- `numeros` (2.6B): _"…é de **1,58 metros**…"_ — pega a coluna do raio errado.
- **1.2B (app)**: _"…a distância segura para média tensão deve ser respeitada conforme o
  Anexo II da NR-10. O valor específico varia…"_ — **a evasão exata do capitão, sem o número**.
- **27B (numeros)**: _"Não sei com base nas normas consultadas."_ — com a tabela ilegível,
  o teto se recusa a chutar.

> **Achado central das perguntas do capitão**: as duas falham por **dado/retrieval**, não
> por prompt. P1 = norma errada no top-2; P2 = tabela do Anexo II mal extraída (nenhum
> LLM, nem o 27B, lê o valor certo). Prompt não conserta nenhuma das duas.

---

## PARTE 2 — Teto do 27B (a pergunta mais importante)

| gerador / contexto | aprov% | cobertura | conv_chunk_hit% | conv_chunk_miss% | tok |
|---|--:|--:|--:|--:|--:|
| 27B baseline · retrieved v4 | 37.7 | 0.479 | 57.1 | 17.6 | 86 |
| 27B `numeros` · retrieved v4 | **44.4** | 0.534 | 64.9 | 23.0 | 108 |
| 27B baseline · **oráculo** | 76.2 | 0.673 | 84.4 | 67.6 | 82 |
| 27B `numeros` · **oráculo** | **84.1** | 0.748 | 83.1 | 85.1 | 104 |
| 2.6B baseline · oráculo | 42.4 | 0.567 | 42.9 | 41.9 | 165 |
| 2.6B `numeros` · oráculo | 46.4 | 0.565 | 53.2 | 39.2 | 214 |
| **1.2B embarcado · retrieved v4 (estilo atual)** | **4.6** | 0.118 | 6.5 | — | 47 |
| 1.2B embarcado `numeros` · retrieved v4 | 2.0 | 0.074 | 3.9 | — | 44 |

**Interpretação (honesta, como pedido):**

1. **A régua NÃO está severa.** Teto do 27B com chunk-ouro = **84,1%** (>>60%). A régua
   honesta é exigente (mede cobertura de fatos, não citação), mas passável por um modelo
   competente com o trecho certo. **Nenhuma recalibração é necessária.**

2. **Separação retrieval × geração:**
   - **Retrieval é o teto duro**: mesmo o 27B cai de **84,1% (oráculo) → 44,4%
     (retrieved v4)** — perde ~40 p.p. só porque a v4 entrega o chunk-ouro em 51% das
     perguntas. O `conv_chunk_miss` do 27B retrieved é 17–23%: sem o chunk certo, nem o
     teto responde.
   - **Geração é o segundo teto**: dado o chunk-ouro, o **2.6B converte só ~53%** contra
     **83% do 27B** — lacuna de capacidade de ~30 p.p. O **1.2B embarcado converte 6,5%**:
     ~13× abaixo do 27B com o mesmo contexto.

3. **Por que o 27B não passa dos 84% no oráculo?** As ~16% restantes são
   **dado ruim** (tabelas mal extraídas como o Anexo II da NR-10) ou subseção específica
   fora do índice grosso — o gabarito exige fatos que o trecho-ouro **fornecido** não
   contém de forma legível. **Não é a régua; é o corpus.** Alavanca: re-extrair tabelas
   das NRs (Anexo II NR-10, tabelas de distância/tensão) para Markdown estruturado.

---

## PARTE 3 — Desenho do próximo treino na Spark (só mapeamento, NÃO treinar)

O capitão perguntou: manter o estilo atual não faz sentido se a conversão é ~9%? **Correto.**
O treino deve **destilar o ESTILO ACIONÁVEL do 27B**, não imitar o comportamento atual.

### Alvo e dados
- **SFT de destilação (recomendado como 1ª rodada)**: alvo = **respostas do 27B
  aprovadas pela régua**. Pares prontos hoje: **40** onde o 27B aprova e o 2.6B reprova
  (30 já com chunk-ouro presente) + **61** no oráculo (contexto perfeito). Expandir com
  o 27B gerando alvos sobre um **pool CLEAN** (fora do holdout 151/20; respeitar as
  muralhas de `finetune2/`: NR-33/16/26 reservadas, Jaccard<0,4) para chegar a
  **800–1500 pares**.
- **DPO (2ª rodada, opcional)**: escolhida = resposta do 27B (acionável, com número);
  rejeitada = resposta **evasiva** do 2.6B/1.2B (as `_baseline_retrieved` desta pasta
  já são uma fonte natural de "rejeitadas", mas **sobre o holdout — não usar como treino**;
  regerar rejeitadas sobre o pool CLEAN, igual ao `finetune2/gen_dpo.py`).
- **Modelo base**: `LiquidAI/LFM2.5-2.6B` (não-QAD, público) — mesmo do `finetune2/`.
  **Aviso do histórico**: `finetune2/` já mostrou que SFT/DPO **regrediram** o 2.6B (base
  0,40 cov → SFT/DPO 0,27–0,29). Desta vez o desenho muda: **alvo é a resposta APROVADA
  do 27B** (não respostas sintéticas estreitas), o que ataca a causa da regressão anterior
  (encolhimento + citação do item errado). Critério de corte pré-treino: cada par-alvo
  DEVE passar a régua honesta antes de entrar no dataset.

### Estilo do alvo (o "estilo acionável")
Do exame das saídas do 27B aprovadas: **ação concreta + valor numérico quando existe +
citação inline curta**, 1–3 frases, sem markdown. É o oposto do "siga as normas". O
prompt de síntese pode ficar no baseline atual (Parte 1 mostra que prompt não é a
alavanca); a **capacidade** vem do treino.

### Custo/tempo estimado na Spark (GB10 121 GB, `ssh walcyrios@spark-b431`)
- Geração dos alvos: **27B já roda no servidor do juiz** (zero token novo na Spark);
  ~1500 alvos em paralelo (16 workers) ≈ 15–25 min.
- SFT LoRA (r=16, 1–2 épocas, 1500 pares) no 2.6B: ~**30–60 min** de treino + requant
  `llama-quantize` (~5 min). DPO (beta 0,1, ~200 pares): +~30 min.
- Avaliação (esta régua, 151, paralela): ~3–5 min por variante.
- **Total realista: meio dia** para SFT+DPO+painel.

### Critério de sucesso (aprovação honesta, NÃO loss)
- **Meta mínima**: 2.6B pós-SFT **> 30% aprov retrieved** e **> 60% conv_chunk_hit**
  (fechar metade da lacuna 2.6B→27B no oráculo, hoje 53% vs 83%).
- **Gate de não-regressão**: cobertura de fatos **não pode cair** vs base 2.6B (0,53
  oráculo); painel OOD de `finetune2/` sem regressão.
- **Só embarca** se passar E couber no orçamento de voz do aparelho (hoje o 2.6B custa
  ~23 s / 4,3 GB no S24+ — ver `android/README_ENGINE_UPGRADE.md`; a decisão de embarque
  é pós-treino e independe deste mapeamento).

---

## Veredito para o `AskPipeline.kt`

**NÃO alterar o `SYNTHESIS_SYSTEM_PROMPT` agora.** A Parte 1 mostra que nenhuma variante
supera o baseline de forma significativa no 2.6B, e as anti-evasão/ação-primeiro
**pioram** (aumentam alucinação). Mudar o prompt para "proibir evasão" no modelo
embarcado (1.2B) só trocaria "não disse nada" por "disse algo errado com confiança" —
pior para o operário. As alavancas reais, em ordem:

1. **Retrieval intra-norma + re-extração de tabelas** (Anexo II NR-10, tabelas de
   distância/tensão): sem o trecho legível, nem o 27B acerta o 13,8 kV.
2. **Capacidade do sintetizador** (destilar o 27B para o 2.6B — Parte 3), depois destravar
   o 2.6B no orçamento de voz.
3. Prompt fica por último (retorno ~zero medido).

---

## Reprodução

```bash
# 0) servidores (RTX 5070): embedding + 2.6B thinking-OFF (+ 1.2B se quiser o baseline app)
llama-server -m ~/models-poc/embed/embeddinggemma-300M-qat-Q4_0.gguf --embedding \
   --pooling mean -b 2048 --port 8399 &
llama-server -m ~/models-poc/LFM2.5-2.6B-Q4_0.gguf -ngl 99 -c 16384 --parallel 4 \
   --reasoning-budget 0 --jinja --port 8410 &

# 1) reconstruir índice v4 + cache denso v4-exp + chunks das 151 (retrieval embarcado)
make -C bench/prompt_teto retrieval

# 2) Parte 1 (6 prompts no 2.6B) + Parte 2 (teto 27B) + oráculo + 1.2B + consolidação
make -C bench/prompt_teto all      # juiz via REGUA_JUDGE_URL (default 10.100.0.111:8005)
```

Artefatos (commitados por serem leves e a análise ser a entrega): `data/consolidation.json`
(tabela mestre + veredito), `data/score_*.json` (régua por config), `data/resp_*.json`
(respostas geradas, inclui casos do capitão). `data/chunks_v4.json` é o cache do retrieval
v4 embarcado (reprodutível). Índice v4 e cache denso são gitignored (reprodutíveis via
`make retrieval`).

Ambiente: `classifier/.venv` (sklearn 1.9.1, requests). Juiz 27B via `REGUA_JUDGE_URL`.
Higiene: paralelo (≥8/16 workers), checkpoint incremental, procedência em cada artefato,
não-interativo. Comentários em PT-BR, código em inglês.
