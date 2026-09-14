# tools_v1 — Tool-calling nativo do LFM2.5 (1.2B primeiro), multiturno

Bancada que responde à intenção do capitão: **o próprio modelo decide quando buscar** (em vez
da heurística de reuso por Jaccard do app), com **followup questions**, **casos sem tool** e a
**chamada já levando um argumento que auxilia a busca**. O produto central do treino é o campo
`consulta` (fala crua → termos de busca). Tudo com o **1.2B** (o único que roda em latência de
voz), régua honesta (`bench/regua`), índice v4 e corpus corrigido.

> Ambientes: geração/validação/avaliação no `classifier/.venv` (sklearn + jinja2, py3.12);
> render de treino/paridade no `.venv-train` (transformers 5.17); treino + export GGUF na **DGX
> Spark** (`walcyrios@spark-b431`, `~/jupyterlab/.venv`). Juiz vLLM 27B `10.100.0.111:8005`
> (roteável deste host). Holdout 151/20 **INTOCÁVEL** (só medição de argumento; nunca treina/
> calibra). Paralelo, checkpoint, não-interativo, procedência. Comentários PT-BR; código em inglês.

---

## PASSO 0 — o formato nativo, confirmado ANTES de gerar (exigência permanente)

O contrato do capitão foi verificado contra o `chat_template.jinja` REAL do
`LiquidAI/LFM2.5-1.2B-Instruct` (`verify_format.py`, saída em `data/passo0_render.txt`):

- **tools no SYSTEM** como `List of tools: [{...}, {...}]` (JSON em texto);
- **chamada** entre tokens especiais:
  `<|tool_call_start|>[buscar_norma(consulta='...', nr='NR-10')]<|tool_call_end|>` (sintaxe de
  função, aspas simples com escape — não JSON);
- **retorno** em turno `role:"tool"`: `<|im_start|>tool\n{conteúdo}<|im_end|>`;
- a região `{% generation %}` marca só os turnos assistant → **assistant_only_loss cobre a
  DECISÃO de chamar + os ARGUMENTOS + a SÍNTESE, e NÃO o turno `tool`** (provado: os tokens do
  chunk injetado ficam fora da máscara).

**Decisão fechada do capitão sobre o gerador (emenda 2): o 27B é gerador de CONTEÚDO, nunca de
FORMATO.** Pedimos a ele só campos semânticos em JSON guiado (`guided_json`) e a **renderização
no formato LFM é NOSSA e determinística** (`tokenizer.apply_chat_template`). Zero regex, zero
string à mão. Para rodar a geração no `classifier/.venv` (sem transformers), `render_jinja.py`
carrega o MESMO `chat_template.jinja` e o renderiza com jinja2 puro — **paridade byte-a-byte
provada** contra `apply_chat_template` (`test_render_parity.py`, `make verify`).

**Contrato da ferramenta (assinatura enxuta, fechada pelo capitão):**
```
buscar_norma(consulta: str, nr: str|None)
```
`consulta` = a REESCRITA da fala em termos de busca (números normalizados: "13,8 kV" →
"13,8 kV 13.8 kV"); `nr` = a norma quando citada/inequívoca, senão `null`. **Não chamar é
comportamento válido e ensinado** (não existe tool "nenhuma"). O `SYSTEM_PROMPT_TOOLS`
(`tool_schema.py`) é **idêntico** no treino e no que o app usaria (exigência da emenda 2).

**Validação de round-trip (parte do PASSO 2):** renderiza a chamada → re-parseia → confere que
`consulta`/`nr` voltam **idênticos** (escape de aspas, acento, vírgula decimal `13,8 kV`). OK
em 100% dos exemplos.

---

## PASSO 1+2 — dataset sintético (5 famílias) com validação MECÂNICA por exemplo

`gen_dialogs.py` fabrica diálogos multiturno; cada exemplo só entra se: (a) a chamada
**parseia** na sintaxe do template (round-trip); (b) o `nr` **existe** no corpus; (c) a
`consulta` gerada **recupera o chunk-ouro no índice v4** (BM25 top-5) — senão o exemplo é
descartado/regerado (o modelo só aprende argumentos que FUNCIONAM na nossa busca); (d) a
resposta final passa pela **régua honesta** (famílias que respondem) ou pelo **critério de
recusa** (`sft_v2`, reusado); (e) o alvo **cita o item** da norma (decisão B — a régua não
reprova por falta, mas o alvo cita).

Muralhas reusadas de `finetune2/common`: holdout intocável, **NR-33/16/26 reservadas**, filtro
lexical Jaccard<0.4 vs holdout; split de chunks train/valtest (`sft_v2.split_chunks`). Dedup
intra-dataset de perguntas ao ciclar o pool. **4.236 diálogos** gerados (alvo 4.500 da emenda 1;
`multiturno_nova_busca` limitado a 411 pela trava dupla — ambos os turnos precisam recuperar o
ouro; as demais no alvo). 0 descartes lexicais (nenhuma fala aproximou-se do holdout).

| família | n | % | o que ensina |
|---|--:|--:|---|
| (1) chamada_simples | 1575 | 37.2 | pergunta → tool_call(consulta,nr) → tool → resposta citando o item |
| (2) sem_ferramenta | 675 | 15.9 | saudação/agradecimento/repetição/bom senso/fora de escopo → responde SEM chamar |
| (3) multiturno_reuso | 900 | 21.2 | 2º turno no MESMO assunto já em contexto → NÃO chama, reusa |
| (4) multiturno_nova_busca | 411 | 9.7 | assunto MUDA → chama de novo com consulta diferente |
| (5) recusa_apos_busca | 675 | 15.9 | tool retorna trecho errado/parcial → recusa/delimita (mantém o ganho do v3) |

**Truncamento medido (`measure_truncation.py`, exigência do brief):** a `max_length=2048`,
**8.71% dos exemplos truncam** — quase todos de `multiturno_nova_busca` (368/411 = 90%: duas
chamadas + dois chunks + duas respostas). A **`max_length=3072` zera o truncamento** (0/4236;
p99=2458, máx=2683). Treino usa **3072**.

---

## PASSO 3 — treino (1.2B, 1 época, varredura de rank + curva de saturação)

Base `LiquidAI/LFM2.5-1.2B-Instruct` (bf16; **não existe checkpoint QAD em HF** — o QAD é só
GGUF). Reusa o motor `sft_v2/train_v2` (rank-configurável) via `train_1_2b_tools.py` (só troca o
nome do artefato). Receita: bs8×ga4, lr 1e-4, cosine, bf16, grad ckpt, alpha=2r,
`assistant_only_loss=True`, val interno. **Asserts publicados ANTES de treinar** (`premises_r*`):
**72 alvos LoRA** (6·4 attn + 16·3 MLP; `feed_forward.w1/w2/w3` + `self_attn.{q,k,v,out}_proj`),
**mlp✓ attn✓ conv✗** em todos os ranks. wandb online: `cemig-tools-1.2b/runs/lh94m0ve`.

**FASE A — curva de saturação (rank fixo r32, degraus ANINHADOS 750/1500/3000/4236, mesma
seed, cada degrau ⊂ maior).** 1 run por degrau (barra de erro NÃO medida aqui — só no final).
**FASE B — varredura de rank r16/32/64/128 no degrau completo (4236).**

| rank | eval_loss interno | train_loss |
|---|--:|--:|
| 16  | 0.7636 | 0.867 |
| 32  | 0.7225 | 0.823 |
| 64  | 0.6874 | 0.785 |
| 128 | 0.6552 | 0.752 |

eval_loss cai monotônica com o rank (não satura na loss — igual ao `sft_1_2b/`), mas a **régua
não acompanha** (ver PASSO 4).

---

## PASSO 4 — métricas NOVAS (as antigas não medem isto)

Harness próprio (`infer_tools.py`) roda o loop REAL: decisão → chamada → **busca v4** (BM25
top-2, `nr` vira filtro-duro) → síntese, exatamente como o app faria. `run_eval.py` +
`score_e2e.py`. Suites 100% fora do treino; a `args` usa o holdout 151 **só para medir**. A
métrica de retrieval usa o `check_hit` OFICIAL (doc+section) — a mesma régua do app (a contagem
por-id pura subconta vizinhos de seção).

### Tabela mestre (Q4_0; e2e sobre 90 answerable de VAL; args sobre as 151)

| candidato | dec F1 | não-chama-à-toa | sintaxe% | **model R@2** | reuso% | novo-tópico% | e2e aprov% | e2e aluc% |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| QAD embarcado (sem tools) | **0.0** | — | — | 0.0 | — | 0.0 | 0.0 | 85.6 |
| Instruct base (sem tools) | **0.0** | — | — | 0.0 | — | 0.0 | 0.0 | 90.0 |
| sft_1_2b r64 (sem tools)  | **0.0** | — | — | 0.0 | — | 0.0 | 0.0 | 84.4 |
| tools r16 | 94.9 | 83.3 | 100.0 | 21.9 | 74.3 | 86.0 | 22.2 | 60.0 |
| tools r32 | 95.4 | 85.0 | 100.0 | 21.9 | 78.6 | 90.0 | 22.2 | 58.9 |
| tools r64 | 95.4 | 85.0 | 100.0 | 19.9 | 75.7 | 98.0 | 22.2 | 63.3 |
| **tools r128** | 94.9 | 83.3 | 99.0 | **24.5** | 82.9 | **100.0** | **24.4** | 57.8 |

Referências de **qualidade do argumento** (recall@2 no v4, check_hit): **pergunta crua 30.5%**,
**27B rewrite 29.8%**, model rewrite ~20-24% (R@5: crua 42.4, 27B **51.7**, model ~37).

### a) DECISÃO DE CHAMAR (matriz de confusão)
r128: TP=93 FN=0 FP=~13 TN=~47 → **recall 100%** (nunca deixa de chamar quando devia),
precision ~88%, **F1 94.9%**. O custo é ~15-17% de "chamou à toa" (chama em alguns casos de
saudação/bom senso). Os **3 baselines sem tool têm F1=0** (n_called=0): não sabem que a
ferramenta existe.

### b) VALIDADE SINTÁTICA
**99-100%** das chamadas emitidas parseiam. (Nota: o llama-server `/completion` não reemite os
tokens especiais como texto; o modelo emite `[buscar_norma(...)]` e o parser runtime aceita a
forma com/sem wrapper — igual ao que o app detectaria.)

### c) QUALIDADE DOS ARGUMENTOS — a métrica central (o delta que o capitão quer isolado)
**Resultado honesto: a reescrita do 1.2B PIORA o retrieval.** delta model−crua = **−8.2 p.p.
R@2** (final r128, 3 runs). O 27B rewrite **empata** com a crua (−0.7 R@2, mas +9.3 R@5). Ou
seja: o 1.2B aprende a CHAMAR e a formatar (F1 95%, sintaxe 100%), mas a consulta que ele produz
recupera menos que a própria fala do operário. Isto **confirma** o achado documentado no
`AGENTS.md` (classifier/retrieval: "Reescrita PIORA — usar fala/keywords brutas"). O rewrite do
1.2B adiciona termos que dispersam o BM25; o poder de reescrita útil está no 27B (e, mesmo nele,
o ganho aparece só em R@5).

### d) REUSO no multiturno
r128 (3 runs): **reuso-correto 84.8%±1.8** (2º turno no mesmo assunto → não chama), **novo-tópico
100%±0** (assunto muda → chama de novo). Custo: ~15-17% de "chamou à toa quando devia reusar".
O comportamento multiturno que o capitão pediu **funciona** — e é aprendido: sobe com dados
(novo-tópico 44%→94% de 750→1500 exemplos) e com rank.

### e) Ponta a ponta (aprovação/alucinação/recusa) — 3 runs do candidato final r128
| métrica | média ± desvio |
|---|--:|
| decisão F1 | 94.9 ± 0.0 |
| sintaxe válida % | 99.3 ± 0.5 |
| model R@2 (argumento) | 22.3 ± 2.5 |
| **delta model−crua R@2** | **−8.2 ± 2.5** |
| reuso-correto % | 84.8 ± 1.8 |
| novo-tópico % | 100.0 ± 0.0 |
| e2e aprovação % | 24.4 ± 3.3 |
| e2e alucinação % | 55.9 ± 3.7 |
| conversão chunk-certo→aprovada % | 35.3 ± 6.2 |

**Muitas diferenças entre ranks estão dentro do ruído** (F1 94.9-95.4, aprovação 22-24 com
desvio ~3): a barra de erro (medida só no final, por custo, conforme emenda 1) mostra que a
escolha de rank é quase indiferente na qualidade — a única melhora consistente do r128 é o
reuso/novo-tópico.

### Curva de saturação (FASE A, r32) — em que volume cada componente para de melhorar?
| exemplos | dec F1 | model R@2 | reuso% | novo-tópico% |
|---|--:|--:|--:|--:|
| 750  | 87.6 | 21.2 | 88.6 | 44.0 |
| 1500 | 94.8 | 17.9 | 62.9 | 94.0 |
| 3000 | 94.9 | 22.5 | 74.3 | 84.0 |
| 4236 | 95.4 | 21.9 | 78.6 | 90.0 |

**Resposta às perguntas da emenda 1, com número:**
- **Decisão de chamar: satura cedo** — pula de 87.6→94.8 entre 750 e 1500 e fica flat. A
  hipótese ("~200-300") era otimista; o joelho está em **~1500**.
- **Validade sintática: satura muito cedo** (100% já em 750). Confirmado.
- **Qualidade dos argumentos (a incógnita): NÃO satura porque NÃO melhora** — oscila 17.9-22.5
  dentro do ruído em TODOS os volumes E TODOS os ranks, sempre **abaixo** da pergunta crua. Não é
  subalimentação (4236 não supera 750); é **teto do 1.2B** para essa tarefa. Isto **não** reabre
  as conclusões do `sft_1_2b` (o teto do 1.2B continua o mesmo): mais dados de rewrite não
  destravam o retrieval.
- **Comportamento multiturno (reuso/novo-tópico): melhora até ~1500-3000** e depois flutua.

**Conclusão de custo de dados:** para o COMPORTAMENTO (decidir/chamar/reusar) **~1500 exemplos
bastam** — treino futuro 3× mais barato. Para a QUALIDADE DO ARGUMENTO, nenhum volume ajuda com
o 1.2B: a alavanca é outro modelo (27B rewriter) ou o retrieval, não mais dados.

---

## PASSO 5 — VEREDITO

**A arquitetura com ferramenta NÃO supera o pipeline fixo atual em RETRIEVAL, mas resolve o que
o capitão pediu para o FUTURO multiturno — e a bancada fica válida.**

1. **O treino de tool-calling habilita uma capacidade que hoje não existe.** Os 3 baselines
   (QAD embarcado, Instruct base, sft_1_2b sem tools) **nunca chamam a ferramenta**
   (F1=0, aprovação e2e=0%, alucinação 84-90%): sem treino, o 1.2B ignora o tool e inventa. Com
   o treino: **decisão F1 95%, sintaxe 100%, reuso 85%, novo-tópico 100%**. A decisão de buscar
   passa a ser **do modelo** (fim da heurística de Jaccard), com followup e casos sem-tool.
2. **MAS o argumento gerado pelo 1.2B PIORA o retrieval** (−8.2 p.p. R@2 vs a fala crua; o 27B
   empata). Como no pipeline fixo atual a busca usa a **fala bruta** (que recupera mais), a
   arquitetura com ferramenta **não melhora o retrieval** — que é o teto prático da POC. Ganho
   nenhum vem "de graça" do rewrite do modelo pequeno.
3. **Recomendação:** manter o **pipeline fixo (fala bruta) para a BUSCA** no curto prazo; adotar
   a **arquitetura de ferramenta pela sua DECISÃO/multiturno** (não pelo argumento) quando o
   produto multiturno entrar — e, nesse caso, **usar a fala bruta como `consulta`** (ou o 27B
   como rewriter na nuvem), não a reescrita do 1.2B. A bancada, os dados e o harness ficam
   prontos para reabrir quando (a) o 2.6B couber na voz, ou (b) houver um rewriter que realmente
   supere a fala crua.
4. **Escolha de candidato**, se for embarcar tool-calling: **r128 Q4** (melhor reuso/novo-tópico
   e argumento marginalmente melhor, mesma latência/RAM dos demais — arch/quant/tamanho iguais
   ao QAD embarcado). Ranks menores estão dentro do ruído; **r32** é a opção barata equivalente.

### Amostras literais (modelo tools_1_2b_r128 Q4, determinístico, busca v4 real)
Ver `data/samples_model_r128.txt` (casos do capitão) e `data/samples_dataset.txt` (1 diálogo
renderizado por família). Destaques:
- **13,8 kV** → chama `buscar_norma(consulta='distância segura trabalho proximidade rede 13,8 kV
  13.8 kV')`, recupera **NR-10 Anexo II** (a tabela certa) e cita "0,20 a 0,70 m, NR-10 Anexo II"
  — a normalização numérica levou ao chunk certo.
- **poste** → chama, recupera NR-22, dá a **ação certa** ("Paralise imediatamente… conforme a
  NR-22, item 22.13.3.2"). (No multiturno, porém, às vezes trata "poste" como reuso e não
  rechama — é o ~15% de erro de reuso.)
- **camiseta rasgada** → chama, recupera norma errada (NR-36/NR-04) e **RECUSA corretamente**
  ("a norma consultada não responde… seria preciso verificar…") — **não** comete o erro de
  "adornos" do v1; a família de recusa transferiu.
- **saudação/agradecimento** → **não chama**, responde cordial (sem citar norma).

---

## Reprodução

```bash
PY=../classifier/.venv/bin/python; PYT=../.venv-train/bin/python
# PASSO 0 (render + paridade + máscara)         -> make verify
# PASSO 1+2 (4500 diálogos, validação mecânica) -> make data
# prep de treino (tools no system + degraus)    -> make prep ; make truncation
# suites de avaliação                            -> make evalsets
# treino na Spark (curva + varredura de rank):
bash sync_to_spark.sh
ssh walcyrios@spark-b431 "cd ~/cemig-poc && WANDB_MODE=online setsid bash tools_v1/run_train_tools.sh > logs/train_tools.log 2>&1 < /dev/null &"
# avaliação (deste host, via tailscale direto):
bash run_all_evals.sh '~/cemig-poc/models/lfm2.5-1.2b-tools_1_2b_r128-Q4_0.gguf' tools_r128
$PY consolidate.py --final-label tools_r128_final
```

### Artefatos
- **Dados/scripts locais** (gitignore para pesados; leves commitados): `data/passo0_render.txt`,
  `data/prep_manifest.json`, `data/truncation*.json`, `data/eval_*.json`, `data/e2e_*.json`,
  `data/consolidation.json`, `data/samples_*.txt`. Dataset completo `data/dialogs.jsonl` (4236,
  reprodutível) gitignored.
- **Pesos na Spark** (`~/cemig-poc/models/`, gitignored): `lfm2.5-1.2b-tools_1_2b_r{16,32,64,128}
  -{bf16,Q4_0}.gguf` + `tools_1_2b_step{750,1500,3000}_r32-{bf16,Q4_0}.gguf`. Dataset:
  `~/cemig-poc/tools_v1/data/train_{full,step*}.jsonl`. Log: `~/cemig-poc/logs/train_tools.log`.
  wandb `cemig-tools-1.2b` (run `lh94m0ve`).

### Arquivos
```
tool_schema.py        contrato da tool + system prompt (idêntico treino/prod) + construtores de mensagem
render.py             render via transformers + parser round-trip (estrito) + parser runtime (tolerante)
render_jinja.py       render via chat_template oficial em jinja2 puro (paridade byte-a-byte, sem transformers)
verify_format.py      PASSO 0: 1 diálogo/família renderizado + prova da máscara assistant-only
test_render_parity.py prova jinja == apply_chat_template (byte-a-byte)
retrieval_check.py    filtro PASSO 2: a consulta recupera o chunk-ouro no índice v4 (check_hit oficial)
common_tools.py       núcleo (muralhas, guided_json 27B, régua honesta, juízes de recusa reusados)
gen_dialogs.py        PASSO 1+2: 5 famílias, validação mecânica por exemplo
prep_train.py         tools no system + degraus aninhados da curva de saturação
measure_truncation.py mede truncamento a max_length (2048 vs 3072)
train_1_2b_tools.py   wrapper do train_v2 (nome do GGUF); run_train_tools.sh: curva + varredura (Spark)
infer_tools.py        loop tool-calling (decisão→chamada→busca v4→síntese)
build_eval_sets.py    suites: decisão / argumentos(151) / reuso+novo-tópico
metrics.py            decisão(matriz/F1) / sintaxe / argumento(recall@k) / reuso
run_eval.py score_e2e.py  orquestra as métricas por candidato
run_all_evals.sh eval_all_candidates.sh  sobe server na Spark + avalia (tailscale direto)
consolidate.py        tabela mestre + curva + média/desvio dos 3 runs finais
extract_samples.py    amostras literais renderizadas
```
