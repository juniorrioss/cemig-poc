# PREFLIGHT — Baseline F0: LFM3-1.7B bf16 no benchmark_mobile

Verificação prévia obrigatória (ordem do capitão) ANTES da bateria completa 1536×4 células.

## (c) Ambiente e versões

- **Execução da geração**: DGX Spark (`walcyrios@spark-b431`, NVIDIA GB10, 121 GB unificada), venv novo `~/cemig-poc/lfm3_f0/.venv` (uv, **Python 3.12.16 managed** — o python do sistema não tem headers p/ o triton JIT).
- **Versões**: transformers **5.20.2** (o config.json do modelo declara 5.12.1; o venv antigo `~/jupyterlab/.venv` tem 4.57.6 = insuficiente, NÃO usado), torch **2.14.1+cu130**, CUDA disponível ✓.
- **Modelo**: `LiquidAI/LFM3-1.7B`, snapshot `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`, **bf16, sem quantização**, `trust_remote_code=True` (auto_map `Lfm3ForCausalLM`).
- **Sampling do model card**: temperature=0.1, top_k=50, repetition_penalty=1.1, do_sample=True. EOS 124900 (`<|im_end|>`) respeitado (assert de sanidade passou).
- **Thinking**: o LFM3 é modelo de raciocínio; o chat template oficial com `add_generation_prompt=True` abre `<think>` por default — **AS IS = thinking-ON**. A resposta avaliada é a parte pós-`</think>`; o rascunho é salvo em `thinking`.
- **max_new_tokens=3072** (calibrado no smoke: 1024 truncava 60% do distrator — o modelo rumina longamente com contexto errado; 3072 reduz a ~30% no smoke do distrator e 0% no oráculo). Truncamento sem resposta final é reportado como `truncated` e NUNCA conta como recusa nem aprovação (honestidade da régua).
- **Scoring/juiz**: local (WSL), `classifier/.venv`; juiz Qwen/Qwen3.8-27B-FP8 em http://10.100.0.111:8005/v1 — **inacessível DA SPARK**, por isso geração na Spark e julgamento local.

## (b) Contagem exata de itens por célula

Dataset: `UFGCEMIGONA/benchmark_mobile`, config `normas_regulamentadoras`, split train,
parquet do cache local (snapshot `5a5aa32`), copiado via scp para a Spark (sem login HF lá).

| célula | itens | contexto |
|---|---|---|
| closed_book | 1536 (384 base_qa_id × 4 vocabulários) | nenhum |
| oraculo | 1536 | 1 passagem correct (chunks_512) |
| ranking | 1536 | 1 correct + 4 incorrect, embaralhados por id (seed 42+sha256(id)) |
| distrator | 1536 | só os 4 incorrect, embaralhados |

Verificado: 1536 linhas; 384 base_qa_id × 4 vocab (clean/technical/operational/lay);
question_type {conditional 716, factual 380, procedural 236, multievidence 204};
TODAS as linhas têm exatamente 1 correct + 4 incorrect em chunks_512;
answer idêntico nos 4 fraseados de cada base_qa_id (por isso o gabarito é por base_qa_id: 384 itens).

Splits extras do repo (`data/*_informal.parquet`) IGNORADOS nesta fase, conforme o brief.

## (a) Smoke de 10 itens por célula + prompts renderizados

Smoke rodado na GPU da Spark (10 itens/célula), pontuado E2E com a régua honesta:

| célula | aprovação smoke | observação |
|---|---|---|
| closed_book | 0/10 | modelo recusa/erra sem contexto (esperado: é a régua do "antes") |
| oraculo | 10/10 | teto de síntese alto com a passagem correta |
| ranking | 8/10 | robustez a distrator razoável no smoke |
| distrator | 2/10 recusa-correta | o modelo tende a responder com o contexto errado (alucinação) |

Artefatos do smoke: `data/smoke/gen_*.jsonl` + `data/smoke/scored_*.jsonl` (commitados).
Prompts integrais byte a byte: `data/preflight_prompts.json`.

### Célula `closed_book` (id 34763c0ad7a87d26f37a6870, 776 chars, shuffle_order=[], correct_position=None)

```
<|startoftext|><|im_start|>system
Você é um assistente técnico de segurança do trabalho, especialista em Normas Regulamentadoras (NRs) brasileiras. Responda sempre em português do Brasil, de forma objetiva e curta. Quando houver contexto normativo fornecido, baseie sua resposta exclusivamente nele e cite a fonte (norma ou documento). Se o contexto fornecido não contiver a informação necessária, diga isso claramente em vez de inventar. Quando não houver contexto, responda com seu conhecimento das normas, citando a norma quando souber.<|im_end|>
<|im_start|>user
Pergunta:
Conforme a NR-33, qual é a obrigação do Responsável Técnico quando o SESMT ou a CIPA solicitam alterações nos procedimentos de entrada para espaços confinados?<|im_end|>
<|im_start|>assistant
<think>
[... truncado para o PREFLIGHT; integral em data/preflight_prompts.json ...]
```

### Célula `oraculo` (id 34763c0ad7a87d26f37a6870, 2697 chars, shuffle_order=[0], correct_position=0)

```
<|startoftext|><|im_start|>system
Você é um assistente técnico de segurança do trabalho, especialista em Normas Regulamentadoras (NRs) brasileiras. Responda sempre em português do Brasil, de forma objetiva e curta. Quando houver contexto normativo fornecido, baseie sua resposta exclusivamente nele e cite a fonte (norma ou documento). Se o contexto fornecido não contiver a informação necessária, diga isso claramente em vez de inventar. Quando não houver contexto, responda com seu conhecimento das normas, citando a norma quando souber.<|im_end|>
<|im_start|>user
Contexto normativo consultado:
[1] (nr-33_guia_tecnico_da_nr_33):
modificação nos procedimentos. As medidas de ordem técnica, administrativa e pessoal, bem como a capacitação dos trabalhadores, plano de emergência e resgate, e procedimentos de entrada precisam ser alterados. Em caso de incidente ou condição não prevista durante a entrada, as falhas nos procedimentos de entrada devem ser investigadas e corrigidas.

33.3.3.5 “d” - qualquer mudança na atividade desenvolvida ou na configuração do espaço confinado;

Comentário:

A mudança na configuração do espaço confinado, seus acessos ou a realização de atividade não prevista n
[... truncado para o PREFLIGHT; integral em data/preflight_prompts.json ...]
```

### Célula `ranking` (id 34763c0ad7a87d26f37a6870, 8426 chars, shuffle_order=[0, 1, 3, 2, 4], correct_position=0)

```
<|startoftext|><|im_start|>system
Você é um assistente técnico de segurança do trabalho, especialista em Normas Regulamentadoras (NRs) brasileiras. Responda sempre em português do Brasil, de forma objetiva e curta. Quando houver contexto normativo fornecido, baseie sua resposta exclusivamente nele e cite a fonte (norma ou documento). Se o contexto fornecido não contiver a informação necessária, diga isso claramente em vez de inventar. Quando não houver contexto, responda com seu conhecimento das normas, citando a norma quando souber.<|im_end|>
<|im_start|>user
Contexto normativo consultado:
[1] (nr-33_guia_tecnico_da_nr_33):
modificação nos procedimentos. As medidas de ordem técnica, administrativa e pessoal, bem como a capacitação dos trabalhadores, plano de emergência e resgate, e procedimentos de entrada precisam ser alterados. Em caso de incidente ou condição não prevista durante a entrada, as falhas nos procedimentos de entrada devem ser investigadas e corrigidas.

33.3.3.5 “d” - qualquer mudança na atividade desenvolvida ou na configuração do espaço confinado;

Comentário:

A mudança na configuração do espaço confinado, seus acessos ou a realização de atividade não prevista n
[... truncado para o PREFLIGHT; integral em data/preflight_prompts.json ...]
```

### Célula `distrator` (id 34763c0ad7a87d26f37a6870, 6536 chars, shuffle_order=[0, 1, 3, 2], correct_position=None)

```
<|startoftext|><|im_start|>system
Você é um assistente técnico de segurança do trabalho, especialista em Normas Regulamentadoras (NRs) brasileiras. Responda sempre em português do Brasil, de forma objetiva e curta. Quando houver contexto normativo fornecido, baseie sua resposta exclusivamente nele e cite a fonte (norma ou documento). Se o contexto fornecido não contiver a informação necessária, diga isso claramente em vez de inventar. Quando não houver contexto, responda com seu conhecimento das normas, citando a norma quando souber.<|im_end|>
<|im_start|>user
Contexto normativo consultado:
[1] (nr-33-atualizada-2022-_retificada):
da NR-01, visando ao atendimento do disposto nesta NR.

33.3.2 Compete ao responsável técnico:

a) identificar e elaborar o cadastro de espaços confinados;

b) adaptar o modelo da Permissão de Entrada e Trabalho - PET de modo a contemplar as

peculiaridades dos espaços confinados da organização;

c) elaborar os procedimentos de segurança relacionados ao espaço confinado;

d) indicar os equipamentos para trabalho em espaços confinados;

e) elaborar o plano de resgate; e

f) coordenar a capacitação inicial e periódica dos supervisores de entrada, vigias,

t
[... truncado para o PREFLIGHT; integral em data/preflight_prompts.json ...]
```

## System prompt FIXO (todas as células)

```
Você é um assistente técnico de segurança do trabalho, especialista em Normas
Regulamentadoras (NRs) brasileiras. Responda sempre em português do Brasil, de forma
objetiva e curta. Quando houver contexto normativo fornecido, baseie sua resposta
exclusivamente nele e cite a fonte (norma ou documento). Se o contexto fornecido não
contiver a informação necessária, diga isso claramente em vez de inventar. Quando não
houver contexto, responda com seu conhecimento das normas, citando a norma quando souber.
```

## (d) Assert do juiz num item conhecido

- `judge_call` (bench/regua/common.py) contra http://10.100.0.111:8005/v1 OK; o juiz emite
  thinking antes do JSON e o parser pega o ÚLTIMO bloco JSON válido (corrigido o caso de
  JSON aninhado `{"resposta": {"fatos": [...]}}` observado em 2/384 itens do gabarito).
- Gabarito completo construído: **384/384 itens, média 4.10 fatos/item, 0 vazios**
  (`data/gabarito_384.jsonl`, fonte = answer+evidence do dataset, juiz 27B).
- Scoring E2E validado no smoke (40 respostas julgadas, 0 erros): item conhecido
  `34763c0a…` (oráculo) → coverage 1.0, aprovado, sem alucinação — resposta de fato contém
  "avaliar a proposta" + "efetuar modificação" do gabarito. ✓
