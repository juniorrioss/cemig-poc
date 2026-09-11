# M3 — Rewriter Limpo e Verificação da Trilha LoRA (Fase 2)

## Resolução do ALERTA DE INTEGRIDADE do capitão

O relatório anterior (`finetune/README.md`) anunciava "92% de recall". A auditoria
confirma o alerta:

1. **O "92%" era a coluna ZERO-SHOT**, não o LoRA. O `finetune/results/evaluation_report.json`
   registra `zero_shot recall@2 = 92.0%` e `lora_finetuned recall@2 = 80.0%` — ou seja,
   o LoRA **REGREDIU 12 p.p.** já no relatório original.
2. **A avaliação estava contaminada por construção**: testava contra o próprio dataset
   sintético invertido (`test.jsonl`), cujas falas carregam o vocabulário do chunk
   (as `golden_keywords` incluíam `doc`, `section` e `title` literais). Por isso "92%"
   sintético vs **~14-28% real**.
3. Os pesos do adapter anterior (`adapter_model.safetensors`) **nunca foram commitados**
   — só `adapter_config.json`/tokenizer. A manchete não era reproduzível.

## Dataset descontaminado (procedência registrada)

- `finetune/gen_dataset_clean.py` gera via **vLLM do capitão** (`Qwen/Qwen3.8-27B-FP8`,
  `enable_thinking=false`) 1.378 pares em 700 chunks priorizados. Cada linha carrega
  `generator` (procedência) e `contamination_score`.
- **Descontaminação**: o prompt PROÍBE reutilizar os substantivos técnicos do trecho e
  exige paráfrase de leigo; a lista de radicais proibidos é injetada no prompt. Filtro
  pós-hoc rejeita falas com Jaccard de radicais fala-vs-chunk > 0.35. Contaminação média
  final = **0.134** (vs a trilha antiga, que copiava o vocabulário).
- O **alvo de treino (assistant)** é `canonical_terms` (termos técnicos da NORMA que um
  especialista usaria), NÃO keywords derivadas do chunk. O modelo aprende a MAPEAR fala
  leiga → norma, não a copiar.
- `finetune/validate_dataset_clean.py` valida por **execução BM25 no caminho do app**
  (95.8% de aprovação), balanceia por NR (`--cap-per-nr 80`, evita domínio da NR-12) e
  faz split por NR (holdout de normas não vistas). Ancoragem de síntese 10% (qa_v2).
- **Holdout de avaliação REAL**: sempre `corpus/qa_pairs_v2.jsonl` (151 perguntas),
  fonte distinta, nunca vista em treino/val/test.

## Gate duplo — RESULTADO: FALHOU (ambos r=8 e r=16)

`finetune/eval_rewriter_clean.py` compara zero-shot vs LoRA nas 151 reais (caminho do
app, com e sem filtro de norma detectada):

| Config | R@1 | R@2 | R@3 | R@5 |
|---|---:|---:|---:|---:|
| Zero-shot (base LFM2.5-1.2B) | 7.3% | **14.6%** | 17.9% | 21.2% |
| LoRA r=8 (melhor variante) | 4.6% | 6.0% | 7.9% | 12.6% |
| LoRA r=16 (melhor variante) | 5.3% | 9.9% | 12.6% | 16.6% |

- **Critério**: LoRA deve superar zero-shot em **≥10 p.p.** de R@2.
- **r=8**: Δ = **−8.6 p.p.** (FAIL). **r=16**: Δ = **−4.6 p.p.** (FAIL).
- Conforme ordem explícita do capitão ("se falhar → r=16 uma vez → se falhar de novo,
  PARE o treino e registre"), **o treino foi interrompido**.

### Por que o LoRA de 1.2B não supera o zero-shot no dado real
O 1.2B aprende o FORMATO (termos curtos + código de NR) mas **não infere a NR correta**
a partir de fala leiga (classificação de norma zero-shot mede 27.2%). Treinar em termos
curtos ainda degrada o BM25 quando a NR prevista está errada. É um limite de porte/tarefa,
não de hiperparâmetro — coerente com a Fase 1 (o gargalo é semântico, não de chunking).

## Fallback de porte — LFM2.5-2.6B DESCARTADO

`corpus/eval_fino.py --reasoning` mediu o `LFM2.5-2.6B-Q4_0.gguf` (o `Q4_K_M` oficial
gera lixo "666..." no build atual `434ddbb`; usar Q4_0):

| Modelo | R@2 (151) | Latência de rewrite | Veredito |
|---|---:|---:|---|
| LFM2.5-1.2B zero-shot | **14.6%** | ~0.07 s | referência |
| LFM2.5-1.2B LoRA r=16 | 9.9% | ~0.4 s | pior |
| LFM2.5-2.6B zero-shot | 7.9% | **~4.0 s** (1.010 tokens de raciocínio) | **inviável** |

O 2.6B **Instruct é um modelo de raciocínio**: sempre "pensa" antes de responder
(~1.010 tokens de geração ≈ 4 s no desktop, dezenas de segundos no S24+, estourando o
orçamento de 10 s de voz), alucina a NR ("nuclear") e acerta a norma só 46/151. Porte
maior NÃO é o gargalo; é pior.

## Conclusão da Fase 2
Nenhuma opção de rewriter on-device supera o **zero-shot do LFM2.5-1.2B** no dado real.
Recomendação: **manter o rewriter zero-shot** (já embarcado) e não introduzir LoRA no
release. Todos os artefatos de treino são preservados no branch (`adapter_r8/`,
`adapter_r16/`, datasets, caches, relatórios) com procedência para auditoria futura.

## Comandos
```bash
python3 finetune/gen_dataset_clean.py --vllm-url http://10.100.0.111:8005/v1 --max-chunks 700
python3 finetune/validate_dataset_clean.py --cap-per-nr 80
.venv-train/bin/python finetune/train_lora.py --suffix _clean --no-extra-maint --lora-r 16 --lora-alpha 32 --lr 1e-4 --epochs 3
.venv-train/bin/python finetune/eval_rewriter_clean.py --adapter finetune/adapter_r16 --out finetune/results/eval_clean_r16.json
```
