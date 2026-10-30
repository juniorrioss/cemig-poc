# data/eval/ — diretório CANÔNICO dos resultados de avaliação (F0 + F2)

Um parquet por **checkpoint × célula**: `lfm3-<checkpoint>-<celula>.parquet`
(`checkpoint`: `baseline` = F0 as-is; `A-r8-ep5` = braço A LoRA r8 época 5;
`B-base-r8-ep4-sum` = task vector; `C-k2v-...` = RL).

Schema (transparência total — ninguém compara células sem saber o que o
modelo viu):

| grupo | colunas |
|---|---|
| benchmark | id, base_qa_id, question, answer_gold, question_type, source_id, vocabulary_level, query_features, source_file, evidence_text |
| saída do modelo | answer_model (resposta final avaliada), raw_output (saída INTEGRAL: thinking+resposta), thinking, n_gen_tokens, stopped_eos |
| veredito (régua honesta) | coverage, n_facts, tautology, hallucination, refused, truncated, refusal_correct, refusal_undue, citation_present, approved, answered_correct_anyway, judge_note |
| protocolo | context_given (texto EXATO do contexto; vazio em closed-book), context_type (none/oracle/ranking_1de5/distractor_only), system_prompt_sha + system_prompt_ref (texto em system_prompts.json), checkpoint, cell, gen_params (JSON) |

Notas:
- `context_given` é reconstruído deterministicamente (embaralhamento por
  seed sha256(f"42:{id}") — idêntico ao da geração; ver
  `bench/lfm3_f0/common_f0.py:build_cell_context`).
- O baseline (F0) cobre as 1536; as células F2 cobrem o recorte das 6 normas
  do experimento (344 = tratadas NR-10/18/34 + controle NR-13/31/33).
- Gerador: `eval/export_parquet.py` (`--baseline`, `--all`, ou `--tag+--cell`).
