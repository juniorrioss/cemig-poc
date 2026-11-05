# bench/lfm3_judge — decision model × LLM-as-judge

Compara o juiz atual (Qwen3.8-27B na régua honesta) com dois decision models, o
**pplx-decider-v1.1-27b** e o **vela3-flash**. Os três julgam os mesmos itens com verdade
conhecida. Resultados e recomendação estão em **[RESULTS.md](RESULTS.md)**; a tabela-mestra
fica em `data/consolidation.json`.

| arquivo | papel |
|---|---|
| `serve/decider_server.py` | Servidor HTTP com lote entre itens. `/decide` atende o pplx (usa o `autojev.DecisionModel` oficial com atenção não-causal; sem vLLM). `/system_one` atende o vela (agrupa as perguntas do mesmo state). Ordem de lote determinística. |
| `rubric_j.py` | Rubrica portável (choice/noul/score) da régua honesta: verdict, alucinação contra o trecho-ouro, recusa, fatos, grounded, citação e context_sufficient. Inclui o eixo extra `span` (só Vela). |
| `variants_j.py` | 12 tipos de variante sintética. Evolui o `jaopredos/eval-vela` com negação, deve→pode, tautologia, extra_true e off-topic da mesma norma. |
| `build_sets.py` | Monta os conjuntos V (variantes), R (respostas reais da F6 com veredito do Qwen) e B (benchmark_mobile_results: 4 retrievers + o nosso; árbitro gpt-oss-120b). |
| `run_decider.py` / `run_qwen.py` | Rodam os julgadores com checkpoint. A opção `--rep` mede determinismo. `--lean` mede o perfil de produção. |
| `analyze.py` | Calcula AUROC, κ (limiar calibrado em split), árbitro, triagem, determinismo, velocidade e o comparativo de retrievers. |
| `disagreements.py` | Amostra discordâncias para leitura manual. Os rótulos ficam em `data/disagreements_labels.json`. |

Interpretador local: `classifier/.venv`. Para reproduzir, `make sets pplx vela qwen analyze`
(servidores na H100, ver §Reprodução do RESULTS). Os brutos são versionados como `.gz`;
`analyze.py` lê esses arquivos de forma transparente.

-- revisado segundo as diretrizes da fábrica
