# Rubricas do decision model por pipeline avaliado: histogramas

**Figura principal: `rubricas_vela.png`** (vela3-flash). A mesma grade com o
pplx-decider-v1.1-27b está em `rubricas_pplx.png`, para comparação.

Como ler a figura:
- **Linhas** são os pipelines avaliados:
  - os 5 retrievers do benchmark B (300 perguntas do `benchmark_mobile_results`, respostas do Energy-Mobile): naive, bm25, mobilerag, mobilerag_plus e **ours** (v4/F6-op);
  - as 2 células reais do NOSSO pipeline (conjunto R, F6 recorte 344, `f4-r32-ep2`): op e ctrl.
- **Colunas** são os eixos da rubrica (`rubric_j.py`).
- **Eixo X** é a nota em [0,1], em 10 bins, onde **1 é sempre o bom**.
- Na margem esquerda de cada linha estão as taxas binárias de referência: aprovação do
  Qwen-régua e, no B, "correct" do gpt-oss-120b.
- O título de cada célula mostra a média (μ) e o n.

Dados:
- `scores_long.parquet`: uma linha por julgador, pipeline, item e eixo, com a nota.
- `summary.csv`: n, média, mediana, p10, p90 e fração ≥0,5 por célula.
- `hist_counts.csv`: contagens por bin.
- `verdicts_by_pipeline.csv`: Qwen, gpt-oss e gate binário (limiar calibrado no R).
- Para regenerar: `python make_report.py` (requer matplotlib, pandas e pyarrow; o venv do classifier não tem).

## Eixos

| coluna | pergunta ao decision model (tipo) | nota |
|---|---|---|
| contexto suficiente | "O contexto recuperado contém o necessário para chegar à resposta de referência?" (noul ×2 paráfrases) | p(sim). Avalia o **retriever**, sem a resposta. Único eixo disponível para `ours` |
| veredito | "A resposta está correta frente à referência e ao trecho oficial?" (choice: correta / parcial / incorreta / recusa ×2) | valor esperado: correta=1, parcial=0,5, incorreta=recusa=0 |
| sem alucinação | "Contradiz o trecho oficial ou inventa exigência, número ou procedimento? Omitir não conta." (noul ×2) | 1 − p(sim) |
| respondeu | "Recusa ou diz que a informação não está disponível?" (noul) | 1 − p(sim) |
| cobertura de fatos | "A resposta comunica este FATO?", um noul por fato do `gabarito_384` | média de p(fato), equivalente à cobertura da régua honesta |
| responde à pergunta | "Trata do que foi perguntado e traz informação além de repetir a pergunta?" (noul) | p(sim) |
| apoiada no contexto | "Todas as afirmações técnicas estão apoiadas nos trechos do contexto?" (noul) | p(sim). Contexto **recuperado**, não o ouro: mede o leitor dado o contexto |
| cita NR/item | "Cita explicitamente a norma ou o item?" (noul) | p(sim). **Informativo, fora do gate** (decisão do capitão) |
| span (só Vela) | spans `unsupported` da resposta contra o trecho-ouro (span adapter nativo) | 1 − fração de caracteres marcados. O pplx não tem `span`; no B não foi rodado |
| GATE | veredito × (1 − p_aluc) × (1 − p_recusa) | análogo contínuo do "aprovado" da régua. O limiar (vela 0,17, pplx 0,11) vem da calibração contra o Qwen |

## Leitura dos histogramas

1. **Quase todos os eixos são bimodais**, com massa nos bins 0 e 1. O decision model decide com
   confiança e a média esconde isso: o μ de cada célula é, na prática, a **fração de itens
   bons**. O meio da distribuição (0,3 a 0,7) é a faixa de dúvida que a triagem manda ao Qwen.
2. **Contexto suficiente: ours fica à frente.** No vela, μ=0,74 e 74% dos itens ≥0,5, contra
   0,66–0,67 do naive e do mobilerag_plus e 0,62–0,64 do mobilerag e do bm25. No pplx a ordem é
   a mesma (0,76 × 0,69). A vantagem vem das normas, como descrito no §5 do RESULTS; em manual o
   nosso perde.
3. **Entre os 4 retrievers do benchmark, a ordem das respostas é estável em todos os eixos**:
   mobilerag_plus ≥ naive > mobilerag ≈ bm25. É a mesma ordem do gpt-oss e do Qwen. O **bm25**
   tem o maior pico em 0 em **respondeu** (μ 0,76): o Energy-Mobile "explica o que falta"
   quando o BM25 não acha nada nas perguntas leigas.
4. **Apoiada no contexto ≈ 0,93 nos 4 retrievers**: o leitor do outro time quase não inventa
   fora do contexto. Os erros dele vêm de contexto errado ou incompleto. Mesmo assim, "sem
   alucinação" contra o trecho-ouro fica em 0,63–0,67.
5. **Nosso pipeline real, F6 op × ctrl**: o op desloca massa do bin 0 para o 1 no veredito
   (0,57 → 0,68), na cobertura de fatos (0,60 → 0,71) e em respondeu (0,74 → 0,86). Recusa
   menor e mais acerto, coerente com a F6 (Qwen 58% → 68%). Já **sem alucinação** é o eixo mais
   espalhado das duas células (μ 0,55–0,60, massa no meio). O `f4-r32-ep2` acrescenta
   detalhes de outros itens, e o decider fica em dúvida se isso é invenção. Leia junto com a
   coluna span, que fica concentrada em ~0,9: só ~10% dos caracteres são marcados como sem
   apoio.
6. **Citação separa os dois leitores**: o nosso cita a NR em ~92% dos casos, contra ~35% do
   Energy-Mobile, que usa `[1]`/`[2]`. Está fora do gate.
7. **GATE**: é bimodal com um platô intermediário nas células reais. O platô vem da alucinação
   em dúvida (item 5). Com os limiares calibrados, a aprovação do gate acompanha a do Qwen
   (`verdicts_by_pipeline.csv`). Exemplo: op 70% (vela) × 68% (Qwen), ctrl 58% × 58%.

-- revisado segundo as diretrizes da fábrica
