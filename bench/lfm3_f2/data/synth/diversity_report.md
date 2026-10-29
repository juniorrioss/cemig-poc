# Relatório de diversidade — F2 geração sintética

- QAs aprovados pelo crítico: 5526
- Removidos por contaminação (J>=0.5 vs benchmark): 0
- Near-dups intra removidos (J>=0.8): 6 (0.1%)
- QAs finais: 5520; parafraseios: 98; mistura: {'raw': 49, 'paraphrase': 98, 'qa': 5520}

| prefixo | entropia (bits) | máx possível | distintos | benchmark ref |
|---|---|---|---|---|
| n=3 | 11.006 | 11.727 | 3390 | 7.664 |
| n=4 | 11.715 | 12.07 | 4300 | 9.381 |
| n=5 | 12.091 | 12.252 | 4877 | 9.99 |

- persona_dist: {'engenheiro': 576, 'auditor': 608, 'socratic': 620, 'supervisor': 612, 'cipeiro': 582, 'trabalhador_leigo': 676, 'tecnico_seg': 652, 'eletricista': 655, 'comprador': 537, '...': 2}
- cenario_dist: {'conformidade': 634, 'incidente': 790, 'aplicacao': 174, 'por_que': 135, 'e_se': 138, 'limite': 173, 'discussao': 1011, 'duvida_direta': 807, 'treinamento': 916, 'situacao_campo': 740, '...': 2}
- style_dist: {'com_ref': 2737, 'sem_ref': 2783}
- origin_dist: {'expand': 4900, 'socratic:aplicacao': 174, 'socratic:por_que': 135, 'socratic:e_se': 138, 'socratic:limite': 173}
- per_nr: {10: 1207, 18: 3052, 34: 1261}
- cobertura por unidade: {'NR-10': {'units': 172, 'covered': 162, 'coverage': 0.942}, 'NR-18': {'units': 472, 'covered': 446, 'coverage': 0.945}, 'NR-34': {'units': 198, 'covered': 185, 'coverage': 0.934}}
