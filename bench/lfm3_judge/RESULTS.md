# RESULTS — decision model × LLM-as-judge (`bench/lfm3_judge/`)

**Pedido do capitão:** hoje confiamos às cegas no Qwen3.8-27B para dizer se uma resposta está certa
ou errada. Ele queria (1) evoluir o pipeline de decision model do `jaopredos/eval-vela`,
(2) comparar o **pplx-decider-v1.1-27b** (prioridade) e o **vela3-flash** com o Qwen-as-judge
mantendo tudo determinístico, (3) medir se dá para julgar dezenas de perguntas no mesmo tempo e
(4) (adendo) confirmar por outro caminho o veredito do smoke de retrievers (v4 ≥ MobileRAG+).

Tabela-mestra: `data/consolidation.json`. Leitura das discordâncias: `data/disagreements_*.jsonl`
e `data/disagreements_labels.json`. Janela de GPU: H100 GPU 0, de 01:33 a 04:00 UTC (~2,5 h das 12 h
liberadas). A GPU foi **devolvida vazia** (0 MiB) logo depois da bateria.

## TL;DR (recomendação)

**COMPLEMENTAR, não substituir. O decision model entra como segundo juiz determinístico e
auditor do Qwen. O Qwen fica como juiz oficial da régua honesta.** Se for para escolher um
decider só, fico com o **vela3-flash**: a hipótese do capitão se confirma com número, e ele
supera o modelo maior onde há verdade independente.

| julgador | AUROC variantes (verdade por construção) | κ vs Qwen-régua (real F6, n=344) | κ vs **gpt-oss-120b** (árbitro, n=608) | determinismo 2 execuções | itens/s (perfil de produção) | GPU |
|---|--:|--:|--:|---|--:|---|
| **Qwen27b-régua** (atual) | 0,940 | — | 0,706 [0,65–0,76] | veredito 99,0% igual, texto 32–36% igual; temp 0,7: 91,3% (κ 0,82) | 6,3 (12 workers) · 12,5 (48) · 17,2 (128) | 1 H100 dedicada (vLLM FP8, compartilhada) |
| **pplx-decider-v1.1-27b** | **0,997** | **0,783** [0,72–0,85] | 0,684 (limiar transferido) · 0,742 (calibrado) | **100% bit-idêntico** (22.873 decisões) | 3,9 (19,6 decisões/s) | 49–65 GB de uma H100, bf16 |
| **vela3-flash (9B)** | 0,992 | 0,720 [0,64–0,80] | **0,790** [0,74–0,84] (transferido) · **0,853** (calibrado) | **100% bit-idêntico** (15.624 decisões) | **7,8** (39,2 decisões/s) | 18,6 GB |
| *ref.: rubrica binária do Qwen (outro time)* | — | — | 0,858 | — | — | — |

O que esses números dizem:

1. **Os decision models discriminam muito melhor quando a verdade é conhecida.** Nas 1.191
   variantes sintéticas (12 tipos de defeito) o gate do decider tem AUROC 0,99+ contra 0,94 da
   régua com Qwen. O Qwen deixa passar **18% das tautologias** (resposta que só repete a pergunta),
   **33% dos itens trocados** (NR-10 → NR-12) e **45% de "deve → pode"**. Os deciders deixam
   passar 1%, 9–13% e 46–49%, respectivamente. Nenhum dos três pega bem a troca de obrigação.
2. **Em respostas reais a vantagem some e o Qwen é mais criterioso.** Contra o árbitro
   independente (gpt-oss-120b), o Qwen-régua fica em κ 0,71 e o pplx com limiar transferido em
   κ 0,68, um empate técnico. O vela3-flash chega a **κ 0,79**, acima dos dois. Na leitura
   manual de 46 discordâncias pplx × Qwen, **o Qwen tinha razão em 26, o decider em 12 e 8 eram
   ambíguas** (detalhe no §4).
3. **Determinismo: os deciders são bit-idênticos com uma condição.** A composição do lote
   precisa ser a mesma. Com lote 8 contra lote 32, só 44% das decisões ficam bit-idênticas
   (Δprob máx. 0,037), mas o veredito muda em só 2 de 688 itens. Com lote fixo, 100%. O Qwen
   em temperatura 0 já não reproduz o texto (só 32–36% igual) e o veredito flutua em ~1% dos
   itens. Em temperatura 0,7 flutua em 8,7%.
4. **Velocidade: a promessa de "dezenas no mesmo tempo" NÃO se confirma contra o Qwen bem
   servido.** O Qwen no vLLM (FP8, batching contínuo) julga 12–17 itens/s com 48–128
   requisições paralelas. O decider precisa de 5 decisões por item (gate) e roda via
   transformers, porque não há suporte no vLLM. Faz 3,9 itens/s (pplx) e 7,8 itens/s (vela).
   A vantagem real é outra: uma decisão sai de **um único forward**, sem gerar tokens, com
   probabilidade calibrada e sem parser (o Qwen deu `parse_fail` em casos reais). O vela cabe
   em 19 GB.

## 1. O que foi feito

| etapa | detalhe |
|---|---|
| Base | `jaopredos/eval-vela@0993e1e`: rubrica de 11 eixos (choice/noul/score/span) para comparar 4 retrievers, validada por variantes sintéticas (AUROC ≥0,85, spread ≤0,15). Dados: `UFGCEMIGONA/benchmark_mobile_results@faeb942` (o `responses.parquet` dele é um recorte desse dataset). |
| Arquitetura verificada antes de rodar | **pplx-decider-v1.1-27b**: backbone `Qwen3_5Model` + `readout.safetensors` [255×5120] sobre códigos de opção, **atenção não-causal** nas camadas full-attention e temperatura de calibração 1,0087. **Não serve no vLLM** (o card exige export próprio com atenção preservada). Rodou via transformers 5.17 com o `autojev.model.DecisionModel` oficial, num servidor de lote próprio (`serve/decider_server.py`), porque o servidor oficial atende 1 requisição por vez. **vela3-flash**: `d3-flash@8b35c853` congelado + LoRAs com gate, `trust_remote_code`. |
| **Assimetria (adendo do capitão, confirmada no código)** | o pplx aceita só `choice`/`noul`/`score` (`autojev.types.Question`), **não existe `span`**. O vela aceita os quatro. A rubrica comparada usa **apenas eixos portáveis**, idênticos nos dois. O `span` ('unsupported' da resposta contra o trecho-ouro) entra **só no Vela**, como eixo extra (AUROC 0,895), fora do κ. |
| Infra | Container irmão `cemig_decider_walcy` (`--gpus device=0`), artefatos em `/raid/aluno_walcy/cemig_mobile/decider`, venv py3.12 com as dependências pinadas do card (torch 2.14 cu130, transformers 5.17, fla 0.5.2). Portas 8006 (pplx) e 8007 (vela). Os pesos foram baixados com token só em env e o arquivo de token foi apagado no fim. |

### 1.1 Rubrica adaptada à régua honesta (`rubric_j.py`)

A rubrica do eval-vela julgava retrievers. A nossa julga **respostas** com as mesmas perguntas
da régua honesta: "contém a informação pedida?", alucinação, recusa e citação (informativa,
fora do gate, por decisão do capitão).

| eixo | tipo | state | papel |
|---|---|---|---|
| `verdict` | choice {correta, parcialmente, incorreta, **recusa**} ×2 paráfrases | pergunta+referência+trecho oficial+resposta | gate |
| `hallucination` | noul ×2 | pergunta+trecho oficial+resposta | gate ("contradiz ou inventa; omitir não conta", igual ao prompt da régua) |
| `refusal` | noul | pergunta+resposta | gate (substitui a regex `looks_like_refusal`) |
| `fact` | noul por fato do `gabarito_384` | pergunta+resposta | cobertura da régua (média de p(fato)) |
| `answers_question`, `grounded` (contra o contexto recuperado), `citation` | noul | — | diagnóstico |
| `context_sufficient` | noul ×2 | pergunta+referência+contexto | avalia o **retriever** (tarefa extra) |
| `span_unsupported` | span (só Vela) | contexto=trecho-ouro, answer=resposta | eixo extra |

Gate contínuo: `verdict_EV × (1−p_aluc) × (1−p_recusa)`. É o análogo multiplicativo de
"cobertura ∧ ¬alucinação ∧ ¬recusa". O limiar é calibrado em metade dos `base_qa_id` do R
contra o Qwen e aplicado no resto (κ honesto). Para o árbitro, o limiar é **transferido do R**
e nunca vê o gpt-oss.

### 1.2 Fragilidades do eval-vela que foram corrigidas

| fragilidade (relatório/código do eval-vela) | correção aqui |
|---|---|
| AUROC alto puxado por defeitos triviais: só 13 `wrong_number` e 9 `wrong_item` em 549 variantes; o grosso era gold×off_topic/refusal | 1.191 variantes, 12 tipos, incluindo **negação**, **deve→pode**, **tautologia**, **acréscimo verdadeiro** (`extra_true`, para não confundir "acrescentou" com "alucinou"), off-topic **da mesma norma** e 4 redações de recusa (o original usava 1 frase fixa) |
| `correctness` sem opção de recusa (recusa caía em "incorreta" ou "parcial" por acaso) | `verdict` com opção `recusa` explícita |
| cobertura por choice de 4 níveis com valores arbitrários (0,33/0,67) | 1 `noul` por fato do gabarito (a pergunta binária da régua honesta) |
| grounding julgado contra o contexto do retriever, que pune a resposta pelo erro do retriever | alucinação julgada contra o **trecho-ouro**. `grounded` contra o contexto fica só como diagnóstico |
| 1 modelo por eixo e sem 2º decision model | pplx e vela com a **mesma** rubrica; span separado |
| sem medição de determinismo nem de velocidade em lote | rep2 bit a bit; servidor com lote entre itens; itens/s medidos |

## 2. Conjuntos com verdade conhecida (`build_sets.py`, `data/sets/`)

| conjunto | n | verdade |
|---|--:|---|
| **V** variantes | 1.191 (150 perguntas × 12 tipos aplicáveis) | por construção, com o contexto = evidência-ouro (oráculo, como no original) |
| **R** respostas reais | 688 = F6 recorte344 × {op, ctrl}, `f4-r32-ep2` | veredito Qwen salvo no parquet da F6 (reexecutado: κ 0,975 com o original) |
| **B** benchmark | 300 perguntas (75 grupos × 4 vocabulários) × 5 retrievers = 1.500. Respostas do Energy-Mobile para naive/bm25/mobilerag/mobilerag_plus; só contexto para `ours` | **gpt-oss-120b** (veredito original do dataset, árbitro independente) + rubrica binária do Qwen (outro time) |

Árbitro, com limitação registrada: o "humano-proxy" é o gpt-oss-120b, um modelo de outra
família que nenhum dos julgadores comparados viu. Ele julga sozinho, então é árbitro e não
verdade absoluta. A leitura manual (§4) cobre o resto.

## 3. Resultados

### 3.1 Variantes (AUROC, verdade por construção)

| eixo | pplx | vela | Qwen-régua |
|---|--:|--:|--:|
| **gate** (bom × mau) | **0,997** | 0,992 | 0,940 |
| verdict | 0,998 | 0,990 | — |
| hallucination | 0,976 | 0,879 | 0,925 |
| refusal | 1,000 | 1,000 | — |
| cobertura por fatos | 0,999 | 0,999 | — |
| grounded | 0,989 | 0,980 | — |
| span_unsupported (só Vela) | — | 0,895 | — |
| spread entre paráfrases (verdict / aluc.) | 0,012 / 0,024 | 0,013 / 0,068 | — |

Taxa de aprovação por variante (o que deveria ser 1 para gold/paraphrase/extra_true e 0 para o resto):

| variante (n) | Qwen-régua | pplx (gate médio) | vela (gate médio) |
|---|--:|--:|--:|
| gold (150) / paraphrase (100) / extra_true (53) | 0,97 / 0,99 / 1,00 | 0,99 / 0,98 / 0,98 | 0,96 / 0,96 / 0,92 |
| hallucination_added (150) | 0,03 | 0,00 | 0,10 |
| negation (104) | 0,03 | 0,01 | 0,04 |
| **swap_obligation** deve→pode (97) | **0,45** | **0,46** | **0,49** |
| **tautology** (150) | **0,18** | 0,01 | 0,01 |
| **wrong_item** (15) | **0,33** | 0,09 | 0,13 |
| wrong_number (28) | 0,07 | 0,01 | 0,05 |
| refusal / off_topic | 0,00 / 0,01 | 0,00 / 0,00 | 0,00 / 0,00 |

### 3.2 Respostas reais e árbitro independente

| comparação (split de teste) | κ | acurácia | aprovação (decider / ref.) |
|---|--:|--:|---|
| pplx × Qwen-régua — R (344) | **0,783** [0,72–0,85] | 0,904 | 68,6 / 65,4% |
| vela × Qwen-régua — R (344) | 0,720 [0,64–0,80] | 0,875 | 67,4 / 65,4% |
| pplx × vela — R / B | 0,840 / 0,829 | — | — |
| **Qwen-régua × gpt-oss** — B (608) | 0,706 [0,65–0,76] | 0,859 | 66,5 / 55,9% |
| **pplx × gpt-oss**, limiar do R | 0,684 [0,63–0,74] | 0,849 | 68,1 / 55,9% |
| **vela × gpt-oss**, limiar do R | **0,790** [0,74–0,84] | **0,898** | 62,8 / 55,9% |
| vela × gpt-oss, limiar calibrado no B | 0,853 | 0,928 | — |
| ref.: rubrica binária do Qwen (outro time) × gpt-oss | 0,858 | 0,929 | 52,8 / 55,9% |

Em todas as discordâncias decider × Qwen-régua no B (limiar transferido), quem concorda com o gpt-oss:

| | discordâncias | decider certo | Qwen certo |
|---|--:|--:|--:|
| pplx | 62 | 28 | 34 |
| **vela** | 64 | **44** | 20 |

Leituras:
- A **régua do Qwen** (fatos lexicais + desempate) é **mais leniente** que o gpt-oss (+10,6 p.p.
  de aprovação). A **rubrica binária** do outro time, com o mesmo Qwen, concorda muito mais com
  o gpt-oss (κ 0,86). A diferença está no **desenho da régua**, não no modelo.
- O vela concorda mais com o árbitro do que o próprio Qwen-régua. O pplx concorda mais com o Qwen.
- **Triagem** (decider decide fora da faixa de dúvida, o Qwen decide dentro dela; faixa calibrada
  contra o Qwen): o decider resolve 80–86% dos itens e o Qwen só o resto, com κ 0,66 (pplx) /
  0,72 (vela) contra o gpt-oss. Com o vela, empata com o Qwen sozinho (0,71) usando só ~20% das
  chamadas ao 27B; com o pplx, perde (0,66).

### 3.3 Determinismo (rodar 2×)

| julgador | unidade | idêntico | observação |
|---|---|--:|---|
| pplx (lote 8 fixo) | 22.873 decisões V+R | **100%** bit a bit | Δprob máx. 0 |
| vela (lote 16 fixo) | 15.624 decisões V | **100%** bit a bit | Δprob máx. 0 |
| pplx lote 8 × lote 32 | 3.440 decisões R | 44,2% bit a bit | Δprob máx. 0,037. **Veredito igual em 686/688**: o padding altera o bf16 |
| Qwen-régua temp 0 (12 workers) | 1.879 itens V+R | veredito 99,0% · texto 32–36% | κ rep1×rep2 0,98 |
| Qwen temp 0, 48 / 128 workers × 12 workers | 688 itens R | veredito 98,4 / 98,0% | a concorrência muda o batching do vLLM e o veredito |
| Qwen temp 0,7 | 688 itens R | veredito 91,3% · texto 0,15% | κ 0,82 |

### 3.4 Velocidade e custo

| julgador | configuração | itens/s | decisões/s | memória |
|---|---|--:|--:|---|
| Qwen-régua | vLLM FP8, 12 / 48 / 128 workers | 6,3 / 12,5 / 17,2 | — | H100 inteira (GPU 3, servidor compartilhado) |
| pplx | 5 decisões/item (gate), lote 8 / 32 | 3,9 / 3,6 | 19,6 / 18,0 | 49 GB de pesos, pico 53–65 GB |
| pplx | rubrica completa (~12 decisões/item, contextos de ~1.700 chars) | 1,6 | 19,0 | — |
| vela | 5 decisões/item (gate) | **7,8** | **39,2** | **18,6 GB** |
| vela | rubrica completa + span | 3,6 | 45,1 | — |

O decider não é mais rápido que um Qwen bem servido. Os gargalos: transformers sem kernels
otimizados (`causal_conv1d` não compila no container CUDA 12.8 contra torch cu130) e sem
batching contínuo/paged-attention. O ganho de vazão viria de um export para vLLM (pendente no
card do pplx) ou de rodar o vela em fila de múltiplas réplicas (cabem 4 numa H100).

## 4. Discordâncias lidas à mão (46 casos, pplx × Qwen, evidência-ouro ao lado)

`data/disagreements_labels.json`: 30 do R e 16 do B (no B, o gpt-oss é o 3º voto).

| direção | Qwen certo | decider certo | ambíguo |
|---|--:|--:|--:|
| R: decider aprova, Qwen reprova (15) | 10 | 3 | 2 |
| R: Qwen aprova, decider reprova (15) | 6 | 3 | 6 |
| B: decider aprova, Qwen reprova (8) | 7 | 1 | 0 |
| B: Qwen aprova, decider reprova (8) | 3 | 5 | 0 |
| **total** | **26** | **12** | **8** |

Padrões encontrados:
- **Onde o decider erra.** (a) Inconsistências internas sutis, como "dispensado da alínea d…
  logo **não pode** ter ligação direta" (4 casos no R, todos aprovados pelo decider). O Qwen
  pega, porque raciocina. (b) Acréscimos de outro item tomados como contradição. (c) **Pune
  citação de NR errada com conteúdo correto** como alucinação (NR-18 no lugar de NR-31). A
  régua tirou citação do gate, então nesses casos o Qwen está alinhado à régua.
- **Onde o Qwen erra.** (a) **Falso positivo da regex de recusa** (`looks_like_refusal`):
  frases como "a norma não especifica, nesse trecho, outras exigências…" viram recusa. São 11
  itens do R (1,6%) que a régua reprova sendo corretos, ou seja, **a F6 subestima a aprovação
  em ~1,6 p.p.** O eixo `refusal` do decider (AUROC 1,0) resolveria isso. (b) Casamento lexical
  frouxo dos fatos aprova respostas que não dizem a ação pedida (caldeira sem TH, benzeno
  respondido com máquinas). (c) Inconsistência: aprova "NR-30/PGRTA" num item e chama a mesma
  resposta de alucinação em outro. (d) `parse_fail` do JSON.

## 5. Tarefa extra: 5 retrievers pela via decision model (2ª confirmação do smoke)

Mesmas 300 perguntas do benchmark, mesmo eixo `context_sufficient` ("o contexto recuperado
contém o necessário para chegar à resposta de referência?") nos dois deciders. O eixo foi
validado separando contexto com evidência (cobertura ≥0,9) de contexto sem evidência (<0,3):
AUROC 1,00. Contra o veredito do gpt-oss sobre a resposta do Energy-Mobile, deu AUROC 0,99.
O `ours` é o ponto de operação F6 (`thr0.7_f2+man1.75+margin`, top-5, orçamento de 5.000
chars ≈ a mediana dos contextos deles).

| retriever | pplx: contexto suficiente | vela: contexto suficiente | cobertura lexical da evidência | resposta: gpt-oss | resposta: Qwen-régua | resposta: gate pplx / vela |
|---|--:|--:|--:|--:|--:|--:|
| naive | 0,675 | 0,658 | 0,829 | 0,668 | 0,700 | 0,572 / 0,557 |
| bm25 | 0,641 | 0,642 | 0,796 | 0,630 | 0,637 | 0,573 / 0,552 |
| mobilerag | 0,627 | 0,615 | 0,805 | 0,648 | 0,670 | 0,552 / 0,524 |
| mobilerag_plus | 0,687 | 0,673 | 0,850 | **0,698** | **0,717** | **0,598 / 0,574** |
| **ours (v4 + F6)** | **0,764** | **0,743** | **0,859** | n/d* | n/d* | n/d* |

Pareado ours × mobilerag_plus (mesmas 300 perguntas, empate se |Δ| ≤ 0,05):

| julgador | Δ médio | vitórias / derrotas / empates | teste de sinal |
|---|--:|---|--:|
| pplx | **+0,077** | 80 / 55 / 165 | p = 0,038 |
| vela | **+0,070** | 101 / 65 / 134 | p = 0,006 |
| cobertura lexical | +0,009 | 82 / 76 | p = 0,69 |

Por tipo de gabarito (pplx; o vela segue o mesmo padrão):

| | naive | mobilerag_plus | **ours** |
|---|--:|--:|--:|
| norma (240) | 0,653 | 0,677 | **0,822** |
| manual (60) | 0,762 | 0,728 | **0,535** |

**Veredito: o smoke se confirma por esta via independente.** No 1º estágio (contexto que chega ao
leitor), os dois decision models dão **v4/F6 > MobileRAG+** com significância (p 0,006–0,04),
e a 1ª da ordem pelos dois deciders e pelo gpt-oss sobre as respostas é a mesma
(mobilerag_plus > naive > bm25 ≈ mobilerag). Duas ressalvas: (1) a vantagem é toda **em
norma**; **em manual o nosso perde** (0,54 contra 0,73), o que é coerente com a F6 (manual
hit@5 47%, o ponto de operação protegeu as normas); (2) a cobertura lexical sozinha não separa
os dois (p 0,69). O sinal vem do julgamento semântico. *Não há resposta do Energy-Mobile para
o nosso contexto (o leitor LiteRT não roda neste host), por isso as colunas de resposta ficam
vazias para `ours`.

## 6. Recomendação fundamentada

1. **Manter o Qwen como juiz oficial da régua honesta.** Em respostas reais ele erra menos que o
   pplx na leitura manual (26 × 12) e empata com o pplx contra o árbitro. Trocar o juiz agora
   quebraria a comparabilidade com todas as fases anteriores, sem ganho medido.
2. **Adotar o vela3-flash como 2º juiz determinístico** (não o pplx). Motivos: maior acordo com
   o árbitro independente (κ 0,79 contra 0,71 do Qwen e 0,68 do pplx), AUROC 0,99 nas variantes,
   2× mais rápido que o pplx, 19 GB contra 50–65 GB, bit-idêntico e com o eixo `span` que aponta
   o trecho alucinado. **A hipótese do capitão ("o vela3-flash já deve ser suficiente") se
   confirma.** O modelo maior não paga: só é melhor nas variantes sintéticas (0,997 × 0,992) e
   no acordo com o Qwen (0,78 × 0,72).
3. **Uso concreto:** (a) **auditoria**: os itens em que o vela e o Qwen discordam (~10%) são a
   fila de revisão; (b) **triagem**: o vela decide os ~80% confiantes e o Qwen só a faixa de
   dúvida, com qualidade equivalente contra o árbitro (κ 0,72 × 0,71) e ~5× menos chamadas ao 27B;
   (c) **corrigir a régua** já: trocar a regex `looks_like_refusal` pelo eixo `refusal` do
   decider (falso positivo de 1,6%) e endurecer o casamento lexical de fatos (o Qwen aprova 18%
   das tautologias sintéticas).
4. **Não vender "dezenas de vezes mais rápido".** O decider é determinístico e barato em
   memória, mas não é mais rápido que o vLLM do Qwen com batching. Para escalar: réplicas do
   vela (4 por H100) ou um export do pplx para vLLM.

## 7. Limites

- O árbitro é um único modelo (gpt-oss-120b) e não humano. A leitura manual cobriu 46
  discordâncias, só do pplx (o vela não foi lido à mão).
- O conjunto R é uma única célula do mesmo modelo (`f4-r32-ep2`), com distribuição de erros
  própria. O B usa o leitor Energy-Mobile (de outro time).
- Os limiares do gate foram calibrados contra o Qwen. O "limiar transferido" para o B evita
  calibrar no árbitro, mas o gate multiplicativo é um desenho nosso, não da biblioteca.
- `swap_obligation` (deve→pode) continua não detectado por nenhum dos três (~45–49% passa).
- A velocidade dos deciders é **pessimista** (sem `causal_conv1d` e sem graphs). A do Qwen
  depende de um servidor compartilhado.
- O `ours` no B é avaliado só no contexto (sem resposta). O orçamento de 5.000 chars é uma
  aproximação do corte de 1.664 tokens do Energy-Mobile.

## Relatório de rubricas (histogramas)

`report_rubricas/RUBRICAS.md`: grade pipelines × eixos com a distribuição das notas (vela e pplx), mais os dados agregados em parquet e CSV.

## Reprodução

```bash
# H100 (container com --gpus device=0; venv py3.12 com requirements do card do pplx)
python serve/decider_server.py --backend pplx --checkpoint <pplx-decider-v1.1-27b> --port 8006
python serve/decider_server.py --backend vela --checkpoint <vela3-flash> --port 8007
# local (classifier/.venv)
make -C bench/lfm3_judge sets pplx vela qwen analyze disagreements
```
`data/sets/*.jsonl.gz` e `data/raw/*.jsonl.gz` versionam os itens e todas as saídas brutas.
`analyze.py` reproduz `consolidation.json` byte a byte só a partir dos `.gz`. Os insumos
gitignored ficam em `cache/` (o parquet do `benchmark_mobile_results`, `rank_*.jsonl` e
`manuals.db` da F5).

-- revisado segundo as diretrizes da fábrica
