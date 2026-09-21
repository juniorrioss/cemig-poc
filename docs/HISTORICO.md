# HISTÓRICO — mapa cronológico dos experimentos da POC CEMIG

> Mapa **lean** dos resultados prévios: uma linha por experimento respondendo **que pergunta
> respondeu / veredito / onde está a evidência**. Sem repetir tabelas — só o mapa e o ponteiro.
> Ordenado cronologicamente (ordem dos commits). **Quem é o vigente**: a última entrega está em
> [`AS_IS.md`](AS_IS.md) (pipeline híbrido de tool-calling, 1.2B r128 Q4 + Nemotron ASR). Tudo
> nesta página é **histórico** — parte foi **superado** (indicado por quem), parte permanece como
> **decisão/fundação** que o vigente ainda usa.

Legenda de status:
- **VIGENTE** — o resultado ainda vale e está no app / é a base do AS_IS.
- **FUNDAÇÃO** — decisão de plataforma reusada por todo o resto (não superada, não é "o vigente").
- **SUPERADO por X** — outra frente entregou um resultado melhor/mais completo.
- **DESCARTADO** — medido e rejeitado por evidência (resultado negativo útil).

| # | Experimento (pasta) | Pergunta que respondeu | Veredito | Status | Evidência |
|--:|---|---|---|---|---|
| 1 | `corpus/` | Como ingerir/chunkar/indexar 36 NRs para RAG offline? Chunk grosso ou fino? | Índice FTS5 grosso `index_hf_36nr.db`; **fino piora o recall** (grosso R@2 teto 77,5% vs fino 51%). | **FUNDAÇÃO** | `corpus/README.md`, `corpus/README_M3.md` |
| 2 | `bench/` (v1/v2) | Que SLM ≤1,2B cabe na voz? Tool-calling vs RAG clássico vs Modo C? | LFM2.5 1.2B viável; abismo lexical (fala bruta recupera 20,8%); topk2 acelera \~30%. | SUPERADO (bancada base p/ tudo) | `AGENTS.md` (bench), `bench/` |
| 3 | `asr/` | Qual ASR offline PT-BR para campo? | **Whisper Base Q5_1** recomendado p/ S21 (197 MB); Nemotron 3.5 melhor em termos técnicos (91,9%) mas 792 MB. | **FUNDAÇÃO** (Nemotron viria a ser o vigente do S24+) | `asr/README.md` |
| 4 | `finetune/` (rewriter LoRA) | Fine-tune do Turno-1 rewriter destrava o retrieval? | **NÃO** — regrediu no holdout limpo (r8 −8,6 p.p.; r16 −4,6 p.p.); o "92%" era dataset contaminado. Enterrado. | **DESCARTADO** · removido na limpeza ([tag](#artefatos-removidos-na-limpeza-camada-12)) | `poc-completa-pre-limpeza:finetune/README.md`, `…:finetune/README_M3.md` |
| 5 | Android M2/M3 | O app responde no aparelho? Por que dizia "Não sei"? | Bug era **FTS5 ausente** no SQLite do Android → fix elevou p/ 8/10 fundamentadas. Rewrite do Turno 1 removido (piorava a busca). | **FUNDAÇÃO** | `android/README_M3.md`, `android/README_CONSOLIDACAO.md`, `android/README_ASR_FIX.md` |
| 6 | `classifier/` | Saber a NR certa (classificador leve) melhora a busca com fala crua? | **SIM** — confiança-gated **R@2 13,9 → 29,8%** (+15,9 p.p., 2,1×), Kotlin puro. Rewrite piora; prompt-classify descartado. | **VIGENTE** (ainda no pipeline) | `classifier/README.md` |
| 7 | `bench/judge151/` | Qual a QUALIDADE da resposta (não só retrieval)? 1.2B vs 2.6B? | Gargalo duplo; dado o chunk certo, o 2.6B converte 3–7× mais que o 1.2B. | SUPERADO por `bench/regua` (métrica honesta) · removido na limpeza ([tag](#artefatos-removidos-na-limpeza-camada-12)) | `poc-completa-pre-limpeza:bench/judge151/README.md` |
| 8 | `retrieval3/` | Como fechar o abismo de vocabulário? (capitão: R@2 ≥ 55%) | Expansão de doc + fusão RRF 3-sinais (EmbeddingGemma) = **R@2 52,3% / R@5 63,6%** on-device. Reranker 27B só na nuvem (descartado, offline). | SUPERADO por `retrieval4/` (v4 embarcado) | `retrieval3/README.md` |
| 9 | Android v3-integration | v3 roda no mesmo engine (embedding)? | SIM — encoder no `libllama_engine.so`, brute-force `.bin` DVEC1; E2E 10/10 ≤10 s, PSS 2,40 GB. | SUPERADO por v4/tools (integração posterior) | `android/README_V3_INTEGRATION.md` |
| 10 | `bench/sintese2/` | Shootout de sintetizadores (+MiniCPM5); engenharia da síntese ajuda? | 2.6B thinking-OFF vence; MiniCPM5-2B pior no domínio; engenharia ajuda o fraco, satura no forte. | SUPERADO por `bench/regua` (rejulgamento) · removido na limpeza ([tag](#artefatos-removidos-na-limpeza-camada-12)) | `poc-completa-pre-limpeza:bench/sintese2/README.md` |
| 11 | Android engine-upgrade | Dá p/ rodar o 2.6B thinking-OFF no JNI? | SIM (fix logit-bias + `<think></think>`), mas **23,3 s / 4,32 GB** no S24+ — inviável p/ voz. | **FUNDAÇÃO** (destravou o 2.6B; segue fora do orçamento de voz) | `android/README_ENGINE_UPGRADE.md` |
| 12 | `finetune2/` (SFT+DPO 2.6B) | Treino de síntese melhora o 2.6B? | **NÃO** — SFT e DPO **regrediram** (base 23,2% → dpo 11,9% gate); é o treino, não a quantização. | **DESCARTADO** · removido na limpeza ([tag](#artefatos-removidos-na-limpeza-camada-12)) | `poc-completa-pre-limpeza:finetune2/README.md` |
| 13 | `retrieval4/` | Como melhorar retriever sem mascarar o real? | **Frente A (classificador+pares-de-graça) DESCARTADA** (mascarada +35–47 p.p.); **Frente B (expansão ASR-robusta) EMBARCADA** — R@2 sob ASR 41,1 → 49,0% (+7,9). | **VIGENTE** (índice v4 + RRF k=10 w=2,1,2) | `retrieval4/README.md` |
| 14 | `bench/ctx_topk/` | Adiantou enxergar mais trechos (topK 2→5)? | **NÃO** com o 1.2B — visão sobe (R@2 51,7 → R@5 65,6%), qualidade flat, TTFT triplica. **Manter topK=2 full.** | **VIGENTE** (default confirmado) | `bench/ctx_topk/README.md` |
| 15 | `bench/regua/` | Quanto da "qualidade" histórica era oca (citação sem informação)? | Régua honesta (fatos, citação fora do gate): **32% das aprovações antigas reprovam**; tautologia rara (0,1%). | **FUNDAÇÃO** (métrica oficial de qualidade) | `bench/regua/README.md` |
| 16 | `docai/` | Reprocessar tabelas das NRs (Document AI) melhora o retrieval? | Anexo II NR-10 reconstruído 18/18; **aditivar tudo POLUI (R@2 13,9 → 11,9)**. Reparo cirúrgico + tokenizer numérico. | **FUNDAÇÃO** (dado disponível; não substitui o índice) | `docai/README.md` |
| 17 | `bench/prompt_teto/` | Prompt engineering resolve a evasão? Qual o teto do 27B? | Prompt **retorno \~zero** (anti-evasão piora); teto do 27B em oráculo **84,1%** (régua passável). | **FUNDAÇÃO** (prompt do app inalterado) | `bench/prompt_teto/README.md` |
| 18 | `slm_oraculo/` | Com contexto perfeito e SFT de destilação, o 2.6B chega ao 27B? | Salto real (Q4 45 → 58,3% em oráculo) mas gap 33 p.p. persiste; **na voz é proibitivo** (\~23 s). Teto refeito 82,1%. | SUPERADO por `sft_v2`/`v3` (treino melhorado) · removido na limpeza ([tag](#artefatos-removidos-na-limpeza-camada-12)); o módulo `corpus_fix.py` foi movido p/ `corpus/` (segue vivo) | `poc-completa-pre-limpeza:slm_oraculo/README.md` |
| 19 | `sft_v2/` | 4 famílias (com/sem oráculo) + varredura de rank + recusa | Rank **move a agulha** (r16→r64 +12,6 p.p.); **recusa destravada** (47 → 100%), mas over-refusal 52% de recusa custa recusa-indevida 23–27%. | SUPERADO por `sft_v3/` (rebalanceamento) | `sft_v2/README.md` |
| 20 | `sft_v3/` | Baixar a recusa p/ ≤25% sem perder o perfil cauteloso? | **SIM** — recusa 52 → 24,5%; recusa-indevida 23–27 → 10–13%, F1 90,7 → 95,7; sem catastrophic forgetting no 2.6B. | **VIGENTE** (dados/receita base do tool-calling) | `sft_v3/README.md` |
| 21 | `sft_1_2b/` | A receita do v3 levanta o 1.2B embarcado? | **SIM** vs o próprio base (oráculo 2,7 → \~22%), mas continua abaixo do 2.6B e **HÁ forgetting** (−0,47 geral); paridade de latência/RAM. | SUPERADO por `tools_v1` (tool-calling no mesmo 1.2B) | `sft_1_2b/README.md` |
| 22 | `tools_v1/` | O 1.2B pode DECIDIR quando buscar? O argumento reescrito ajuda? | Decisão/multiturno **funcionam** (F1 95%, sintaxe 100%); **argumento PIORA a busca** (R@2 22 < crua 30,5). | SUPERADO por `tools_oraculo` (fecha oráculo/recusa/híbrido) | `tools_v1/README.md` |
| 23 | `tools_oraculo/` | O tool-calling atrapalha a síntese? O híbrido bate o pipeline fixo? | Tool **ajuda** a síntese (oráculo 31,4 vs 22,3); **híbrido bate tools puro** (19,2 vs 7–8%) e não perde p/ o fixo. **Adotar o híbrido.** | **VIGENTE** (desenho embarcado) | `tools_oraculo/README.md` |
| 24 | Android tools-hybrid | O híbrido roda no aparelho? | SIM — 5/5 casos E2E corretos; PSS 3,86 GB; reuso \~7 s, busca 10–12 s. Renderização nativa provada byte-a-byte. | **VIGENTE** (a última entrega) | `android/README_TOOLS_HYBRID.md` |
| 25 | `tools_2_6b/` | E o 2.6B com tool-calling? (2ª opção de embarque) | Qualidade maior (oráculo 46–54%) mas **custo proibitivo p/ voz** (18–29 s, 4,3–4,75 GB) + forgetting. Fica por flag. | **VIGENTE** (opção 2, `-Pcemig.toolModel=2.6b`) | `tools_2_6b/README.md` |

---

## Como ler a linha do tempo

1. **Fundação** (1–5, 15–17): corpus, ASR, app funcionando, métrica honesta, teto e dado — o
   terreno onde tudo roda. Não é "o vigente", mas o vigente depende disso.
2. **Retrieval** (6, 8, 13, 14): a maior alavanca da POC. `classifier` + `retrieval3` → `retrieval4`
   (v4 embarcado) → `ctx_topk` confirma topK=2. **Vigente: v4 + classificador gated + topK=2.**
3. **Síntese/treino** (7, 10, 12, 18–21): a busca pela capacidade do sintetizador. `finetune2`
   (negativo) → `slm_oraculo` (destilação funciona) → `sft_v2` → `sft_v3` (recusa controlada) →
   `sft_1_2b` (leva ao 1.2B). **Vigente: dados/receita do v3.**
4. **Tool-calling** (22–25): o modelo decide quando buscar. `tools_v1` → `tools_oraculo` (híbrido) →
   app → `tools_2_6b` (2ª opção). **Vigente: híbrido 1.2B r128 Q4.**

Para o "por quê" de cada escolha com número e impacto de mudar, ver o **Registro de Decisões** em
[`AS_IS.md`](AS_IS.md).

---

## Artefatos removidos na limpeza (camada 1+2)

> Para manter o repo enxuto (só o último experimento), a limpeza da **camada 1+2** removeu do
> `HEAD` os experimentos abandonados/superados e os intermediários pesados. **O histórico de
> commits foi PRESERVADO** (sem rewrite, sem force-push) e **todo o material continua recuperável**
> pela tag anotada abaixo. Nada do que o vigente (última entrega) usa foi tocado.

- **Tag de arquivo**: `poc-completa-pre-limpeza`
- **Commit etiquetado**: `3bbd5299d21af59dba71ad7f8def70b5941928d6` (main com o dataset publicado
  `tools_v1/data/train_full.jsonl`, sha256 `be60946f…`).
- **Recuperação** (qualquer arquivo/diretório removido):
  ```sh
  git show    poc-completa-pre-limpeza:<caminho>          # ver um arquivo
  git checkout poc-completa-pre-limpeza -- <caminho>       # restaurar no worktree
  ```
- **Economia**: footprint rastreado 332,1 MB → 193,9 MB (**−138,2 MB**).

### Camada 1 — experimentos abandonados (removidos por inteiro)

| Diretório | Pergunta que respondeu | Veredito / quem superou | Arquivos | Recuperar |
|---|---|---|--:|---|
| `finetune/` | Fine-tune do Turno-1 rewriter destrava o retrieval? | **DESCARTADO** — regrediu no holdout limpo (r8 −8,6 p.p.; r16 −4,6 p.p.); o "92%" era dataset contaminado. Contém 17 pesos LoRA versionados (adapter_r8/r16). | 37 | `git checkout poc-completa-pre-limpeza -- finetune` |
| `finetune2/` | Treino de síntese (SFT+DPO) melhora o 2.6B? | **DESCARTADO** — SFT e DPO **regrediram** (base 23,2% → dpo 11,9% gate); é o treino, não a quantização. | 29 | `git checkout poc-completa-pre-limpeza -- finetune2` |
| `slm_oraculo/` | Com contexto perfeito + SFT de destilação, o 2.6B chega ao 27B? | **SUPERADO por `sft_v2`/`sft_v3`** — salto real (Q4 45→58,3% em oráculo) mas gap 33 p.p.; na voz é proibitivo (~23 s). Teto refeito **82,1%**. | 50 (1 movido) | `git checkout poc-completa-pre-limpeza -- slm_oraculo` |

> **Exceção viva do `slm_oraculo/`**: o módulo `corpus_fix.py` (resolve o trecho-ouro do corpus
> corrigido / Anexo II NR-10) era importado em runtime por `sft_v2/score_v2.py` e
> `tools_oraculo/harness_oraculo.py` (ambos VIGENTES). Foi **movido para `corpus/corpus_fix.py`**
> (lugar semanticamente correto) e os dois imports foram ajustados — segue vivo no `HEAD`, não
> depende da tag.

### Camada 2 — intermediários pesados dentro de pastas que ficam (removido só o indicado)

| Caminho | O que era | Onde a conclusão sobrevive | Arquivos | Recuperar |
|---|---|---|--:|---|
| `retrieval3/results/vec0_text_rank.json`, `vec0_exponly_rank.json` | Rankings densos intermediários (~19 MB cada) via sqlite-vec. | `retrieval3/README.md` (conclusões); os equivalentes **`bin_*_rank.json` FICARAM** (input vivo do `retrieval4/` e `bench/prompt_teto/`). | 2 | `git checkout poc-completa-pre-limpeza -- retrieval3/results/vec0_text_rank.json retrieval3/results/vec0_exponly_rank.json` |
| `bench/results/` | Saídas do harness/juiz do bench v1/v2 (8 SLMs × modos). | `AGENTS.md` (bench); rejulgado em `bench/regua/` (tabela commitada). | 9 | `git checkout poc-completa-pre-limpeza -- bench/results` |
| `bench/judge151/` | Painel de qualidade da resposta (1.2B vs 2.6B). | **Substituído pela régua honesta `bench/regua/`**; ver linha 7 acima. | 32 | `git checkout poc-completa-pre-limpeza -- bench/judge151` |
| `bench/sintese2/` | Shootout de sintetizadores (+MiniCPM5) + engenharia da síntese. | **SUPERADO por `bench/regua/`** (rejulgamento); ver linha 10 acima. | 27 | `git checkout poc-completa-pre-limpeza -- bench/sintese2` |

> **Nota de reprodutibilidade**: alguns scripts que FICAM ainda citam esses caminhos como
> ferramenta de re-geração (`bench/regua/analyze.py` faz glob dos painéis; `sft_v2/Makefile`
> copia packs de `slm_oraculo/data/`). As **saídas dessas ferramentas já estão commitadas**
> (`bench/regua/data/`, `sft_v2/data/*_151.json`), então o vigente não quebra; para re-rodar do
> zero, restaure o material pela tag.
