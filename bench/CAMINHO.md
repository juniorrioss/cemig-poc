# O caminho que deu certo — do LFM3-1.7B cru ao APK v1.2

> Este documento conta **só a linha vencedora**, do começo ao fim. Cada passo traz o que se fez,
> o número que mudou e onde está o artefato. Os desvios testados e descartados ficam no relatório
> completo, [`bench/REPORT.md`](REPORT.md). Todos os números foram copiados dos `RESULTS.md` de
> cada fase.
>
> **Régua única**: benchmark `UFGCEMIGONA/benchmark_mobile` (1536 perguntas = 384 × 4
> vocabulários: limpo, técnico, operacional e leigo). Juiz Qwen 27B, régua honesta: aprovada é a
> resposta que cobre ≥ 50% dos fatos pedidos, sem alucinar, sem recusar e sem truncar. O
> "recorte 344" são 6 NRs (10/18/34/13/31/33) usadas na avaliação rápida. As "151" são as
> perguntas coloquiais reais da POC.

## A linha em uma tabela

| # | passo | o que moveu | escopo |
|--:|---|---|---|
| 0 | F0: baseline as-is | closed-book 5,9 · oráculo 83,7 · ranking 65,8 · recusa correta no distrator 30,5 | 1536 |
| 1 | F3: robustez com retriever real | ranking **65,8 → 75,2** · recusa correta **30,5 → 59,1** (2×) · truncamento 35,4 → 0 | 1536 |
| 2 | F3b: o dado domina | ranking **77,0** · oráculo 85,4 (campeão `data2x-ep4`) | 1536 |
| 3 | F4: tool-calling XML nativo | decisão F1 **97,2** · sintaxe **100%** · ranking **79,6** · oráculo 87,8 | 1536 |
| 4 | Smoke de retrieval | v4 hit@5 0,874 × 0,804 do retriever do MobileRAG+ → **mantém o v4** | 1136 (norma) |
| 5 | F5: gate + compressão | citação do item **65,2 → 73,2** · prompt **1635 → 1100 tok (−33%)** · aprovação 55,6 → 57,5 | recorte 344 |
| 6 | F6: manuais como aba no RRF | fim a fim **57,5 → 68,0** · manual 18,3 → 53,3 | recorte 344 |
| 7 | F7: embarque no S24+ | **8/8** texto e voz · 1º token 7,0–8,4 s · PSS ~3,7 GB | aparelho |
| 8 | F8: UI "Assistente CEMIG" | 8/8 pela UI · PSS 3,76 GB | aparelho |
| 9 | Q4_0 por latência | −4,9 p.p. vs bf16, aceito pelo prefill mais rápido (190 t/s no S24+) | recorte 344 / aparelho |
| 10 | P1: filtro de aprendibilidade | **49,2%** do universo descartado (fácil ou mal especificado) | 11.313 itens |
| 11 | P2: RLVR verificável (`v3s400`) | distrator aluc. **23,8 → 15,7** · fim a fim **64,1 → 71,8** · 151 **12,6 → 21,2** | 1536 + 151 |
| 12 | APK v1.1 | 8/8 texto (×2) e voz · 1º token 8,1–8,3 s · PSS 3,69 GB | aparelho |
| 13 | APK v1.2: instala limpo | setup baixa 2,01 GB em 3,6 min (SHA-256, retomada) · 3/3 norma/manual/voz | aparelho |

---

## 0. F0 — a régua do "antes"

**O que se fez.** Medimos o `LiquidAI/LFM3-1.7B` bf16 sem nenhuma adaptação, nas 4 células do
benchmark: closed-book (sem contexto), oráculo (só o trecho certo), ranking (o certo entre 5) e
distrator (só trechos errados da mesma NR; a resposta certa é recusar).
**Número.** closed-book **5,9%** · oráculo **83,7%** · ranking **65,8%** · recusa correta no
distrator **30,5%** (alucinação 28,3%, truncamento 35,4% por ruminação no thinking).
**Leitura.** O modelo quase não sabe as NRs de cor, mas lê bem quando recebe o trecho certo. O
problema está em ler no meio de distratores e em saber recusar.
**Artefato.** `bench/lfm3_f0/` (README + parquets).

## 1. F3 — robustez de uso de contexto com o retriever REAL

**O que se fez.** SFT LoRA r32 só no MLP (`w1/w2/w3`) sobre o instruct. O dataset tem 6.972
exemplos ancorados no **retriever v4 real** da POC, com k variável (3/5/8 trechos) e quatro
casos: ouro presente, ouro ausente (recusa), parcial e closed-book. As respostas-alvo foram
geradas pelo 27B vendo só o trecho-âncora.
**Número.** ranking **65,8 → 75,2%** · recusa correta no distrator **30,5 → 59,1%** (2×) ·
alucinação 28,3 → 22,4% · truncamento **35,4 → 0%** · oráculo e closed-book intactos (83,5 / 6,2).
**Artefato.** `bench/lfm3_f3/RESULTS.md`; checkpoint `f3-r32-ep4`.

## 2. F3b — sweep: o dado domina

**O que se fez.** Variamos um fator por vez sobre a receita da F3 (dado, dose de recusa, âncora
geral, rank, alvos e LR). Só os eixos de **dado** moveram a agulha. O vencedor dobra o dataset
(12 NRs novas).
**Número.** Campeão `data2x-ep4`: ranking **77,0%** (+1,8) · oráculo **85,4%** · closed 7,4%.
**Artefato.** `bench/lfm3_f3b/RESULTS.md`; base mesclada da F4 (md5 `ae5f304d…`).

## 3. F4 — tool-calling multiturno no formato nativo

**O que se fez.** LoRA r32 MLP sobre o `data2x-ep4` mesclado, com 4.624 diálogos em 6 famílias:
buscar, não buscar, reusar contexto, trocar de NR, recusar após a busca e multihop. O formato é o
**XML `<tool_call>` emitido pelo próprio `apply_chat_template`**, com paridade byte a byte. A
consulta da 1ª chamada é a **fala crua** do usuário.
**Número.** decisão F1 **97,2** · sintaxe XML **100%** (0/285) · reuso **100%** · troca de NR
100/96,4. O modo sem ferramenta **melhorou**: ranking 77,0 → **79,6%**, oráculo 85,4 → **87,8%**,
recusa correta no distrator **66,6%**.
**Artefato.** `bench/lfm3_f4/RESULTS.md`; candidato **`f4-r32-ep2`** (`checkpoint-284`).

## 4. Smoke de retrieval — fica o v4

**O que se fez.** Rodamos o retriever v4 da POC (BM25 com expansão ASR-robusta + 2 densos
EmbeddingGemma-300M, RRF) nas 1536 perguntas, com o harness de métricas do MobileRAG+ validado
contra os números publicados por eles.
**Número.** Nas 1136 com ouro em norma: v4 **hit@5 0,874** contra **0,804** do melhor 1º estágio
do MobileRAG+ (Qwen3-Embedding fine-tuned, mesmo corpus de normas). O v4 vence em todas as
métricas e vocabulários. Decisão: **não trocar o retriever** e atacar 3 pontos dentro do v4
(gate do classificador, pós-recuperação e manuais).
**Artefato.** `bench/lfm3_retrieval_smoke/REPORT.md`.

## 5. F5 — gate 0,7/2× + compressão top-5 → 50%

**O que se fez.** (a) Gate do classificador de NR com limiar 0,7 e boost 2×, o único ajuste que
sobe nas duas réguas (151 e benchmark). (b) A **hipótese do capitão**: buscar mais (top-5) e
comprimir o contexto para 50% dos tokens do top-2 cru. A compressão seleciona unidades normativas
pelo próprio EmbeddingGemma e preserva o código do item.
**Número.** Fim a fim (recorte 344, `f4-r32-ep2`, formato do app): aprovação 55,6 → **57,5%**
(empate dentro do ruído) · **citação do item-ouro 65,2 → 73,2% (+8 p.p.)** · prompt **1635 →
1100 tokens (−33%)**. A hipótese foi provada: menos contexto, mais denso, sem perder resposta.
**Artefato.** `bench/lfm3_f5/RESULTS.md` (`compress.py`, `retriever_f5.py`).

## 6. F6 — manuais do MTE como aba separada no RRF

**O que se fez.** 26% das 1536 perguntas têm a resposta num dos 9 manuais oficiais, que não
estavam indexados. Entraram como sinais próprios no RRF (BM25 + denso, peso 1,75), ligados só
quando o melhor manual empata ou supera a melhor norma na similaridade densa (margem 0). A
compressão ganhou as janelas dos manuais.
**Número.** manual hit@5 **0 → 46,8%** com norma −0,44 p.p. e as 151 intactas. Fim a fim (recorte
344): **57,5 → 68,0%** (manual 18,3 → 53,3%, operacional 55,4 → 65,1%), **sem um token a mais de
prompt**.
**Artefato.** `bench/lfm3_f6/RESULTS.md`; ponto de operação `thr0.7_f2+man1.75+margin` → top-5 →
compressão 50%.

## 7. F7 — embarque no Galaxy S24+

**O que se fez.** O `f4-r32-ep2` virou GGUF Q4_0 (llama.cpp v0.9.0). O porte para o app Android
inclui o template XML do LFM3 em Kotlin (byte a byte), o retriever do ponto de operação e o
compressor (janelas fp16), tudo com paridade verificada contra o espelho Python: 40/40 contextos
na JVM e top-5/contexto 8/8 no próprio aparelho. ASR Nemotron 3.5 on-device.
**Número.** Smoke Q4_0 = bf16 (15/20). No S24+: **8/8 aprovadas em texto e em voz** · 1º token
visível **~8,4 s** (6,99 s na revalidação de produção) · total ~10–12 s · **pico PSS ~3,7 GB**.
**Artefato.** `bench/lfm3_f7/RESULTS.md`; `android/` (`HybridRetriever.searchOp`,
`ContextCompressor`, `Lfm3ToolFormat`).

## 8. F8 — UI "Assistente CEMIG"

**O que se fez.** A interface foi recriada em Compose no estilo do app de referência: chat,
paleta verde CEMIG, fontes NR/item expansíveis, bolha "Pensando" ligada ao tool-calling e
segurar-para-falar. O motor ficou intocado.
**Número.** Checklist de regressão 100% · 8 perguntas-exemplo pela UI 8/8 com a fonte esperada ·
PSS 3,76 GB.
**Artefato.** `bench/lfm3_f8/RESULTS.md`; APK v1.0 publicado no HF.

## 9. Formato de produção: Q4_0, por latência

**O que se fez.** Medimos a retenção de qualidade do GGUF contra o bf16 no recorte 344, com o
ponto de operação congelado, e a velocidade no S24+.
**Número.** Q4_0 = **−4,9 p.p.** contra o bf16 (68,0 → 63,1) e prefill de **190 t/s** no S24+.
**Decisão do capitão:** o Q4_0 fica, porque o 1º token pesa mais que a qualidade marginal.
**Artefato.** `bench/lfm3_quant/RESULTS.md`, `bench/lfm3_q4km/RESULTS.md`.

## 10. P1 — filtro de aprendibilidade

**O que se fez.** Rodamos o `f4-r32-ep2` 4 vezes por item, com uma régua automática sem juiz
(cita o item-ouro, cobre os fatos, não inventa números, recusa só quando deve). Separamos cada
item em fácil (acerta ≥ 80%), produtivo, difícil (o 27B resolve) e mal especificado (nem o 27B
resolve).
**Número.** Num universo de 11.313 itens, **49,2% descartados**: 39,5% já dominados e 9,7% mal
especificados. O RL treina só onde há o que aprender.
**Artefato.** `bench/lfm3_p1p2/p1_pipeline.py`, `data/p1_report.json`.

## 11. P2 — RLVR com reward verificável (`v3s400`)

**O que se fez.** GRPO com perda DAPO (TRL 1.17, clip-higher 0,2/0,28, máscara de truncadas,
β = 0). LoRA novo r32 MLP sobre o `f4-r32-ep2` mesclado, 8 prompts × 8 gerações, lr 5e-6, 400
steps (~75 min numa H100). O **reward é binário e verificável por regra**, sem juiz no loop:
- com o ouro no contexto, vale 1 se cita o item, cobre ≥ 50% dos fatos e todo número existe no
  contexto;
- sem o ouro, vale 1 se recusa explicitamente no início.

O pool de prompts saiu do P1 e cobre os dois formatos de uso: o turno de ferramenta do app (top-5
comprimido) e o contexto estático.
**Número (bateria completa, base regerada no mesmo harness):**

| | f4-r32-ep2 | **v3s400** |
|---|--:|--:|
| distrator ×1536: alucinação | 23,8% | **15,7%** |
| fim a fim do app ×1536 | 64,1% | **71,8%** (pareado 177/59) |
| oráculo ×1536 / ranking ×1536 | 88,5 / 78,1 | 90,0 / 81,6 |
| 151 coloquiais da POC | 12,6% | **21,2%** (p = 0,011) |

O ganho se repete fora das normas do recorte (distrator 24,2 → 15,9; app 62,7 → 70,8). Custo: +14%
de tokens por resposta. Nas 151 o modelo recusa menos (53 → 35%) e alucina um pouco mais (28,5 →
31,1%).
**Artefato.** `bench/lfm3_p1p2/RESULTS.md` (`train_p2_rlvr.py`, `verifier.py`);
checkpoint `lfm3-p2-rlvr-v3/checkpoint-400`.

## 12. APK v1.1 — o v3s400 no app

**O que se fez.** O LoRA foi mesclado no `f4-r32-ep2` e convertido para GGUF **Q4_0** com o
llama.cpp do próprio app. Só o GGUF muda; motor, assets e UI ficam iguais. O APK v1.1 foi
publicado no HF junto com os pesos.
**Número.** Smoke 20 operacionais: Q4_0 15/20 × bf16 16/20. No S24+: **8/8 em texto (2 rodadas)
e 8/8 em voz** · 1º token visível **8,1–8,3 s**, o mesmo do f4 na mesma sessão (8,1 s) · total
~12 s · **pico PSS 3,69 GB**.
**Artefato.** `bench/lfm3_f9/RESULTS.md`. No HF `juniorrios/automobil-ultra-semnexo`:
`v1.1-rlvr/` (safetensors + GGUF Q4_0) e `apk/cemig-assistente-v1.1-rlvr.apk`.

## 13. APK v1.2 — instalação limpa, sem PC

**O que se fez.** Os APKs anteriores dependiam de `adb push` dos pesos. A v1.2 ganhou a tela
"Configuração inicial", que baixa o GGUF v3s400, o EmbeddingGemma e o Nemotron do HF para o
armazenamento interno (retomada por HTTP Range, SHA-256 conferido, checagem de espaço). Os índices
densos passaram a vir nos assets. O modelo é o mesmo da v1.1.
**Número.** Instalação limpa no S24+: download de **2,01 GB em 3,6 min**, retomada testada,
**3/3** perguntas-chave (norma, manual e voz) com resposta idêntica à da v1.1.
**Artefato.** `bench/lfm3_f10/RESULTS.md`; `android/.../ModelDownloader.kt`. No HF:
`apk/cemig-assistente-v1.2.apk` (o v1.1 foi removido).

---

## O que ficou de pé (a receita, em uma frase por camada)

- **Modelo**: LFM3-1.7B → SFT de robustez com o retriever real (dado em dobro) → SFT de
  tool-calling XML nativo → RLVR curto com reward verificável, num pool filtrado por
  aprendibilidade. Tudo com LoRA r32 só no MLP.
- **Busca**: v4 (BM25 com expansão + 2 densos EmbeddingGemma, RRF) + gate 0,7/2× + manuais como
  aba com margem densa → top-5 → compressão para 50% preservando o código do item.
- **Aparelho**: GGUF Q4_0 + llama.cpp v0.9.0 (KleidiAI) + Nemotron ASR, 100% offline, ~8 s até o
  1º token e ~3,7 GB de RAM no Galaxy S24+.

Limites honestos que seguem abertos: fala coloquial longa (as 151 ficam em 21%), multihop e metade
das perguntas de manual sem o trecho recuperado. Detalhes e todos os caminhos testados e
descartados estão em [`bench/REPORT.md`](REPORT.md).
