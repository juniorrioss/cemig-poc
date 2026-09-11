# Etapa 3 — Seleção própria de modelos de embedding (critério MOBILE-FIRST)

> Ordem do capitão: NÃO usar bge/e5 como default; seleção própria pensando em
> otimização mobile (tamanho, latência de encode on-device, memória, int8, dim).
> Nada dos `vector_db_*` da UFGCEMIGONA. bge/e5 só entram se um experimento próprio
> provar que batem os mobile-first COM viabilidade mobile igual.

## Critério de triagem (aplicado ANTES de qualquer teste)

Um candidato só vai para o experimento se satisfizer TODOS:

| Critério | Limite | Razão mobile |
|---|---|---|
| Multilíngue / PT real | obrigatório | perguntas e normas em PT-BR |
| Parâmetros | ≤ 350M (ideal ≤ 150M) | RAM/CPU do S24+/S21; coexistir com SLM 700 MB + ASR 197 MB |
| Quantizável int8 | sim | reduzir disco e RAM; ONNX int8 é o alvo Android |
| Dimensionalidade | ≤ 768 | tamanho do índice vetorial (2202 vetores) e custo de dot-product |
| Latência de encode de query on-device | < 300 ms | 1 query ≈ 1 forward de ~30 tokens; teto de voz 10 s |

## Candidatos avaliados na triagem

| Modelo | Params | Dim | PT? | Veredito de triagem |
|---|---|---|---|---|
| **model2vec / potion (estático)** | ~0 (lookup) | 256–512 | destilável PT | **ENTRA** — estático, encode <10 ms CPU, <50 MB int8; teto de qualidade mais baixo, mas custo mobile imbatível |
| **EmbeddingGemma-300M** | 300M | 768 (MRL→128) | multilíngue (100+) | **ENTRA** — mobile-first do Google, Matryoshka permite dim 128/256; ~limite superior de params |
| **LFM2.5-Embedding-350M** | 350M | 768 | multilíngue | **ENTRA (se disponível)** — mesmo ecossistema Liquid do SLM; reuso de runtime |
| **snowflake-arctic-embed-s** | 33M | 384 | multilíngue fraco | **ENTRA** — 33M é ótimo p/ mobile; PT limitado (risco) |
| **paraphrase-multilingual-MiniLM-L12-v2** | 118M | 384 | multilíngue (50+) forte | **ENTRA** — destilado, 118M, 384 dim, PT sólido, int8 ONNX maduro |
| **intfloat/multilingual-e5-small** | 118M | 384 | multilíngue | triagem OK mas **e5 é PROIBIDO como default** (ordem); só como controle comparativo |
| bge-m3 / bge-large | 560M+ | 1024 | multilíngue | **FORA** — >350M, dim 1024, proibido por ordem |

## Modelos efetivamente testados neste experimento

A escolha final de quais rodar depende de disponibilidade offline/HF e do tempo de
encode na 5070. Priorizamos, em ordem:

1. `paraphrase-multilingual-MiniLM-L12-v2` (118M, 384) — melhor custo/benefício mobile comprovado, PT forte.
2. `model2vec` destilado do MiniLM PT (estático, <50 MB) — piso de custo mobile.
3. `EmbeddingGemma-300M` (dim 256 via MRL) — mobile-first do Google, se baixar a tempo.
4. Controle proibido-como-default: `multilingual-e5-small` — só para PROVAR se bge/e5 batem os mobile-first (exigência da ordem para poder descartá-los com evidência).

Cada modelo testado reporta: tamanho (fp32/int8), dim, tamanho do índice vetorial
(2202 vetores), latência de encode 5070 e latência projetada S24+ por query.

Procedência: task poc-retrieval-v3, branch fm/poc-retrieval-v3.
