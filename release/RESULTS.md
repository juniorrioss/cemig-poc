# RESULTS — release autocontida do melhor modelo (teste de replicação do zero)

Pedido do capitão: subir tudo o que é preciso para **replicar a inferência** do melhor
modelo (ponto de operação F6 + `f4-r32-ep2`), de forma autocontida e pública: um código
Python que baixa modelo + embeddings/bases do Hugging Face e responde.

## Entregáveis

| o quê | onde |
|---|---|
| modelo fundido bf16 (safetensors) + GGUF Q4_0 + `inference.py` + requirements + model card | **https://huggingface.co/juniorrios/cemig-lfm3-1.7b-nr** (público, licença LFM Open v1.0, commit `dd71f54`) |
| bases de busca: `index.db` (2202 trechos de 36 NRs), `manuals.db` (1092 trechos de 9 manuais), densos `dense_text/exponly.bin`, `manuals_dense.npz`, janelas do compressor (`units.json` + `units_win3.f16.npy`), classificador (`nr_classifier.bin` + `classic_winner.pkl`), `config.json` | **https://huggingface.co/datasets/juniorrios/cemig-nr-retrieval-assets** (público, commit `1475313`) |
| código canônico (espelho do que está no HF) | `release/inference.py`, `release/requirements*.txt`, `release/config.json`, `release/README.md` (= model card) |
| empacotamento dos assets / prova de paridade / replicação | `release/tools/build_assets.py`, `release/tools/parity_check.py`, `release/validation/` |

Quickstart (está no model card):

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r https://huggingface.co/juniorrios/cemig-lfm3-1.7b-nr/resolve/main/requirements.txt
curl -LO https://huggingface.co/juniorrios/cemig-lfm3-1.7b-nr/resolve/main/inference.py
python inference.py "Quando trocar de turno, preciso trocar os bloqueios e as etiquetas?"
python inference.py --backend hf "..."   # transformers bf16
```

> **Nota (pós-entrega):** o model repo foi renomeado para `juniorrios/automobil-ultra-semnexo`
> (fora desta lane, ver `bench/lfm3_q4km/RESULTS.md` §5). O nome antigo segue resolvendo por redirect
> (conferido via `hf_hub_download`), e o `inference.py` e o quickstart continuam funcionando sem mudança.
> O card ganhou o `Q4_K_M.gguf` opcional; o padrão continua Q4_0. `release/README.md` foi sincronizado com o card.

## Decisões (curtas)

- **Licença**: a LFM Open License v1.0 da base permite redistribuir derivados (Seção 4). A
  condição é incluir a licença, marcar a modificação e manter as atribuições. Fizemos as
  três: `LICENSE` no repo, aviso "modificado a partir de LiquidAI/LFM3-1.7B" no card e
  atribuição. O limite comercial (Seção 5, receita de US$ 10 milhões/ano) está declarado no card.
- **Dados limpos**: as bases contêm só as NRs e os 9 manuais/guias públicos do MTE. Uma busca
  por `cemig`, `benchmark` e `UFGCEMIG` nos textos deu 0 ocorrências. As expansões vêm do
  `retrieval4`, com prompt genérico por trecho. Não subimos dados de treino nem nada derivado
  do `benchmark_mobile`: as 10 perguntas de validação ficam só no git.
- **llama-cpp-python**: a versão 0.3.36 do PyPI vendoriza um llama.cpp de 01/10/2026, anterior
  ao suporte a LFM3 (`19cce69`, 27/10). O requirements fixa o commit `906afdd` (PR
  abetlen/llama-cpp-python#2398, llama.cpp b12032), que carrega o GGUF. Ele compila do fonte
  em ~1,5 min.
- **Sampler da via GGUF**: o código monta a cadeia na ordem do `llama-server`, onde o Q4 foi
  validado na F7: penalties(64, 1,1) → top_k 50 → top_p 1 → min_p 0 → temp 0,1 → dist(seed),
  com o prompt no histórico de penalidade. O `create_completion` padrão do binding usa
  min_p 0,05 e typical, e não sabe o prompt.
- **Template sem transformers**: na via GGUF o prompt é renderizado com jinja2 a partir do
  `chat_template.jinja` do modelo, como o `apply_chat_template` faz. O resultado é
  byte-idêntico ao template da F6 (ver paridade). A via hf usa o próprio `apply_chat_template`.
- **Janelas em fp16** e não int8: como na F7, fp16 reproduz 344/344 contextos, int8 só 332/344.
- **Encoder em tempo de execução**: o EmbeddingGemma vem do repo oficial
  `ggml-org/embeddinggemma-300M-qat-q4_0-GGUF`, com revisão fixada e sha256 idêntico ao usado
  nos índices. Não o redistribuímos (Gemma ToU).

## Teste de replicação do zero (DGX Spark, aarch64, GB10)

`release/validation/run_from_scratch.sh`: venv novo (uv, Python 3.12.16), `HF_HOME` vazio,
**sem token HF** (acesso anônimo = público de fato), `uv pip install --no-cache`. Baixa só
`inference.py` + `requirements*.txt` do model repo e roda as 10 perguntas operacionais
(`validation/questions10.jsonl`: as 8 perguntas-exemplo do app F7, 5 de norma e 3 de manual,
mais 2 do smoke F7). Cada via rodou 2 vezes com seed 42. Log bruto em `validation/out/`.

| etapa | resultado |
|---|---|
| install llama-cpp-python (compila) + deps | 94 s |
| download do HF (GGUF + safetensors + assets + encoder) | 4,6 GB no cache |
| determinismo, via GGUF Q4_0 (CPU), rodada 1 × rodada 2 | **10/10 respostas idênticas byte a byte** |
| determinismo, via transformers bf16 (GPU), rodada 1 × rodada 2 | **10/10 idênticas** |
| busca + compressão + prompt × prompts congelados da F6 (`op_d50`), 10 perguntas | 9/10 idênticos byte a byte |
| idem, recorte inteiro (344) | 327/344 contexto e prompt idênticos (311/344 ids top-5) |
| idem, recorte inteiro, com o MESMO vetor de query da F5 (`qemb.npz`) | **344/344 contexto, ids e prompt idênticos** |
| latência mediana (Spark): busca + compressão / geração | 0,67 s / GGUF CPU 27 s · bf16 GPU 6,4 s |

**Régua honesta** (juiz Qwen 27B, mesma régua de `score_f6.py`/`score_smoke.py`), nas 10
perguntas (`validation/validation_report.json`):

| célula | aprovadas | alucin. | recusa | cobertura média |
|---|--:|--:|--:|--:|
| **release, GGUF Q4_0 (CPU)**, rodadas 1 e 2 | **10/10** | 0 | 0 | 0,950 |
| **release, transformers bf16 (GPU)**, rodadas 1 e 2 | **9/10** | 1 | 0 | 0,967 |
| repo F6 (transformers bf16, batch 8), rodadas 1/2/3 | 10/10 · 10/10 · 10/10 | 0 | 0 | 0,917–0,967 |
| repo F7 smoke Q4_0 (llama-server CUDA), 7 ids em comum | 7/7 | 0 | 0 | 0,929 |

### Leitura honesta da paridade

- **O porte é exato.** Com os mesmos vetores de query, o `inference.py` reproduz os 344
  prompts da F6 byte a byte: gate, BM25, densos, manuais, RRF, orçamento, compressão e
  template.
- **A única fonte de diferença na busca é a numérica do encoder.** O cache da F5 foi
  calculado num `llama-server` CUDA em lotes de 32. A release calcula na CPU, uma query por
  vez. Cosseno entre os dois vetores ≥ 0,9998, mas em ~5% das perguntas isso troca a ordem de
  2 trechos quase empatados no RRF. Em 9 das 20 divergências listadas o conjunto do top-5 é o
  mesmo; em 17 o top-2 é o mesmo. Na validação de 10, 1 pergunta (NR-33, bloqueios) teve o
  contexto alterado e a resposta seguiu correta e aprovada.
- **A geração é determinística por máquina e backend, não entre backends.** Com prompt
  idêntico e seed 42, a via GGUF repete 10/10. Contra o smoke da F7 (mesmo GGUF, mesmo prompt,
  mas CUDA) ficam idênticas 3/7: as outras 4 divergem a partir de algumas palavras, porque a
  aritmética CPU×CUDA muda o logit e a amostragem a temp 0,1 se separa. As 3 rodadas HF da
  própria F6 só coincidem entre si em 1/10 perguntas (batch e padding diferentes). A
  replicação fiel se mede pela régua, e nela a release empata com o repo: 10/10 e 9/10
  contra 10/10.
- A reprovação da via bf16 é **1 alucinação** (escada fixa vertical): a resposta cita a NR-22
  quando a pergunta veio da NR-18 e o trecho ouro estava em outro item. O caso mostra o ruído
  de amostragem já documentado na F6 (±3 p.p. entre rodadas com prompt idêntico), não um
  defeito do porte. A mesma pergunta passa na via GGUF.

## Reprodução

```bash
# 1) assets (repo + caches F5/F6 gitignored) -> HF dataset
classifier/.venv/bin/python release/tools/build_assets.py --out /tmp/cemig-assets \
    --f5-cache bench/lfm3_f5/cache --f6-cache bench/lfm3_f6/cache
cp release/DATASET_CARD.md /tmp/cemig-assets/README.md
hf upload juniorrios/cemig-nr-retrieval-assets /tmp/cemig-assets . --repo-type dataset
# 2) modelo (Spark: merged + GGUF da F7) -> HF model; inference.py/requirements/README/LICENSE/pipeline_config.json junto
#    (na Spark use HF_XET_CACHE=<dir próprio>: ~/.cache/huggingface/xet pertence ao root)
# 3) paridade local (encoder via llama-server :8399 ou llama-cpp-python)
classifier/.venv/bin/python release/tools/parity_check.py \
    --prompts bench/lfm3_f6/data/e2e/prompts_recorte.jsonl.gz --emb-url http://127.0.0.1:8399
# 4) replicação do zero (máquina limpa com uv) + régua
bash release/validation/run_from_scratch.sh
classifier/.venv/bin/python release/validation/score_validation.py --out-dir <W>/out
```
