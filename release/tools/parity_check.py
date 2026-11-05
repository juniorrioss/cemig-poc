#!/usr/bin/env python3
"""
parity_check.py — prova que o porte autocontido (release/inference.py) reproduz o pipeline
do repo: monta o contexto (busca + compressão) de cada pergunta e compara byte a byte com
o turno `tool` dos prompts congelados da F6 (célula op_d50, recorte 344) e com o prompt
renderizado inteiro (template LFM3).

Uso:
  python release/tools/parity_check.py --prompts bench/lfm3_f6/data/e2e/prompts_recorte.jsonl.gz \
      [--assets DIR] [--emb-url http://127.0.0.1:8399] [--limit N]
Sem --assets baixa do HF. Com --emb-url usa um llama-server de embedding (mesmo GGUF) em vez
do llama-cpp-python (útil onde ele não está instalado).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference as I  # noqa: E402


class HttpEmbedder:
    def __init__(self, url: str):
        self.url = url

    def query(self, text: str) -> np.ndarray:
        import requests
        r = requests.post(f"{self.url}/v1/embeddings",
                          json={"input": [I.QUERY_PROMPT + text], "model": "emb"}, timeout=60)
        r.raise_for_status()
        v = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
        return v / max(np.linalg.norm(v), 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--cell", default="op_d50")
    ap.add_argument("--assets", default=None)
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (default: HF)")
    ap.add_argument("--template", default=None, help="chat_template.jinja (default: HF)")
    ap.add_argument("--emb-url", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    assets = Path(a.assets) if a.assets else I.download_assets()
    cfg = json.loads((assets / "config.json").read_text(encoding="utf-8"))
    emb = HttpEmbedder(a.emb_url) if a.emb_url else I.Embedder(I.download_embedder())
    tok = a.tokenizer or hf_hub_download(I.MODEL_REPO, "tokenizer.json", revision=I.MODEL_REV)
    tmpl = a.template or hf_hub_download(I.MODEL_REPO, "chat_template.jinja", revision=I.MODEL_REV)

    pipe = I.Pipeline.__new__(I.Pipeline)
    pipe.assets, pipe.cfg, pipe.embedder = assets, cfg, emb
    pipe.rtv = I.Retriever(assets, cfg, emb)
    pipe.comp = I.Compressor(assets, pipe.rtv.chunks, tok)

    opener = gzip.open if a.prompts.endswith(".gz") else open
    with opener(a.prompts, "rt", encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows if r["cell"] == a.cell]
    if a.limit:
        rows = rows[:a.limit]
    same_ctx = same_ids = same_prompt = 0
    diffs = []
    for r in rows:
        q = r["messages"][1]["content"]
        c = pipe.context(q)
        ok_ctx = c["context"] == r["messages"][-1]["content"]
        ok_ids = c["retrieved_ids"] == r["ctx_ids"]
        mine = I.render_prompt(I.build_messages(q, c["context"]), tmpl)
        ref = I.render_prompt(r["messages"], tmpl)
        ok_p = mine == ref
        same_ctx += ok_ctx
        same_ids += ok_ids
        same_prompt += ok_p
        if not (ok_ctx and ok_ids and ok_p):
            diffs.append({"key": r["key"], "ctx": ok_ctx, "ids": ok_ids, "prompt": ok_p,
                          "mine_ids": c["retrieved_ids"], "ref_ids": r["ctx_ids"]})
    rep = {"n": len(rows), "context_identical": same_ctx, "ids_identical": same_ids,
           "prompt_identical": same_prompt, "embedder": "http" if a.emb_url else "llama-cpp-python",
           "diffs": diffs[:20]}
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
