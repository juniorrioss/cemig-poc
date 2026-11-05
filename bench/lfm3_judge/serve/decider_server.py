#!/usr/bin/env python3
"""
decider_server.py — servidor HTTP de decisão em LOTE para os decision models.

Por que não o `autojev.server` oficial: ele processa UMA requisição por vez (lock
não-bloqueante, 529 se ocupado) e só agrupa as perguntas do MESMO state. Para julgar
dezenas de respostas "no mesmo tempo" precisamos de lote ENTRE itens. Este servidor
recebe uma lista de linhas {state, question} e roda `DecisionModel.prepare/forward`
em lotes ordenados por comprimento (menos padding), devolvendo as probabilidades
por opção exatamente como `autojev.model.answer` (mesma temperatura de calibração).

Determinismo: o forward é um único passe sem amostragem. Para que 2 execuções
sejam bit-idênticas, a composição dos lotes também precisa ser idêntica: a ordem
é determinística (comprimento em tokens, desempate pelo índice) e o tamanho de
lote é fixo por requisição. Kernels determinísticos ligados quando possível.

Backends:
  --backend pplx  -> perplexity-ai/pplx-decider-v1.1-27b (backbone Qwen3_5Model +
                     readout [255,5120], atenção NÃO-causal nas camadas full-attention;
                     NÃO serve via vLLM — ver README).
  --backend vela  -> vllm-sr/vela3-flash (AutoModel trust_remote_code, system_one),
                     inclui perguntas `span` (exclusivas do Vela).

Uso (dentro do container, GPU 0):
  python decider_server.py --backend pplx --checkpoint /workspace/cemig/decider/models/pplx-decider-v1.1-27b --port 8006
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="cemig-decider")
STATE: dict[str, Any] = {"model": None, "backend": None, "name": None, "lock": threading.Lock()}


class DecideRequest(BaseModel):
    rows: list[dict[str, Any]]          # [{state, question, [images]}]
    batch_size: int = 8


class SystemOneRequest(BaseModel):
    items: list[dict[str, Any]]         # [{state, questions:{k:q}}] (vela)


def _set_determinism() -> None:
    """Liga o que existe de determinismo sem quebrar kernels sem implementação."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass


def load_pplx(checkpoint: str):
    sys.path.insert(0, str(Path(checkpoint) / "source" / "src"))
    from autojev.model import DecisionModel  # noqa: E402
    return DecisionModel(checkpoint, device="cuda")


def load_vela(checkpoint: str, batch_size: int):
    from transformers import AutoModel
    return AutoModel.from_pretrained(checkpoint, trust_remote_code=True, device="cuda:0",
                                     batch_size=batch_size)


def decide_pplx(model, rows: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    from autojev.model import answer
    # Ordem determinística por comprimento aproximado (chars) para reduzir padding.
    order = sorted(range(len(rows)), key=lambda i: (len(json.dumps(rows[i], ensure_ascii=False)), i))
    out: list[dict[str, Any] | None] = [None] * len(rows)
    with torch.inference_mode():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            batch_rows = [rows[i] for i in idx]
            try:
                batch = model.prepare(batch_rows)
            except ValueError as e:  # entrada longa demais: nunca truncar, devolver erro por linha
                for i in idx:
                    out[i] = {"error": "max_length_exceeded", "message": str(e)}
                continue
            logits = model(batch) / model.temperature
            dist = logits.softmax(-1).cpu().tolist()
            raw = logits.cpu().tolist()
            for j, i in enumerate(idx):
                n = batch.counts[j]
                ans = answer(rows[i]["question"], dist[j][:n])
                ans["logits"] = raw[j][:n]
                out[i] = ans
    return out  # type: ignore[return-value]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ready" if STATE["model"] is not None else "loading",
            "backend": STATE["backend"], "name": STATE["name"],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mem_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
            "mem_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None}


@app.post("/decide")
def decide(req: DecideRequest) -> dict[str, Any]:
    if STATE["model"] is None:
        raise HTTPException(503, "loading")
    if STATE["backend"] != "pplx":
        raise HTTPException(400, "/decide só no backend pplx; use /system_one")
    with STATE["lock"]:
        t0 = time.perf_counter()
        answers = decide_pplx(STATE["model"], req.rows, max(1, req.batch_size))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return {"model": STATE["name"], "answers": answers, "seconds": dt}


@app.post("/system_one")
def system_one(req: SystemOneRequest) -> dict[str, Any]:
    if STATE["model"] is None:
        raise HTTPException(503, "loading")
    if STATE["backend"] != "vela":
        raise HTTPException(400, "/system_one só no backend vela")
    res = []
    with STATE["lock"]:
        t0 = time.perf_counter()
        for it in req.items:
            try:
                r = STATE["model"].system_one(state=it["state"], questions=it["questions"])
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)[:300]}
            res.append(r)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return {"model": STATE["name"], "results": res, "seconds": dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["pplx", "vela"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, default=8006)
    ap.add_argument("--vela-batch", type=int, default=16)
    args = ap.parse_args()
    _set_determinism()
    STATE["backend"] = args.backend
    t0 = time.time()
    if args.backend == "pplx":
        STATE["model"] = load_pplx(args.checkpoint)
        STATE["name"] = "pplx-decider-v1.1-27b"
    else:
        STATE["model"] = load_vela(args.checkpoint, args.vela_batch)
        STATE["name"] = "vela3-flash"
    print(f"[load] {STATE['name']} em {time.time() - t0:.1f}s", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
