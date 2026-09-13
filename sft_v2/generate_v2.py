#!/usr/bin/env python3
"""
generate_v2.py — Geração de respostas do ALUNO sobre um pack (v2).

Igual em espírito ao slm_oraculo/generate.py, mas com o SYSTEM PROMPT configurável:
  --system v2      : SYNTHESIS_SYSTEM_PROMPT_V2 (permite recusa útil — o que o v2 treinou).
  --system numeros : prompt acionável do teto (compat. v1; usado no 27B/base p/ referência).
  --system prod    : SYNTHESIS_SYSTEM_PROMPT de PRODUÇÃO (recusa preguiçosa; baseline v1).

Serve os packs de 151 (oracle/retrieved) E os packs de avaliação v2 (val/refusal), que têm
o mesmo schema mínimo {item_id, question, context}. Higiene: paralelo, checkpoint, retry,
não-interativo. Comentários PT-BR; código em inglês.

Uso (2.6B via llama-server tunelado OU 27B via juiz):
  ../classifier/.venv/bin/python generate_v2.py --pack data/val_pack.json \
      --url http://127.0.0.1:8471/v1 --model "" --label sft_r16_val --system v2 \
      --out data/gen_sft_r16_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import requests

_HERE = Path(__file__).resolve().parent
import common_v2 as C  # noqa: E402
sys.path.insert(0, str(_HERE.parent / "bench" / "prompt_teto"))
from prompts import NUMEROS  # noqa: E402

SYSTEMS = {
    "v2": C.SYNTHESIS_SYSTEM_PROMPT_V2,
    "numeros": NUMEROS,
    "prod": C._PROD_SYNTHESIS_PROMPT,
}

LFM_SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05}
DET_SAMPLING = {"temperature": 0.0}


def synth(url: str, model: str, system: str, user: str, max_tokens: int,
          timeout: int, sampling: Dict[str, Any], think_off: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, **sampling,
    }
    if model:
        payload["model"] = model
    if think_off:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.perf_counter()
    r = requests.post(f"{url}/chat/completions", json=payload, timeout=timeout)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    return {"response": (msg.get("content") or "").strip(),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "wall_s_gpu": round(wall, 3),
            "finish_reason": d["choices"][0].get("finish_reason", "")}


def run(args: argparse.Namespace) -> None:
    system = SYSTEMS[args.system]
    pack = json.loads(Path(args.pack).read_text(encoding="utf-8"))
    all_items = pack["items"]
    sampling = DET_SAMPLING if args.deterministic else LFM_SAMPLING
    url = args.url.rstrip("/")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: Dict[str, Any] = {}
    if out_path.exists() and not args.overwrite:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for it in prev.get("items", []):
            if it.get("response") or it.get("__error__") is None:
                done[it["item_id"]] = it

    meta = {"task": "poc-sft-v2", "label": args.label, "model": args.model,
            "system": args.system, "pack": Path(args.pack).name,
            "sampling": sampling, "max_tokens": args.max_tokens,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pack_meta": pack.get("metadata", {})}

    def work(rec: Dict[str, Any]) -> Dict[str, Any]:
        user = C.build_user(rec["context"], rec["question"])
        last_err = None
        for attempt in range(4):
            try:
                s = synth(url, args.model, system, user, args.max_tokens,
                          args.timeout, sampling, args.think_off)
                out = {k: rec.get(k) for k in
                       ("item_id", "question", "doc", "section", "family", "facts",
                        "gold_text", "answerable", "expects_refusal", "retrieval_hit",
                        "golden_answer", "captain_case") if k in rec}
                out.update({"response": s["response"],
                            "prompt_tokens": s["prompt_tokens"],
                            "completion_tokens": s["completion_tokens"],
                            "wall_s_gpu": s["wall_s_gpu"],
                            "finish_reason": s["finish_reason"]})
                return out
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        return {"item_id": rec["item_id"], "question": rec.get("question", ""),
                "response": "", "__error__": str(last_err),
                "family": rec.get("family"), "answerable": rec.get("answerable"),
                "expects_refusal": rec.get("expects_refusal")}

    pending = [r for r in all_items if r["item_id"] not in done]
    results: Dict[str, Any] = dict(done)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r["item_id"] for r in pending}
        for fut in as_completed(futs):
            rec = fut.result()
            results[rec["item_id"]] = rec
            completed += 1
            if completed % 20 == 0:
                order = {r["item_id"]: i for i, r in enumerate(all_items)}
                srt = sorted(results.values(), key=lambda r: order.get(r["item_id"], 9999))
                out_path.write_text(json.dumps({"metadata": meta, "items": srt},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  [{args.label}] {completed}/{len(pending)}", flush=True)

    order = {r["item_id"]: i for i, r in enumerate(all_items)}
    srt = sorted(results.values(), key=lambda r: order.get(r["item_id"], 9999))
    out_path.write_text(json.dumps({"metadata": meta, "items": srt},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(srt)
    errs = sum(1 for it in srt if it.get("__error__"))
    avg = sum(it.get("completion_tokens", 0) for it in srt) / max(1, n)
    print(f"[ok] {args.label}: {n} itens ({errs} erros), tok médio={avg:.0f} -> {out_path}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--system", default="v2", choices=list(SYSTEMS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--think-off", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
