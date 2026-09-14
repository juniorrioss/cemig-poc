#!/usr/bin/env python3
"""
harness_oraculo.py — Harness das 4 medições da task poc-tools-oraculo.

Roda o modelo TOOL do tools_v1 (checkpoint já treinado, sem retreinar) sobre as 151 do holdout
em três modos de injeção de contexto no turno `tool` — a diferença é DE ONDE vem o trecho que
volta ao modelo após ele decidir chamar a ferramenta:

  --mode oracle : entrega o CHUNK-OURO direto no turno `tool` (bypass da busca). Responde a
                  MEDIÇÃO 1 (o capitão): com contexto perfeito, o treino de tool ajudou ou
                  atrapalhou a síntese vs o sft_1_2b sem tool?
  --mode hibrido: quando o modelo decide chamar, o harness IGNORA a `consulta` gerada e usa o
                  contexto do pipeline v4 EXTERNO (fala crua + classificador NR + RRF v4), lido
                  do cache `bench/prompt_teto/data/chunks_v4.json`. Responde a MEDIÇÃO 3 (o
                  desenho híbrido do capitão).
  --mode pure   : reproduz o tools_v1 puro — a busca usa a `consulta` do PRÓPRIO modelo (BM25 v4
                  top-K, nr vira filtro). É a configuração (B) da MEDIÇÃO 3 (comparação).

Em TODOS os modos o MODELO decide chamar/reusar/não-chamar (a arquitetura de tool). O que muda
é só o retorno do turno `tool`. A saída é gravada no MESMO schema do sft_v2/generate_v2.py
(campos item_id/question/doc/section/family/facts/gold_text/answerable/expects_refusal/response)
para ser pontuada por sft_v2/score_v2.py e sft_v3/refusal_metrics.py SEM alterá-los (reuso).

Além da resposta, grava metadados da DECISÃO (called_tool, consulta, nr, got_gold, ctx_source)
para as MEDIÇÕES 2 e 4 (recusa e decomposição da alucinação).

Higiene: paralelo, checkpoint (idempotente por item_id), retry, não-interativo, procedência.
Comentários PT-BR; identificadores em inglês.

Uso:
  ../classifier/.venv/bin/python harness_oraculo.py \
      --url http://100.79.169.101:8500 --label tools_r128_oracle_q4 \
      --mode oracle --pack 151 --out data/gen_tools_r128_oracle_q4.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE.parent / "tools_v1"))
sys.path.insert(0, str(_ROOT / "bench" / "regua"))
sys.path.insert(0, str(_ROOT / "slm_oraculo"))

import render_jinja as RJ  # noqa: E402  (tools_v1/render_jinja)
from render import parse_tool_calls_runtime  # noqa: E402
from tool_schema import (TOOLS, format_tool_result, make_assistant_message,  # noqa: E402
                         make_system_message, make_tool_call_message,
                         make_tool_result_message, make_user_message)

LFM_SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05}
DET_SAMPLING = {"temperature": 0.0}
STOP = ["<|im_end|>", "<|tool_call_end|>"]
TOOL_MARKERS = ("<|tool_call_start|>", "buscar_norma(")

V4_CACHE = _ROOT / "bench" / "prompt_teto" / "data" / "chunks_v4.json"
GABARITO = _ROOT / "bench" / "regua" / "data" / "gabarito_151.jsonl"


# ---------------------------------------------------------------------------
# Fontes de contexto por modo.
# ---------------------------------------------------------------------------
def _load_v4_cache() -> Dict[str, Any]:
    return json.loads(V4_CACHE.read_text(encoding="utf-8"))["items"]


def _complete(url: str, prompt: str, max_tokens: int, sampling: Dict[str, Any],
              timeout: int = 240, retries: int = 5) -> Dict[str, Any]:
    """Chama /completion (texto cru) do llama-server, com retry (rede oscila sob carga)."""
    payload = {"prompt": prompt, "n_predict": max_tokens, "stop": STOP,
               "cache_prompt": True, **sampling}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{url}/completion", json=payload, timeout=timeout)
            r.raise_for_status()
            d = r.json()
            return {"content": d.get("content", ""),
                    "tokens_predicted": d.get("tokens_predicted", 0)}
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"/completion falhou após {retries}: {last}")


def _search_pure(consulta: str, nr: Optional[str], topk: int) -> List[Dict[str, Any]]:
    """Busca do tools_v1 PURO: BM25 v4 top-K com a consulta DO MODELO; nr vira filtro-duro."""
    import retrieval_check as RC
    from corpus.eval_fino import app_fts_query
    ret = RC._retriever()
    q = app_fts_query(consulta)
    doc_filter = [nr.strip().lower()] if nr else None
    res = ret.search(q, topk, doc_filter=doc_filter)
    if not res and doc_filter:
        res = ret.search(q, topk)
    return [{"id": c["id"], "doc": c["doc"], "section": c["section"],
             "title": c.get("title", ""), "text": c.get("text", "")} for c in res]


def run_item(url: str, item: Dict[str, Any], mode: str, v4item: Optional[Dict[str, Any]],
             sampling: Dict[str, Any], topk: int, max_tokens: int,
             timeout: int) -> Dict[str, Any]:
    """Roda UM turno (single-turn) do usuário sob o modo escolhido. Retorna registro de saída."""
    question = item["question"]
    msgs = [make_system_message(), make_user_message(question)]
    prompt = RJ.render(msgs, tools=TOOLS, add_generation_prompt=True)
    first = _complete(url, prompt, max_tokens, sampling, timeout)
    raw = first["content"]
    tokens = first["tokens_predicted"]

    called = any(mk in raw for mk in TOOL_MARKERS)
    calls = parse_tool_calls_runtime(raw) if called else []
    syntactic_ok = bool(calls) and calls[0]["name"] == "buscar_norma" \
        and bool(str(calls[0]["arguments"].get("consulta", "")).strip())

    rec: Dict[str, Any] = {
        "called_tool": called, "syntactic_ok": syntactic_ok,
        "consulta": None, "nr": None, "ctx_source": None,
        "retrieved": [], "got_gold": None,
    }

    if called and syntactic_ok:
        args = calls[0]["arguments"]
        consulta = str(args.get("consulta", "")).strip()
        nr = args.get("nr")
        nr = str(nr).strip() if nr else None
        rec["consulta"], rec["nr"] = consulta, nr

        # Retorno do turno `tool` conforme o modo (a decisão de chamar foi do MODELO).
        if mode == "oracle":
            chunks = v4item["gold_chunks"]
            rec["ctx_source"] = "oracle_gold_chunk"
            rec["got_gold"] = True  # por construção, entregamos o ouro
        elif mode == "hibrido":
            chunks = v4item["chunks"]  # fala crua + classificador + RRF v4 (pipeline externo)
            rec["ctx_source"] = "hybrid_v4_rawquestion"
            rec["got_gold"] = bool(v4item.get("retrieval_hit"))
        elif mode == "pure":
            chunks = _search_pure(consulta, nr, topk)  # consulta DO MODELO
            rec["ctx_source"] = "pure_model_consulta"
            rec["got_gold"] = _got_gold_by_checkhit(chunks, item)
        else:
            raise ValueError(f"modo desconhecido: {mode}")

        rec["retrieved"] = [{"id": c["id"], "doc": c["doc"], "section": c["section"]}
                            for c in chunks]
        msgs2 = [make_system_message(), make_user_message(question),
                 make_tool_call_message(consulta, nr),
                 make_tool_result_message(format_tool_result(chunks))]
        prompt2 = RJ.render(msgs2, tools=TOOLS, add_generation_prompt=True)
        second = _complete(url, prompt2, max_tokens, sampling, timeout)
        answer = second["content"].strip()
        tokens += second["tokens_predicted"]
    else:
        # não chamou: resposta direta (limpa resíduo de tokens de tool).
        answer = raw
        for mk in TOOL_MARKERS:
            answer = answer.split(mk)[0]
        answer = answer.strip()
        rec["ctx_source"] = "no_tool_call"
        rec["got_gold"] = False

    # schema compatível com sft_v2/score_v2 + campos de decisão para as medições 2 e 4.
    out = {k: item.get(k) for k in
           ("item_id", "question", "doc", "section", "family", "facts", "gold_text",
            "answerable", "expects_refusal", "golden_answer", "captain_case") if k in item}
    out.update({
        "response": answer, "completion_tokens": tokens,
        "called_tool": rec["called_tool"], "syntactic_ok": rec["syntactic_ok"],
        "consulta": rec["consulta"], "nr": rec["nr"], "ctx_source": rec["ctx_source"],
        "retrieved": rec["retrieved"], "got_gold": rec["got_gold"],
    })
    return out


def _got_gold_by_checkhit(chunks: List[Dict[str, Any]], item: Dict[str, Any]) -> bool:
    """check_hit OFICIAL (doc+section OU id) — mesma régua do app."""
    from corpus.eval_retrieval import check_hit
    gold = {"doc": str(item.get("doc", "")).lower(), "section": item.get("section", ""),
            "chunk_id": item.get("gold_chunk_id", -1),
            "relevant_chunk_ids": item.get("relevant_chunk_ids", [])}
    return any(check_hit({"id": c["id"], "doc": c["doc"], "section": c["section"], "text": ""},
                         gold) for c in chunks)


# ---------------------------------------------------------------------------
# Packs.
# ---------------------------------------------------------------------------
def _load_gabarito() -> Dict[str, Dict[str, Any]]:
    return {json.loads(l)["id"]: json.loads(l)
            for l in GABARITO.read_text(encoding="utf-8").splitlines() if l.strip()}


def _load_gold_texts_151() -> Dict[str, str]:
    """Trecho-ouro do corpus CORRIGIDO por pergunta (checagem de alucinação)."""
    from corpus_fix import CorrectedGoldResolver
    from common import load_holdout
    resolver = CorrectedGoldResolver()
    out = {}
    for q in load_holdout():
        gc = resolver.resolve(q["doc"], q["section"], q["question"])
        out[q["id"]] = "\n\n".join(c["text"] for c in gc)
    return out


def build_pack_151(mode: str) -> List[Dict[str, Any]]:
    """As 151 do holdout com facts (gabarito) + gold_text (corpus corrigido) + gold/hybrid chunks."""
    v4 = _load_v4_cache()
    gab = _load_gabarito()
    gold_txt = _load_gold_texts_151()
    items = []
    for iid, v in v4.items():
        items.append({
            "item_id": iid, "question": v["question"], "doc": v["doc"], "section": v["section"],
            "family": "oracle", "answerable": True, "expects_refusal": False,
            "facts": gab.get(iid, {}).get("facts", []),
            "gold_text": gold_txt.get(iid, ""),
            "golden_answer": v.get("golden_answer", ""),
            "gold_chunk_id": (v["gold_chunks"][0]["id"] if v.get("gold_chunks") else -1),
            "captain_case": False,
            # contexto pré-computado por modo (oracle/hibrido); pure ignora estes.
            "gold_chunks": v.get("gold_chunks", []),
            "chunks": v.get("chunks", []),
            "retrieval_hit": v.get("retrieval_hit", False),
        })
    return items


def build_pack_refusal() -> List[Dict[str, Any]]:
    """Pack de recusa do sft_v2 (85 itens, 4 famílias). O contexto vem DENTRO do pack."""
    pack = json.loads((_ROOT / "sft_v2" / "data" / "refusal_test_pack.json")
                      .read_text(encoding="utf-8"))
    return pack["items"]


def run(args: argparse.Namespace) -> None:
    url = args.url.rstrip("/")
    sampling = DET_SAMPLING if args.deterministic else LFM_SAMPLING

    if args.pack == "151":
        items = build_pack_151(args.mode)
        v4_by_id = {it["item_id"]: it for it in items}
    elif args.pack == "refusal":
        items = build_pack_refusal()
        v4_by_id = {}
    else:
        raise ValueError(f"pack desconhecido: {args.pack}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: Dict[str, Any] = {}
    if out_path.exists() and not args.overwrite:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for it in prev.get("items", []):
            if it.get("response") and not it.get("__error__"):
                done[it["item_id"]] = it

    meta = {"task": "poc-tools-oraculo", "label": args.label, "model": args.model,
            "mode": args.mode, "pack": args.pack, "system": "tools",
            "sampling": sampling, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    def work(item: Dict[str, Any]) -> Dict[str, Any]:
        v4item = None
        if args.pack == "151":
            v4item = v4_by_id.get(item["item_id"])
        # pack refusal: contexto DENTRO do item -> força o turno tool com esse contexto.
        last_err = None
        for attempt in range(4):
            try:
                if args.pack == "refusal":
                    return _run_refusal_item(url, item, sampling, args.max_tokens, args.timeout)
                return run_item(url, item, args.mode, v4item, sampling, args.topk,
                                args.max_tokens, args.timeout)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        return {"item_id": item["item_id"], "question": item.get("question", ""),
                "response": "", "__error__": str(last_err),
                "family": item.get("family"), "answerable": item.get("answerable"),
                "expects_refusal": item.get("expects_refusal")}

    pending = [it for it in items if it["item_id"] not in done]
    results: Dict[str, Any] = dict(done)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it["item_id"] for it in pending}
        for fut in as_completed(futs):
            rec = fut.result()
            results[rec["item_id"]] = rec
            completed += 1
            if completed % 20 == 0:
                _flush(out_path, meta, results, items)
                print(f"  [{args.label}] {completed}/{len(pending)}", flush=True)
    _flush(out_path, meta, results, items)
    n = len(results)
    errs = sum(1 for it in results.values() if it.get("__error__"))
    called = sum(1 for it in results.values() if it.get("called_tool"))
    print(f"[ok] {args.label}: {n} itens ({errs} erros), called={called} -> {out_path}",
          flush=True)


def _run_refusal_item(url: str, item: Dict[str, Any], sampling: Dict[str, Any],
                      max_tokens: int, timeout: int) -> Dict[str, Any]:
    """No pack de recusa o contexto JÁ está no item (família recusa/distrator/parcial/oracle).

    O modelo TOOL sempre precisa 'chamar' para receber o contexto — então emulamos a decisão de
    chamar (usando a fala crua como consulta) e injetamos o contexto do pack no turno `tool`.
    Assim medimos a SÍNTESE/RECUSA da arquitetura de tool sobre EXATAMENTE o mesmo contexto que
    o sft_1_2b/v3 receberam (comparabilidade da MEDIÇÃO 2)."""
    question = item["question"]
    context = item.get("context", "")
    # 1º turno: deixamos o modelo decidir; mas para recusa precisamos do turno tool -> forçamos
    # a chamada com a fala crua e injetamos o contexto do pack (o retorno do tool é o pack).
    msgs = [make_system_message(), make_user_message(question),
            make_tool_call_message(question, item.get("nr")),
            make_tool_result_message(context)]
    prompt = RJ.render(msgs, tools=TOOLS, add_generation_prompt=True)
    second = _complete(url, prompt, max_tokens, sampling, timeout)
    answer = second["content"].strip()
    out = {k: item.get(k) for k in
           ("item_id", "question", "doc", "section", "family", "facts", "gold_text",
            "answerable", "expects_refusal", "context", "captain_case") if k in item}
    out.update({"response": answer, "completion_tokens": second["tokens_predicted"],
                "called_tool": True, "ctx_source": "refusal_pack_context"})
    return out


def _flush(out_path: Path, meta: Dict[str, Any], results: Dict[str, Any],
           order_items: List[Dict[str, Any]]) -> None:
    order = {it["item_id"]: i for i, it in enumerate(order_items)}
    srt = sorted(results.values(), key=lambda r: order.get(r["item_id"], 9999))
    out_path.write_text(json.dumps({"metadata": meta, "items": srt},
                                   ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", required=True, choices=["oracle", "hibrido", "pure"])
    ap.add_argument("--pack", default="151", choices=["151", "refusal"])
    ap.add_argument("--model", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
