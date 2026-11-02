#!/usr/bin/env python3
"""
run_dialog_harness.py — harness de DIÁLOGO do F4: roda o modelo com o
simulador de tool REAL (retriever v4 da POC respondendo de verdade, não mock)
e grava traces por suite para as métricas de tool-calling.

Loop por turno do usuário:
  render (apply_chat_template, enable_thinking=False) -> generate -> parse:
    - emitiu <tool_call>? executa buscar_norma(consulta) no retriever v4
      (top-k=5), injeta <tool_response>, gera de novo (máx 3 chamadas/turno);
    - sem tool_call -> resposta final do turno.

Suites (data/eval/suite_*.jsonl, de gen_dialogs_f4 --stage build):
  decision  : 1 turno; mede chamar-vs-devia (F1) + sintaxe.
  reuse     : histórico injetado (ciclo tool completo) + follow-up; NÃO deve chamar.
  switch    : histórico + pergunta de outra NR; DEVE chamar de novo.
  refusal   : pergunta cujo retriever REAL erra; deve chamar e depois RECUSAR.
  multihop  : pergunta cuja 1ª busca (hop1_ids congelados do dataset) é parcial;
              DEVE re-chamar com argumento DIFERENTE e sintetizar das duas.
              (hop1 é injetado congelado p/ garantir a parcialidade; a 2ª
              chamada roda no retriever REAL ao vivo.)

Saída: data/eval/trace_{tag}_{suite}.jsonl (checkpoint incremental por id).

Uso (local, .venv-f4, RTX 5070; llama-server :8399 no ar):
  .venv-f4/bin/python bench/lfm3_f4/eval/run_dialog_harness.py \
      --model /tmp/f4_base/lfm3-1.7b-data2x-ep4-merged \
      [--adapter /tmp/f4_adapters/checkpoint-XXX] --tag f4-r32-ep2 \
      [--suites decision,reuse,switch,refusal,multihop]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent            # bench/lfm3_f4/eval
F4 = HERE.parent
sys.path.insert(0, str(F4))

import common_f4 as C  # noqa: E402
from tool_parser import parse_tool_calls, strip_tool_calls, split_thinking  # noqa: E402
from tool_schema_f4 import (format_tool_result, make_assistant_message,  # noqa: E402
                            make_baked_system_message, make_tool_call_message,
                            make_tool_result_message, make_user_message)

EVAL = F4 / "data" / "eval"
MAX_CALLS_PER_TURN = 3
TOP_K = 5
EOS_ID = None   # resolvido do tokenizer


class Engine:
    """Geração local (transformers) com sampling idêntico ao gen_f2 (0.1/50/1.1)."""

    def __init__(self, model_path: str, adapter: str = ""):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="cuda")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            print(f"[engine] adapter: {adapter}")
        self.model.eval()
        self.eos_id = self.tok.convert_tokens_to_ids("<|im_end|>")

    def generate(self, messages: List[Dict[str, Any]], max_new: int = 1024) -> Dict[str, Any]:
        text = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new, do_sample=True,
                temperature=0.1, top_k=50, repetition_penalty=1.1,
                pad_token_id=self.eos_id, eos_token_id=self.eos_id)
        gen = out[0][enc["input_ids"].shape[1]:].tolist()
        stopped = self.eos_id in gen
        if stopped:
            gen = gen[: gen.index(self.eos_id)]
        raw = self.tok.decode(gen, skip_special_tokens=False)
        _, body = split_thinking(raw)
        return {"raw": raw, "body": body, "n_tokens": len(gen),
                "stopped_eos": stopped}


def run_turn(eng: Engine, rtv, messages: List[Dict[str, Any]],
             live_retrieval: bool = True,
             injected_first_result: Optional[str] = None) -> Dict[str, Any]:
    """Executa UM turno do usuário até resposta final (com loop de tool).

    injected_first_result: quando setado, a 1ª chamada do turno recebe este
    conteúdo em vez do retriever ao vivo (multihop: hop1 congelado parcial).
    """
    calls: List[Dict[str, Any]] = []
    raw_gens: List[str] = []
    final = ""
    syntactic_fail = False
    for ci in range(MAX_CALLS_PER_TURN + 1):
        g = eng.generate(messages)
        raw_gens.append(g["raw"])
        parsed = parse_tool_calls(g["body"])
        has_tag = "<tool_call>" in g["body"]
        if has_tag and not parsed:
            syntactic_fail = True
            final = strip_tool_calls(g["body"])
            break
        if not parsed:
            final = g["body"].strip()
            break
        call = parsed[0]
        consulta = str(call["arguments"].get("consulta") or "")
        nr = call["arguments"].get("nr")
        if injected_first_result is not None and ci == 0:
            result = injected_first_result
            chunk_ids: List[int] = []
        else:
            chunks = rtv.search_text(consulta, top_k=TOP_K) if live_retrieval else []
            result = format_tool_result(chunks)
            chunk_ids = [c["id"] for c in chunks]
        calls.append({"name": call["name"], "consulta": consulta, "nr": nr,
                      "chunk_ids": chunk_ids, "injected": injected_first_result is not None and ci == 0})
        messages = messages + [
            make_tool_call_message(consulta, nr),
            make_tool_result_message(result)]
    return {"final": final, "calls": calls, "n_calls": len(calls),
            "syntactic_fail": syntactic_fail, "raw_gens": raw_gens}


def hydrate_history(chunks_db: Dict[int, Dict], item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstrói o histórico (ciclo tool completo do 1º turno) de uma suite."""
    q1 = item["history_question"]
    ctx = [chunks_db[i] for i in item["history_ids"] if i in chunks_db]
    return [make_user_message(q1),
            make_tool_call_message(q1, C.detect_nr(q1)),
            make_tool_result_message(format_tool_result(ctx)),
            make_assistant_message(item["history_answer"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suites", default="decision,reuse,switch,refusal,multihop")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--emb-url", default="http://127.0.0.1:8399")
    args = ap.parse_args()

    from retriever_lazy import LazyRetrieverV4
    rtv = LazyRetrieverV4(emb_url=args.emb_url)
    eng = Engine(args.model, args.adapter)
    chunks_db = C.load_chunks()
    sys_msg = make_baked_system_message()

    for suite in args.suites.split(","):
        items = C.read_jsonl(EVAL / f"suite_{suite}.jsonl")
        if args.limit:
            items = items[: args.limit]
        out_path = EVAL / f"trace_{args.tag}_{suite}.jsonl"
        done = {r["id"] for r in C.read_jsonl(out_path)}
        todo = [it for it in items if it["id"] not in done]
        print(f"[harness] suite={suite} n={len(items)} pendentes={len(todo)}")
        t0 = time.time()
        for i, item in enumerate(todo):
            if suite in ("decision", "refusal"):
                msgs = [sys_msg, make_user_message(item["question"])]
                res = run_turn(eng, rtv, msgs)
            elif suite == "reuse":
                msgs = [sys_msg] + hydrate_history(chunks_db, item) + \
                    [make_user_message(item["followup"])]
                res = run_turn(eng, rtv, msgs)
            elif suite == "switch":
                msgs = [sys_msg] + hydrate_history(chunks_db, item) + \
                    [make_user_message(item["new_question"])]
                res = run_turn(eng, rtv, msgs)
            elif suite == "multihop":
                hop1 = [chunks_db[i] for i in item["hop1_ids"] if i in chunks_db]
                msgs = [sys_msg, make_user_message(item["question"])]
                res = run_turn(eng, rtv, msgs,
                               injected_first_result=format_tool_result(hop1))
            else:
                raise ValueError(suite)
            rec = {"id": item["id"], "suite": suite, "tag": args.tag, **res}
            # raw_gens só do último (traço completo pesa; o último basta p/ debug)
            rec["raw_last"] = rec.pop("raw_gens")[-1][-2000:]
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  ... {i+1}/{len(todo)} ({rate:.2f} it/s)", flush=True)
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
