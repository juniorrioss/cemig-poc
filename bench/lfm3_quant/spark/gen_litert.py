#!/usr/bin/env python3
"""gen_litert.py — Spark: gera os 344 prompts op_d50 (MESMO texto congelado dos GGUF) no
LiteRT-LM (Python 0.19.0, venv do scout lfm3-litert).

Desenho (decidido por sondas, ver RESULTS.md §LiteRT-LM — armadilhas):
- UMA Engine NOVA POR PROMPT: reusar a Engine entre conversas/sessões CONTAMINA o estado da
  shortconv (LiteRT-LM issue #3165 — reproduzido aqui: a mesma pergunta muda de resposta depois
  de outra conversa na mesma Engine). Com Engine nova a saída é determinística (seed fixo).
- Conversation com template IDENTIDADE (concatena o conteúdo): o prompt já vem renderizado pelo
  chat template HF; `render_message_to_string` confere 344/344 byte a byte e o runtime põe o BOS.
  A Conversation é o único caminho com repetition penalty (Session não expõe) — e SEM ela o
  int4 entra em laço ("35.5.1.1.1.1…"). Penalty 1.1 / janela 64 = llama.cpp (repeat_last_n 64).
- Sampler: top_k 50 / top_p 1 / temp 0.1 / seed. max_output_tokens 768.
Decode t/s = BenchmarkInfo do próprio runtime (enable_benchmark=True).

Uso: python gen_litert.py --model X.litertlm --backend cpu|gpu --fmt litert_int4_gpu \
        --tag litert_int4_gpu-r1 --seed 42 [--threads 6] [--shard 0/2]
"""
import argparse
import json
import time
from pathlib import Path

import litert_lm

Q = Path.home() / "cemig-poc/lfm3_quant"
BOS = "<|startoftext|>"
IDENT = ("{% for m in messages %}{% if m.content is string %}{{ m.content }}{% else %}"
         "{% for c in m.content %}{{ c.text }}{% endfor %}{% endif %}{% endfor %}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--fmt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rep", type=float, default=1.1)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    items = [json.loads(l) for l in open(Q / "prompts_op344_rendered.jsonl")][si::sn]
    outp = Q / "gen" / f"gen_{a.tag}.s{si}.jsonl"
    outp.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["key"] for l in open(outp)} if outp.exists() else set()
    todo = [it for it in items if it["key"] not in done]
    print(f"[gen] {a.tag} shard {a.shard} pendentes={len(todo)}", flush=True)
    fh = open(outp, "a")
    for k, it in enumerate(todo):
        p = it["prompt_text"]
        assert p.startswith(BOS)
        p = p[len(BOS):]
        be = litert_lm.Backend.GPU() if a.backend == "gpu" else litert_lm.Backend.CPU(thread_count=a.threads)
        t0 = time.time()
        eng = litert_lm.Engine(a.model, backend=be, max_num_tokens=2048,
                               cache_dir=str(Path(a.model).parent), enable_benchmark=True)
        conv = eng.create_conversation(
            chat_template=IDENT, max_output_tokens=768,
            sampler_config=litert_lm.SamplerConfig(top_k=50, top_p=1.0, temperature=0.1, seed=a.seed))
        assert conv.render_message_to_string({"role": "user", "content": p}) == p
        parity = eng.tokenize(BOS + p) == it["hf_ids"]
        rp = (litert_lm.RepetitionPenaltyConfig(repetition_penalty=a.rep, window_size=64)
              if a.rep > 1.0 else None)
        t1 = time.time()
        m = conv.send_message({"role": "user", "content": p}, repetition_penalty_config=rp)
        t2 = time.time()
        bi = conv.get_benchmark_info()
        conv.close()
        eng.close()
        txt = "".join(c.text for c in m.contents if hasattr(c, "text")) if m.contents else ""
        n_gen = bi.last_decode_token_count
        rec = {"key": it["key"], "id": it["id"], "engine_format": a.fmt, "tag": a.tag, "seed": a.seed,
               "tok_parity": parity, "n_prompt_tokens": it["n_prompt"],
               "n_prefill_runtime": bi.last_prefill_token_count, "raw": txt,
               "n_gen_tokens": n_gen, "stopped_eos": n_gen < 768,
               "decode_tps": round(bi.last_decode_tokens_per_second, 2),
               "prefill_tps": round(bi.last_prefill_tokens_per_second, 2),
               "init_s": round(t1 - t0, 2), "wall_s": round(t2 - t1, 2)}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        if k % 25 == 0:
            print(f"[gen] {k}/{len(todo)} parity={parity} dec={rec['decode_tps']} init={rec['init_s']}",
                  flush=True)
    print("[ok]", flush=True)


if __name__ == "__main__":
    main()
