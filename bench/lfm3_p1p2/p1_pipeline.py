#!/usr/bin/env python3
"""
p1_pipeline.py — P1: FILTRO DE APRENDIBILIDADE (ferramenta reutilizável).

Ideia (paper Rufus-Air, "a escolha mais importante do RL"): só ensina o que a
política ATUAL às vezes acerta. Para cada item do universo roda-se a política
n=4 vezes (temp 1.0) e mede-se a taxa de acerto pela RÉGUA AUTOMÁTICA
(`verifier.py`, sem juiz). Bandas:

  acc >= 0.8        -> DESCARTA ("easy": não ensina)
  0 < acc < 0.8     -> BANDA PRODUTIVA (mantém)
  acc == 0          -> filtro de CORRETUDE: o 27B com o MESMO contexto resolve
                       (pela mesma régua, 2 tentativas)? sim -> mantém como
                       "hard"; não -> item mal especificado, DESCARTA e lista.

Subcomandos (cada um idempotente/reentrante; caminhos via --work):

  build        [local]     monta o universo (data2x SFT + famílias F4 + pool RLVR)
  facts        [Qwen]      extrai fatos obrigatórios da resposta-referência (27B)
  calibrate    [local]     régua aplicada às próprias referências (sanidade do contrato)
  rollout      [GPU]       n amostras da política por item (HF generate em lote)
  score        [local]     acc por item pela régua
  correctness  [Qwen]     27B resolve os acc==0? (mesmo contexto, mesma régua)
  bands        [local]     relatório de bandas + dataset-P1 filtrado + prompts P2

Universo (fontes):
  data2x  : SFT F3b-data2x (11.151 linhas, casos gold/absent/partial/closed)
  f4      : diálogos F4 (4.533, 6 famílias) — avalia-se o ÚLTIMO turno assistant
  rlvr    : pool de prompts candidatos a RLVR — perguntas do data2x com contexto do
            RETRIEVER OP (F6: gate 0,7/2x + manuais 1,75 margem, top-5 -> 50%) no
            formato do APP (tool turn); expect=answer se o fato da unidade está no
            contexto comprimido (containment >= 0,70), refuse se < 0,30, senão fora.

Uso típico:
  PY=classifier/.venv/bin/python
  $PY bench/lfm3_p1p2/p1_pipeline.py build --work /mnt/d/cemig-scratch/p1p2
  $PY bench/lfm3_p1p2/p1_pipeline.py facts --work ... --workers 64
  (container) python p1_pipeline.py rollout --work /workspace/cemig/p1p2 --model M
  $PY ... score / correctness / bands
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

SEED = 20261109
DEFAULT_WORK = Path(os.environ.get("P1_WORK", "/mnt/d/cemig-scratch/p1p2"))
BANDS = {"easy": 0.8}           # acc >= 0.8 descarta


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------
def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def uid_of(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _load_qa() -> Dict[str, Dict[str, Any]]:
    qa = {}
    for p in sorted(glob.glob(str(REPO / "bench" / "lfm3_f*" / "data" / "synth" / "qa_passed_nr*.jsonl"))):
        for r in read_jsonl(Path(p)):
            qa.setdefault(r["qa_id"], r)
    return qa


def _load_units() -> Dict[str, Dict[str, Any]]:
    units = {}
    for p in sorted(glob.glob(str(REPO / "bench" / "lfm3_f*" / "data" / "synth" / "units_nr*.jsonl"))):
        for u in read_jsonl(Path(p)):
            units.setdefault(u["unit_id"], u)
    return units


def _gold_items(ref: str, target: str, context: str = "", question: str = "") -> List[str]:
    """Códigos de item-ouro VERIFICÁVEIS: união ref da unidade + códigos do alvo 27B
    (escrito com o chunk-ouro à vista), RESTRITA aos códigos citáveis no contexto ou
    na pergunta (igual, ou o contexto traz um filho mais específico). Código-ouro
    que não aparece no contexto não é verificável por offsets -> fora. Lista vazia
    = citação dispensada. Sem contexto (closed) = sem restrição."""
    from verifier import item_codes
    cand = item_codes(ref) | item_codes(target)
    if not context:
        return sorted(cand)
    visible = item_codes(context) | item_codes(question)
    return sorted(g for g in cand
                  if any(v == g or v.startswith(g + ".") for v in visible))


DATA2X_EXPECT = {"gold": "answer", "absent": "refuse", "partial": "partial", "closed": "closed"}
F4_EXPECT = {"chamada_simples": "answer", "recusa": "refuse", "reuso": "answer",
             "troca": "answer", "multihop": "answer", "sem_ferramenta": "no_tool"}


def build_data2x(work: Path, qa: Dict) -> List[Dict[str, Any]]:
    rows = read_jsonl(work / "data2x" / "train.jsonl")
    out = []
    for i, r in enumerate(rows):
        msgs = r["messages"]
        q = qa[r["qa_id"]]
        target = msgs[-1]["content"]
        user = msgs[1]["content"]
        ctx = "" if r["case"] == "closed" else user.split("\nPergunta:\n")[0]
        out.append({
            "uid": uid_of("data2x", r["qa_id"], str(i)), "source": "data2x",
            "family": r["case"], "qa_id": r["qa_id"], "nr": r["nr"], "k": r["k"],
            "expect": DATA2X_EXPECT[r["case"]], "prompt": msgs[:-1], "target": target,
            "question": q["question"], "ref": q["ref"], "unit_id": q["unit_id"],
            "context": ctx, "gold_items": _gold_items(q["ref"], target, ctx, q["question"])
            if r["case"] in ("gold", "closed") else [],
            "row_index": i,
        })
    return out


def build_f4(work: Path, qa: Dict) -> List[Dict[str, Any]]:
    rows = read_jsonl(work / "f4" / "train.jsonl")
    out = []
    for i, r in enumerate(rows):
        msgs = r["messages"]
        last = len(msgs) - 1
        assert msgs[last]["role"] == "assistant" and not msgs[last].get("tool_calls")
        prompt = msgs[:last]
        users = [m for m in prompt if m["role"] == "user"]
        question = users[-1]["content"]
        # contexto = turnos tool visíveis ao responder (troca: só o ciclo do 2º tópico
        # também está no histórico — o modelo pode citar ambos; mantém todos)
        ctx = "\n\n".join(m["content"] for m in prompt if m["role"] == "tool")
        qa_id = r["qa_id"].split("+")[-1]
        q = qa.get(qa_id, {})
        fam = r["family"]
        target = msgs[last]["content"]
        ref = q.get("ref", "") if fam in ("chamada_simples", "troca", "multihop") else ""
        out.append({
            "uid": uid_of("f4", r["qa_id"], str(i)), "source": "f4", "family": fam,
            "qa_id": r["qa_id"], "nr": r.get("nr"), "expect": F4_EXPECT[fam],
            "prompt": prompt, "target": target, "question": question, "ref": ref,
            "unit_id": q.get("unit_id"), "context": ctx,
            "gold_items": _gold_items(ref, target, ctx, question) if F4_EXPECT[fam] == "answer" else [],
            "row_index": i,
        })
    return out


def _build_rlvr_absent(work, qa, units, b, E, norm_tokens, n_absent: int) -> None:
    """Ouro AUSENTE com contexto REAL: mesma pergunta, ranking do retriever op
    (top-12) SEM os chunks que contêm o fato da unidade (containment >= 0,30),
    top-5 restante -> compressão 50% com o MESMO orçamento da célula op.
    Simula o miss do retriever (o caso em que o app deve recusar)."""
    from compress import ntok  # noqa: PLC0415
    from eval_retrieval import variant_cfg  # noqa: PLC0415
    src = [r for r in read_jsonl(work / "universe_rlvr.raw.jsonl") if r["containment"] >= 0.70]
    random.Random(SEED + 7).shuffle(src)
    src = src[:n_absent]
    out_path = work / "universe_rlvr_absent.raw.jsonl"
    done = {r["qa_id"] for r in read_jsonl(out_path)}
    with out_path.open("a", encoding="utf-8") as fh:
        for r in src:
            if r["qa_id"] in done:
                continue
            q = qa[r["qa_id"]]
            u = units.get(q["unit_id"]) or {}
            ft = set(norm_tokens(u.get("fato", "")))
            if not ft:
                continue
            qtext = q["question"]
            qv = b.rtv.qc.get(qtext)
            ctrl = b.rtv.search(qtext, variant_cfg("v4app"), top_k=5)["ids"][:2]
            budget = int(ntok(b.comp.render_raw(ctrl)) * E.RATIO)
            ids = b.rtv.search(qtext, variant_cfg(E.OP), top_k=12)["ids"]
            keep = []
            for cid in ids:
                txt = b.rtv.chunk(cid).get("text", "")
                if len(ft & set(norm_tokens(txt))) / len(ft) < 0.30:
                    keep.append(cid)
            keep = keep[:5]
            if len(keep) < 3:
                continue
            c = b.comp.compress(qv, keep, budget, q=qtext, **E.SCORER)
            cont = len(ft & set(norm_tokens(c["text"]))) / len(ft)
            fh.write(json.dumps({"qa_id": r["qa_id"], "ctx": c["text"], "ids": keep,
                                 "containment": round(cont, 4), "gold_removed": True,
                                 "messages": b.messages(qtext, c["text"])},
                                ensure_ascii=False) + "\n")


def build_rlvr(work: Path, qa: Dict, units: Dict, n_max: int,
               n_absent: int = 1500) -> List[Dict[str, Any]]:
    """Pool RLVR com o retriever OP da F6 no formato do APP (prompt = system baked +
    user + tool_call(fala crua) + tool(top5->50%)). Precisa do encoder :8399."""
    sys.path.insert(0, str(REPO / "bench" / "lfm3_f6"))
    os.environ.setdefault("LFM3_TOKENIZER", str(next(Path.home().glob(
        ".cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/*/tokenizer.json"))))
    sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "synth"))
    import e2e_build_f6 as E  # noqa: PLC0415
    from common_synth import norm_tokens  # noqa: PLC0415

    assigns = read_jsonl(REPO / "bench" / "lfm3_f3b" / "data" / "train" / "data2x" / "assignments.jsonl")
    cands = [a for a in assigns if a["case"] in ("gold", "absent", "partial")]
    rng = random.Random(SEED)
    rng.shuffle(cands)
    cands = cands[:n_max]
    out_path = work / "universe_rlvr.raw.jsonl"
    done = {r["qa_id"] for r in read_jsonl(out_path)}
    b = E.Builder()
    t0, n = time.time(), 0
    with out_path.open("a", encoding="utf-8") as fh:
        for a in cands:
            if a["qa_id"] in done:
                continue
            q = qa[a["qa_id"]]
            c = b.cell(q["question"], E.OP, "c5")
            u = units.get(q["unit_id"]) or {}
            ft = set(norm_tokens(u.get("fato", "")))
            ct = set(norm_tokens(c["text"]))
            cont = len(ft & ct) / len(ft) if ft else 0.0
            fh.write(json.dumps({"qa_id": a["qa_id"], "ctx": c["text"], "ids": c["ids"],
                                 "containment": round(cont, 4),
                                 "messages": b.messages(q["question"], c["text"])},
                                ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"[rlvr] {n} ({n / (time.time() - t0):.1f}/s)", flush=True)
                b.rtv.qc.save()
    b.rtv.qc.save()
    _build_rlvr_absent(work, qa, units, b, E, norm_tokens, n_absent)
    out = []
    for r in read_jsonl(out_path) + read_jsonl(work / "universe_rlvr_absent.raw.jsonl"):
        q = qa[r["qa_id"]]
        if r["containment"] >= 0.70 and not r.get("gold_removed"):
            expect = "answer"
        elif r["containment"] < 0.30:
            expect = "refuse"
        else:
            continue                          # ambíguo: fora do pool binário
        tagr = "rlvr_absent" if r.get("gold_removed") else "rlvr"
        out.append({
            "uid": uid_of(tagr, r["qa_id"]), "source": "rlvr",
            "family": "rlvr_refuse_removed" if r.get("gold_removed") else f"rlvr_{expect}",
            "qa_id": r["qa_id"], "nr": q["nr"], "expect": expect, "prompt": r["messages"],
            "target": q["answer"] if expect == "answer" else "", "question": q["question"],
            "ref": q["ref"], "unit_id": q["unit_id"], "context": r["ctx"],
            "gold_items": _gold_items(q["ref"], q["answer"], r["ctx"], q["question"])
            if expect == "answer" else [],
            "containment": r["containment"], "ctx_ids": r["ids"],
        })
    return out


def cmd_build(args) -> None:
    work = Path(args.work)
    qa, units = _load_qa(), _load_units()
    parts = {}
    if "data2x" in args.sources:
        parts["data2x"] = build_data2x(work, qa)
    if "f4" in args.sources:
        parts["f4"] = build_f4(work, qa)
    if "rlvr" in args.sources:
        parts["rlvr"] = build_rlvr(work, qa, units, args.rlvr_max, args.rlvr_absent)
    for k, rows in parts.items():
        write_jsonl(work / f"universe_{k}.jsonl", rows)
        print(k, len(rows), dict(collections.Counter(r["family"] for r in rows)))


# ---------------------------------------------------------------------------
# facts (27B): fatos obrigatórios da resposta-referência
# ---------------------------------------------------------------------------
FACTS_PROMPT = """Você é um auditor técnico de Normas Regulamentadoras brasileiras (NRs).
A partir da RESPOSTA-REFERÊNCIA abaixo (correta, escrita com o trecho da norma à vista),
extraia a lista dos FATOS OBRIGATÓRIOS que QUALQUER resposta correta à pergunta precisa
conter.

Um FATO OBRIGATÓRIO é um item concreto e verificável: um equipamento, uma etapa, uma
condição, um valor/limite, uma exigência, um prazo, uma proibição.

REGRAS ESTRITAS:
- SOMENTE fatos presentes na RESPOSTA-REFERÊNCIA. Não invente, não use conhecimento externo.
- Cada fato CURTO (1 a 6 palavras), ATÔMICO, no vocabulário da norma.
- Foque no que responde à PERGUNTA. De 2 a 5 fatos.
- Não inclua o número da norma/item como fato (a citação é verificada à parte).

[PERGUNTA]
{question}

[RESPOSTA-REFERÊNCIA]
{answer}

Retorne EXCLUSIVAMENTE um JSON estrito: {{"fatos": ["fato 1", "fato 2"]}}"""


def cmd_facts(args) -> None:
    sys.path.insert(0, str(REPO / "bench" / "regua"))
    sys.path.insert(0, str(REPO / "bench" / "lfm3_f0"))
    from common import judge_call, parallel_map  # noqa: PLC0415
    from build_gabarito_f0 import _parse_facts  # noqa: PLC0415

    work = Path(args.work)
    items = []
    for k in args.sources:
        items += read_jsonl(work / f"universe_{k}.jsonl")
    # chave = (pergunta, referência) — o pool rlvr reusa os fatos do QA
    need = {}
    for it in items:
        if it["expect"] in ("answer", "partial", "closed") and it["target"]:
            key = uid_of(it["question"], it["target"])
            need[key] = {"key": key, "question": it["question"], "answer": it["target"]}
    ckpt = work / "facts.ckpt.json"
    existing = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    print(f"[facts] {len(need)} chaves ({len(existing)} prontas)", flush=True)

    def worker(x):
        raw = judge_call(FACTS_PROMPT.format(question=x["question"], answer=x["answer"]),
                         max_tokens=400, temperature=0.0)
        return {"facts": _parse_facts(raw)}

    res = parallel_map(worker, list(need.values()), max_workers=args.workers,
                       checkpoint_path=ckpt, key_fn=lambda x: x["key"], existing=existing)
    bad = sum(1 for v in res.values() if "__error__" in v or not v.get("facts"))
    print(f"[facts] ok={len(res) - bad} vazios/erros={bad}")


def attach_facts(work: Path, items: List[Dict[str, Any]]) -> None:
    facts = json.loads((work / "facts.ckpt.json").read_text())
    for it in items:
        if it["expect"] in ("answer", "partial", "closed"):
            f = facts.get(uid_of(it["question"], it["target"]), {})
            it["facts"] = f.get("facts", [])
        else:
            it["facts"] = []


def load_universe(work: Path, sources: List[str]) -> List[Dict[str, Any]]:
    items = []
    for k in sources:
        items += read_jsonl(work / f"universe_{k}.jsonl")
    attach_facts(work, items)
    return items


# ---------------------------------------------------------------------------
# calibrate: a régua nas próprias referências (devem passar quase sempre)
# ---------------------------------------------------------------------------
def cmd_calibrate(args) -> None:
    from verifier import verify  # noqa: PLC0415
    work = Path(args.work)
    items = [it for it in load_universe(work, args.sources) if it["target"]]
    out = {}
    for x in (0.34, 0.5, 0.67, 0.8):
        by = collections.defaultdict(collections.Counter)
        for it in items:
            v = verify(it["target"], it, x=x)
            by[f'{it["source"]}:{it["expect"]}'][v.reason] += 1
        out[str(x)] = {k: {"n": sum(c.values()), "pass_pct": round(100 * c["ok"] / sum(c.values()), 1)
                           if "ok" in c or True else 0,
                           "reasons": dict(c.most_common(5))} for k, c in by.items()}
    (work / "calibrate_refs.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for x, d in out.items():
        print("X =", x)
        for k, v in sorted(d.items()):
            print(f"   {k:<22} n={v['n']:<6} ok={v['reasons'].get('ok', 0) / v['n'] * 100:5.1f}%  {v['reasons']}")


# ---------------------------------------------------------------------------
# rollout (GPU): n amostras por item, HF generate em lote
# ---------------------------------------------------------------------------
def cmd_rollout(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work = Path(args.work)
    items = []
    for k in args.sources:
        items += read_jsonl(work / f"universe_{k}.jsonl")
    if args.limit:
        random.Random(SEED).shuffle(items)
        items = items[: args.limit]
    out_path = work / f"rollouts_{args.tag}.jsonl"
    done = {r["uid"] for r in read_jsonl(out_path)}
    todo = [it for it in items if it["uid"] not in done]
    print(f"[rollout] tag={args.tag} pendentes={len(todo)} feitos={len(done)}", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    for it in todo:
        it["_text"] = tok.apply_chat_template(it["prompt"], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
        it["_n"] = len(tok(it["_text"], add_special_tokens=False)["input_ids"])
    todo.sort(key=lambda x: x["_n"])
    torch.manual_seed(SEED)
    t0, dn = time.time(), 0
    i = 0
    while i < len(todo):
        # lote por orçamento de tokens (prompts longos -> lotes menores)
        bs = max(1, min(args.batch, args.token_budget // max(1, todo[min(i + args.batch, len(todo)) - 1]["_n"] * args.n)))
        batch = todo[i:i + bs]
        i += bs
        enc = tok([b["_text"] for b in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=args.temperature, top_k=0, top_p=1.0,
                                 num_return_sequences=args.n, pad_token_id=pad,
                                 eos_token_id=eos)
        L = enc["input_ids"].shape[1]
        with out_path.open("a", encoding="utf-8") as fh:
            for j, b in enumerate(batch):
                samples = []
                for s in range(args.n):
                    gen = out[j * args.n + s][L:].tolist()
                    stopped = eos in gen
                    if stopped:
                        gen = gen[: gen.index(eos)]
                    while gen and gen[-1] == pad:
                        gen.pop()
                    samples.append({"text": tok.decode(gen, skip_special_tokens=False),
                                    "n_tok": len(gen), "stopped": stopped})
                fh.write(json.dumps({"uid": b["uid"], "n_prompt": b["_n"], "samples": samples},
                                    ensure_ascii=False) + "\n")
        dn += len(batch)
        el = time.time() - t0
        print(f"[rollout] {dn}/{len(todo)} bs={bs} ({dn / el:.2f} it/s, "
              f"ETA {(len(todo) - dn) / (dn / el) / 60:.1f} min)", flush=True)
    print(f"[ok] {out_path}", flush=True)


# ---------------------------------------------------------------------------
# score: acc por item (régua automática)
# ---------------------------------------------------------------------------
def cmd_score(args) -> None:
    from verifier import verify  # noqa: PLC0415
    work = Path(args.work)
    items = {it["uid"]: it for it in load_universe(work, args.sources)}
    out = []
    for r in read_jsonl(work / f"rollouts_{args.tag}.jsonl"):
        it = items.get(r["uid"])
        if it is None:
            continue
        vs = [verify(s["text"], it, stopped=s["stopped"], x=args.x) for s in r["samples"]]
        out.append({"uid": r["uid"], "source": it["source"], "family": it["family"],
                    "expect": it["expect"], "nr": it["nr"],
                    "acc": sum(v.reward for v in vs) / len(vs),
                    "rewards": [v.reward for v in vs], "reasons": [v.reason for v in vs],
                    "n_tok": [s["n_tok"] for s in r["samples"]],
                    "n_facts": len(it.get("facts") or [])})
    write_jsonl(work / f"scores_{args.tag}.jsonl", out)
    c = collections.Counter(("easy" if s["acc"] >= BANDS["easy"] else
                             "zero" if s["acc"] == 0 else "prod") for s in out)
    print(f"[score] n={len(out)} {dict(c)}")
    rs = collections.Counter(x for s in out for x in s["reasons"])
    print("razões:", rs.most_common(10))


# ---------------------------------------------------------------------------
# correctness (27B): os acc==0 são resolvíveis com o MESMO contexto?
# ---------------------------------------------------------------------------
def _flatten_prompt(prompt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converte o diálogo (com tool_calls/tool) em mensagens simples p/ o 27B
    sem ferramentas: o conteúdo é o MESMO, só muda o invólucro."""
    msgs = []
    sys_txt = prompt[0]["content"] if prompt and prompt[0]["role"] == "system" else ""
    sys_txt = sys_txt.split("\nVocê tem a ferramenta")[0] if "List of tools" in sys_txt or "<tools>" in sys_txt or "buscar_norma" in sys_txt else sys_txt
    sys_txt += ("\nResponda à ÚLTIMA pergunta do usuário. Se os trechos de norma fornecidos "
                "não contiverem a informação, diga claramente que não encontrou nas normas "
                "recuperadas. Ao responder, cite a NR e o número do item.")
    msgs.append({"role": "system", "content": sys_txt})
    buf = []
    for m in prompt[1:]:
        if m["role"] == "user":
            buf.append(f"[USUÁRIO]\n{m['content']}")
        elif m["role"] == "tool":
            buf.append(f"[TRECHOS DE NORMA RECUPERADOS]\n{m['content']}")
        elif m["role"] == "assistant" and m.get("tool_calls"):
            a = m["tool_calls"][0]["function"]["arguments"]
            buf.append(f"[BUSCA REALIZADA: {a.get('consulta')}]")
        elif m["role"] == "assistant":
            buf.append(f"[ASSISTENTE]\n{m['content']}")
    msgs.append({"role": "user", "content": "\n\n".join(buf)})
    return msgs


def cmd_correctness(args) -> None:
    sys.path.insert(0, str(REPO / "bench" / "regua"))
    from common import parallel_map  # noqa: PLC0415
    import requests  # noqa: PLC0415
    from verifier import verify  # noqa: PLC0415

    url = os.environ.get("REGUA_JUDGE_URL", "http://10.100.0.111:8005/v1")
    work = Path(args.work)
    items = {it["uid"]: it for it in load_universe(work, args.sources)}
    zeros = [s for s in read_jsonl(work / f"scores_{args.tag}.jsonl") if s["acc"] == 0]
    ckpt = work / f"correctness_{args.tag}.ckpt.json"
    existing = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    print(f"[correct] acc==0: {len(zeros)} ({len(existing)} prontos)", flush=True)

    def worker(s):
        it = items[s["uid"]]
        payload = {"model": "Qwen/Qwen3.8-27B-FP8", "messages": _flatten_prompt(it["prompt"]),
                   "max_tokens": 600, "temperature": 0.7, "n": args.tries,
                   "chat_template_kwargs": {"enable_thinking": False}}
        for att in range(3):
            try:
                r = requests.post(f"{url}/chat/completions", json=payload, timeout=180)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(2 * (att + 1))
        ch = r.json()["choices"]
        texts = [c["message"]["content"] for c in ch]
        vs = [verify(t, it, stopped=c.get("finish_reason") == "stop", x=args.x)
              for t, c in zip(texts, ch)]
        return {"solved": any(v.reward for v in vs), "reasons": [v.reason for v in vs],
                "answer": texts[0][:1500]}

    res = parallel_map(worker, zeros, max_workers=args.workers, checkpoint_path=ckpt,
                       key_fn=lambda s: s["uid"], existing=existing)
    ok = sum(1 for v in res.values() if v.get("solved"))
    print(f"[correct] 27B resolve {ok}/{len(res)}")


# ---------------------------------------------------------------------------
# bands: relatório + datasets finais
# ---------------------------------------------------------------------------
def band_of(s: Dict[str, Any], corr: Dict[str, Any]) -> str:
    if s["acc"] >= BANDS["easy"]:
        return "easy_drop"
    if s["acc"] > 0:
        return "productive"
    c = corr.get(s["uid"])
    if c is None or "__error__" in c:
        return "zero_unchecked"
    return "hard_keep" if c["solved"] else "ill_drop"


def cmd_bands(args) -> None:
    work = Path(args.work)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = {it["uid"]: it for it in load_universe(work, args.sources)}
    scores = read_jsonl(work / f"scores_{args.tag}.jsonl")
    cp = work / f"correctness_{args.tag}.ckpt.json"
    corr = json.loads(cp.read_text()) if cp.exists() else {}
    rows = []
    for s in scores:
        it = items[s["uid"]]
        rows.append({**s, "band": band_of(s, corr), "qa_id": it["qa_id"]})

    def table(key_fn):
        t = collections.defaultdict(collections.Counter)
        for r in rows:
            t[key_fn(r)][r["band"]] += 1
        res = {}
        for k, c in sorted(t.items(), key=lambda kv: str(kv[0])):
            n = sum(c.values())
            res[str(k)] = {"n": n, **{b: c[b] for b in ("easy_drop", "productive", "hard_keep",
                                                       "ill_drop", "zero_unchecked")},
                           "drop_pct": round(100 * (c["easy_drop"] + c["ill_drop"]) / n, 1),
                           "mean_acc": round(sum(r["acc"] for r in rows if str(key_fn(r)) == str(k)) / n, 3)}
        return res

    report = {
        "tag": args.tag, "x_coverage": args.x, "n_samples": len(scores[0]["rewards"]) if scores else 0,
        "thresholds": {"easy_drop": "acc>=0.8", "productive": "0<acc<0.8",
                       "hard_keep": "acc==0 & 27B resolve", "ill_drop": "acc==0 & 27B não resolve"},
        "total": table(lambda r: "all")["all"],
        "by_source": table(lambda r: r["source"]),
        "by_family": table(lambda r: f'{r["source"]}:{r["family"]}'),
        "by_nr": table(lambda r: f'NR-{r["nr"]}' if r["nr"] else "NR-?"),
        "acc_hist": dict(collections.Counter(round(r["acc"], 2) for r in rows)),
    }
    (out_dir / "p1_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_jsonl(work / f"bands_{args.tag}.jsonl", rows)
    keep = {r["uid"] for r in rows if r["band"] in ("productive", "hard_keep")}
    # datasets finais (versionáveis: só índices/uids + banda; o texto é regenerável)
    manifest = [{"uid": r["uid"], "source": r["source"], "family": r["family"], "qa_id": r["qa_id"],
                 "row_index": items[r["uid"]].get("row_index"), "acc": r["acc"], "band": r["band"]}
                for r in rows]
    write_jsonl(out_dir / "p1_manifest.jsonl", manifest)
    ill = [{"uid": r["uid"], "source": r["source"], "family": r["family"], "qa_id": r["qa_id"],
            "question": items[r["uid"]]["question"], "ref": items[r["uid"]].get("ref"),
            "reasons_policy": r["reasons"], "reasons_27b": corr.get(r["uid"], {}).get("reasons")}
           for r in rows if r["band"] == "ill_drop"]
    write_jsonl(out_dir / "p1_ill_specified.jsonl", ill)
    # dataset-P1 filtrado (texto completo, fora do git: work/)
    for src in ("data2x", "f4"):
        orig = read_jsonl(work / src / "train.jsonl")
        sel = [orig[items[u]["row_index"]] for u in keep if items[u]["source"] == src]
        if sel:
            write_jsonl(work / f"p1_{src}_train.jsonl", sel)
    rl = [items[u] for u in keep if items[u]["source"] == "rlvr"]
    write_jsonl(work / "p2_prompts.jsonl", rl)
    print(json.dumps(report["total"], indent=1))
    print("by_source:", json.dumps(report["by_source"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, tag=True):
        p.add_argument("--work", default=str(DEFAULT_WORK))
        p.add_argument("--sources", nargs="+", default=["data2x", "f4", "rlvr"])
        if tag:
            p.add_argument("--tag", default="f4ep2")
        return p

    p = common(sub.add_parser("build"), tag=False)
    p.add_argument("--rlvr-max", type=int, default=6000)
    p.add_argument("--rlvr-absent", type=int, default=1500)
    p = common(sub.add_parser("facts"), tag=False)
    p.add_argument("--workers", type=int, default=64)
    common(sub.add_parser("calibrate"), tag=False)
    p = common(sub.add_parser("rollout"))
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default="")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--token-budget", type=int, default=600_000)
    p.add_argument("--limit", type=int, default=0)
    p = common(sub.add_parser("score"))
    p.add_argument("--x", type=float, default=None)
    p = common(sub.add_parser("correctness"))
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--tries", type=int, default=2)
    p.add_argument("--x", type=float, default=None)
    p = common(sub.add_parser("bands"))
    p.add_argument("--x", type=float, default=None)
    p.add_argument("--out", default=str(HERE / "data"))
    args = ap.parse_args()
    {"build": cmd_build, "facts": cmd_facts, "calibrate": cmd_calibrate,
     "rollout": cmd_rollout, "score": cmd_score, "correctness": cmd_correctness,
     "bands": cmd_bands}[args.cmd](args)


if __name__ == "__main__":
    main()
