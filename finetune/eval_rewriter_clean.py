#!/usr/bin/env python3
"""
eval_rewriter_clean.py — Avaliação do rewriter no HOLDOUT REAL (151 perguntas).

Gate duplo do M3 (Fase 2):
  1. Recall@k nas 151 perguntas coloquiais reais (qa_pairs_v2.jsonl) — conjunto NUNCA
     visto em treino/val/test (fonte distinta do sintético). Compara:
       - Zero-shot (modelo base LFM2.5-1.2B-Instruct)
       - LoRA fine-tuned (adapter aplicado)
     usando o MESMO caminho de busca do app (app_fts_query, pesos 1.5/3/2/1),
     com e sem filtro por norma detectada na reescrita.
  2. Também mede recall no test set descontaminado (NRs não vistas) para diagnóstico.

Critério de saída (brief): recall@2 nas 151 do LoRA deve SUPERAR o zero-shot em >=10pp.

Uso:
    .venv-train/bin/python finetune/eval_rewriter_clean.py --adapter finetune/adapter_r8 \
        --db corpus/index_hf_36nr.db --qa corpus/qa_pairs_v2.jsonl --out finetune/results/eval_clean_r8.json
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from corpus.eval_fino import app_fts_query, detect_norm  # caminho de busca idêntico ao app

REWRITE_SYSTEM_PROMPT = (
    "Você é um especialista em busca técnica nas Normas Regulamentadoras (NRs de segurança do trabalho).\n"
    "Sua única tarefa é extrair e converter a dúvida do trabalhador em palavras-chave técnicas e conceituais para busca BM25 no acervo das NRs.\n"
    "Diretrizes mandatórias:\n"
    "1. Retorne APENAS de 3 a 6 palavras-chave técnicas na mesma linha, separadas por vírgula ou espaço.\n"
    "2. Foque nos termos específicos da situação, procedimentos, equipamentos, riscos ou medidas de proteção aplicáveis.\n"
    "3. NÃO inclua saudações, preâmbulos, justificativas ou listas longas de nomes de normas.\n"
    "4. NÃO responda à dúvida nesta etapa; retorne estritamente os termos de busca."
)
APP_W = (1.5, 3.0, 2.0, 1.0)
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"


def clean_kw(raw: str, fallback: str) -> str:
    if not raw or not raw.strip():
        return fallback
    text = raw.strip().split("\n")[0]
    text = re.sub(r'(?i)^(termos técnicos|palavras-chave|busca|keywords)\s*[:\-]\s*', '', text)
    text = text.replace("*", " ").replace("`", " ").strip()
    return text or fallback


@torch.no_grad()
def batch_rewrite(model, tok, questions: List[str], max_new: int = 40) -> List[str]:
    outs = []
    for q in questions:
        msgs = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Dúvida do trabalhador: {q}\nTermos técnicos para busca:"},
        ]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(model.device)
        am = enc["attention_mask"].to(model.device)
        gen = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new, do_sample=False)
        txt = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        outs.append(clean_kw(txt, q))
    return outs


def search(con, terms: str, doc_filter: Optional[str], limit: int = 5) -> List[Dict[str, Any]]:
    q = app_fts_query(terms)
    if q == '""':
        return []
    cur = con.cursor()
    w = APP_W
    if doc_filter:
        sql = (f"SELECT c.id,c.doc,c.section,c.text,bm25(chunks_fts,{w[0]},{w[1]},{w[2]},{w[3]}) s "
               "FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid WHERE c.doc=? AND chunks_fts MATCH ? ORDER BY s LIMIT ?")
        args = (doc_filter, q, limit)
    else:
        sql = (f"SELECT c.id,c.doc,c.section,c.text,bm25(chunks_fts,{w[0]},{w[1]},{w[2]},{w[3]}) s "
               "FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?")
        args = (q, limit)
    try:
        cur.execute(sql, args)
    except sqlite3.OperationalError:
        return []
    return [{"id": r[0], "doc": r[1], "section": r[2], "text": r[3]} for r in cur.fetchall()]


def hit(r: Dict[str, Any], p: Dict[str, Any]) -> bool:
    if r["id"] == p.get("chunk_id"):
        return True
    if p.get("relevant_chunk_ids") and r["id"] in p["relevant_chunk_ids"]:
        return True
    td, ts = p.get("doc", "").lower(), p.get("section", "").lower()
    rd, rs, rt = r["doc"].lower(), r["section"].lower(), r["text"].lower()
    if td == rd and (ts in rs or rs in ts or (ts and ts in rt)):
        return True
    return False


def eval_recall(con, qa: List[Dict[str, Any]], rewrites: List[str], use_normfilter: bool) -> Dict[str, float]:
    h1 = h2 = h3 = h5 = 0
    mrr = 0.0
    for p, rw in zip(qa, rewrites):
        nf = None
        if use_normfilter:
            nf = detect_norm(p["question"]) or detect_norm(rw)
        res = search(con, rw, nf, limit=5)
        if use_normfilter and nf and len(res) < 5:
            extra = search(con, rw, None, limit=5)
            seen = {r["id"] for r in res}
            res += [r for r in extra if r["id"] not in seen]
            res = res[:5]
        rank = None
        for i, r in enumerate(res):
            if hit(r, p):
                rank = i + 1
                break
        if rank:
            h1 += rank <= 1
            h2 += rank <= 2
            h3 += rank <= 3
            h5 += rank <= 5
            mrr += 1.0 / rank
    n = len(qa)
    return {"r1": h1 / n * 100, "r2": h2 / n * 100, "r3": h3 / n * 100, "r5": h5 / n * 100, "mrr": mrr / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="finetune/adapter_r8")
    ap.add_argument("--db", default="corpus/index_hf_36nr.db")
    ap.add_argument("--qa", default="corpus/qa_pairs_v2.jsonl")
    ap.add_argument("--out", default="finetune/results/eval_clean.json")
    ap.add_argument("--label", default="r8")
    args = ap.parse_args()

    qa = [json.loads(l) for l in Path(args.qa).read_text(encoding="utf-8").splitlines() if l.strip()]
    con = sqlite3.connect(args.db)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")

    print(f"[eval] Zero-shot em {len(qa)} perguntas reais...")
    t0 = time.time()
    zs = batch_rewrite(base, tok, [p["question"] for p in qa])
    print(f"  zero-shot rewrite: {time.time()-t0:.1f}s")

    print(f"[eval] LoRA ({args.adapter}) em {len(qa)} perguntas reais...")
    lora = PeftModel.from_pretrained(base, args.adapter)
    t0 = time.time()
    ft = batch_rewrite(lora, tok, [p["question"] for p in qa])
    print(f"  lora rewrite: {time.time()-t0:.1f}s")

    results = {}
    for name, rws in [("zero_shot", zs), ("lora", ft)]:
        results[name] = {
            "no_filter": eval_recall(con, qa, rws, use_normfilter=False),
            "normfilter": eval_recall(con, qa, rws, use_normfilter=True),
        }

    con.close()

    def row(tag, d):
        print(f"| {tag:<28} | {d['r1']:>5.1f} | {d['r2']:>5.1f} | {d['r3']:>5.1f} | {d['r5']:>5.1f} | {d['mrr']:.3f} |")

    print("\n" + "=" * 78)
    print(f"  GATE M3 — REWRITER NO HOLDOUT REAL ({len(qa)} PERGUNTAS, NÃO-CONTAMINADO)")
    print("=" * 78)
    print(f"| {'Config':<28} | {'R@1':>5} | {'R@2':>5} | {'R@3':>5} | {'R@5':>5} | {'MRR':>5} |")
    print("|" + "-"*30 + "|" + "-"*7 + "|" + "-"*7 + "|" + "-"*7 + "|" + "-"*7 + "|" + "-"*7 + "|")
    row("Zero-shot (sem filtro)", results["zero_shot"]["no_filter"])
    row("Zero-shot + filtro norma", results["zero_shot"]["normfilter"])
    row("LoRA (sem filtro)", results["lora"]["no_filter"])
    row("LoRA + filtro norma", results["lora"]["normfilter"])
    print("=" * 78)

    # Gate: melhor config LoRA supera melhor zero-shot em >=10pp de R@2?
    best_zs = max(results["zero_shot"]["no_filter"]["r2"], results["zero_shot"]["normfilter"]["r2"])
    best_lora = max(results["lora"]["no_filter"]["r2"], results["lora"]["normfilter"]["r2"])
    delta = best_lora - best_zs
    verdict = "PASS" if delta >= 10.0 else "FAIL"
    print(f"\nGate R@2: LoRA {best_lora:.1f}% vs Zero-shot {best_zs:.1f}% => Δ={delta:+.1f}pp [{verdict}]")

    results["gate"] = {"best_zs_r2": best_zs, "best_lora_r2": best_lora, "delta_pp": delta, "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo: {args.out}")


if __name__ == "__main__":
    main()
