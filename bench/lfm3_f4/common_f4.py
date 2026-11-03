#!/usr/bin/env python3
"""
common_f4.py — utilitários do F4 (tool-calling multiturno sobre o data2x-ep4).

Reuso máximo (ordem do brief): pools de QA/targets/assignments do F3/F3b,
cliente 27B + parallel_map + jaccard da F2 (common_synth), retriever real v4
(retriever_lazy), render nativo do LFM3 (tool_schema_f4 + apply_chat_template).

Papéis:
  - carregar pools (assignments data2x, targets, units, chunks do índice v4);
  - muralha anticontaminação: dedup Jaccard>=0.5 vs as 1536 do benchmark;
  - GATE de render por exemplo: renderiza o diálogo com o chat_template real e
    re-parseia TODAS as tool calls (round-trip); falha -> descarta;
  - particionar o pool em TRAIN x EVAL-SUITES (disjunto por qa_id, seed fixa).

Comentários PT-BR; identificadores em inglês.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent              # bench/lfm3_f4
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "synth"))
sys.path.insert(0, str(HERE))

import common_synth as cs  # noqa: E402  (cliente 27B, jaccard, parallel_map)
from tool_parser import parse_tool_calls  # noqa: E402
from tool_schema_f4 import make_baked_system_message  # noqa: E402

F3 = REPO / "bench" / "lfm3_f3"
F3B = REPO / "bench" / "lfm3_f3b"
DATA = HERE / "data"
SYNTH = DATA / "synth"
TRAIN = DATA / "train"
EVAL = DATA / "eval"

V4_DB = REPO / "android" / "app" / "src" / "main" / "assets" / "index.db"
BENCH_PARQUET = Path.home() / (
    ".cache/huggingface/hub/datasets--UFGCEMIGONA--benchmark_mobile/snapshots/"
    "5a5aa32ae4087481be828e071aff169827ec2993/normas_regulamentadoras/"
    "train-00000-of-00001.parquet")

SEED = 20260211
NR_RE = re.compile(r"\bnr[\s\-]?(\d{1,2})\b", re.IGNORECASE)

# re-export p/ os estágios
gen_json = cs.gen_json
gen_call = cs.gen_call
parallel_map = cs.parallel_map
read_jsonl = cs.read_jsonl
write_jsonl = cs.write_jsonl
norm_tokens = cs.norm_tokens
jaccard = cs.jaccard


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------
def load_chunks() -> Dict[int, Dict[str, Any]]:
    import sqlite3
    con = sqlite3.connect(str(V4_DB))
    con.row_factory = sqlite3.Row
    out = {r["id"]: dict(r) for r in con.execute(
        "SELECT id, doc, section, title, text FROM chunks")}
    con.close()
    return out


def load_assignments() -> List[Dict[str, Any]]:
    """Assignments do data2x (campeão F3b): caso, k, context_ids, labels."""
    return read_jsonl(F3B / "data" / "train" / "data2x" / "assignments.jsonl")


def load_targets() -> Dict[str, str]:
    """Alvos 27B (gold/partial) do data2x, por qa_id."""
    return {t["qa_id"]: t["target"]
            for t in read_jsonl(F3B / "data" / "train" / "data2x" / "targets.jsonl")}


def load_units() -> Dict[str, Dict[str, Any]]:
    """Unidades de conhecimento (fato literal + ref) dos funis F2/F3/F3b."""
    units: Dict[str, Dict[str, Any]] = {}
    for base in (REPO / "bench" / "lfm3_f2" / "data" / "synth",
                 F3 / "data" / "synth", F3B / "data" / "synth"):
        for p in sorted(base.glob("units_nr*.jsonl")):
            for u in read_jsonl(p):
                units[u["unit_id"]] = u
    return units


def load_bench_questions() -> List[str]:
    import pyarrow.parquet as pq
    rows = pq.read_table(str(BENCH_PARQUET)).to_pylist()
    return [r["question"] for r in rows]


class ContaminationFilter:
    """Dedup Jaccard >= 0.5 vs as 1536 do benchmark (mesmo critério s6/F3)."""

    def __init__(self):
        self.bench_tok = [set(norm_tokens(q)) for q in load_bench_questions()]

    def contaminated(self, q: str) -> bool:
        qt = set(norm_tokens(q))
        if not qt:
            return False
        for bt in self.bench_tok:
            inter = len(qt & bt)
            if inter and inter / len(qt | bt) >= 0.5:
                return True
        return False


# ---------------------------------------------------------------------------
# Política determinística do argumento 'nr' (fala crua; sem reescrita criativa)
# ---------------------------------------------------------------------------
def detect_nr(question: str) -> Optional[str]:
    m = NR_RE.search(question)
    return f"NR-{int(m.group(1))}" if m else None


# ---------------------------------------------------------------------------
# GATE de render round-trip por exemplo
# ---------------------------------------------------------------------------
_tok = None


def get_tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B",
                                             trust_remote_code=False)
    return _tok


def roundtrip_ok(messages: List[Dict[str, Any]]) -> bool:
    """Renderiza o diálogo (system baked, sem tools=) e confere que TODAS as
    tool_calls dos messages re-parseiam idênticas no texto final."""
    tok = get_tokenizer()
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       enable_thinking=False)
    except Exception:
        return False
    want = []
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            fn = tc["function"]
            want.append((fn["name"], fn["arguments"].get("consulta"),
                         fn["arguments"].get("nr")))
    got = [(c["name"], c["arguments"].get("consulta"), c["arguments"].get("nr"))
           for c in parse_tool_calls(text)]
    return want == got


def count_tokens(messages: List[Dict[str, Any]]) -> int:
    tok = get_tokenizer()
    out = tok.apply_chat_template(messages, tokenize=True,
                                  enable_thinking=False)
    # transformers 5.x retorna BatchEncoding; len() direto contaria as CHAVES
    ids = out["input_ids"] if not isinstance(out, list) else out
    return len(ids)


# ---------------------------------------------------------------------------
# Partição TRAIN x EVAL (disjunta por qa_id, seed fixa)
# ---------------------------------------------------------------------------
def split_pool(items: List[Dict[str, Any]], n_eval: int,
               key: str = "qa_id") -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(SEED)
    items = sorted(items, key=lambda x: x[key])
    rng.shuffle(items)
    return items[n_eval:], items[:n_eval]


def ensure_dirs() -> None:
    for d in (DATA, SYNTH, TRAIN, EVAL):
        d.mkdir(parents=True, exist_ok=True)
