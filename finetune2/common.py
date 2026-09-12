#!/usr/bin/env python3
"""
common.py — utilidades compartilhadas do treino de síntese do LFM2.5-2.6B (finetune2/).

Concentra as TRÊS MURALHAS anti-contaminação e a infra de geração via vLLM:

  MURALHA 1 (holdout intocável): 151 (corpus/qa_pairs_v2.jsonl) + 20 (bench/data/smoke_qa_20.jsonl)
     ficam FORA de todo treino/ajuste; só entram no juízo final (eval_panel.py). Aqui expomos
     os textos das perguntas e os chunk_ids-ouro do holdout para os filtros.

  MURALHA 2 (split por NR): as NRs 33/16/26 são RESERVADAS (RESERVED_NRS). Nenhum chunk delas
     entra no dataset SFT; elas viram o painel OOD estrutural na avaliação.

  MURALHA 3 (filtro lexical automático): toda fala de treino gerada é comparada, por Jaccard de
     n-grams de palavras, contra TODAS as perguntas do holdout; itens com Jaccard >= LEX_THR são
     descartados e registrados (procedência + score) — evita que a geração "copie" o holdout.

Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent

# ---------------------------------------------------------------------------
# Caminhos canônicos
# ---------------------------------------------------------------------------
QA_V2_PATH = ROOT / "corpus" / "qa_pairs_v2.jsonl"
SMOKE_PATH = ROOT / "bench" / "data" / "smoke_qa_20.jsonl"
INDEX_DB = ROOT / "corpus" / "index_hf_36nr.db"

DATA_DIR = _HERE / "data"
LOG_DIR = _HERE / "logs"

# ---------------------------------------------------------------------------
# MURALHA 2 — NRs reservadas (OOD estrutural). Nunca entram no treino.
# ---------------------------------------------------------------------------
RESERVED_NRS: Tuple[str, ...] = ("nr-33", "nr-16", "nr-26")

# ---------------------------------------------------------------------------
# MURALHA 3 — limiar do filtro lexical (Jaccard de 3-grams de palavras).
# ---------------------------------------------------------------------------
LEX_THR = 0.4
LEX_NGRAM = 3

# ---------------------------------------------------------------------------
# vLLM (juiz/gerador/editor). VPN oscila -> retry com backoff.
# ---------------------------------------------------------------------------
DEFAULT_VLLM_URL = "http://10.100.0.111:8005/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3.8-27B-FP8"

# System prompt de síntese de PRODUÇÃO (AskPipeline v1_rigido) — alvo de estilo do SFT.
SYNTHESIS_SYSTEM_PROMPT = (
    "Você é o assistente técnico de campo da CEMIG. O eletricista OUVE sua resposta por voz, "
    "então seja curto e direto.\n"
    "REGRAS OBRIGATÓRIAS (nunca viole):\n"
    "- Responda em NO MÁXIMO 4 frases curtas.\n"
    "- PROIBIDO usar markdown, títulos, negrito, listas, bullets ou numeração. Escreva em prosa corrida.\n"
    "- Cite a norma e o item DENTRO da frase (ex.: 'conforme a NR-10, item 10.5.1, ...').\n"
    "- Baseie-se EXCLUSIVAMENTE no contexto normativo fornecido; não invente procedimentos.\n"
    "- Se o contexto não responder, diga apenas: 'Não sei com base nas normas consultadas.'"
)


# ---------------------------------------------------------------------------
# Carregamento do holdout (MURALHA 1)
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        recs.append(json.loads(s))
    return recs


def load_holdout() -> List[Dict[str, Any]]:
    """151 (qa_v2) + 20 (smoke), normalizados para {id, text, doc, source}."""
    items: List[Dict[str, Any]] = []
    for r in load_jsonl(QA_V2_PATH):
        items.append({"id": r["id"], "text": r["question"], "doc": str(r.get("doc", "")).lower(),
                      "source": "qa_v2"})
    for r in load_jsonl(SMOKE_PATH):
        items.append({"id": r["id"], "text": r["question"], "doc": str(r.get("doc", "")).lower(),
                      "source": "smoke"})
    return items


def holdout_gold_chunk_ids() -> Set[int]:
    """chunk_ids-ouro do holdout — excluídos da amostragem de chunks do SFT."""
    ids: Set[int] = set()
    for path in (QA_V2_PATH, SMOKE_PATH):
        for r in load_jsonl(path):
            if isinstance(r.get("chunk_id"), int) and r["chunk_id"] >= 0:
                ids.add(r["chunk_id"])
            for cid in r.get("relevant_chunk_ids", []) or []:
                if isinstance(cid, int):
                    ids.add(cid)
    return ids


# ---------------------------------------------------------------------------
# MURALHA 3 — filtro lexical por Jaccard de n-grams de palavras
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _word_ngrams(text: str, n: int = LEX_NGRAM) -> Set[str]:
    toks = _WORD_RE.findall(text.lower())
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class LexicalFilter:
    """Compara falas geradas contra o holdout inteiro; descarta e registra as próximas demais."""

    def __init__(self, thr: float = LEX_THR, n: int = LEX_NGRAM):
        self.thr = thr
        self.n = n
        holdout = load_holdout()
        self.holdout_grams: List[Tuple[str, Set[str]]] = [
            (h["id"], _word_ngrams(h["text"], n)) for h in holdout
        ]
        self.discards: List[Dict[str, Any]] = []

    def max_overlap(self, text: str) -> Tuple[float, Optional[str]]:
        g = _word_ngrams(text, self.n)
        best = 0.0
        best_id: Optional[str] = None
        for hid, hg in self.holdout_grams:
            j = jaccard(g, hg)
            if j > best:
                best, best_id = j, hid
        return best, best_id

    def accept(self, text: str, meta: Optional[Dict[str, Any]] = None) -> bool:
        """True se a fala pode entrar no treino (Jaccard < thr). Senão registra descarte."""
        score, hid = self.max_overlap(text)
        if score >= self.thr:
            self.discards.append({
                "text": text, "max_jaccard": round(score, 4),
                "closest_holdout_id": hid, **(meta or {}),
            })
            return False
        return True

    def dump_discards(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "filter": "word_ngram_jaccard", "n": self.n, "threshold": self.thr,
            "total_discarded": len(self.discards), "items": self.discards,
        }, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# vLLM com retry (VPN oscila)
# ---------------------------------------------------------------------------
def call_vllm(messages: List[Dict[str, str]], *, url: str = DEFAULT_VLLM_URL,
              model: str = DEFAULT_VLLM_MODEL, temperature: float = 0.7,
              top_p: float = 0.95, max_tokens: int = 512, thinking: bool = False,
              retries: int = 5, timeout: int = 120) -> str:
    """Chamada OpenAI-compat ao vLLM 27B com backoff exponencial. Retorna o content (str)."""
    payload = {
        "model": model, "messages": messages, "temperature": temperature,
        "top_p": top_p, "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    data = json.dumps(payload).encode("utf-8")
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{url}/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            msg = out["choices"][0]["message"]
            return (msg.get("content") or msg.get("reasoning_content") or "").strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = min(2 ** attempt, 30)
            time.sleep(wait)
    raise RuntimeError(f"vLLM falhou após {retries} tentativas: {last_err}")


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Extrai o último objeto JSON válido do texto (robusto a preâmbulo/thinking)."""
    text = raw.strip()
    for block in reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)):
        try:
            return json.loads(block)
        except Exception:
            pass
    for m in reversed(re.findall(r"(\{(?:[^{}]|\{[^{}]*\})*\})", text, re.DOTALL)):
        try:
            return json.loads(m)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Chunks do índice
# ---------------------------------------------------------------------------
def load_chunks(db_path: Path = INDEX_DB) -> List[Dict[str, Any]]:
    import sqlite3
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT id, doc, section, title, text FROM chunks").fetchall()]
    con.close()
    return rows


if __name__ == "__main__":
    ho = load_holdout()
    gold = holdout_gold_chunk_ids()
    chunks = load_chunks()
    reserved = [c for c in chunks if c["doc"] in RESERVED_NRS]
    print(f"holdout: {len(ho)} perguntas | chunk-ouro do holdout: {len(gold)}")
    print(f"chunks totais: {len(chunks)} | reservados (OOD {RESERVED_NRS}): {len(reserved)}")
    lf = LexicalFilter()
    # sanity: uma pergunta do holdout deve bater alto contra si mesma
    s, hid = lf.max_overlap(ho[0]["text"])
    print(f"sanity filtro: fala holdout[0] tem max_jaccard={s:.3f} vs {hid} (esperado ~1.0)")
