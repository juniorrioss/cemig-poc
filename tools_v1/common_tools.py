#!/usr/bin/env python3
"""
common_tools.py — Núcleo compartilhado da bancada de TOOL-CALLING do LFM2.5 (tools_v1/).

Reúso máximo do que já existe (ordem do brief):
  - muralhas anti-contaminação + split por NR + call_vllm + load_chunks: finetune2/common;
  - régua honesta (cobertura de fatos, tautologia, alucinação): bench/regua/ruler + juiz;
  - juízes das famílias de recusa: sft_v2/common_v2 (approve_nonoracle, judge_nonoracle);
  - retrieval v4 (filtro da consulta): tools_v1/retrieval_check;
  - renderização nativa + parser round-trip: tools_v1/render + tool_schema.

EMENDA 2 do capitão (decisão fechada): o 27B é gerador de CONTEÚDO, nunca de FORMATO.
Pedimos a ele apenas campos SEMÂNTICOS em JSON simples (guided decoding) e a renderização
no formato LFM é NOSSA (apply_chat_template). Zero regex de conversão, zero string à mão.

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "classifier"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# finetune2/common — muralhas, call_vllm, load_chunks, extract_json
_ft = _load_module("ft2_common", _ROOT / "finetune2" / "common.py")
DEFAULT_VLLM_URL = _ft.DEFAULT_VLLM_URL
DEFAULT_VLLM_MODEL = _ft.DEFAULT_VLLM_MODEL
RESERVED_NRS = _ft.RESERVED_NRS
LexicalFilter = _ft.LexicalFilter
call_vllm = _ft.call_vllm
extract_json = _ft.extract_json
holdout_gold_chunk_ids = _ft.holdout_gold_chunk_ids
load_chunks = _ft.load_chunks

from nr_taxonomy import CLASSES, NONE_LABEL, normalize_nr  # noqa: E402

# régua honesta
sys.path.insert(0, str(_ROOT / "bench" / "regua"))
from ruler import deterministic_pass  # noqa: E402
from judge_regua import judge_disambiguate  # noqa: E402

# juízes de recusa (sft_v2) — reúso direto do módulo v2
sys.path.insert(0, str(_ROOT / "sft_v2"))
import common_v2 as V2  # noqa: E402


# ---------------------------------------------------------------------------
# Guided JSON via vLLM (garante JSON válido; validamos com json.loads + schema).
# ---------------------------------------------------------------------------
def call_vllm_json(messages: List[Dict[str, str]], schema: Dict[str, Any], *,
                   url: str = DEFAULT_VLLM_URL, model: str = DEFAULT_VLLM_MODEL,
                   temperature: float = 0.7, top_p: float = 0.95,
                   max_tokens: int = 700, retries: int = 4,
                   timeout: int = 150) -> Optional[Dict[str, Any]]:
    """Chamada ao 27B com guided_json (schema fixo). Retorna dict validado ou None."""
    import time
    payload = {
        "model": model, "messages": messages, "temperature": temperature,
        "top_p": top_p, "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "guided_json": schema,
    }
    data = json.dumps(payload).encode("utf-8")
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{url}/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            content = out["choices"][0]["message"].get("content") or ""
            obj = extract_json(content)
            return obj
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    return None


# ---------------------------------------------------------------------------
# NR válida no corpus (PASSO 2: o 'nr' precisa existir).
# ---------------------------------------------------------------------------
def corpus_nrs() -> List[str]:
    """NRs presentes no corpus (docs distintos), em minúsculas ('nr-10')."""
    docs = {c["doc"] for c in load_chunks()}
    return sorted(docs)


def nr_display(doc_lower: str) -> str:
    """'nr-10' -> 'NR-10' (formato que o app/usuário citam)."""
    if not doc_lower:
        return ""
    d = doc_lower.strip().lower()
    if d.startswith("nr-"):
        return "NR-" + d[3:]
    return doc_lower


# ---------------------------------------------------------------------------
# Régua honesta sobre a resposta FINAL (oráculo/reuso/nova busca).
# Mesma lógica do sft_v2.gen_oracle: determinístico + desempate 27B + alucinação.
# ---------------------------------------------------------------------------
def approve_answer(question: str, answer: str, facts: List[str], gold_text: str,
                   threshold: float = 0.5) -> bool:
    """Régua honesta: cobertura>=thr, sem tautologia, sem alucinação, não-vazia."""
    r = deterministic_pass(question, answer, facts, threshold)
    if r.empty:
        return False
    if r.needs_judge or (r.approved and not r.empty):
        try:
            jr = judge_disambiguate(question, answer, r.facts_ambiguous, gold_text)
            amb_low = {a.lower(): a for a in r.facts_ambiguous}
            for fp in jr.get("fatos_presentes", []):
                key = fp.strip().lower()
                for al, orig in amb_low.items():
                    if key == al or al.startswith(key[:12]) or key.startswith(al[:12]):
                        r.facts_present.append(orig)
            r.facts_present = list(set(r.facts_present))
            r.hallucination = bool(jr.get("alucinacao", False))
        except Exception:
            pass
    coverage = len(r.facts_present) / max(1, len(facts))
    return coverage >= threshold and not r.tautology and not r.hallucination


if __name__ == "__main__":
    nrs = corpus_nrs()
    print(f"NRs no corpus: {len(nrs)} -> {nrs[:8]}...")
    print(f"reservadas (OOD): {RESERVED_NRS}")
    print(f"holdout gold chunks: {len(holdout_gold_chunk_ids())}")
