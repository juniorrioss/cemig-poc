"""
loaders.py — Descoberta e normalização de TODAS as respostas salvas no repo.

Varre bench/ e finetune2/ e devolve uma lista de "candidatos", cada um com:
  { "candidate": <rótulo único>, "source": <arquivo>, "family": <familia>,
    "items": { item_id -> {"question","doc","section","response","golden_answer",
                            "retrieval_hit"} } }

Formatos suportados:
  1) bench/**/responses_*.json  (dict com metadata + items[])
  2) finetune2/data/panel_*.json (dict com holdout.items[])
  3) bench/results/v2/harness_results_v2.json (models{}->results{mode}->[itens], 101)
  4) bench/results/harness_results.json (v1, se tiver response)

NÃO gera resposta nova — só lê disco (ordem do capitão).
Comentários em português; identificadores em inglês.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _norm_item(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normaliza um item de resposta. Retorna None se não houver resposta utilizável."""
    resp = it.get("response")
    if resp is None:
        return None
    return {
        "question": it.get("question", ""),
        "doc": it.get("doc", ""),
        "section": it.get("section", ""),
        "response": resp if isinstance(resp, str) else str(resp),
        "golden_answer": it.get("golden_answer", ""),
        "retrieval_hit": it.get("retrieval_hit"),
    }


def _load_responses_file(path: Path) -> Optional[Dict[str, Any]]:
    """Formato bench/**/responses_*.json (metadata + items[])."""
    d = json.loads(path.read_text(encoding="utf-8"))
    items = d.get("items")
    if not isinstance(items, list):
        return None
    meta = d.get("metadata", {})
    norm = {}
    for it in items:
        n = _norm_item(it)
        if n and it.get("item_id"):
            norm[it["item_id"]] = n
    if not norm:
        return None
    label = _label_from_path(path, meta)
    return {"candidate": label, "source": str(path.relative_to(REPO_ROOT)),
            "family": meta.get("family") or _family_from_label(label),
            "items": norm, "meta": meta}


def _load_panel_file(path: Path) -> Optional[Dict[str, Any]]:
    """Formato finetune2/data/panel_*.json (holdout.items[])."""
    d = json.loads(path.read_text(encoding="utf-8"))
    hold = d.get("holdout")
    if not isinstance(hold, dict):
        return None
    items = hold.get("items")
    if not isinstance(items, list):
        return None
    variant = d.get("metadata", {}).get("variant", path.stem.replace("panel_", ""))
    norm = {}
    for it in items:
        n = _norm_item(it)
        if n and it.get("item_id"):
            norm[it["item_id"]] = n
    if not norm:
        return None
    label = f"finetune2/{variant}"
    return {"candidate": label, "source": str(path.relative_to(REPO_ROOT)),
            "family": "lfm2.6b-train", "items": norm, "meta": d.get("metadata", {})}


def _load_v2_harness(path: Path) -> List[Dict[str, Any]]:
    """Formato bench/results/v2/harness_results_v2.json: shootout dos 8 SLMs x modos."""
    d = json.loads(path.read_text(encoding="utf-8"))
    models = d.get("models", {})
    out = []
    # Modos que representam o pipeline RAG comparável (classic e topk2).
    keep_modes = {"classic_rag", "classic_rag_topk2",
                  "query_rewrite_inject", "query_rewrite_inject_topk2",
                  "tool_calling"}
    for model_name, mdata in models.items():
        results = mdata.get("results", {})
        for mode, items in results.items():
            if mode not in keep_modes or not isinstance(items, list):
                continue
            norm = {}
            for it in items:
                n = _norm_item(it)
                if n and it.get("item_id"):
                    norm[it["item_id"]] = n
            if not norm:
                continue
            label = f"shootout-v2/{model_name}/{mode}"
            out.append({"candidate": label, "source": str(path.relative_to(REPO_ROOT)),
                        "family": model_name, "items": norm, "meta": {"mode": mode}})
    return out


def _label_from_path(path: Path, meta: Dict[str, Any]) -> str:
    """Rótulo único e legível para um arquivo de respostas."""
    parent = path.parent.parent.name  # ex.: judge151, sintese2, ctx_topk
    stem = path.stem.replace("responses_", "")
    ck = meta.get("config_key")
    cond = meta.get("condition")
    if ck and cond:
        return f"{parent}/{ck}_{cond}"
    return f"{parent}/{stem}"


def _family_from_label(label: str) -> str:
    low = label.lower()
    if "2.6b" in low:
        return "lfm2.6b"
    if "1.2b" in low or "lfm1.2b" in low:
        return "lfm1.2b"
    if "minicpm2b" in low:
        return "minicpm2b"
    if "minicpm1b" in low:
        return "minicpm1b"
    return "outro"


def discover_candidates() -> List[Dict[str, Any]]:
    """Descobre todos os candidatos com respostas salvas no repo."""
    cands: List[Dict[str, Any]] = []

    # 1) responses_*.json em bench/**
    for p in sorted(REPO_ROOT.glob("bench/**/responses_*.json")):
        try:
            c = _load_responses_file(p)
            if c:
                cands.append(c)
        except Exception as e:  # noqa: BLE001
            print(f"[loaders] falha {p}: {e}")

    # 2) panels do finetune2
    for p in sorted(REPO_ROOT.glob("finetune2/data/panel_*.json")):
        try:
            c = _load_panel_file(p)
            if c:
                cands.append(c)
        except Exception as e:  # noqa: BLE001
            print(f"[loaders] falha {p}: {e}")

    # 3) shootout v2 (8 SLMs)
    v2 = REPO_ROOT / "bench" / "results" / "v2" / "harness_results_v2.json"
    if v2.exists():
        try:
            cands.extend(_load_v2_harness(v2))
        except Exception as e:  # noqa: BLE001
            print(f"[loaders] falha {v2}: {e}")

    return cands


if __name__ == "__main__":
    cs = discover_candidates()
    print(f"Total de candidatos: {len(cs)}")
    for c in cs:
        print(f"  {c['candidate']:<48} n={len(c['items'])} family={c['family']}")
