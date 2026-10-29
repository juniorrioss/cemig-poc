"""
common_synth.py — utilitários do funil de geração sintética do F2.

- Cliente 27B (Qwen3.8-27B-FP8 em 10.100.0.111:8005) com thinking ANTES do
  JSON: o parser pega o ÚLTIMO bloco JSON válido da resposta (lição da POC).
- Carga do corpus das normas tratadas (nrs.parquet: conteudo + manual).
- Normalização PT-BR e tokenização barata p/ dedup/entropia.

Comentários em português; identificadores em inglês (padrão do projeto).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# requests é importado tardiamente em gen_call (validate_stage roda com python3
# do sistema, sem requests).

HERE = Path(__file__).resolve().parent          # bench/lfm3_f2/synth
F2 = HERE.parent                                 # bench/lfm3_f2
REPO = F2.parents[1]                             # repo root
DATA = F2 / "data"
SYNTH = DATA / "synth"

NRS_PARQUET = Path("/home/rios/projetos/cemig-mobile-llm/firstmate/data/nrs_hf/nrs.parquet")

GEN_URL = os.environ.get("F2_GEN_URL", "http://10.100.0.111:8005/v1")
GEN_MODEL = os.environ.get("F2_GEN_MODEL", "Qwen/Qwen3.8-27B-FP8")

SELECTION = json.loads((DATA / "norm_selection.json").read_text(encoding="utf-8")) \
    if (DATA / "norm_selection.json").exists() else {}


# ---------------------------------------------------------------------------
# Cliente do gerador 27B
# ---------------------------------------------------------------------------
def gen_call(prompt: str, system: str | None = None, max_tokens: int = 4096,
             temperature: float = 0.7, thinking: bool = True,
             retries: int = 4, timeout: int = 300) -> str:
    """Chamada ao 27B. thinking=True deixa o modelo raciocinar antes do JSON."""
    import requests
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": GEN_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{GEN_URL}/chat/completions", json=payload,
                              timeout=timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last = f"HTTP {r.status_code}: {r.text[:160]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"gen_call falhou: {last}")


def extract_last_json(text: str) -> Any:
    """
    Pega o ÚLTIMO bloco JSON válido do texto (o 27B pensa antes do JSON).
    Procura blocos ```json ...``` primeiro; senão varre objetos/arrays brutos.
    """
    fenced = re.findall(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    for cand in reversed(fenced):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    # varredura bruta: tenta do último '[' ou '{' até o fim, recuando
    for opener in ("[", "{"):
        starts = [m.start() for m in re.finditer(re.escape(opener), text)]
        for s in reversed(starts):
            chunk = text[s:].strip()
            # tenta recortes decrescentes a partir do fim
            closer = "]" if opener == "[" else "}"
            ends = [m.start() for m in re.finditer(re.escape(closer), chunk)]
            for e in reversed(ends):
                try:
                    return json.loads(chunk[: e + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"nenhum JSON válido na resposta ({len(text)} chars): "
                     f"{text[-300:]!r}")


def gen_json(prompt: str, system: str | None = None, max_tokens: int = 4096,
             temperature: float = 0.7, retries: int = 3,
             thinking: bool = True) -> Any:
    """
    gen_call + extract_last_json com retry de parse (re-chama o modelo).
    Fallback: se o thinking estourar o budget sem JSON (lição do s1: 30k chars
    de rascunho truncado), a última tentativa re-chama com thinking OFF.
    """
    last = None
    for attempt in range(retries):
        think = thinking and attempt < retries - 1
        raw = gen_call(prompt, system=system, max_tokens=max_tokens,
                       temperature=temperature, thinking=think)
        try:
            return extract_last_json(raw)
        except ValueError as e:
            last = e
    raise RuntimeError(f"gen_json: parse falhou {retries}x: {last}")


# ---------------------------------------------------------------------------
# Paralelismo com checkpoint (mesma semântica de bench/regua.common)
# ---------------------------------------------------------------------------
def parallel_map(fn: Callable[[Any], Any], items: List[Any],
                 max_workers: int = 16,
                 checkpoint_path: Optional[Path] = None,
                 key_fn: Optional[Callable[[Any], str]] = None) -> Dict[str, Any]:
    key_fn = key_fn or (lambda x: str(x))
    results: Dict[str, Any] = {}
    if checkpoint_path and checkpoint_path.exists():
        results = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        print(f"[ckpt] retomando com {len(results)} prontos")
    pending = [it for it in items if key_fn(it) not in results]
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, it): key_fn(it) for it in pending}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                results[k] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[k] = {"__error__": str(e)}
            done += 1
            if checkpoint_path and done % 8 == 0:
                _dump(results, checkpoint_path)
            if done % 25 == 0:
                print(f"  ... {done}/{len(pending)}")
    if checkpoint_path:
        _dump(results, checkpoint_path)
    return results


def _dump(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Corpus das normas
# ---------------------------------------------------------------------------
def load_norm_corpus(nr: int) -> Dict[str, str]:
    """Texto limpo da NR (+ manual quando existir) do nrs.parquet."""
    import pyarrow.parquet as pq
    rows = pq.read_table(str(NRS_PARQUET)).to_pylist()
    for r in rows:
        if int(r["numero"]) == nr:
            return {"norma": r["norma"], "titulo": r["titulo"],
                    "conteudo": r["conteudo"], "manual": r["manual"] or ""}
    raise KeyError(f"NR-{nr} não está no nrs.parquet")


def treated_norms() -> List[int]:
    return SELECTION["treated"]


def control_norms() -> List[int]:
    return SELECTION["control"]


# ---------------------------------------------------------------------------
# Normalização / tokenização barata (dedup e entropia)
# ---------------------------------------------------------------------------
def deaccent(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def norm_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", deaccent(text.lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = set(norm_tokens(a)), set(norm_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
