"""
common.py — Utilitários compartilhados da régua honesta (bench/regua).

Contém:
- Resolução do TRECHO-OURO (gold chunk) de cada uma das 151 do holdout a partir
  do índice canônico corpus/index_hf_36nr.db (o `relevant_chunk_ids` do JSONL
  está obsoleto; usa-se doc+seção com fallback textual robusto).
- Carga do holdout (151 perguntas reais de corpus/qa_pairs_v2.jsonl).
- Normalização de texto PT-BR (remoção de acentos, tokenização barata).
- Cliente paralelo do juiz vLLM 27B (ThreadPoolExecutor, checkpoint incremental).

Comentários em português; identificadores em inglês (padrão do projeto).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

# Raiz do repo (bench/regua/common.py -> repo root = parents[2])
REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DB = REPO_ROOT / "corpus" / "index_hf_36nr.db"
HOLDOUT_151 = REPO_ROOT / "corpus" / "qa_pairs_v2.jsonl"

# Juiz vLLM 27B. Na Spark o acesso é via túnel reverso (127.0.0.1:18005);
# localmente é o IP direto. Configurável por env.
JUDGE_URL = os.environ.get("REGUA_JUDGE_URL", "http://10.100.0.111:8005/v1")
JUDGE_MODEL = os.environ.get("REGUA_JUDGE_MODEL", "Qwen/Qwen3.8-27B-FP8")

# Numerais romanos usados nos anexos das NRs.
_ROMAN = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v", "6": "vi",
          "7": "vii", "8": "viii", "9": "ix", "10": "x"}


def deaccent(text: str) -> str:
    """Remove acentos (NFD) preservando o texto base."""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def load_holdout() -> List[Dict[str, Any]]:
    """Carrega as 151 perguntas reais do holdout (FIXO, nunca usado em treino)."""
    return [json.loads(line) for line in HOLDOUT_151.read_text(encoding="utf-8").splitlines() if line.strip()]


class GoldResolver:
    """Resolve o texto do trecho-ouro por (doc, seção) sobre o índice canônico."""

    def __init__(self, db_path: Path = INDEX_DB):
        self.con = sqlite3.connect(str(db_path))
        self.con.row_factory = sqlite3.Row

    def _rows(self, sql: str, params: tuple) -> List[sqlite3.Row]:
        return self.con.execute(sql, params).fetchall()

    def resolve(self, doc: str, section: str) -> List[Dict[str, Any]]:
        """
        Retorna a lista de chunks-ouro (até 2) para (doc, seção).
        Estratégia em cascata:
          1) subseção numérica citada no texto do chunk (ex.: '10.5.1');
          2) anexo por rótulo romano/arábico;
          3) seção-nível exata ou por prefixo do capítulo;
          4) glossário/textual.
        """
        doc = doc.strip().lower()
        sec = section.strip().lower()

        # 1) subseção numérica (ex.: 10.5.1, 1.5.5.1.1)
        m = re.match(r"^[\d.]+", sec)
        if m:
            num = m.group(0).rstrip(".")
            rows = self._rows(
                "SELECT id,doc,section,title,text FROM chunks WHERE doc=? AND text LIKE ?",
                (doc, f"%{num}%"))
            if rows:
                return [self._as_dict(r) for r in rows[:2]]

        # 2) anexo (ex.: 'Anexo 4, item 1, alínea b')
        am = re.search(r"anexo\s+([ivx0-9]+)", deaccent(sec))
        if am:
            n = am.group(1)
            n = _ROMAN.get(n, n)
            rows = self._rows(
                "SELECT id,doc,section,title,text FROM chunks WHERE doc=? AND lower(section) LIKE ?",
                (doc, f"anexo {n}%"))
            if rows:
                return [self._as_dict(r) for r in rows[:2]]

        # 3) seção-nível exata / prefixo de capítulo
        rows = self._rows(
            "SELECT id,doc,section,title,text FROM chunks WHERE doc=? AND (section=? OR section LIKE ?)",
            (doc, sec, (sec.split(".")[0] + ".%")))
        if rows:
            return [self._as_dict(r) for r in rows[:1]]

        # 4) glossário e textual
        rows = self._rows(
            "SELECT id,doc,section,title,text FROM chunks WHERE doc=? AND lower(section) LIKE ?",
            (doc, "%" + deaccent(sec.split(",")[0]) + "%"))
        if rows:
            return [self._as_dict(r) for r in rows[:2]]

        # Fallback final: qualquer chunk da norma (glossário costuma cair aqui)
        rows = self._rows(
            "SELECT id,doc,section,title,text FROM chunks WHERE doc=? LIMIT 3", (doc,))
        return [self._as_dict(r) for r in rows]

    @staticmethod
    def _as_dict(r: sqlite3.Row) -> Dict[str, Any]:
        return {"id": r["id"], "doc": r["doc"], "section": r["section"],
                "title": r["title"], "text": r["text"]}


# ---------------------------------------------------------------------------
# Cliente do juiz vLLM (paralelo, robusto, com retry curto)
# ---------------------------------------------------------------------------

def judge_call(prompt: str, url: str = None, model: str = None,
               max_tokens: int = 1024, temperature: float = 0.0,
               retries: int = 3, timeout: int = 60) -> str:
    """Chama o juiz vLLM (chat/completions, thinking-OFF). Retorna texto bruto."""
    url = url or JUDGE_URL
    model = model or JUDGE_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_err = None
    for attempt in range(retries):
        try:
            res = requests.post(f"{url}/chat/completions", json=payload, timeout=timeout)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            last_err = f"HTTP {res.status_code}: {res.text[:120]}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"judge_call falhou após {retries} tentativas: {last_err}")


def parallel_map(fn: Callable[[Any], Any], items: List[Any],
                 max_workers: int = 16,
                 checkpoint_path: Optional[Path] = None,
                 key_fn: Optional[Callable[[Any], str]] = None,
                 existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Aplica `fn` em paralelo sobre `items`, com checkpoint incremental opcional.
    Retorna dict {key -> resultado}. `key_fn` extrai a chave estável de cada item.
    """
    key_fn = key_fn or (lambda x: str(x))
    results: Dict[str, Any] = dict(existing or {})
    pending = [it for it in items if key_fn(it) not in results]
    if not pending:
        return results

    done_since_save = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, it): key_fn(it) for it in pending}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                results[k] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[k] = {"__error__": str(e)}
            done_since_save += 1
            if checkpoint_path and done_since_save >= 16:
                _atomic_dump(results, checkpoint_path)
                done_since_save = 0
    if checkpoint_path:
        _atomic_dump(results, checkpoint_path)
    return results


def _atomic_dump(obj: Any, path: Path) -> None:
    """Escrita atômica de checkpoint (evita corromper em interrupção)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
