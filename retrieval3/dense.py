#!/usr/bin/env python3
"""
dense.py — Etapa 3: busca densa mobile-first (encode 5070 -> índice sqlite-vec).

Fluxo:
  1. encode dos 2202 chunks (texto normativo; opcionalmente + expansão) na RTX 5070;
  2. índice vetorial sqlite-vec (vec0) — o MESMO motor viável no Android arm64;
  3. busca densa por similaridade de cosseno (vetores L2-normalizados);
  4. relatório de custo mobile: tamanho do modelo, dim, tamanho do índice,
     latência de encode 5070 e projeção S24+ (1 query = 1 forward ~30 tokens).

Reuso Android: sqlite-vec compila para arm64 (mesma família do libsqliteX já embarcado);
o modelo de embedding é exportável p/ ONNX int8 (dim <=768). Aqui medimos qualidade e
custo; a integração no app corre em paralelo (não faz parte desta task).

Procedência: task poc-retrieval-v3. PT-BR nos comentários; inglês no código.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _load_st(model_name: str, device: str = "cuda"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device, trust_remote_code=True)


# Prefixos/prompts assimétricos por família de modelo (query vs passagem).
# Modelos de RETRIEVAL exigem prefixos distintos p/ pergunta e documento; sem isso
# a qualidade despenca (MiniLM-paraphrase é simétrico e falha em retrieval).
PREFIXES = {
    "e5": ("query: ", "passage: "),
    "gte": ("", ""),
    "bge": ("", ""),
}


def resolve_prefixes(model_name: str, override: Optional[str] = None) -> Tuple[str, str]:
    """Devolve (prefixo_query, prefixo_doc) conforme a família do modelo."""
    if override == "e5":
        return PREFIXES["e5"]
    if override == "none":
        return ("", "")
    ml = model_name.lower()
    if "e5" in ml:
        return PREFIXES["e5"]
    return ("", "")


def _pack_f32(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32).tolist())


def load_chunks(db_path: str, with_expansion: bool = False,
                expansions: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT id, doc, section, title, text FROM chunks ORDER BY id").fetchall()]
    con.close()
    if with_expansion and expansions:
        for r in rows:
            ex = expansions.get(r["id"], "")
            if ex:
                r["text"] = r["text"] + " " + ex
    return rows


def build_dense_index(model_name: str, src_db: str, out_db: str,
                      with_expansion: bool = False, expansions_path: Optional[str] = None,
                      dim_trunc: Optional[int] = None, batch: int = 64,
                      device: str = "cuda", prefix_kind: Optional[str] = None,
                      query_prompt_name: Optional[str] = None,
                      doc_prompt_name: Optional[str] = None,
                      expansion_only: bool = False) -> Dict[str, Any]:
    """Codifica chunks e grava índice sqlite-vec. Retorna metadados de custo.

    expansion_only=True: codifica APENAS a expansão coloquial (perguntas-que-o-chunk-
    responde). Assim a busca vira query↔query (fala do operário × perguntas geradas),
    regime em que embeddings simétricos brilham. Fallback ao texto normativo se um
    chunk não tiver expansão.
    """
    import sqlite_vec
    q_pre, d_pre = resolve_prefixes(model_name, prefix_kind)

    expansions = None
    if (with_expansion or expansion_only) and expansions_path:
        expansions = {}
        for line in Path(expansions_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            expansions[r["id"]] = " ".join(r.get("perguntas", []) + r.get("sinonimos", []))

    chunks = load_chunks(src_db, with_expansion and not expansion_only, expansions)
    if expansion_only:
        # texto de indexação = expansão coloquial (fallback ao normativo se vazio)
        texts = [(expansions.get(c["id"]) or c["text"]) if expansions else c["text"]
                 for c in chunks]
    else:
        texts = [c["text"] for c in chunks]

    t0 = time.time()
    model = _load_st(model_name, device=device)
    load_s = time.time() - t0

    enc_kwargs: Dict[str, Any] = dict(batch_size=batch, normalize_embeddings=True,
                                      show_progress_bar=False, convert_to_numpy=True)
    if doc_prompt_name:
        enc_kwargs["prompt_name"] = doc_prompt_name
        doc_texts = texts
    else:
        doc_texts = [d_pre + t for t in texts] if d_pre else texts
    t0 = time.time()
    emb = model.encode(doc_texts, **enc_kwargs)
    encode_s = time.time() - t0
    full_dim = emb.shape[1]
    if dim_trunc and dim_trunc < full_dim:
        # Matryoshka: trunca + re-normaliza
        emb = emb[:, :dim_trunc]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    dim = emb.shape[1]

    out_path = Path(out_db)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    con = sqlite3.connect(str(out_path))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("""
        CREATE TABLE meta (id INTEGER PRIMARY KEY, doc TEXT, section TEXT,
                           title TEXT, text TEXT)
    """)
    con.executemany("INSERT INTO meta VALUES (?,?,?,?,?)",
                    [(c["id"], c["doc"], c["section"], c["title"], c["text"]) for c in chunks])
    con.execute(f"CREATE VIRTUAL TABLE vec USING vec0(embedding float[{dim}])")
    con.executemany("INSERT INTO vec(rowid, embedding) VALUES (?, ?)",
                    [(c["id"], _pack_f32(v)) for c, v in zip(chunks, emb)])
    con.commit()
    con.close()

    # Latência de encode de 1 query (forward curto ~30 tokens)
    q = "posso trabalhar sem desligar a energia no poste?"
    # warmup
    model.encode([q], normalize_embeddings=True)
    t0 = time.time()
    for _ in range(20):
        model.encode([q], normalize_embeddings=True)
    q_ms_gpu = (time.time() - t0) / 20 * 1000

    idx_mb = out_path.stat().st_size / 1e6
    meta = {
        "model": model_name,
        "query_prefix": q_pre,
        "doc_prefix": d_pre,
        "query_prompt_name": query_prompt_name or "",
        "doc_prompt_name": doc_prompt_name or "",
        "full_dim": int(full_dim),
        "dim": int(dim),
        "n_chunks": len(chunks),
        "with_expansion": with_expansion,
        "load_s": round(load_s, 1),
        "encode_all_s": round(encode_s, 1),
        "encode_query_ms_gpu": round(q_ms_gpu, 1),
        "index_mb": round(idx_mb, 2),
        "index_path": str(out_path),
    }
    return meta


class DenseRetriever:
    """Busca densa no índice sqlite-vec (cosseno via distância L2 em vetores normalizados)."""

    def __init__(self, index_db: str, model_name: str, dim_trunc: Optional[int] = None,
                 device: str = "cuda", prefix_kind: Optional[str] = None,
                 query_prompt_name: Optional[str] = None):
        import sqlite_vec
        self.con = sqlite3.connect(index_db)
        self.con.row_factory = sqlite3.Row
        self.con.enable_load_extension(True)
        sqlite_vec.load(self.con)
        self.con.enable_load_extension(False)
        self.model = _load_st(model_name, device=device)
        self.dim_trunc = dim_trunc
        self.q_pre, _ = resolve_prefixes(model_name, prefix_kind)
        self.query_prompt_name = query_prompt_name

    def close(self) -> None:
        self.con.close()

    def encode_query(self, text: str) -> np.ndarray:
        if self.query_prompt_name:
            e = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True,
                                  prompt_name=self.query_prompt_name)[0]
        else:
            e = self.model.encode([self.q_pre + text], normalize_embeddings=True,
                                  convert_to_numpy=True)[0]
        if self.dim_trunc and self.dim_trunc < len(e):
            e = e[:self.dim_trunc]
            e = e / np.linalg.norm(e)
        return e

    def search(self, question: str, limit: int = 5) -> List[Dict[str, Any]]:
        qv = self.encode_query(question)
        rows = self.con.execute(
            """SELECT v.rowid AS id, v.distance AS dist, m.doc, m.section, m.title, m.text
               FROM vec v JOIN meta m ON m.id = v.rowid
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance ASC""",
            (_pack_f32(qv), limit),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["score"] = -float(d.pop("dist"))  # menor distância = melhor
            out.append(d)
        return out

    def search_vec(self, qv: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        rows = self.con.execute(
            """SELECT v.rowid AS id, v.distance AS dist, m.doc, m.section, m.title, m.text
               FROM vec v JOIN meta m ON m.id = v.rowid
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance ASC""",
            (_pack_f32(qv), limit),
        ).fetchall()
        return [dict(r, score=-float(r["dist"])) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-expansion", action="store_true")
    ap.add_argument("--expansions", default=str(_HERE / "data" / "expansions.jsonl"))
    ap.add_argument("--dim-trunc", type=int, default=0)
    ap.add_argument("--prefix-kind", default=None, help="e5 | none (auto por nome se omitido)")
    ap.add_argument("--query-prompt-name", default=None, help="prompt_name p/ query (ex.: EmbeddingGemma)")
    ap.add_argument("--doc-prompt-name", default=None, help="prompt_name p/ documento")
    ap.add_argument("--expansion-only", action="store_true",
                    help="indexa APENAS a expansão coloquial (busca query↔query)")
    ap.add_argument("--meta-out", default="")
    args = ap.parse_args()

    meta = build_dense_index(
        args.model, args.src, args.out,
        with_expansion=args.with_expansion,
        expansions_path=args.expansions if (args.with_expansion or args.expansion_only) else None,
        dim_trunc=args.dim_trunc or None,
        prefix_kind=args.prefix_kind,
        query_prompt_name=args.query_prompt_name,
        doc_prompt_name=args.doc_prompt_name,
        expansion_only=args.expansion_only,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if args.meta_out:
        Path(args.meta_out).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
