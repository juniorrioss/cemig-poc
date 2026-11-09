#!/usr/bin/env python3
"""
inference.py — inferência AUTOCONTIDA do melhor pipeline da POC CEMIG (RAG de NRs PT-BR).

Pipeline (= ponto de operação F6 + modelo f4-r32-ep2, ver config.json):
  fala crua -> classificador de NR (gate 0,7 / boost 2x, override de NR citada)
            -> BM25 FTS5 (36 NRs) + 2 densos EmbeddingGemma (texto, expansão)
            -> + manuais (BM25 + denso, peso 1,75) quando a margem densa liga
            -> RRF k=10 -> top-5 -> compressão a 50% (janelas-3, híbrido denso+lexical)
            -> LFM3-1.7B f4-r32-ep2 (turno tool injetado, thinking OFF) -> resposta citada.

Duas vias de LM: `--backend llama` (GGUF Q4_0 via llama-cpp-python, CPU/GPU) e
`--backend hf` (transformers bf16, trust_remote_code). A busca é idêntica nas duas.

Uso:
  python inference.py "Qual a distância de segurança para 13,8 kV?"
  python inference.py --backend hf --json "pergunta"

Porte fiel (sem importar bench/) de: bench/lfm3_f5/retriever_f5.py, compress.py,
manuals.py, common_f5.py, bench/lfm3_f6/compress_f6.py + e2e_build_f6.py,
bench/lfm3_f5/tool_schema_f4.py e gen_e2e.py, classifier/export_kotlin.py (NRC2).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import struct
import sys
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------------------------
# Artefatos no Hugging Face
# ------------------------------------------------------------------------------------
MODEL_REPO = os.environ.get("CEMIG_MODEL_REPO", "juniorrios/cemig-lfm3-1.7b-nr")
MODEL_REV = os.environ.get("CEMIG_MODEL_REV", None)
# Variante do modelo no repo. Padrão (raiz) = f4-r32-ep2 (APK v1.0). CEMIG_MODEL_VARIANT=v1.1-rlvr
# usa a subpasta v1.1-rlvr/ = v3s400 (RLVR sobre o f4; modelo do APK v1.1): mesmo template,
# tokenizer e ponto de operação, só os pesos mudam.
MODEL_SUBFOLDER = os.environ.get("CEMIG_MODEL_VARIANT", "") or None
GGUF_FILE = ("cemig-lfm3-1.7b-nr-v3s400-Q4_0.gguf" if MODEL_SUBFOLDER == "v1.1-rlvr"
             else "cemig-lfm3-1.7b-nr-Q4_0.gguf")
ASSETS_REPO = os.environ.get("CEMIG_ASSETS_REPO", "juniorrios/cemig-nr-retrieval-assets")
ASSETS_REV = os.environ.get("CEMIG_ASSETS_REV", None)
ASSET_FILES = ["config.json", "index.db", "manuals.db", "dense_text.bin", "dense_exponly.bin",
               "manuals_dense.npz", "units.json", "units_win3.f16.npy", "nr_classifier.bin"]
EMB_REPO = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF"
EMB_REV = "8dd0ca2a66a8f14470acb0e2a71f801afbc5fb73"
EMB_FILE = "embeddinggemma-300M-qat-Q4_0.gguf"

MANUAL_ID_OFFSET = 1_000_000
NONE_LABEL = "nenhuma"
BOS_TOKEN = "<|startoftext|>"
EOS_TOKEN = "<|im_end|>"


def download_assets() -> Path:
    """Baixa (com cache do HF) os artefatos de busca; devolve o diretório local."""
    from huggingface_hub import hf_hub_download
    path = None
    for f in ASSET_FILES:
        path = hf_hub_download(ASSETS_REPO, f, repo_type="dataset", revision=ASSETS_REV)
    return Path(path).parent


def download_embedder() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(EMB_REPO, EMB_FILE, revision=EMB_REV)


# ------------------------------------------------------------------------------------
# Texto: consulta FTS5 do app (corpus/eval_fino.py) e tokens lexicais
# ------------------------------------------------------------------------------------
APP_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "ate", "até",
    "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "do", "dos",
    "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa", "essas",
    "esse", "esses", "esta", "estas", "este", "estes", "eu", "foi", "fomos", "foram",
    "ha", "há", "isso", "isto", "ja", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo",
    "meu", "meus", "minha", "minhas", "muito", "na", "nas", "nao", "não", "no", "nos",
    "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "por", "qual", "quais", "quando", "que", "quem",
    "se", "seja", "sem", "so", "só", "sua", "suas", "seu", "seus", "tambem", "também",
    "te", "tem", "têm", "temos", "ter", "teu", "teus", "tua", "tuas", "um", "uma",
    "voce", "você", "voces", "vocês",
    "norma", "normas", "regulamentar", "regulamentares", "regulamentaria", "regulamentarias",
    "seguranca", "segurança", "trabalho", "item", "artigo",
}


def deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def app_fts_query(raw_query: str) -> str:
    """Fts5Retriever.sanitizeFts5Query do app: stem 6 + prefixo* + OR."""
    raw_tokens = re.findall(r"[\w.-]+", deaccent(raw_query).lower())
    filtered = [t for t in raw_tokens if t not in APP_STOPWORDS and len(t) > 2]
    if not filtered:
        filtered = [t for t in raw_tokens if len(t) > 2]
    if not filtered:
        return '""'
    parts = []
    for tok in filtered:
        stem = tok[:6] if len(tok) > 6 else tok
        parts.append(f'"{stem}"*' if ("-" in stem or "." in stem) else f"{stem}*")
    return " OR ".join(parts)


_TOK = re.compile(r"\w+")


def lex_tokens(text: str) -> List[str]:
    return [t for t in (deaccent(m.group(0)).lower() for m in _TOK.finditer(text))
            if len(t) > 2 and t not in APP_STOPWORDS]


_EXPLICIT_NR = re.compile(r"\bnr[\s._-]*(\d{1,2})\b", re.IGNORECASE)


def detect_explicit_nr(text: str) -> Optional[str]:
    m = _EXPLICIT_NR.search(text or "")
    return "nr-%02d" % int(m.group(1)) if m else None


def chunk_body(text: str) -> Tuple[str, str, int]:
    """Separa o prefixo 'NR-XX · seção título — ' do corpo."""
    i = text.find(" — ")
    if i < 0 or i > 300:
        return "", text, 0
    return text[: i + 3], text[i + 3:], i + 3


# ------------------------------------------------------------------------------------
# Classificador de NR (TF-IDF char_wb 2-5 + word 1-2 + LogReg), formato NRC2, numpy puro
# ------------------------------------------------------------------------------------
class NrClassifier:
    def __init__(self, path: Path):
        data = path.read_bytes()
        assert data[:4] == b"NRC2", "nr_classifier.bin inválido"
        n_cls, = struct.unpack_from("<H", data, 4)
        n_feat, n_char = struct.unpack_from("<II", data, 6)
        off = 14
        self.classes = []
        for _ in range(n_cls):
            ln, = struct.unpack_from("<H", data, off)
            self.classes.append(data[off + 2:off + 2 + ln].decode("utf-8"))
            off += 2 + ln
        self.char_map: Dict[str, int] = {}
        self.word_map: Dict[str, int] = {}
        idf = np.empty(n_feat, dtype=np.float32)
        for col in range(n_feat):
            ln, = struct.unpack_from("<H", data, off)
            ng = data[off + 2:off + 2 + ln].decode("utf-8")
            idf[col], = struct.unpack_from("<f", data, off + 2 + ln)
            off += 6 + ln
            (self.char_map if col < n_char else self.word_map)[ng] = col
        self.idf = idf
        self.coef = np.frombuffer(data, dtype=np.float32, count=n_cls * n_feat,
                                  offset=off).reshape(n_cls, n_feat)
        self.intercept = np.frombuffer(data, dtype=np.float32, count=n_cls,
                                       offset=off + n_cls * n_feat * 4)
        self.n_char = n_char

    @staticmethod
    def _char_wb(text: str) -> List[str]:
        out = []
        for token in text.split():
            w = " " + token + " "
            for n in range(2, 6):
                offset = 0
                out.append(w[0:n])
                while offset + n < len(w):
                    offset += 1
                    out.append(w[offset:offset + n])
                if offset == 0:
                    break
        return out

    @staticmethod
    def _word(text: str) -> List[str]:
        toks = re.findall(r"(?u)\b\w\w+\b", text)
        return [" ".join(toks[i:i + n]) for n in (1, 2) for i in range(len(toks) - n + 1)]

    def predict_proba(self, text: str) -> np.ndarray:
        t = deaccent(text).lower().strip()
        vec = np.zeros(self.idf.shape[0], dtype=np.float64)
        for ng in self._char_wb(t):
            c = self.char_map.get(ng)
            if c is not None:
                vec[c] += 1.0
        for ng in self._word(t):
            c = self.word_map.get(ng)
            if c is not None:
                vec[c] += 1.0
        nz = vec > 0
        vec[nz] = 1.0 + np.log(vec[nz])
        vec *= self.idf
        for sl in (slice(0, self.n_char), slice(self.n_char, None)):
            n = np.linalg.norm(vec[sl])
            if n > 0:
                vec[sl] /= n
        logits = self.coef.astype(np.float64) @ vec + self.intercept
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def top2(self, text: str) -> Tuple[List[str], float]:
        p = self.predict_proba(text)
        order = np.argsort(-p)[:2]
        return [self.classes[j] for j in order], float(p.max())


# ------------------------------------------------------------------------------------
# Encoder denso (EmbeddingGemma-300M GGUF via llama-cpp-python)
# ------------------------------------------------------------------------------------
QUERY_PROMPT = "task: search result | query: "


class Embedder:
    def __init__(self, gguf: str, n_threads: Optional[int] = None):
        from llama_cpp import LLAMA_POOLING_TYPE_MEAN, Llama
        self.m = Llama(gguf, embedding=True, pooling_type=LLAMA_POOLING_TYPE_MEAN, n_ctx=2048,
                       n_batch=2048, n_ubatch=2048, n_gpu_layers=0, n_threads=n_threads,
                       verbose=False)

    def query(self, text: str) -> np.ndarray:
        v = np.asarray(self.m.embed(QUERY_PROMPT + text, normalize=False), dtype=np.float32)
        n = np.linalg.norm(v)
        return v / (n if n > 0 else 1.0)


def load_dvec(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Formato DVEC1: magic + dim(i32) + count(i32) + count*[id i32, vec f32*dim]."""
    data = path.read_bytes()
    assert data[:5] == b"DVEC1", path
    dim, count = struct.unpack_from("<ii", data, 5)
    rec = np.frombuffer(data, dtype=np.dtype([("id", "<i4"), ("v", "<f4", (dim,))]),
                        count=count, offset=13)
    return rec["id"].astype(np.int32), np.ascontiguousarray(rec["v"], dtype=np.float32)


def rrf_fuse(rankings: List[List[Dict[str, Any]]], weights: List[float], k: float,
             limit: int) -> List[int]:
    scores: Dict[int, float] = {}
    for rank_list, w in zip(rankings, weights):
        for pos, r in enumerate(rank_list):
            scores[r["id"]] = scores.get(r["id"], 0.0) + w / (k + pos + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:limit]]


# ------------------------------------------------------------------------------------
# Retriever do ponto de operação
# ------------------------------------------------------------------------------------
class Retriever:
    def __init__(self, assets: Path, cfg: Dict[str, Any], embedder: Embedder):
        self.cfg = cfg["retrieval"]
        self.ctrl = cfg["budget_control"]
        self.emb = embedder
        self.clf = NrClassifier(assets / "nr_classifier.bin")
        self.con = sqlite3.connect(str(assets / "index.db"), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.mcon = sqlite3.connect(str(assets / "manuals.db"), check_same_thread=False)
        self.mcon.row_factory = sqlite3.Row
        self.ids_t, self.mat_t = load_dvec(assets / "dense_text.bin")
        self.ids_e, self.mat_e = load_dvec(assets / "dense_exponly.bin")
        z = np.load(assets / "manuals_dense.npz")
        self.ids_m, self.mat_m = z["ids"], z["vecs"]
        self.chunks: Dict[int, Dict[str, Any]] = {r["id"]: dict(r) for r in self.con.execute(
            "SELECT id, doc, section, title, page, text FROM chunks")}
        for r in self.mcon.execute("SELECT id, doc, section, title, page, text FROM mchunks"):
            self.chunks[r["id"]] = dict(r)
        bm = self.cfg["bm25"]
        self.w_norm = ", ".join(str(x) for x in bm["field_weights_norms"])
        self.w_man = ", ".join(str(x) for x in bm["field_weights_manuals"])
        self.pool = int(bm["pool"])

    # ---------------- BM25 (normas)
    def _bm(self, fq: str, limit: int, doc_filter: Optional[List[str]] = None):
        if fq == '""':
            return []
        cols = "c.id, c.doc, c.section, c.title, c.page, c.text"
        if doc_filter:
            ph = ",".join("?" for _ in doc_filter)
            sql = (f"SELECT {cols}, bm25(chunks_fts, {self.w_norm}) AS score FROM chunks_fts "
                   f"JOIN chunks c ON c.id = chunks_fts.rowid WHERE c.doc IN ({ph}) AND "
                   f"chunks_fts MATCH ? ORDER BY score ASC LIMIT ?;")
            params: Tuple = (*doc_filter, fq, limit)
        else:
            sql = (f"SELECT {cols}, bm25(chunks_fts, {self.w_norm}) AS score FROM chunks_fts "
                   f"JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ? "
                   f"ORDER BY score ASC LIMIT ?;")
            params = (fq, limit)
        try:
            return [dict(r) for r in self.con.execute(sql, params).fetchall()]
        except sqlite3.OperationalError:
            return []

    def _hard(self, fq: str, nr: str):
        res = self._bm(fq, self.pool, doc_filter=[nr])
        if len(res) < self.pool:
            seen = {r["id"] for r in res}
            res += [r for r in self._bm(fq, self.pool) if r["id"] not in seen]
        return res[:self.pool]

    def _soft(self, fq: str, nrs: List[str], factor: float):
        res = self._bm(fq, self.pool)
        bset = set(nrs)
        for r in res:
            r["adj"] = r["score"] * factor if r["doc"] in bset else r["score"]
        return sorted(res, key=lambda r: r["adj"])

    def bm25_signal(self, q: str, conf_thr: float, factor: float):
        fq = app_fts_query(q)
        if self.cfg["gate"]["explicit_nr_override"]:
            ex = detect_explicit_nr(q)
            if ex is not None:
                return self._hard(fq, ex), "explicit"
        preds, p1 = self.clf.top2(q)
        top1 = preds[0] if preds else NONE_LABEL
        if top1 == NONE_LABEL:
            return self._bm(fq, self.pool), "none"
        if p1 >= conf_thr:
            return self._hard(fq, top1), "hard"
        return self._soft(fq, [p for p in preds[:2] if p != NONE_LABEL], factor), "soft"

    # ---------------- densos / manuais
    def _knn(self, qv, ids, mat):
        sims = mat @ qv
        top = np.argsort(-sims)[:self.pool]
        return [{"id": int(ids[j]), "score": float(sims[j])} for j in top]

    def _manual_bm25(self, q: str):
        fq = app_fts_query(q)
        if fq == '""':
            return []
        try:
            cur = self.mcon.execute(
                f"SELECT m.id, bm25(mchunks_fts, {self.w_man}) AS score FROM mchunks_fts "
                "JOIN mchunks m ON m.id = mchunks_fts.rowid WHERE mchunks_fts MATCH ? "
                "ORDER BY score LIMIT ?", (fq, self.pool))
        except sqlite3.OperationalError:
            return []
        return [{"id": r["id"], "score": r["score"]} for r in cur]

    def search(self, q: str, qv: np.ndarray, conf_thr: float, factor: float, manuals: bool,
               top_k: int = 5) -> Dict[str, Any]:
        s1, mode = self.bm25_signal(q, conf_thr, factor)
        sT = self._knn(qv, self.ids_t, self.mat_t)
        sE = self._knn(qv, self.ids_e, self.mat_e)
        signals, weights = [s1, sT, sE], list(self.cfg["rrf"]["weights"])
        man_on = False
        if manuals:
            mc = self.cfg["manuals"]
            top_m = float((self.mat_m @ qv).max())
            man_on = mc["gate"] == "always" or top_m >= sT[0]["score"] + float(mc["margin"])
            if man_on:
                signals += [self._manual_bm25(q), self._knn(qv, self.ids_m, self.mat_m)]
                weights += [float(mc["weight"])] * 2
        ids = rrf_fuse(signals, weights, k=float(self.cfg["rrf"]["k"]), limit=max(top_k, 5))
        return {"ids": ids, "mode": mode, "manuals_on": man_on}


# ------------------------------------------------------------------------------------
# Compressor de contexto (janelas-3, seleção gulosa até o orçamento)
# ------------------------------------------------------------------------------------
ITEM_RE = re.compile(r"(?<![\w.,])(\d{1,2}(?:\.\d{1,3}){1,5})(?![\w,]|\.\d)")


class Compressor:
    def __init__(self, assets: Path, chunks: Dict[int, Dict[str, Any]], tokenizer_json: str):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(tokenizer_json)
        raw = json.loads((assets / "units.json").read_text(encoding="utf-8"))
        self.units = {int(c): {"prefix": u["prefix"],
                               "units": [{"a": a, "b": b, "item": it, "starts_item": bool(s)}
                                         for a, b, it, s in u["units"]]}
                      for c, u in raw.items()}
        self.vecs = np.load(assets / "units_win3.f16.npy").astype(np.float32)
        self.chunks = chunks
        self.base: Dict[int, int] = {}
        k = 0
        for cid in sorted(c for c in self.units if c < MANUAL_ID_OFFSET) + \
                sorted(c for c in self.units if c >= MANUAL_ID_OFFSET):
            self.base[cid] = k
            k += len(self.units[cid]["units"])
        assert k == len(self.vecs), (k, len(self.vecs))

    def ntok(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    @staticmethod
    def source_label(ch: Dict[str, Any]) -> str:
        src = f"{ch['doc']} - {ch['section']}"
        if ch.get("title"):
            src += f" - {ch['title']}"
        return src

    def render_raw(self, cids: List[int]) -> str:
        return "\n\n".join(f"[{i}] ({self.source_label(self.chunks[c])}):\n{self.chunks[c]['text']}"
                           for i, c in enumerate(cids, 1))

    def _unit_plain(self, cid: int, j: int) -> str:
        u = self.units[cid]["units"][j]
        _, body, _ = chunk_body(self.chunks[cid]["text"])
        return body[u["a"]:u["b"]]

    def _unit_text(self, cid: int, j: int) -> str:
        u = self.units[cid]["units"][j]
        t = self._unit_plain(cid, j)
        return f"(item {u['item']}) {t}" if (not u["starts_item"] and u["item"]) else t

    def render_selected(self, sel: Dict[int, List[int]], order: List[int]) -> str:
        parts, i = [], 0
        for cid in order:
            js = sorted(sel.get(cid, []))
            if not js:
                continue
            i += 1
            prefix = self.units[cid]["prefix"].rstrip(" —").strip()
            pieces, prev = [], None
            for j in js:
                if prev is not None and j != prev + 1:
                    pieces.append("[…]")
                pieces.append(self._unit_text(cid, j) if (prev is None or j != prev + 1)
                              else self._unit_plain(cid, j))
                prev = j
            parts.append(f"[{i}] ({self.source_label(self.chunks[cid])}):\n{prefix} — "
                         + " ".join(pieces))
        return "\n\n".join(parts)

    def _lex(self, q: str, cid: int) -> np.ndarray:
        qs = {t[:6] for t in lex_tokens(q)}
        us = self.units[cid]["units"]
        if not qs:
            return np.zeros(len(us))
        _, body, _ = chunk_body(self.chunks[cid]["text"])
        toks = [{t[:6] for t in lex_tokens(body[u["a"]:u["b"]])} for u in us]
        return np.array([len(qs & set().union(*toks[max(0, i - 1):i + 2])) / len(qs)
                         for i in range(len(us))])

    def compress(self, qv: np.ndarray, cids: List[int], budget: int, q: str,
                 lam: float, prior: float) -> Dict[str, Any]:
        cand = []
        for r, cid in enumerate(cids):
            b, n = self.base[cid], len(self.units[cid]["units"])
            sc = self.vecs[b:b + n] @ qv
            sc = sc + lam * self._lex(q, cid)
            for j, s in enumerate(sc):
                cand.append((float(s) + prior / (r + 1), r, cid, j))
        cand.sort(key=lambda x: -x[0])
        hdr = {cid: self.ntok(f"[9] ({self.source_label(self.chunks[cid])}):\n"
                              f"{self.units[cid]['prefix'].rstrip(' —').strip()} — ") + 2
               for cid in cids}
        sel: Dict[int, List[int]] = {}
        used, picked = 0, []
        for s, r, cid, j in cand:
            cost = self.ntok(self._unit_text(cid, j)) + 3 + (hdr[cid] if cid not in sel else 0)
            if used + cost > budget:
                continue
            sel.setdefault(cid, []).append(j)
            used += cost
            picked.append((s, cid, j))
        if not sel:
            s, r, cid, j = cand[0]
            sel, picked = {cid: [j]}, [(s, cid, j)]
        text = self.render_selected(sel, cids)
        while self.ntok(text) > budget and len(picked) > 1:
            s, cid, j = picked.pop()
            sel[cid].remove(j)
            if not sel[cid]:
                del sel[cid]
            text = self.render_selected(sel, cids)
        return {"text": text, "sel": sel, "ntok": self.ntok(text)}


# ------------------------------------------------------------------------------------
# Prompt: ferramenta buscar_norma (contrato do f4-r32-ep2) + template LFM3
# ------------------------------------------------------------------------------------
TOOL_BUSCAR_NORMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "buscar_norma",
        "description": (
            "Busca trechos das Normas Regulamentadoras (NRs) brasileiras de segurança do "
            "trabalho para fundamentar a resposta. Chame quando precisar de um dado "
            "normativo que ainda não está na conversa; chame DE NOVO com outros termos se "
            "os trechos retornados não bastarem para responder. NÃO chame para saudações, "
            "agradecimentos, conversa geral, nem quando os trechos já consultados nesta "
            "conversa respondem à nova pergunta."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": (
                        "Os termos de busca. Use a própria pergunta do usuário como ela é "
                        "(fala crua); numa SEGUNDA busca da mesma pergunta, use termos "
                        "diferentes, focados no que faltou."
                    ),
                },
                "nr": {
                    "type": ["string", "null"],
                    "description": (
                        "A norma no formato 'NR-10' quando citada ou inequívoca; senão null."
                    ),
                },
            },
            "required": ["consulta", "nr"],
        },
    },
}

SYSTEM_PROMPT = (
    "Você é um assistente técnico de segurança do trabalho, especialista em "
    "Normas Regulamentadoras (NRs) brasileiras. Responda sempre em português do "
    "Brasil, de forma objetiva e curta.\n"
    "Você tem a ferramenta buscar_norma(consulta, nr) para consultar as NRs. "
    "Decida sozinho quando usá-la:\n"
    "- CHAME quando precisar de um dado normativo que ainda não está na conversa. "
    "No campo 'consulta', use a pergunta do usuário como ela é; preencha 'nr' só "
    "quando a norma for citada ou inequívoca, senão null.\n"
    "- Se os trechos retornados responderem só PARTE da pergunta, CHAME DE NOVO "
    "com uma consulta diferente, focada no que faltou, antes de responder.\n"
    "- NÃO CHAME para saudações, agradecimentos, conversa geral, nem quando os "
    "trechos já consultados nesta conversa respondem à nova pergunta (reuse o "
    "contexto).\n"
    "Na resposta final: baseie-se exclusivamente nos trechos consultados e cite a "
    "fonte (norma e item). Se os trechos não contiverem a informação, diga isso "
    "claramente em vez de inventar. Quando não houver contexto, responda com seu "
    "conhecimento das normas, citando a norma quando souber."
)


def baked_system_content() -> str:
    """Tools BAKED no system (== apply_chat_template com tools=, provado na F4)."""
    fn = TOOL_BUSCAR_NORMA["function"]
    items = [f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
             for k, v in fn.items() if k != "defer_loading"]
    return SYSTEM_PROMPT + "\n\n<tools>\n" + "{" + ", ".join(items) + "}" + "\n</tools>"


def build_messages(q: str, context: str) -> List[Dict[str, Any]]:
    ex = detect_explicit_nr(q)
    nr_arg = f"NR-{int(ex[3:])}" if ex else None
    return [
        {"role": "system", "content": baked_system_content()},
        {"role": "user", "content": q},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
            "name": "buscar_norma", "arguments": {"consulta": q, "nr": nr_arg}}}]},
        {"role": "tool", "content": context},
    ]


@lru_cache(maxsize=4)
def _jinja_template(path: str):
    """chat_template.jinja do modelo renderizado como o transformers faz (sandbox,
    trim/lstrip_blocks, tojson sem escape) — sem depender do transformers."""
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def _raise(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, \
        sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                                    separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = _raise
    src = Path(path).read_text(encoding="utf-8")
    src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)   # tags de máscara de treino
    return env.from_string(src)


def render_prompt(messages: List[Dict[str, Any]], template_path: str) -> str:
    return _jinja_template(template_path).render(
        messages=messages, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN,
        add_generation_prompt=True, enable_thinking=False)


def clean_answer(text: str) -> str:
    """Remove <think>…</think>, corta re-chamada de ferramenta e o <|im_end|>."""
    t = re.sub(r"^\s*<think>.*?</think>", "", text or "", flags=re.DOTALL)
    i = t.find("<tool_call>")
    if i >= 0:
        t = t[:i]
    return t.replace(EOS_TOKEN, "").strip()


# ------------------------------------------------------------------------------------
# LMs
# ------------------------------------------------------------------------------------
class LlamaCppLM:
    """GGUF Q4_0 via llama-cpp-python, sampler na ordem do llama-server
    (penalties -> top_k -> top_p -> min_p -> temp -> dist) com o prompt no histórico
    de penalidade (= llama-server onde o GGUF foi validado na F7)."""

    def __init__(self, gguf: str, gen: Dict[str, Any], n_gpu_layers: int = 0,
                 n_threads: Optional[int] = None, n_ctx: int = 4096):
        from llama_cpp import Llama
        self.m = Llama(gguf, n_ctx=n_ctx, n_batch=512, n_gpu_layers=n_gpu_layers,
                       n_threads=n_threads, verbose=False)
        self.gen = gen
        self.eos = self.m.tokenize(EOS_TOKEN.encode(), add_bos=False, special=True)[0]

    def _chain(self, seed: int):
        import llama_cpp as L
        g = self.gen
        chain = L.llama_sampler_chain_init(L.llama_sampler_chain_default_params())
        L.llama_sampler_chain_add(chain, L.llama_sampler_init_penalties(
            self.m.n_vocab(), int(g["repeat_last_n"]), float(g["repetition_penalty"]), 0.0, 0.0))
        L.llama_sampler_chain_add(chain, L.llama_sampler_init_top_k(int(g["top_k"])))
        L.llama_sampler_chain_add(chain, L.llama_sampler_init_top_p(float(g["top_p"]), 1))
        L.llama_sampler_chain_add(chain, L.llama_sampler_init_min_p(float(g["min_p"]), 1))
        L.llama_sampler_chain_add(chain, L.llama_sampler_init_temp(float(g["temperature"])))
        L.llama_sampler_chain_add(chain, L.llama_sampler_init_dist(seed))
        return chain

    def n_tokens(self, prompt: str) -> int:
        return len(self.m.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))

    def generate(self, prompt: str, seed: int) -> Tuple[str, int]:
        import llama_cpp as L
        toks = self.m.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        chain = self._chain(seed)
        try:
            for t in toks:
                L.llama_sampler_accept(chain, t)
            self.m.reset()
            self.m.eval(toks)
            out: List[int] = []
            for _ in range(int(self.gen["max_new_tokens"])):
                t = L.llama_sampler_sample(chain, self.m.ctx, -1)
                if t == self.eos or L.llama_vocab_is_eog(self.m._model.vocab, t):
                    break
                out.append(t)
                self.m.eval([t])
        finally:
            L.llama_sampler_free(chain)
        return self.m.detokenize(out, special=True).decode("utf-8", errors="replace"), len(out)


class HfLM:
    """transformers bf16 (trust_remote_code), mesma chamada de generate do gen_e2e.py."""

    def __init__(self, repo: str, gen: Dict[str, Any], device: Optional[str] = None,
                 revision: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"subfolder": MODEL_SUBFOLDER} if MODEL_SUBFOLDER else {}
        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, revision=revision, **kw)
        # sem device_map (dispensa a dependência accelerate)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.bfloat16, trust_remote_code=True,
            revision=revision, **kw).to(self.device).eval()
        self.gen = gen
        self.eos = self.tok.convert_tokens_to_ids(EOS_TOKEN)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos

    def render(self, messages: List[Dict[str, Any]]) -> str:
        return self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=False, enable_thinking=False)

    def n_tokens(self, prompt: str) -> int:
        return len(self.tok(prompt, add_special_tokens=False)["input_ids"])

    def generate(self, prompt: str, seed: int) -> Tuple[str, int]:
        torch, g = self.torch, self.gen
        torch.manual_seed(seed)
        enc = self.tok([prompt], return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=int(g["max_new_tokens"]),
                                      do_sample=True, temperature=float(g["temperature"]),
                                      top_k=int(g["top_k"]),
                                      repetition_penalty=float(g["repetition_penalty"]),
                                      pad_token_id=self.pad, eos_token_id=self.eos)
        gen = out[0][enc["input_ids"].shape[1]:].tolist()
        if self.eos in gen:
            gen = gen[: gen.index(self.eos)]
        return self.tok.decode(gen, skip_special_tokens=False), len(gen)


# ------------------------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------------------------
class Pipeline:
    def __init__(self, backend: str = "llama", n_gpu_layers: int = 0,
                 n_threads: Optional[int] = None, device: Optional[str] = None,
                 retrieval_only: bool = False):
        from huggingface_hub import hf_hub_download
        self.assets = download_assets()
        self.cfg = json.loads((self.assets / "config.json").read_text(encoding="utf-8"))
        self.embedder = Embedder(download_embedder(), n_threads=n_threads)
        self.rtv = Retriever(self.assets, self.cfg, self.embedder)
        tok_json = hf_hub_download(MODEL_REPO, "tokenizer.json", subfolder=MODEL_SUBFOLDER, revision=MODEL_REV)
        self.template = hf_hub_download(MODEL_REPO, "chat_template.jinja", subfolder=MODEL_SUBFOLDER,
                                        revision=MODEL_REV)
        self.comp = Compressor(self.assets, self.rtv.chunks, tok_json)
        self.backend = backend
        self.lm = None
        if retrieval_only:
            return
        gen = self.cfg["generation"]
        if backend == "llama":
            self.lm = LlamaCppLM(hf_hub_download(MODEL_REPO, GGUF_FILE, subfolder=MODEL_SUBFOLDER,
                                                 revision=MODEL_REV),
                                 gen, n_gpu_layers=n_gpu_layers, n_threads=n_threads)
        elif backend == "hf":
            self.lm = HfLM(MODEL_REPO, gen, device=device, revision=MODEL_REV)
        else:
            raise ValueError(backend)

    def context(self, q: str) -> Dict[str, Any]:
        """Busca + compressão (determinístico; idêntico nas duas vias)."""
        rc, cc = self.cfg["retrieval"], self.cfg["compression"]
        t0 = time.time()
        qv = self.embedder.query(q)
        ctrl = self.rtv.search(q, qv, conf_thr=0.5, factor=5.0, manuals=False, top_k=5)
        budget = int(self.comp.ntok(self.comp.render_raw(ctrl["ids"][:2]))
                     * float(self.cfg["budget_control"]["ratio"]))
        op = self.rtv.search(q, qv, conf_thr=float(rc["gate"]["conf_thr"]),
                             factor=float(rc["gate"]["soft_boost_factor"]), manuals=True,
                             top_k=int(rc["top_k"]))
        ids = op["ids"][:int(rc["top_k"])]
        c = self.comp.compress(qv, ids, budget, q=q, lam=float(cc["lambda_lexical"]),
                               prior=float(cc["rank_prior"]))
        used = [cid for cid in ids if c["sel"].get(cid)]
        return {"context": c["text"], "ctx_ntok": c["ntok"], "budget": budget,
                "retrieved_ids": ids, "used_ids": used, "gate_mode": op["mode"],
                "manuals_on": op["manuals_on"],
                "sources": [{"id": cid, "kind": "manual" if cid >= MANUAL_ID_OFFSET else "norma",
                             "label": self.comp.source_label(self.rtv.chunks[cid])}
                            for cid in used],
                "retrieval_s": round(time.time() - t0, 3)}

    def ask(self, q: str, seed: Optional[int] = None) -> Dict[str, Any]:
        seed = int(self.cfg["generation"]["seed"] if seed is None else seed)
        res = self.context(q)
        messages = build_messages(q, res["context"])
        prompt = (self.lm.render(messages) if isinstance(self.lm, HfLM)
                  else render_prompt(messages, self.template))
        t0 = time.time()
        raw, n_gen = self.lm.generate(prompt, seed)
        res.update({"question": q, "backend": self.backend, "seed": seed,
                    "n_prompt_tokens": self.lm.n_tokens(prompt), "n_gen_tokens": n_gen,
                    "raw": raw, "answer": clean_answer(raw),
                    "generation_s": round(time.time() - t0, 2)})
        return res


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG de NRs (CEMIG POC) — LFM3-1.7B f4-r32-ep2")
    ap.add_argument("question", nargs="*", help="pergunta (ou --file com 1 por linha)")
    ap.add_argument("--file", help="arquivo .txt (1 pergunta/linha) ou .jsonl (campo question)")
    ap.add_argument("--backend", choices=["llama", "hf"], default="llama")
    ap.add_argument("--n-gpu-layers", type=int, default=0, help="só backend llama (99 = tudo)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default=None, help="só backend hf (cuda|cpu)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--retrieval-only", action="store_true", help="só busca + compressão")
    ap.add_argument("--json", action="store_true", help="saída JSON (1 objeto por linha)")
    a = ap.parse_args()

    qs = [" ".join(a.question)] if a.question else []
    if a.file:
        for line in Path(a.file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                qs.append(json.loads(line)["question"] if a.file.endswith(".jsonl") else line.strip())
    if not qs:
        ap.error("informe uma pergunta")
    pipe = Pipeline(a.backend, n_gpu_layers=a.n_gpu_layers, n_threads=a.threads,
                    device=a.device, retrieval_only=a.retrieval_only)
    for q in qs:
        r = pipe.context(q) | {"question": q} if a.retrieval_only else pipe.ask(q, seed=a.seed)
        if a.json:
            print(json.dumps(r, ensure_ascii=False), flush=True)
            continue
        print(f"\n### Pergunta\n{q}")
        if not a.retrieval_only:
            print(f"\n### Resposta ({r['backend']}, seed {r['seed']})\n{r['answer']}")
        print("\n### Fontes usadas")
        for s in r["sources"]:
            print(f"- [{s['kind']}] {s['label']}")
        print(f"\n### Trechos entregues ao modelo ({r['ctx_ntok']} tokens, orçamento {r['budget']})")
        print(r["context"])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
