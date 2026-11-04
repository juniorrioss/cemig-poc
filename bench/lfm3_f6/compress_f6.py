#!/usr/bin/env python3
"""
compress_f6.py — compressor da F5 (SCR-Gemma, janela de 3 unidades) ESTENDIDO aos 1092
chunks de MANUAL (cache/manuals.db da F5).

O compress.Compressor só conhece os 2202 chunks de norma (units.json + units_win3.npz).
Aqui geramos, com a MESMA segmentação/atribuição de item/janela/prompt de documento, as
unidades e janelas dos chunks de manual -> cache/units_manual.json + units_manual_win3.npz,
e CompressorF6 concatena os dois índices (ids de manual = 1_000_000 + idx, ver manuals.py).

Diferenças deliberadas para manual (documentadas):
  - item de cabeçalho vazio (seção 'manual' não é código citável) -> "(item X)" só aparece
    quando o próprio manual cita um código normativo (comentário do item X);
  - spans de fonte: o corpo do chunk de manual é o texto nativo com espaços normalizados;
    realinhamos com o MESMO align_chunk da F5 contra o arquivo do manual (offsets nativos)
    -> evidence recall pós-compressão vale também para o ouro em manual.

Uso (build, llama-server :8399): classifier/.venv/bin/python bench/lfm3_f6/compress_f6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_f6 as F6  # noqa: E402
import common_f5 as C  # noqa: E402
from align import _src_index, align_chunk  # noqa: E402
from compress import ITEM_RE, Compressor, segment  # noqa: E402
from manuals import ManualIndex  # noqa: E402

UNITS_M_JSON = F6.CACHE / "units_manual.json"
UNITS_M_NPZ = F6.CACHE / "units_manual_win3.npz"


def build_manual_units() -> None:
    S = C.load_sources()
    man = ManualIndex.load()
    idx_cache: Dict[str, Any] = {}
    out: Dict[str, Any] = {}
    win_texts: List[str] = []
    cov_low = 0
    for cid in sorted(man.meta):
        ch = man.meta[cid]
        prefix, body, _ = C.chunk_body(ch["text"])
        f = ch["file"]
        if f not in idx_cache:
            idx_cache[f] = _src_index(S["by_file"][f])
        _, cov, _, tokmap = align_chunk(body, idx_cache[f], want_map=True)
        cov_low += cov < 0.9
        item = ""
        units = []
        for a, b in segment(body):
            text = body[a:b]
            m = ITEM_RE.match(text)
            if m:
                item = m.group(1)
            src = sorted((s2, e2) for (s1, e1, s2, e2) in tokmap if s1 >= a and e1 <= b)
            spans: List[List[int]] = []
            for s2, e2 in src:
                if spans and s2 - spans[-1][1] <= 40:
                    spans[-1][1] = max(spans[-1][1], e2)
                else:
                    spans.append([s2, e2])
            units.append({"a": a, "b": b, "item": item, "starts_item": bool(m), "spans": spans})
        for i in range(len(units)):
            lo, hi = max(0, i - 1), min(len(units), i + 2)
            win_texts.append(f"{prefix}{body[units[lo]['a']:units[hi - 1]['b']]}")
        out[str(cid)] = {"prefix": prefix, "units": units}
    F6.CACHE.mkdir(parents=True, exist_ok=True)
    C.write_json(UNITS_M_JSON, out)
    n = len(win_texts)
    print(f"[units-manual] {n} unidades em {len(out)} chunks (média {n/len(out):.1f}/chunk); "
          f"align cov<0.9: {cov_low}")
    vecs = []
    for i in range(0, n, 32):
        batch = [C.DOC_PROMPT + t for t in win_texts[i:i + 32]]
        r = requests.post(f"{C.EMB_URL}/v1/embeddings", json={"input": batch, "model": "emb"},
                          timeout=600)
        r.raise_for_status()
        d = sorted(r.json()["data"], key=lambda x: x["index"])
        e = np.array([x["embedding"] for x in d], dtype=np.float32)
        vecs.append(e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9))
        if (i // 32) % 100 == 0:
            print(f"  ... {i}/{n}", flush=True)
    np.savez(UNITS_M_NPZ, vecs=np.vstack(vecs))
    print(f"[ok] {UNITS_M_JSON} + {UNITS_M_NPZ}")


class CompressorF6(Compressor):
    """Compressor da F5 com os chunks/unidades/janelas de manual anexados."""

    def __init__(self):
        super().__init__()
        man = ManualIndex.load()
        um = json.loads(UNITS_M_JSON.read_text(encoding="utf-8"))
        k = len(self.vecs)
        for cid in sorted(man.meta):
            self.chunks[cid] = man.meta[cid]
            self.units[str(cid)] = um[str(cid)]
            self.base[cid] = k
            k += len(um[str(cid)]["units"])
        self.vecs = np.vstack([self.vecs, np.load(UNITS_M_NPZ)["vecs"]])
        assert k == len(self.vecs), (k, len(self.vecs))


if __name__ == "__main__":
    build_manual_units()
