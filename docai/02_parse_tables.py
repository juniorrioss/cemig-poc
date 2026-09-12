"""
02_parse_tables.py — Parse dos shards do Document AI, costura entre páginas, normalização de glifos
e achatamento de linhas de tabela em frases pesquisáveis.

Entrada: shards JSON do Document AI (Document proto serializado em JSON) em --shards-dir.
Saída:
  - <out-dir>/tables_<nr>.json      : tabelas reconstruídas por NR (linhas cell-a-cell).
  - <out-dir>/sentences_<nr>.json   : frases achatadas (prosa pesquisável) por tabela/linha.
  - <out-dir>/parse_report.json     : diagnóstico (glifos, costuras, ambíguas, expansões).

Etapas (ordem do brief do capitão):
  2. COSTURA ENTRE PÁGINAS: Form Parser entrega tabela por página; tabelas longas voltam
     partidas. Costura por cabeçalho repetido + continuidade de nº de colunas.
  3. NORMALIZAÇÃO: mapa determinístico dos glifos de fonte privada (PUA).
  4. ACHATAMENTO: cada linha vira frase em prosa com valores explícitos; faixas numéricas
     de tensão expandidas para os valores usuais de distribuição (13,8/23/34,5/69/138 kV)
     que caem na faixa — SEM inventar valor (só cita a célula).

Comentários em português; identificadores em inglês (padrão do projeto).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ETAPA 3 — Mapa determinístico de glifos de fonte privada (Private Use Area).
# Símbolos matemáticos que o encoder de fonte do PDF jogou para a PUA.
# ---------------------------------------------------------------------------
GLYPH_MAP = {
    "\uf03c": "<",    # menor que
    "\uf03e": ">",    # maior que
    "\uf0b3": ">=",   # maior ou igual (≥)
    "\uf0a3": "<=",   # menor ou igual (≤)
    "\uf0b7": "•",    # bullet
    "\uf0b1": "±",    # mais ou menos
    "\uf0d7": "·",    # ponto de multiplicação
    "\uf02d": "-",    # hífen
    "\uf0b0": "°",    # grau
    "\uf06d": "µ",    # mu (micro)
    "\uf057": "Ω",    # ômega (ohm)
}


def normalize_glyphs(text: str) -> Tuple[str, Dict[str, int]]:
    """Substitui glifos PUA por seus equivalentes ASCII/Unicode padrão. Conta cada ocorrência."""
    found: Dict[str, int] = defaultdict(int)
    out = text
    for glyph, repl in GLYPH_MAP.items():
        n = out.count(glyph)
        if n:
            found[glyph] += n
            out = out.replace(glyph, repl)
    # Detecta glifos PUA residuais não mapeados (0xE000–0xF8FF) para relatório honesto.
    residual: Dict[str, int] = defaultdict(int)
    for ch in out:
        if "\ue000" <= ch <= "\uf8ff":
            residual[ch] += 1
    return out, {"mapped": dict(found), "residual_pua": dict(residual)}


# ---------------------------------------------------------------------------
# Leitura do Document proto (JSON) — extração de texto por cell via textAnchor.
# ---------------------------------------------------------------------------
def _text_from_anchor(full_text: str, layout: Dict[str, Any]) -> str:
    """Extrai o texto de uma célula a partir de layout.textAnchor.textSegments."""
    anchor = (layout or {}).get("textAnchor") or {}
    segs = anchor.get("textSegments") or []
    if not segs:
        # Alguns anchors trazem 'content' direto.
        return (anchor.get("content") or "").strip()
    parts = []
    for s in segs:
        start = int(s.get("startIndex", 0) or 0)
        end = int(s.get("endIndex", 0) or 0)
        parts.append(full_text[start:end])
    return " ".join(p.strip() for p in parts).strip()


def _row_cells(full_text: str, row: Dict[str, Any]) -> List[str]:
    """Devolve o texto (normalizado) de cada célula de uma linha."""
    cells = []
    for cell in row.get("cells", []):
        raw = _text_from_anchor(full_text, cell.get("layout", {}))
        norm, _ = normalize_glyphs(raw)
        norm = re.sub(r"\s+", " ", norm).strip()
        cells.append(norm)
    return cells


def parse_shard(shard: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extrai as tabelas de um shard (Document proto). Devolve:
      - lista de tabelas: {page, header:[...], rows:[[...],...], n_cols}
      - contagem agregada de glifos normalizados no shard inteiro.
    """
    full_text = shard.get("text", "")
    _, glyph_stats_full = normalize_glyphs(full_text)
    tables = []
    for page in shard.get("pages", []):
        page_no = int((page.get("pageNumber") or 0))
        for tbl in page.get("tables", []):
            header_rows = tbl.get("headerRows", []) or []
            body_rows = tbl.get("bodyRows", []) or []
            header = _row_cells(full_text, header_rows[0]) if header_rows else []
            # Cabeçalho multi-linha: concatena verticalmente por coluna.
            if len(header_rows) > 1:
                stacked = [_row_cells(full_text, hr) for hr in header_rows]
                width = max(len(r) for r in stacked)
                header = []
                for c in range(width):
                    col_parts = [r[c] for r in stacked if c < len(r) and r[c]]
                    header.append(" ".join(col_parts).strip())
            rows = [_row_cells(full_text, br) for br in body_rows]
            n_cols = max([len(header)] + [len(r) for r in rows]) if (header or rows) else 0
            tables.append({
                "page": page_no,
                "header": header,
                "rows": rows,
                "n_cols": n_cols,
            })
    return tables, glyph_stats_full.get("mapped", {})


# ---------------------------------------------------------------------------
# ETAPA 2 — Costura entre páginas.
# ---------------------------------------------------------------------------
def _header_key(header: List[str]) -> str:
    """Chave canônica de cabeçalho (para detectar repetição entre páginas)."""
    joined = " ".join(header).lower()
    joined = re.sub(r"[^a-z0-9]+", " ", joined).strip()
    return joined


def stitch_tables(tables: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Costura tabelas partidas entre páginas consecutivas.
    Regra: mesmo cabeçalho (repetido) OU cabeçalho vazio + mesmo nº de colunas na
    página imediatamente seguinte -> continuação. Relata costuradas e ambíguas.
    """
    stitched: List[Dict[str, Any]] = []
    n_stitched = 0
    n_ambiguous = 0
    ambiguous_detail: List[Dict[str, Any]] = []

    for tbl in tables:
        merged = False
        if stitched:
            prev = stitched[-1]
            same_header = tbl["header"] and _header_key(tbl["header"]) == _header_key(prev["header"])
            cont_no_header = (not tbl["header"]) and tbl["n_cols"] == prev["n_cols"] and prev["n_cols"] > 0
            consecutive_page = tbl["page"] == prev["page"] + 1 or tbl["page"] == prev["page"]
            if (same_header or cont_no_header):
                # Continuação clara: anexa as linhas.
                prev["rows"].extend(tbl["rows"])
                prev["stitched_from"] = prev.get("stitched_from", [prev["page"]]) + [tbl["page"]]
                n_stitched += 1
                merged = True
            elif consecutive_page and tbl["n_cols"] == prev["n_cols"] and prev["n_cols"] >= 3:
                # Ambíguo: colunas batem mas cabeçalho difere; costura mas registra.
                prev["rows"].extend(tbl["rows"])
                prev["stitched_from"] = prev.get("stitched_from", [prev["page"]]) + [tbl["page"]]
                n_stitched += 1
                n_ambiguous += 1
                ambiguous_detail.append({"page": tbl["page"], "prev_page": prev["page"],
                                         "n_cols": tbl["n_cols"]})
                merged = True
        if not merged:
            stitched.append(dict(tbl))
    return stitched, {"stitched": n_stitched, "ambiguous": n_ambiguous, "ambiguous_detail": ambiguous_detail}


# ---------------------------------------------------------------------------
# ETAPA 4 — Achatamento em frases + expansão de faixas de tensão.
# ---------------------------------------------------------------------------
# Tensões nominais usuais de distribuição/transmissão citadas explicitamente (kV).
USUAL_VOLTAGES_KV = [0.22, 0.38, 0.44, 2.3, 3.8, 6.6, 11.4, 13.8, 23.0, 34.5,
                     44.0, 69.0, 88.0, 138.0, 230.0, 345.0, 440.0, 500.0]

# Faixa de tensão do tipo ">=10 e <15" ou "10 e 15" ou "≥ 10 e < 15 kV".
RANGE_RE = re.compile(
    r"(?:>=|≥)?\s*(\d+(?:[.,]\d+)?)\s*(?:kv)?\s*(?:e|a|-|–|até)\s*(?:<|<=|≤)?\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
SINGLE_LT_RE = re.compile(r"^(?:<)\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)


def _num(s: str) -> Optional[float]:
    """Converte '13,8' ou '10' em float; None se não numérico."""
    s = s.strip().replace(".", "").replace(",", ".") if s.count(",") == 1 and s.count(".") <= 1 else s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def voltages_in_range(lo: float, hi: float, inclusive_lo: bool = True, inclusive_hi: bool = False) -> List[float]:
    """Tensões usuais que caem na faixa [lo, hi) (padrão do Anexo II: >=lo e <hi)."""
    out = []
    for v in USUAL_VOLTAGES_KV:
        ge = v >= lo if inclusive_lo else v > lo
        le = v <= hi if inclusive_hi else v < hi
        if ge and le:
            out.append(v)
    return out


def _fmt_kv(v: float) -> str:
    """Formata a tensão em pt-BR (13.8 -> '13,8')."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def flatten_table(nr: str, tbl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Gera frases pesquisáveis a partir das linhas de uma tabela.
    Detecta o padrão do Anexo II da NR-10 (faixa de tensão -> raio zona risco/controlada)
    e expande as tensões usuais. Para tabelas genéricas, gera frase "col: valor" por linha.
    """
    sentences: List[Dict[str, Any]] = []
    header = tbl.get("header", [])
    header_join = " ".join(header).lower()
    is_zona = ("zona" in header_join and ("risco" in header_join or "controlada" in header_join)) \
        or (nr == "nr-10" and any("raio" in h.lower() for h in header))

    for row in tbl["rows"]:
        cells = [c for c in row]
        nums = [c for c in cells if re.search(r"\d", c)]
        if not nums:
            continue
        # Primeira célula com faixa/tensão.
        faixa_cell = next((c for c in cells if re.search(r"\d", c)), "")
        # Valores numéricos "de saída" (raios/limites) — células com vírgula decimal.
        value_cells = [c for c in cells if re.search(r"\d[.,]\d", c)]

        m = RANGE_RE.search(faixa_cell)
        m_lt = SINGLE_LT_RE.search(faixa_cell.strip())

        if is_zona and (m or m_lt):
            # Padrão do Anexo II: faixa -> [raio risco, raio controlada].
            # value_cells na ordem: Rr (risco->controlada), Rc (controlada->livre).
            vals = value_cells[:2]
            if len(vals) < 2:
                # fallback: pega os 2 últimos numéricos com decimal.
                vals = value_cells[-2:] if len(value_cells) >= 2 else value_cells
            if m:
                lo = _num(m.group(1)); hi = _num(m.group(2))
            else:
                lo = 0.0; hi = _num(m_lt.group(1))
            if lo is None or hi is None or len(vals) < 2:
                continue
            rr, rc = vals[0], vals[1]
            usual = voltages_in_range(lo, hi)
            inc = ", incluindo " + ", ".join(f"{_fmt_kv(v)} kV" for v in usual) if usual else ""
            faixa_txt = (f"entre {_fmt_kv(lo)} kV e {_fmt_kv(hi)} kV" if m
                         else f"abaixo de {_fmt_kv(hi)} kV")
            sent = (f"Para tensão nominal {faixa_txt}{inc}, "
                    f"conforme o Anexo II da NR-10, o raio da zona de risco é de {rr} metros "
                    f"e o raio da zona controlada é de {rc} metros.")
            sentences.append({
                "nr": nr, "kind": "zona_risco",
                "faixa": faixa_cell, "rr": rr, "rc": rc,
                "voltages_expanded": [_fmt_kv(v) for v in usual],
                "sentence": sent,
                "source_cells": cells,
            })
        else:
            # Frase genérica cabeçalho:valor (mantém rastreabilidade cell-a-cell).
            pairs = []
            for i, c in enumerate(cells):
                h = header[i] if i < len(header) and header[i] else f"coluna {i+1}"
                if c:
                    pairs.append(f"{h}: {c}")
            if pairs:
                sent = f"({nr.upper()}) " + "; ".join(pairs) + "."
                sentences.append({
                    "nr": nr, "kind": "generic",
                    "sentence": sent, "source_cells": cells,
                })
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse+costura+normalização+achatamento das tabelas DocAI.")
    parser.add_argument("--shards-dir", required=True, help="Diretório com os shards JSON do Document AI.")
    parser.add_argument("--out-dir", required=True, help="Diretório de saída (tables/sentences/report).")
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Agrupa shards por NR (nome do PDF original vem no shard ou no nome do arquivo).
    shard_files = sorted(shards_dir.glob("*.json"))
    if not shard_files:
        raise SystemExit(f"Nenhum shard em {shards_dir}")

    report: Dict[str, Any] = {"glyphs_total": defaultdict(int), "per_nr": {}}

    # Um output/ do DocAI tem subpastas por doc; o nome do shard costuma conter o nome de origem.
    by_nr: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sf in shard_files:
        m = re.search(r"(nr-\d{2})", sf.name.lower())
        nr = m.group(1) if m else sf.stem
        shard = json.loads(sf.read_text(encoding="utf-8"))
        by_nr[nr].append(shard)

    for nr, shards in by_nr.items():
        all_tables: List[Dict[str, Any]] = []
        glyph_nr: Dict[str, int] = defaultdict(int)
        for shard in shards:
            tables, glyphs = parse_shard(shard)
            all_tables.extend(tables)
            for g, c in glyphs.items():
                glyph_nr[g] += c
                report["glyphs_total"][g] += c
        # Ordena por página antes de costurar.
        all_tables.sort(key=lambda t: t["page"])
        stitched, stitch_stats = stitch_tables(all_tables)

        sentences: List[Dict[str, Any]] = []
        for tbl in stitched:
            sentences.extend(flatten_table(nr, tbl))

        (out_dir / f"tables_{nr}.json").write_text(
            json.dumps(stitched, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"sentences_{nr}.json").write_text(
            json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8")

        report["per_nr"][nr] = {
            "n_tables_raw": len(all_tables),
            "n_tables_stitched": len(stitched),
            "stitch": stitch_stats,
            "glyphs": dict(glyph_nr),
            "n_sentences": len(sentences),
            "n_zona_sentences": sum(1 for s in sentences if s["kind"] == "zona_risco"),
        }
        logger.info("%s: %d tabelas cruas -> %d costuradas, %d frases (%d zona_risco), glifos=%s",
                    nr, len(all_tables), len(stitched), len(sentences),
                    report["per_nr"][nr]["n_zona_sentences"], dict(glyph_nr))

    report["glyphs_total"] = dict(report["glyphs_total"])
    (out_dir / "parse_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Relatório salvo em %s", out_dir / "parse_report.json")


if __name__ == "__main__":
    main()
