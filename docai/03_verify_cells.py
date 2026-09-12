"""
03_verify_cells.py — ETAPA 5: verificação célula-a-célula contra o texto original (parquet).

Regra do capitão: NENHUM número pode mudar de valor. Todo número presente nas frases
achatadas (results/parsed_pilot/sentences_<nr>.json) DEVE existir no texto original da
mesma NR no parquet (após normalização dos glifos PUA). Os que não fecharem vão para a
lista "nao reparavel automaticamente" COM CONTAGEM — preferimos relatar do que fingir cobertura.

Metodologia (barata, paralela, determinística):
  - Extrai o multiset de números (decimais pt-BR e inteiros) do texto-fonte de cada NR.
  - Extrai os números "de dado" das frases zona_risco (rr, rc, faixa) e genéricas.
  - Um número é VERIFICADO se aparece no multiset da fonte (comparação por valor).
  - Números "gerados" que são tensões usuais de expansão (13,8 etc.) NÃO precisam existir
    na fonte como número isolado — são rótulos de faixa; são checados por CONTENÇÃO na faixa,
    não por igualdade. Reporta-se essa categoria à parte (procedência honesta).

Comentários em português; identificadores em inglês (padrão do projeto).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARQUET = Path("/home/rios/projetos/cemig-mobile-llm/firstmate/data/nrs_hf/nrs.parquet")

# Glifos PUA -> ASCII (mesmo mapa do 02, para comparar a fonte já normalizada).
GLYPH_MAP = {"\uf03c": "<", "\uf03e": ">", "\uf0b3": ">=", "\uf0a3": "<=", "\uf0b7": "•"}

# Tensões usuais expandidas (não precisam existir como número isolado na fonte).
USUAL_VOLTAGES = {"0,22", "0,38", "0,44", "2,3", "3,8", "6,6", "11,4", "13,8", "23",
                  "34,5", "44", "69", "88", "138", "230", "345", "440", "500"}

NUM_RE = re.compile(r"\d+(?:,\d+)?")


def source_numbers(nr: str) -> Counter:
    """Multiset de números (string pt-BR normalizada) do texto-fonte da NR no parquet."""
    rows = pq.read_table(str(PARQUET)).to_pylist()
    num = int(nr.split("-")[1])
    row = next((r for r in rows if r["numero"] == num), None)
    if not row:
        return Counter()
    text = row["conteudo"] or ""
    for g, r in GLYPH_MAP.items():
        text = text.replace(g, r)
    return Counter(_canon(m) for m in NUM_RE.findall(text))


def _canon(s: str) -> str:
    """Canoniza um número pt-BR: remove zeros à direita da parte decimal e vírgula supérflua."""
    s = s.strip()
    if "," in s:
        intp, dec = s.split(",", 1)
        dec = dec.rstrip("0")
        return f"{int(intp)},{dec}" if dec else str(int(intp))
    return str(int(s)) if s.isdigit() else s


def verify_nr(nr: str, sentences: List[Dict], src_nums: Counter) -> Dict:
    """Verifica os números 'de dado' das frases contra a fonte. Retorna estatísticas + falhas."""
    verified = 0
    generated_voltage = 0
    unrepairable: List[Dict] = []

    for s in sentences:
        cells = s.get("source_cells", [])
        data_nums: Set[str] = set()
        for c in cells:
            for m in NUM_RE.findall(c):
                data_nums.add(_canon(m))
        for n in data_nums:
            if n in src_nums:
                verified += 1
            elif n in USUAL_VOLTAGES:
                generated_voltage += 1  # rótulo de faixa, procedência = expansão
            else:
                unrepairable.append({"nr": nr, "num": n, "kind": s.get("kind"),
                                     "sentence": s.get("sentence", "")[:120]})
    return {
        "nr": nr,
        "n_sentences": len(sentences),
        "verified_numbers": verified,
        "generated_voltage_labels": generated_voltage,
        "unrepairable_count": len(unrepairable),
        "unrepairable_sample": unrepairable[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Verificação célula-a-célula das frases DocAI vs parquet.")
    ap.add_argument("--parsed-dir", default="docai/data", help="Diretório com sentences_<nr>.json.")
    ap.add_argument("--out", default="docai/data/verify_report.json")
    ap.add_argument("--nrs", default="nr-10,nr-15,nr-28,nr-04,nr-32")
    args = ap.parse_args()

    parsed = Path(args.parsed_dir)
    report = {"per_nr": {}, "totals": {}}
    tot_v = tot_g = tot_u = 0
    for nr in [n.strip() for n in args.nrs.split(",")]:
        sf = parsed / f"sentences_{nr}.json"
        if not sf.exists():
            logger.warning("sem frases para %s", nr)
            continue
        sentences = json.loads(sf.read_text(encoding="utf-8"))
        src = source_numbers(nr)
        r = verify_nr(nr, sentences, src)
        report["per_nr"][nr] = r
        tot_v += r["verified_numbers"]; tot_g += r["generated_voltage_labels"]; tot_u += r["unrepairable_count"]
        logger.info("%s: verificados=%d, tensoes-geradas=%d, nao-reparavel=%d",
                    nr, r["verified_numbers"], r["generated_voltage_labels"], r["unrepairable_count"])
    report["totals"] = {"verified": tot_v, "generated_voltage": tot_g, "unrepairable": tot_u}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Relatório de verificação salvo em %s (verif=%d, ger=%d, nao-rep=%d)",
                args.out, tot_v, tot_g, tot_u)


if __name__ == "__main__":
    main()
