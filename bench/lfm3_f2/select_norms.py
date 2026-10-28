"""
select_norms.py — fase 1 do F2: sorteio estratificado de 3 normas TRATADAS +
3 CONTROLE, sem viés de frequência, com seed registrada.

Estratificação (duas dimensões, ordem do brief):
  - cobertura no benchmark (nº de base_qa_id da NR no benchmark_mobile);
  - tamanho do doc (chars do texto limpo em nrs.parquet).

Desenho: cada NR elegível recebe um estrato (cobertura_tercil × tamanho_tercil).
Para a comparação tratada-vs-controle ser justa, cada norma TRATADA é pareada
com uma CONTROLE do MESMO estrato (sorteio dentro do estrato, seed fixa).

Elegibilidade: NR presente no nrs.parquet (texto limpo disponível p/ treino) e
com >= 6 base_qa_id no benchmark (senão o recorte de avaliação fica sem poder
estatístico; 6 base_qa × 4 vocab = 24 queries por norma no mínimo).

Saída: data/norm_selection.json (auditável, com estratos e seed).

Uso:
    classifier/.venv/bin/python bench/lfm3_f2/select_norms.py
"""
from __future__ import annotations

import collections
import glob
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
NRS_PARQUET = Path("/home/rios/projetos/cemig-mobile-llm/firstmate/data/nrs_hf/nrs.parquet")

SEED = 20251028  # seed registrada do sorteio (data da fase F2)
MIN_BASE_QA = 6


def find_benchmark_parquet() -> str:
    pats = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                         "datasets--UFGCEMIGONA--benchmark_mobile/snapshots/*/"
                         "normas_regulamentadoras/train-00000-of-00001.parquet"))
    assert pats, "benchmark_mobile parquet não encontrado no cache HF"
    return pats[0]


def nr_of(source_file: str) -> int | None:
    m = re.search(r"nr[-_ ]?(\d+)", source_file.lower())
    return int(m.group(1)) if m else None


def tercile(value: float, sorted_values: list[float]) -> int:
    """Tercil 0/1/2 de value dentro da distribuição."""
    n = len(sorted_values)
    rank = sum(1 for v in sorted_values if v < value)
    return min(2, rank * 3 // n)


def main() -> None:
    bench = pq.read_table(find_benchmark_parquet()).to_pylist()
    base_qa_per_nr: dict[int, set] = collections.defaultdict(set)
    for r in bench:
        nr = nr_of(r["source_file"])
        if nr is not None:
            base_qa_per_nr[nr].add(r["base_qa_id"])

    nrs = pq.read_table(str(NRS_PARQUET)).to_pylist()
    doc_size = {int(r["numero"]): len(r["conteudo"]) for r in nrs}

    eligible = sorted(nr for nr in base_qa_per_nr
                      if nr in doc_size and len(base_qa_per_nr[nr]) >= MIN_BASE_QA)

    covs = sorted(len(base_qa_per_nr[nr]) for nr in eligible)
    sizes = sorted(doc_size[nr] for nr in eligible)

    strata: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for nr in eligible:
        s = (tercile(len(base_qa_per_nr[nr]), covs), tercile(doc_size[nr], sizes))
        strata[s].append(nr)

    # Sorteio: estratos com >= 2 NRs permitem par tratada+controle do MESMO
    # estrato. Sorteamos 3 estratos distintos (espalhando cobertura/tamanho)
    # e, dentro de cada um, 1 tratada + 1 controle.
    rng = random.Random(SEED)
    pairable = sorted([s for s, members in strata.items() if len(members) >= 2])
    # prioriza diversidade: 1 estrato de cobertura alta, 1 média, 1 baixa (se existir)
    chosen_strata: list[tuple[int, int]] = []
    for cov_terc in (2, 1, 0):
        cands = [s for s in pairable if s[0] == cov_terc and s not in chosen_strata]
        if cands:
            chosen_strata.append(rng.choice(cands))
    while len(chosen_strata) < 3:
        rest = [s for s in pairable if s not in chosen_strata]
        chosen_strata.append(rng.choice(rest))

    treated, control = [], []
    for s in chosen_strata:
        members = sorted(strata[s])
        pick = rng.sample(members, 2)
        treated.append(pick[0])
        control.append(pick[1])

    out = {
        "seed": SEED,
        "min_base_qa": MIN_BASE_QA,
        "eligible": {str(nr): {"base_qa": len(base_qa_per_nr[nr]),
                               "doc_chars": doc_size[nr],
                               "stratum": [tercile(len(base_qa_per_nr[nr]), covs),
                                           tercile(doc_size[nr], sizes)]}
                     for nr in eligible},
        "chosen_strata": [list(s) for s in chosen_strata],
        "treated": sorted(treated),
        "control": sorted(control),
    }
    out_path = HERE / "data" / "norm_selection.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TRATADAS: {sorted(treated)}  CONTROLE: {sorted(control)}")
    for nr in sorted(treated) + sorted(control):
        tag = "T" if nr in treated else "C"
        print(f"  [{tag}] NR-{nr:02d}: base_qa={len(base_qa_per_nr[nr])}, "
              f"doc={doc_size[nr]} chars, estrato={out['eligible'][str(nr)]['stratum']}")
    print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
