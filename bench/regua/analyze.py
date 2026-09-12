"""
analyze.py — Consolidação da régua honesta vs. régua velha (dominada por citação).

Responde as 4 perguntas de entrega do capitão:
 (a) quanto da "qualidade" histórica era TAUTOLOGIA;
 (b) o ranking muda com a régua honesta? quem sobe, quem cai;
 (c) SFT/DPO regrediram na métrica ÚTIL, ou a régua velha (citação) é que mentiu;
 (d) qual o gargalo REAL agora.

Cruza:
 - rejudge_table.json (régua honesta: aprovação por cobertura de fatos + antitautologia)
 - old gate (citação-dominado) das fontes com metrics salvos:
     finetune2/data/panel_*.json         (holdout.items[].pass_gate, metrics)
     bench/judge151/data/judge_evaluations*.json  (modes.*.summary.gate_pass_rate_pct)
     bench/sintese2/data/judge_evaluations.json   (idem)

Não gera nada novo. Saída: data/consolidation.json + prints legíveis.
Comentários em português; identificadores em inglês.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent / "data"
REPO_ROOT = Path(__file__).resolve().parents[2]


def old_gate_pass(m: Dict[str, Any]) -> bool:
    """Gate histórico (dominado por citação): idêntico a finetune2/eval_panel.py."""
    g = (m["acerto_factual"] * 0.35 + m["fidelidade_contexto"] * 0.30
         + m["citacao_fonte"] * 0.20 + m["qualidade_pt"] * 0.15)
    return (m["acerto_factual"] >= 3 and m["fidelidade_contexto"] >= 3
            and m["citacao_fonte"] >= 3 and g >= 3.5)


def old_gate_no_citation(m: Dict[str, Any]) -> bool:
    """Contrafactual: mesmo gate SEM a trava de citação (para isolar o efeito)."""
    g = (m["acerto_factual"] * 0.35 + m["fidelidade_contexto"] * 0.30
         + m["citacao_fonte"] * 0.20 + m["qualidade_pt"] * 0.15)
    return (m["acerto_factual"] >= 3 and m["fidelidade_contexto"] >= 3 and g >= 3.5)


def collect_old_gate() -> Dict[str, Dict[str, Any]]:
    """Coleta old-gate% por candidato a partir das fontes com metrics."""
    out: Dict[str, Dict[str, Any]] = {}

    # finetune2 panels (per-item metrics + pass_gate)
    for p in sorted(REPO_ROOT.glob("finetune2/data/panel_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        hold = d.get("holdout")
        if not isinstance(hold, dict):
            continue
        items = hold.get("items", [])
        variant = d.get("metadata", {}).get("variant", p.stem.replace("panel_", ""))
        label = f"finetune2/{variant}"
        n = len(items)
        n_old = sum(1 for i in items if i.get("pass_gate"))
        # recomputa o contrafactual sem citação, item a item
        n_noc = sum(1 for i in items if "metrics" in i and old_gate_no_citation(i["metrics"]))
        cit = [i["metrics"]["citacao_fonte"] for i in items if "metrics" in i]
        out[label] = {
            "old_gate_pct": round(100 * n_old / max(1, n), 1),
            "old_gate_nocit_pct": round(100 * n_noc / max(1, n), 1),
            "citacao_avg": round(sum(cit) / max(1, len(cit)), 2),
        }

    # judge_evaluations dos dois shootouts (summary já traz gate_pass_rate_pct)
    for p, prefix in [
        (REPO_ROOT / "bench/judge151/data/judge_evaluations.json", "judge151"),
        (REPO_ROOT / "bench/sintese2/data/judge_evaluations.json", "sintese2"),
    ]:
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for model_key, mdata in d.get("models", {}).items():
            for mode, mode_d in mdata.get("modes", {}).items():
                items = mode_d.get("items", [])
                summ = mode_d.get("summary", {})
                # rótulo compatível com o loaders.py da régua
                if prefix == "sintese2":
                    label = f"sintese2/{model_key.replace('__', '_')}"
                else:
                    label = f"judge151/{model_key}"
                n = len(items)
                n_noc = sum(1 for i in items if "metrics" in i
                            and old_gate_no_citation(i["metrics"]))
                cit = [i["metrics"]["citacao_fonte"] for i in items if "metrics" in i]
                out[label] = {
                    "old_gate_pct": summ.get("gate_pass_rate_pct"),
                    "old_gate_nocit_pct": round(100 * n_noc / max(1, n), 1) if n else None,
                    "citacao_avg": round(sum(cit) / max(1, len(cit)), 2) if cit else None,
                }
    return out


def main() -> None:
    rj = json.loads((DATA_DIR / "rejudge_table.json").read_text(encoding="utf-8"))
    new_table = {m["candidate"]: m for m in rj["table"]}
    old = collect_old_gate()

    # --- (a) Tautologia histórica: média sobre todos os candidatos ---
    taut_vals = [m["tautology_pct"] for m in new_table.values()]
    taut_mean = sum(taut_vals) / max(1, len(taut_vals))
    taut_max = max(new_table.values(), key=lambda m: m["tautology_pct"])
    hollow = compute_hollow_passes()

    # --- Junção old x new onde houver old-gate ---
    joined = []
    for cand, m in new_table.items():
        o = old.get(cand)
        joined.append({
            "candidate": cand, "family": m["family"], "n": m["n"],
            "new_approval_pct": m["approval_pct"],
            "new_coverage": m["mean_coverage"],
            "tautology_pct": m["tautology_pct"],
            "hallucination_pct": m["hallucination_pct"],
            "citation_pct": m["citation_pct"],
            "retrieval_hit_pct": m["retrieval_hit_pct"],
            "old_gate_pct": (o or {}).get("old_gate_pct"),
            "old_gate_nocit_pct": (o or {}).get("old_gate_nocit_pct"),
            "citacao_avg": (o or {}).get("citacao_avg"),
        })

    # --- (b) Mudança de ranking: comparar por candidatos com old-gate ---
    have_old = [j for j in joined if j["old_gate_pct"] is not None]
    old_rank = sorted(have_old, key=lambda j: -j["old_gate_pct"])
    new_rank = sorted(have_old, key=lambda j: -j["new_approval_pct"])
    old_pos = {j["candidate"]: i for i, j in enumerate(old_rank)}
    new_pos = {j["candidate"]: i for i, j in enumerate(new_rank)}
    movers = []
    for j in have_old:
        c = j["candidate"]
        delta = old_pos[c] - new_pos[c]  # +sobe, -cai
        movers.append({"candidate": c, "old_rank": old_pos[c] + 1,
                       "new_rank": new_pos[c] + 1, "rank_delta": delta,
                       "old_gate_pct": j["old_gate_pct"],
                       "new_approval_pct": j["new_approval_pct"]})
    movers.sort(key=lambda x: -x["rank_delta"])

    # --- (c) SFT/DPO: útil vs. citação ---
    train = {c: new_table[c] for c in new_table if c.startswith("finetune2/")}
    train_view = {}
    for c, m in train.items():
        o = old.get(c, {})
        train_view[c] = {
            "new_approval_pct": m["approval_pct"], "new_coverage": m["mean_coverage"],
            "old_gate_pct": o.get("old_gate_pct"),
            "old_gate_nocit_pct": o.get("old_gate_nocit_pct"),
            "citacao_avg": o.get("citacao_avg"),
            "hallucination_pct": m["hallucination_pct"],
        }

    # --- (d) Gargalo real: decompor por-item onde retrieval acertou vs errou ---
    bottleneck = compute_bottleneck()
    per_model_conv = compute_per_model_conversion([
        "sintese2/lfm2.6b_noth_baseline", "finetune2/base", "judge151/v3_lfm2.6b_noth",
        "finetune2/dpo_f16", "finetune2/sft",
        "sintese2/lfm1.2b_baseline", "judge151/v3_lfm1.2b", "judge151/lfm1.2b",
        "sintese2/minicpm2b_full"])

    payload = {
        "provenance": rj["provenance"], "threshold": rj["threshold"],
        "a_tautology": {
            "mean_tautology_pct_all_candidates": round(taut_mean, 2),
            "max": {"candidate": taut_max["candidate"],
                    "tautology_pct": taut_max["tautology_pct"]},
            "hollow_passes": hollow,
            "note": ("Tautologia PURA (só reordenar a pergunta) é rara; o inflacionamento "
                     "histórico veio da CITAÇÃO no gate, não de tautologia literal. "
                     "'hollow_passes' mede as aprovações antigas que a régua honesta reprova."),
        },
        "b_ranking": {"movers_top": movers[:10], "movers_bottom": movers[-10:]},
        "c_training": train_view,
        "d_bottleneck": bottleneck,
        "d_per_model_conversion": per_model_conv,
        "joined_table": sorted(joined, key=lambda j: -j["new_approval_pct"]),
    }
    (DATA_DIR / "consolidation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # ------- prints -------
    print("=" * 78)
    print("(a) TAUTOLOGIA histórica / QUALIDADE OCA")
    print(f"    média tautologia pura (todos candidatos): {taut_mean:.2f}%")
    print(f"    pico: {taut_max['candidate']} = {taut_max['tautology_pct']}%")
    print(f"    aprovações antigas (old-gate) que a régua honesta REPROVA: "
          f"{hollow['hollow']} / {hollow['old_passes']} = {hollow['hollow_pct']}%")
    print(f"    dessas 'aprovações ocas', % que tinham citação presente: {hollow['hollow_with_citation_pct']}%")
    print()
    print("(b) MUDANÇA DE RANKING (só candidatos com old-gate), top subidas:")
    for x in movers[:6]:
        print(f"    +{x['rank_delta']:>2}  {x['candidate']:<38} old#{x['old_rank']}->new#{x['new_rank']}"
              f"  oldgate={x['old_gate_pct']}% newaprov={x['new_approval_pct']}%")
    print("    maiores quedas:")
    for x in movers[-6:]:
        print(f"    {x['rank_delta']:>3}  {x['candidate']:<38} old#{x['old_rank']}->new#{x['new_rank']}"
              f"  oldgate={x['old_gate_pct']}% newaprov={x['new_approval_pct']}%")
    print()
    print("(c) TREINO SFT/DPO — útil (cobertura de fatos) vs. citação:")
    for c in ["finetune2/base_f16", "finetune2/base", "finetune2/sft",
              "finetune2/dpo_f16", "finetune2/dpo"]:
        if c in train_view:
            t = train_view[c]
            print(f"    {c:<22} new_aprov={t['new_approval_pct']:>5}% cov={t['new_coverage']:.3f}"
                  f" | old_gate={t['old_gate_pct']}% old_gate_SEM_cit={t['old_gate_nocit_pct']}%"
                  f" cit_avg={t['citacao_avg']}")
    print()
    print("(d) GARGALO REAL:")
    print(f"    respostas COM chunk-ouro no top: aprovação nova = {bottleneck['with_hit']['approval_pct']}%"
          f" (n={bottleneck['with_hit']['n']})")
    print(f"    respostas SEM chunk-ouro:        aprovação nova = {bottleneck['without_hit']['approval_pct']}%"
          f" (n={bottleneck['without_hit']['n']})")
    print(f"    reprovações por: cobertura-baixa={bottleneck['fail_reasons']['low_coverage_pct']}%"
          f" alucinacao={bottleneck['fail_reasons']['hallucination_pct']}%"
          f" tautologia={bottleneck['fail_reasons']['tautology_pct']}%"
          f" vazio={bottleneck['fail_reasons']['empty_pct']}%")
    print("    conversão chunk-certo -> aprovada (LM dado o chunk):")
    for c, v in per_model_conv.items():
        print(f"      {c:<34} COM-chunk {v['with_hit_approval_pct']}%  SEM-chunk {v['without_hit_approval_pct']}%")
    print(f"\n[ok] consolidation -> {DATA_DIR / 'consolidation.json'}")


def compute_bottleneck() -> Dict[str, Any]:
    """Decompõe as reprovações usando rejudge_items.jsonl (todos os itens de todos cands)."""
    items = [json.loads(l) for l in
             (DATA_DIR / "rejudge_items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    # Cruza com retrieval_hit por (candidate,item) via loaders para separar hit/miss.
    from loaders import discover_candidates
    hit_map: Dict[str, Optional[bool]] = {}
    for c in discover_candidates():
        for iid, it in c["items"].items():
            hit_map[f"{c['candidate']}|{iid}"] = it.get("retrieval_hit")

    with_hit = {"n": 0, "approved": 0}
    without_hit = {"n": 0, "approved": 0}
    fails = {"low_coverage": 0, "hallucination": 0, "tautology": 0, "empty": 0, "total": 0}

    for it in items:
        key = f"{it['candidate']}|{it['item_id']}"
        h = hit_map.get(key)
        if h is True:
            with_hit["n"] += 1
            if it["approved"]:
                with_hit["approved"] += 1
        elif h is False:
            without_hit["n"] += 1
            if it["approved"]:
                without_hit["approved"] += 1
        if not it["approved"]:
            fails["total"] += 1
            if it["empty"]:
                fails["empty"] += 1
            elif it["tautology"]:
                fails["tautology"] += 1
            elif it["hallucination"]:
                fails["hallucination"] += 1
            else:
                fails["low_coverage"] += 1

    ft = max(1, fails["total"])
    return {
        "with_hit": {"n": with_hit["n"],
                     "approval_pct": round(100 * with_hit["approved"] / max(1, with_hit["n"]), 1)},
        "without_hit": {"n": without_hit["n"],
                        "approval_pct": round(100 * without_hit["approved"] / max(1, without_hit["n"]), 1)},
        "fail_reasons": {
            "low_coverage_pct": round(100 * fails["low_coverage"] / ft, 1),
            "hallucination_pct": round(100 * fails["hallucination"] / ft, 1),
            "tautology_pct": round(100 * fails["tautology"] / ft, 1),
            "empty_pct": round(100 * fails["empty"] / ft, 1),
            "n_fails": fails["total"],
        },
    }


def compute_hollow_passes() -> Dict[str, Any]:
    """Aprovações do gate ANTIGO (citação-dominado) que a régua honesta REPROVA.
    Mede quanto da 'qualidade' histórica era oca (passava sem conter a informação)."""
    new = {}
    for it in (json.loads(l) for l in
               (DATA_DIR / "rejudge_items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()):
        new[(it["candidate"], it["item_id"])] = it

    old: Dict[str, Dict[str, bool]] = {}
    # finetune2 panels (pass_gate por item)
    for p in sorted(REPO_ROOT.glob("finetune2/data/panel_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        hold = d.get("holdout")
        if not isinstance(hold, dict):
            continue
        variant = d.get("metadata", {}).get("variant", p.stem.replace("panel_", ""))
        old[f"finetune2/{variant}"] = {i["item_id"]: bool(i.get("pass_gate"))
                                       for i in hold.get("items", [])}
    # judge_evaluations (recomputa old_gate_pass por item)
    for p, prefix in [
        (REPO_ROOT / "bench/judge151/data/judge_evaluations.json", "judge151"),
        (REPO_ROOT / "bench/sintese2/data/judge_evaluations.json", "sintese2"),
    ]:
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for mk, md in d.get("models", {}).items():
            for mode, mdd in md.get("modes", {}).items():
                label = (f"sintese2/{mk.replace('__', '_')}" if prefix == "sintese2"
                         else f"judge151/{mk}")
                old[label] = {i["item_id"]: old_gate_pass(i["metrics"])
                              for i in mdd.get("items", []) if "metrics" in i}

    tot_old = tot_hollow = tot_hollow_cit = 0
    per = {}
    for cand, om in old.items():
        oldP = hollow = hollow_cit = 0
        for iid, op in om.items():
            ni = new.get((cand, iid))
            if ni is None:
                continue
            if op:
                oldP += 1
                if not ni["approved"]:
                    hollow += 1
                    if ni["citation"]:
                        hollow_cit += 1
        if oldP:
            per[cand] = {"old_passes": oldP, "hollow": hollow,
                         "hollow_pct": round(100 * hollow / oldP, 1)}
            tot_old += oldP
            tot_hollow += hollow
            tot_hollow_cit += hollow_cit
    return {
        "old_passes": tot_old, "hollow": tot_hollow,
        "hollow_pct": round(100 * tot_hollow / max(1, tot_old), 1),
        "hollow_with_citation_pct": round(100 * tot_hollow_cit / max(1, tot_hollow), 1),
        "per_candidate": per,
    }


def compute_per_model_conversion(cands: List[str]) -> Dict[str, Any]:
    """Para cada candidato: aprovação nova SEPARADA por retrieval_hit True/False.
    Isola o LM (dado o chunk certo) do retrieval."""
    items = [json.loads(l) for l in
             (DATA_DIR / "rejudge_items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cand: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        by_cand.setdefault(it["candidate"], []).append(it)

    from loaders import discover_candidates
    hit_map: Dict[str, Optional[bool]] = {}
    for c in discover_candidates():
        for iid, it in c["items"].items():
            hit_map[f"{c['candidate']}|{iid}"] = it.get("retrieval_hit")

    out: Dict[str, Any] = {}
    for cand in cands:
        wh = whn = nh = nhn = 0
        for it in by_cand.get(cand, []):
            h = hit_map.get(f"{cand}|{it['item_id']}")
            if h is True:
                whn += 1
                wh += 1 if it["approved"] else 0
            elif h is False:
                nhn += 1
                nh += 1 if it["approved"] else 0
        if whn == 0 and nhn == 0:
            continue
        out[cand] = {
            "with_hit_approved": wh, "with_hit_n": whn,
            "with_hit_approval_pct": round(100 * wh / max(1, whn), 1),
            "without_hit_approved": nh, "without_hit_n": nhn,
            "without_hit_approval_pct": round(100 * nh / max(1, nhn), 1),
        }
    return out


if __name__ == "__main__":
    main()
