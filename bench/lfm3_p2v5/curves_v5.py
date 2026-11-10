#!/usr/bin/env python3
"""
curves_v5.py — curva de eficiência do v5 (reward, frac_reward_zero_std, tokens) a partir do
log do treino (linhas `[step N] reward=… zero_std=… len=…` do Probe do train_p2_rlvr.py),
comparada com v2 (0,54) / v3 / v4 (0,33) de `bench/lfm3_p1p2/data/efficiency_curves.json`.

Também aplica a regra de parada do capitão: reward ACHATADO por 100+ steps = a média
móvel de 100 steps não sobe mais que `--flat-eps` em relação à janela de 100 anterior.

Uso: python3 bench/lfm3_p2v5/curves_v5.py --log train_v5.log   (log puxado da H100)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
P1P2 = HERE.parent / "lfm3_p1p2" / "data"
STEP_RE = re.compile(r"^\[step (\d+)\] reward=([\d.]+) zero_std=([\d.nan]+) len=(\d+|nan)")


def parse(log: Path):
    rows = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_RE.match(line.strip())
        if m:
            rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return [(s, *rows[s]) for s in sorted(rows)]


def bins(rows, width=50):
    out = []
    for lo in range(1, (rows[-1][0] if rows else 0) + 1, width):
        b = [r for r in rows if lo <= r[0] < lo + width]
        if b:
            out.append({"steps": f"{lo}-{lo + width - 1}",
                        "reward": round(sum(r[1] for r in b) / len(b), 3),
                        "zero_std": round(sum(r[2] for r in b) / len(b), 3),
                        "len": round(sum(r[3] for r in b) / len(b))})
    return out


def flat_check(rows, win=100, eps=0.01):
    """Janela a janela de `win` steps: Δ da média de reward vs a janela anterior."""
    res = []
    for end in range(2 * win, (rows[-1][0] if rows else 0) + 1, win):
        cur = [r[1] for r in rows if end - win < r[0] <= end]
        prev = [r[1] for r in rows if end - 2 * win < r[0] <= end - win]
        if cur and prev:
            d = sum(cur) / len(cur) - sum(prev) / len(prev)
            res.append({"window_end": end, "delta_reward": round(d, 4), "flat": d <= eps})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=str(HERE / "data" / "efficiency_v5.json"))
    ap.add_argument("--flat-eps", type=float, default=0.01)
    args = ap.parse_args()
    rows = parse(Path(args.log))
    (HERE / "data" / "curves").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "curves" / "train_p2_v5.tsv").write_text(
        "".join(f"{s} {r:.3f} {z:.2f} {int(n)}\n" for s, r, z, n in rows))
    ref = json.loads((P1P2 / "efficiency_curves.json").read_text())
    v5 = {"bins": bins(rows), "n_steps": len(rows),
          "zero_std_mean": round(sum(r[2] for r in rows) / max(1, len(rows)), 3),
          "zero_std_mean_1_400": round(sum(r[2] for r in rows if r[0] <= 400) /
                                       max(1, sum(1 for r in rows if r[0] <= 400)), 3),
          "len_mean": round(sum(r[3] for r in rows) / max(1, len(rows)), 1),
          "flat_check": flat_check(rows, eps=args.flat_eps)}
    out = {"v5": v5, "ref_zero_std_mean": {k: ref[k].get("zero_std_mean") for k in ref}}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"v5: {len(rows)} steps, zero_std médio {v5['zero_std_mean']} (1-400: {v5['zero_std_mean_1_400']}),"
          f" ref {out['ref_zero_std_mean']}")
    for b in v5["bins"]:
        print(f"  {b['steps']:>9} reward {b['reward']:.3f} zero_std {b['zero_std']:.2f} len {b['len']}")
    for f in v5["flat_check"]:
        print(f"  janela até {f['window_end']}: Δreward {f['delta_reward']:+.4f} {'ACHATADO' if f['flat'] else ''}")


if __name__ == "__main__":
    main()
