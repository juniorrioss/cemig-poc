#!/usr/bin/env python3
"""
device_bench.py — PARTE 2: custo REAL no S24+ das configs de janela×trechos.

Mede no aparelho (não estima) para as 3 configs de custo (k2_full=default atual,
k5_full=máx trechos, k5_trim=máx trechos com recorte barato):

  A) llama-bench (-p <ptok> -n <gtok> -t 6 -r 3) no tamanho de prompt REPRESENTATIVO de
     cada config (mediana medida nas 151): prefill tok/s, decode tok/s e RAM pico (VmHWM,
     polling 50ms) — mesma metodologia de bench/device-liquid. TTFT = ptok/prefill.
  B) llama-cli em prompts REAIS (20 perguntas: roteiro-10 + 10 extra fora do holdout) p/
     confirmar TTFT/decode/latência ponta a ponta e o ACERTO DE NORMA (a NR citada bate a
     esperada). Captura a linha '[ Prompt: X t/s | Generation: Y t/s ]' e a resposta.

O que o capitão valoriza: TTFT e decode > leitura (~4-7 tok/s) importam mais que o total;
prompt maior infla o TTFT (hoje ~3.9s); se 5 trechos levarem o TTFT a ~8s é regressão de
experiência mesmo com qualidade igual. Este script quantifica isso.

Higiene: não-interativo, timeout por chamada, checkpoint incremental, procedência.
Comentários PT-BR, código em inglês. Procedência: task poc-ctx-topk.

Uso:
  PATH=~/android-sdk/platform-tools:$PATH \
  ../../classifier/.venv/bin/python device_bench.py --serial 192.168.0.6:41073 --all
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
REMOTE_POC = "/data/local/tmp/poc"
REMOTE_CTX = f"{REMOTE_POC}/ctxtopk"
MODEL = "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf"
THREADS = 6

# Prompt-token representativo (mediana medida nas 151, gen_device_prompts) por config de custo.
CONFIG_PTOK = {"k2_full": 1419, "k5_full": 3066, "k5_trim": 1338}
GEN_TOK = 48  # gen médio medido (respostas concisas ~45-48 tok)


def adb(serial: str, args: List[str], timeout: int = 120) -> str:
    # errors='replace': o stream do llama-cli pode cortar um byte UTF-8 no meio (buffer),
    # o que quebrava a decodificação. Substituímos o byte inválido em vez de abortar.
    cmd = ["adb", "-s", serial] + args
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    dec = lambda b: b.decode("utf-8", errors="replace") if b else ""
    return dec(r.stdout) + dec(r.stderr)


def adb_shell(serial: str, script: str, timeout: int = 600) -> str:
    return adb(serial, ["shell", script], timeout=timeout)


def run_llama_bench(serial: str, ptok: int, gtok: int, reps: int = 3) -> Dict[str, Any]:
    """llama-bench no tamanho de prompt da config + captura de RAM pico (VmHWM)."""
    script = f"""
cd {REMOTE_POC}
JSON={REMOTE_CTX}/bench_{ptok}.json
RAM={REMOTE_CTX}/ram_{ptok}.txt
rm -f "$JSON" "$RAM"
./llama-bench -m {MODEL} -p {ptok} -n {gtok} -r {reps} -t {THREADS} -o json > "$JSON" 2>/dev/null &
PID=$!
MAX=0
while [ -d "/proc/$PID" ]; do
  H=$(cat /proc/$PID/status 2>/dev/null | grep VmHWM | awk '{{print $2}}')
  if [ -n "$H" ] && [ "$H" -gt "$MAX" ] 2>/dev/null; then MAX=$H; fi
  sleep 0.05
done
wait $PID
echo "PEAK_RAM_KB=$MAX" > "$RAM"
cat "$JSON"
echo "---RAMSEP---"
cat "$RAM"
"""
    out = adb_shell(serial, script, timeout=900)
    json_part = out.split("---RAMSEP---")[0]
    ram_part = out.split("---RAMSEP---")[-1] if "---RAMSEP---" in out else ""
    peak_kb = 0
    m = re.search(r"PEAK_RAM_KB=(\d+)", ram_part)
    if m:
        peak_kb = int(m.group(1))
    pp = tg = 0.0
    pp_std = tg_std = 0.0
    try:
        data = json.loads(json_part[json_part.index("["):json_part.rindex("]") + 1])
        for e in data:
            if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0:
                pp, pp_std = e.get("avg_ts", 0.0), e.get("stddev_ts", 0.0)
            elif e.get("n_gen", 0) > 0 and e.get("n_prompt", 0) == 0:
                tg, tg_std = e.get("avg_ts", 0.0), e.get("stddev_ts", 0.0)
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] parse llama-bench falhou p/ ptok={ptok}: {e}")
    ttft = ptok / pp if pp > 0 else 0.0
    decode = gtok / tg if tg > 0 else 0.0
    return {
        "prompt_tokens": ptok, "gen_tokens": gtok,
        "prefill_tok_s": round(pp, 2), "prefill_std": round(pp_std, 2),
        "decode_tok_s": round(tg, 2), "decode_std": round(tg_std, 2),
        "ttft_s": round(ttft, 3), "decode_s": round(decode, 3),
        "total_s": round(ttft + decode, 3), "peak_ram_mib": round(peak_kb / 1024.0, 1),
    }


NR_RE = re.compile(r"nr[-\s]?(\d{1,2})", re.IGNORECASE)


def run_real_prompt(serial: str, cfg: str, qid: str, expected_doc: str, n_predict: int = 220) -> Dict[str, Any]:
    """llama-cli em prompt real; captura t/s, resposta e acerto de norma (NR citada)."""
    sys_r = f"{REMOTE_CTX}/{cfg}/{qid}.sys.txt"
    usr_r = f"{REMOTE_CTX}/{cfg}/{qid}.usr.txt"
    script = (f"cd {REMOTE_POC} && ./llama-cli -m {MODEL} -sysf {sys_r} -f {usr_r} -st "
              f"-c 4096 -t {THREADS} -n {n_predict} --temp 0.1 --top-k 50 --repeat-penalty 1.05")
    t0 = time.perf_counter()
    out = adb_shell(serial, script, timeout=300)
    wall = time.perf_counter() - t0
    # Extrai a linha de timing.
    pp = gen = 0.0
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", out)
    if m:
        pp, gen = float(m.group(1)), float(m.group(2))
    # Extrai a resposta: o llama-cli ecoa o prompt do usuário (que termina com a pergunta
    # do eletricista) e depois imprime a GERAÇÃO, seguida de uma linha em branco e "[ Prompt:".
    # A resposta é o último parágrafo antes do timing, após o eco da pergunta.
    answer = ""
    if "[ Prompt:" in out:
        pre = out[:out.rfind("[ Prompt:")].rstrip()
        marker = pre.rfind("Pergunta do eletricista:")
        seg = pre[marker:] if marker >= 0 else pre
        # A geração é o texto após a ÚLTIMA linha em branco do segmento (o eco termina com a
        # pergunta e "(truncated)"/"?" seguido de \n; a geração vem depois, separada por \n).
        # Robusto: pega tudo após a última linha que contém "... (truncated)" OU o último "?".
        lines = [l for l in seg.split("\n")]
        cut = 0
        for i, ln in enumerate(lines):
            if "(truncated)" in ln or ln.strip().endswith("?"):
                cut = i + 1
        gen_lines = [l.strip() for l in lines[cut:] if l.strip()]
        answer = " ".join(gen_lines)
    # Acerto de norma: alguma NR citada na RESPOSTA == expected (só o segmento pós-eco).
    scan = answer if answer else out
    cited = {f"nr-{mm.group(1).zfill(2)}" for mm in NR_RE.finditer(scan)}
    exp_norm = "nr-" + expected_doc.split("-")[-1].zfill(2)
    norm_ok = exp_norm in cited
    return {
        "cfg": cfg, "qid": qid, "expected_doc": expected_doc,
        "prompt_tok_s": pp, "gen_tok_s": gen, "wall_s": round(wall, 2),
        "cited_nrs": sorted(cited), "norm_ok": norm_ok, "answer": answer[:400],
    }


def push_prompts(serial: str) -> None:
    """Envia os prompts reais gerados p/ o aparelho."""
    local = _HERE / "data" / "device_prompts"
    adb_shell(serial, f"mkdir -p {REMOTE_CTX}")
    for cfg in ("k2_full", "k5_full", "k5_trim"):
        adb_shell(serial, f"mkdir -p {REMOTE_CTX}/{cfg}")
        # tar para poucos comandos adb push (mais rápido).
        subprocess.run(["adb", "-s", serial, "push", str(local / cfg) + "/.", f"{REMOTE_CTX}/{cfg}/"],
                       capture_output=True, text=True, timeout=300)
    print(f"Prompts enviados para {REMOTE_CTX}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--all", action="store_true", help="bench sintético + prompts reais")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--real-only", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=str(_HERE / "results" / "device_cost.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}
    if out_path.exists():
        report = json.loads(out_path.read_text(encoding="utf-8"))

    report.setdefault("device", "Samsung Galaxy S24+ (SM-S926B, Exynos 2400)")
    report.setdefault("model", MODEL)
    report.setdefault("threads", THREADS)
    report.setdefault("generated_at", time.strftime("%Y-%m-%d %H:%M:%S"))

    do_syn = args.all or args.synthetic_only
    do_real = args.all or args.real_only

    if do_syn:
        report.setdefault("synthetic", {})
        for cfg, ptok in CONFIG_PTOK.items():
            print(f"=== llama-bench {cfg} (ptok={ptok}) ===")
            report["synthetic"][cfg] = run_llama_bench(args.serial, ptok, GEN_TOK, reps=args.reps)
            print("  ", report["synthetic"][cfg])
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if do_real:
        push_prompts(args.serial)
        manifest = json.loads((_HERE / "data" / "device_prompts" / "manifest.json").read_text(encoding="utf-8"))
        report.setdefault("real", {})
        for cfg in ("k2_full", "k5_full", "k5_trim"):
            items = manifest["configs"][cfg]["items"]
            done = {r["qid"] for r in report["real"].get(cfg, [])}
            recs = report["real"].get(cfg, [])
            print(f"=== prompts reais {cfg} ({len(items)} perguntas) ===")
            for qid, meta in items.items():
                if qid in done:
                    continue
                r = run_real_prompt(args.serial, cfg, qid, meta["expected_doc"])
                recs.append(r)
                report["real"][cfg] = recs
                out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  {qid}: pp={r['prompt_tok_s']:.0f} gen={r['gen_tok_s']:.1f} "
                      f"norm_ok={r['norm_ok']} cited={r['cited_nrs']}")

    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
