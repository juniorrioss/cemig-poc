"""
perf.py — Medição de performance desktop via llama-bench (CPU) para SLMs Tier 1.

Mede:
1. Prefill throughput (pp64 e pp512 em tok/s)
2. Decode throughput (tg16 e tg128 em tok/s)
3. Consumo de memória RAM (MiB) e tamanho dos pesos quantizados (Q4_K_M)
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("bench.perf")

DEFAULT_MODELS_DIR = Path.home() / "models-poc"
DEFAULT_LLAMA_DIR = Path.home() / "llama.cpp"

MODELS = [
    ("qwen3.5-0.8b", "qwen3.5-0.8b-q4_k_m.gguf", "Qwen 3.5 0.8B (Gated DeltaNet)"),
    ("qwen3-0.6b", "qwen3-0.6b-q4_k_m.gguf", "Qwen 3 0.6B Instruct (Baseline)"),
    ("gemma-3-1b", "gemma-3-1b-it-q4_k_m.gguf", "Gemma 3 1B IT"),
    ("lfm2-1.2b", "lfm2-1.2b-rag-q4_k_m.gguf", "Liquid LFM2 1.2B RAG"),
    ("lfm2.5-350m", "lfm2.5-350m-q4_k_m.gguf", "Liquid LFM2.5 350M"),
    ("llama-3.2-1b", "llama-3.2-1b-instruct-q4_k_m.gguf", "Llama 3.2 1B Instruct (Controle)"),
]


def run_llama_bench_model(
    model_alias: str,
    filename: str,
    desc: str,
    models_dir: Path,
    llama_dir: Path
) -> Dict[str, Any]:
    """Executa o llama-bench para um modelo específico e extrai as métricas."""
    model_path = models_dir / filename
    bench_bin = llama_dir / "build" / "bin" / "llama-bench"

    if not model_path.exists():
        logger.error("Modelo não encontrado: %s", model_path)
        return {}

    cmd = [
        str(bench_bin),
        "-m", str(model_path),
        "-p", "64,512",
        "-n", "16,128",
        "-r", "1",
        "-o", "json"
    ]

    logger.info("Executando llama-bench para %s (%s)...", model_alias, filename)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    if proc.returncode != 0:
        logger.error("Erro ao executar llama-bench (%s): %s", model_alias, proc.stderr)
        return {}

    try:
        raw_json = json.loads(proc.stdout)
    except Exception as e:
        logger.error("Falha ao interpretar JSON do llama-bench: %s", e)
        return {}

    # Mapeia resultados por tipo de teste
    pp64, pp512, tg16, tg128 = 0.0, 0.0, 0.0, 0.0
    size_bytes = 0
    n_params = 0
    backend = "CPU"
    threads = 4

    for sample in raw_json:
        n_p = sample.get("n_prompt", 0)
        n_g = sample.get("n_gen", 0)
        speed = sample.get("avg_ts", 0.0)
        size_bytes = sample.get("model_size", size_bytes)
        n_params = sample.get("model_n_params", n_params)
        backend = sample.get("backends", backend)
        threads = sample.get("n_threads", threads)

        if n_p == 64 and n_g == 0:
            pp64 = speed
        elif n_p == 512 and n_g == 0:
            pp512 = speed
        elif n_p == 0 and n_g == 16:
            tg16 = speed
        elif n_p == 0 and n_g == 128:
            tg128 = speed

    size_mib = round(size_bytes / (1024 * 1024), 1)
    params_m = round(n_params / 1_000_000, 1)

    return {
        "model": model_alias,
        "filename": filename,
        "description": desc,
        "params_m": params_m,
        "size_mib": size_mib,
        "backend": backend,
        "threads": threads,
        "prefill_pp64_t_s": round(pp64, 1),
        "prefill_pp512_t_s": round(pp512, 1),
        "decode_tg16_t_s": round(tg16, 1),
        "decode_tg128_t_s": round(tg128, 1),
        "total_bench_duration_s": round(dt, 2),
    }


def run_all_perf(
    models_dir: Path = DEFAULT_MODELS_DIR,
    llama_dir: Path = DEFAULT_LLAMA_DIR,
    output_json: Path = Path("bench/results/llama_bench_metrics.json"),
    output_csv: Path = Path("bench/results/llama_bench_metrics.csv")
) -> List[Dict[str, Any]]:
    """Executa o benchmark relativo de performance para todos os modelos."""
    results = []
    for alias, fn, desc in MODELS:
        res = run_llama_bench_model(alias, fn, desc, models_dir, llama_dir)
        if res:
            results.append(res)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 94)
    print("MÉTRICAS DE PERFORMANCE DESKTOP RELATIVA (llama-bench CPU Ryzen 7 9800X3D)")
    print("Nota: Servem de régua relativa entre modelos, NÃO de número absoluto do celular.")
    print("=" * 94)
    header = "{:<14} | {:<8} | {:<9} | {:<12} | {:<12} | {:<12} | {:<12}"
    print(header.format("Modelo", "Params", "RAM/Tamanho", "Prefill pp64", "Prefill pp512", "Decode tg16", "Decode tg128"))
    print("-" * 94)
    for r in results:
        print(header.format(
            r["model"],
            f"{r['params_m']} M",
            f"{r['size_mib']} MiB",
            f"{r['prefill_pp64_t_s']} t/s",
            f"{r['prefill_pp512_t_s']} t/s",
            f"{r['decode_tg16_t_s']} t/s",
            f"{r['decode_tg128_t_s']} t/s",
        ))
    print("=" * 94 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o llama-bench para SLMs Tier 1.")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--llama-dir", type=str, default=str(DEFAULT_LLAMA_DIR))
    parser.add_argument("--output-json", type=str, default="bench/results/llama_bench_metrics.json")
    parser.add_argument("--output-csv", type=str, default="bench/results/llama_bench_metrics.csv")
    args = parser.parse_args()

    run_all_perf(
        models_dir=Path(args.models_dir),
        llama_dir=Path(args.llama_dir),
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv)
    )


if __name__ == "__main__":
    main()
