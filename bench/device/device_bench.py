#!/usr/bin/env python3
"""
bench/device/device_bench.py — Execução de benchmarks de SLMs Tier 1 no Galaxy S24+ via ADB.

Mede:
1. Throughput de prefill (pp1500) e decode (tg128) via llama-bench (3 repetições)
2. Pico de memória RAM (VmHWM em /proc/<pid>/status)
3. Tempo de carregamento frio (cold load time)
4. Variação térmica (AP, Bateria, Superfície via dumpsys thermalservice)
5. Sanity check de inferência em PT-BR via llama-cli
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("device_bench")

MODELS_SPEC = [
    {
        "alias": "qwen3-0.6b",
        "filename": "qwen3-0.6b-q4_k_m.gguf",
        "name": "Qwen 3 0.6B Instruct (Baseline)",
        "params": "0.6B",
    },
    {
        "alias": "qwen3.5-0.8b",
        "filename": "qwen3.5-0.8b-q4_k_m.gguf",
        "name": "Qwen 3.5 0.8B (Gated DeltaNet)",
        "params": "0.8B",
    },
    {
        "alias": "gemma-3-1b-it",
        "filename": "gemma-3-1b-it-q4_k_m.gguf",
        "name": "Gemma 3 1B IT",
        "params": "1.0B",
    },
    {
        "alias": "lfm2-1.2b-rag",
        "filename": "lfm2-1.2b-rag-q4_k_m.gguf",
        "name": "Liquid LFM2 1.2B RAG",
        "params": "1.2B",
    },
    {
        "alias": "lfm2.5-350m",
        "filename": "lfm2.5-350m-q4_k_m.gguf",
        "name": "Liquid LFM2.5 350M",
        "params": "0.35B",
    },
    {
        "alias": "llama-3.2-1b",
        "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "name": "Meta Llama 3.2 1B Instruct (Controle)",
        "params": "1.2B",
    },
]

FIXED_PT_PROMPT = "Qual é o equipamento de proteção individual essencial para trabalho em altura segundo a NR-35? Responda de forma sucinta em uma frase."
REMOTE_POC_DIR = "/data/local/tmp/poc"


class AdbDevice:
    """Gerencia comandos ADB com reconexão automática e tolerância a falhas."""

    def __init__(self, serial: str, adb_bin: str = "adb"):
        self.serial = serial
        self.adb_bin = adb_bin

    def _cmd(self, args: List[str]) -> List[str]:
        return [self.adb_bin, "-s", self.serial] + args

    def ensure_connected(self, max_retries: int = 3) -> bool:
        """Verifica se o dispositivo está conectado e tenta reconectar se necessário."""
        for attempt in range(1, max_retries + 1):
            res = subprocess.run([self.adb_bin, "devices"], capture_output=True, text=True, errors="replace")
            if self.serial in res.stdout and "\tdevice" in res.stdout:
                return True
            logger.warning("Dispositivo %s não conectado. Tentando 'adb connect' (tentativa %d/%d)...",
                           self.serial, attempt, max_retries)
            subprocess.run([self.adb_bin, "connect", self.serial], capture_output=True, text=True, errors="replace", timeout=10)
            time.sleep(2)
        return False

    def shell(self, cmd: str, timeout: int = 300) -> Tuple[int, str, str]:
        """Executa um comando no shell do dispositivo."""
        self.ensure_connected()
        full_cmd = self._cmd(["shell", cmd])
        try:
            res = subprocess.run(full_cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            logger.error("Timeout de %ds executando shell: %s", timeout, cmd[:80])
            return -1, "", f"Timeout after {timeout}s"

    def push(self, local_path: Path, remote_path: str, timeout: int = 300) -> bool:
        """Transfere arquivo para o dispositivo com medição de velocidade."""
        self.ensure_connected()
        logger.info("Enviando %s para %s...", local_path.name, remote_path)
        t0 = time.time()
        full_cmd = self._cmd(["push", str(local_path), remote_path])
        res = subprocess.run(full_cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
        if res.returncode != 0:
            logger.error("Falha no adb push: %s", res.stderr)
            return False
        elapsed = time.time() - t0
        size_mb = local_path.stat().st_size / (1024 * 1024)
        speed = size_mb / elapsed if elapsed > 0 else 0
        logger.info("Upload concluído: %.1f MB em %.2fs (%.1f MB/s)", size_mb, elapsed, speed)
        return True

    def remove_remote_file(self, remote_path: str) -> None:
        """Remove arquivo remoto no dispositivo."""
        self.shell(f"rm -f {remote_path}")

    def wake_up(self) -> str:
        """Acorda a tela do dispositivo e retorna o estado de vigília."""
        self.shell("input keyevent KEYCODE_WAKEUP")
        time.sleep(0.3)
        code, out, _ = self.shell("dumpsys power | grep 'mWakefulness='")
        state = "unknown"
        if "mWakefulness=" in out:
            state = out.strip().split("=")[-1]
        return state

    def get_temperatures(self) -> Dict[str, float]:
        """Lê temperaturas do HAL (AP, BAT, SKIN) via dumpsys thermalservice."""
        code, out, _ = self.shell("dumpsys thermalservice")
        temps: Dict[str, float] = {}
        if code != 0 or not out:
            return temps

        in_hal = False
        for line in out.splitlines():
            if "Current temperatures from HAL:" in line:
                in_hal = True
                continue
            if in_hal:
                if "Current cooling devices" in line or "Temperature static" in line:
                    break
                m = re.search(r"mValue=([0-9.]+).*?mName=(\w+)", line)
                if m:
                    val_str, name = m.group(1), m.group(2)
                    try:
                        temps[name] = float(val_str)
                    except ValueError:
                        pass
        return temps


def run_benchmark_for_model(
    adb: AdbDevice,
    model_info: Dict[str, str],
    models_dir: Path,
    threads: int = 6,
    repetitions: int = 3,
    prompt_tokens: int = 1500,
    gen_tokens: int = 128
) -> Dict[str, Any]:
    """Executa o pipeline completo de medição de performance para um único modelo GGUF."""
    alias = model_info["alias"]
    filename = model_info["filename"]
    local_model_path = models_dir / filename
    remote_model_path = f"{REMOTE_POC_DIR}/{filename}"

    if not local_model_path.exists():
        logger.error("Arquivo do modelo não encontrado: %s", local_model_path)
        return {"error": f"File not found: {local_model_path}"}

    file_size_bytes = local_model_path.stat().st_size
    file_size_mib = file_size_bytes / (1024 * 1024)

    logger.info("=========================================================")
    logger.info("Iniciando benchmark do modelo: %s (%s, %.1f MiB)", alias, model_info["name"], file_size_mib)
    logger.info("=========================================================")

    # 1. Acordar tela e estado térmico inicial
    wake_state = adb.wake_up()
    temps_before = adb.get_temperatures()
    ap_before = temps_before.get("AP", 0.0)
    bat_before = temps_before.get("BAT", 0.0)
    skin_before = temps_before.get("SKIN", 0.0)
    logger.info("Estado da tela: %s | Temp inicial: AP=%.1f°C, BAT=%.1f°C, SKIN=%.1f°C",
                wake_state, ap_before, bat_before, skin_before)

    # 2. Transferir modelo para /data/local/tmp/poc/
    push_ok = adb.push(local_model_path, remote_model_path, timeout=300)
    if not push_ok:
        return {"error": "adb push failed"}

    # 3. Executar llama-bench monitorando pico de RAM (VmHWM) e tempo de carga
    # Script no device executa llama-bench em background e coleta VmHWM a cada 50ms
    bench_remote_script = f"""
MODEL_PATH="{remote_model_path}"
BENCH_BIN="{REMOTE_POC_DIR}/llama-bench"
JSON_OUT="{REMOTE_POC_DIR}/bench.json"
LOG_OUT="{REMOTE_POC_DIR}/bench.log"
RAM_OUT="{REMOTE_POC_DIR}/ram.txt"

rm -f "$JSON_OUT" "$LOG_OUT" "$RAM_OUT"

$BENCH_BIN -m "$MODEL_PATH" -p {prompt_tokens} -n {gen_tokens} -r {repetitions} -t {threads} -v -o json > "$JSON_OUT" 2> "$LOG_OUT" &
BENCH_PID=$!
MAX_HWM=0

while [ -d "/proc/$BENCH_PID" ]; do
    HWM=$(cat "/proc/$BENCH_PID/status" 2>/dev/null | grep VmHWM | awk '{{print $2}}')
    if [ -n "$HWM" ] && [ "$HWM" -gt "$MAX_HWM" ] 2>/dev/null; then
        MAX_HWM=$HWM
    fi
    sleep 0.05
done

wait $BENCH_PID
BENCH_RET=$?
echo "PEAK_RAM_KB=$MAX_HWM" > "$RAM_OUT"
echo "BENCH_EXIT=$BENCH_RET" >> "$RAM_OUT"
"""

    logger.info("Executando llama-bench (-p %d -n %d -r %d -t %d)...",
                prompt_tokens, gen_tokens, repetitions, threads)
    adb.wake_up()
    t_start_bench = time.time()
    code, out, err = adb.shell(bench_remote_script, timeout=600)
    bench_wall_time = time.time() - t_start_bench

    # Ler saídas geradas no device
    _, json_raw, _ = adb.shell(f"cat {REMOTE_POC_DIR}/bench.json")
    _, log_raw, _ = adb.shell(f"cat {REMOTE_POC_DIR}/bench.log")
    _, ram_raw, _ = adb.shell(f"cat {REMOTE_POC_DIR}/ram.txt")

    # Extrair pico de RAM
    peak_ram_kb = 0
    for line in ram_raw.splitlines():
        if line.startswith("PEAK_RAM_KB="):
            try:
                peak_ram_kb = int(line.split("=")[-1].strip())
            except ValueError:
                pass
    peak_ram_mib = peak_ram_kb / 1024.0
    logger.info("Pico de RAM capturado: %.1f MiB (%d kB)", peak_ram_mib, peak_ram_kb)

    # Extrair cold load time do log
    cold_load_s = 0.0
    load_matches = re.findall(r"load time\s*=\s*([0-9.]+)\s*ms", log_raw)
    if load_matches:
        cold_load_s = float(load_matches[0]) / 1000.0
        logger.info("Tempo de carregamento (load time): %.3fs", cold_load_s)

    # Parsear métricas do JSON do llama-bench
    bench_data = []
    try:
        bench_data = json.loads(json_raw)
    except json.JSONDecodeError as e:
        logger.error("Erro ao decodificar JSON do llama-bench: %s\nSaída bruta:\n%s", e, json_raw[:300])

    pp_avg_ts = 0.0
    pp_stddev_ts = 0.0
    pp_samples = []
    tg_avg_ts = 0.0
    tg_stddev_ts = 0.0
    tg_samples = []

    for entry in bench_data:
        n_prompt = entry.get("n_prompt", 0)
        n_gen = entry.get("n_gen", 0)
        avg_ts = entry.get("avg_ts", 0.0)
        std_ts = entry.get("stddev_ts", 0.0)
        samples = entry.get("samples_ts", [])

        if n_prompt > 0 and n_gen == 0:
            pp_avg_ts = avg_ts
            pp_stddev_ts = std_ts
            pp_samples = samples
        elif n_gen > 0 and n_prompt == 0:
            tg_avg_ts = avg_ts
            tg_stddev_ts = std_ts
            tg_samples = samples

    logger.info("Resultados llama-bench: Prefill = %.2f tok/s (±%.2f) | Decode = %.2f tok/s (±%.2f)",
                pp_avg_ts, pp_stddev_ts, tg_avg_ts, tg_stddev_ts)

    # 4. Geração de Sanity via llama-cli com prompt PT fixo curto
    logger.info("Executando sanity check via llama-cli...")
    adb.wake_up()
    cli_cmd = (
        f"{REMOTE_POC_DIR}/llama-cli "
        f"-m '{remote_model_path}' "
        f"-p '{FIXED_PT_PROMPT}' "
        f"-n 64 --temp 0 --no-warmup -t {threads} --single-turn < /dev/null"
    )
    cli_code, cli_stdout, cli_stderr = adb.shell(cli_cmd, timeout=120)

    sanity_output = ""
    cli_prompt_ts = 0.0
    cli_gen_ts = 0.0

    if cli_stdout:
        # Extrair texto entre o prompt e o rodapé [ Prompt: ... ]
        lines = cli_stdout.splitlines()
        capturing = False
        captured_lines = []
        for line in lines:
            if FIXED_PT_PROMPT in line:
                capturing = True
                remainder = line.split(FIXED_PT_PROMPT, 1)[-1].strip()
                if remainder:
                    captured_lines.append(remainder)
                continue
            if capturing:
                if line.strip().startswith("[ Prompt:") or "Exiting..." in line:
                    break
                captured_lines.append(line.strip())
        sanity_output = " ".join([l for l in captured_lines if l]).strip()

        # Extrair throughput reportado pelo llama-cli se presente
        m_perf = re.search(r"\[ Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s \]", cli_stdout)
        if m_perf:
            cli_prompt_ts = float(m_perf.group(1))
            cli_gen_ts = float(m_perf.group(2))

    logger.info("Sanity output (%s): %s", alias, sanity_output[:120])

    # 5. Estado térmico final e delta
    temps_after = adb.get_temperatures()
    ap_after = temps_after.get("AP", 0.0)
    bat_after = temps_after.get("BAT", 0.0)
    skin_after = temps_after.get("SKIN", 0.0)

    delta_ap = round(ap_after - ap_before, 2)
    delta_bat = round(bat_after - bat_before, 2)
    delta_skin = round(skin_after - skin_before, 2)

    logger.info("Temp final: AP=%.1f°C (Δ%+.1f), BAT=%.1f°C (Δ%+.1f), SKIN=%.1f°C (Δ%+.1f)",
                ap_after, delta_ap, bat_after, delta_bat, skin_after, delta_skin)

    # 6. Limpeza imediata do arquivo GGUF no dispositivo (espaço finito)
    logger.info("Removendo modelo do dispositivo: %s...", remote_model_path)
    adb.remove_remote_file(remote_model_path)
    adb.shell(f"rm -f {REMOTE_POC_DIR}/bench.json {REMOTE_POC_DIR}/bench.log {REMOTE_POC_DIR}/ram.txt")

    # 7. Cálculo de latência estimada no S24+ e projeção S21
    # Cenário CEMIG: 1500 tokens de contexto recuperado (prefill) + 100 tokens de síntese de resposta
    t_prefill_s24 = (prompt_tokens / pp_avg_ts) if pp_avg_ts > 0 else 999.0
    t_decode_s24 = (100 / tg_avg_ts) if tg_avg_ts > 0 else 999.0
    t_total_s24 = t_prefill_s24 + t_decode_s24
    fits_10s_s24 = t_total_s24 <= 10.0

    # Projeção S21 (fator conservador de slowdown 2.25x devido ao Exynos 2100 vs Exynos 2400)
    s21_factor = 2.25
    t_total_s21_proj = t_total_s24 * s21_factor
    fits_10s_s21_proj = t_total_s21_proj <= 10.0

    result = {
        "alias": alias,
        "name": model_info["name"],
        "params": model_info["params"],
        "filename": filename,
        "file_size_bytes": file_size_bytes,
        "file_size_mib": round(file_size_mib, 2),
        "threads": threads,
        "repetitions": repetitions,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "screen_wake_state": wake_state,
        "prefill_tok_per_sec": round(pp_avg_ts, 2),
        "prefill_stddev": round(pp_stddev_ts, 2),
        "prefill_samples": pp_samples,
        "decode_tok_per_sec": round(tg_avg_ts, 2),
        "decode_stddev": round(tg_stddev_ts, 2),
        "decode_samples": tg_samples,
        "peak_ram_mib": round(peak_ram_mib, 2),
        "cold_load_sec": round(cold_load_s, 3),
        "bench_wall_time_sec": round(bench_wall_time, 2),
        "temperatures": {
            "ap_before_c": ap_before,
            "ap_after_c": ap_after,
            "delta_ap_c": delta_ap,
            "bat_before_c": bat_before,
            "bat_after_c": bat_after,
            "delta_bat_c": delta_bat,
            "skin_before_c": skin_before,
            "skin_after_c": skin_after,
            "delta_skin_c": delta_skin,
        },
        "sanity": {
            "prompt": FIXED_PT_PROMPT,
            "response": sanity_output,
            "cli_prompt_ts": cli_prompt_ts,
            "cli_gen_ts": cli_gen_ts,
        },
        "latency_budget": {
            "target_budget_sec": 10.0,
            "s24_total_sec": round(t_total_s24, 2),
            "s24_prefill_sec": round(t_prefill_s24, 2),
            "s24_decode_sec": round(t_decode_s24, 2),
            "s24_fits_budget": fits_10s_s24,
            "s21_projection_factor": s21_factor,
            "s21_total_sec_proj": round(t_total_s21_proj, 2),
            "s21_fits_budget_proj": fits_10s_s21_proj,
        }
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark de SLMs Tier 1 diretamente no dispositivo Android")
    parser.add_argument("--serial", default="192.168.0.6:41073", help="Serial ADB do dispositivo")
    parser.add_argument("--adb-bin", default=str(Path.home() / "android-sdk/platform-tools/adb"), help="Caminho do adb")
    parser.add_argument("--models-dir", default=str(Path.home() / "models-poc"), help="Diretório de pesos GGUF")
    parser.add_argument("--bin-dir", default=str(Path(__file__).parent / "build-android/bin"), help="Diretório dos binários")
    parser.add_argument("--results-dir", default=str(Path(__file__).parent / "results"), help="Diretório de resultados")
    parser.add_argument("--models", nargs="*", default=None, help="Lista de aliases de modelos para executar")
    parser.add_argument("--threads", type=int, default=6, help="Número de threads de inferência (default 6 para Exynos 2400)")
    parser.add_argument("--repetitions", type=int, default=3, help="Número de repetições por teste no llama-bench")
    parser.add_argument("--prompt-tokens", type=int, default=1500, help="Tokens de prefill")
    parser.add_argument("--gen-tokens", type=int, default=128, help="Tokens de geração")
    parser.add_argument("--clean-only", action="store_true", help="Apenas limpa /data/local/tmp/poc e encerra")
    parser.add_argument("--force", action="store_true", help="Reprocessa mesmo se já houver resultado salvo")

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    bin_dir = Path(args.bin_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    adb = AdbDevice(args.serial, adb_bin=args.adb_bin)

    if not adb.ensure_connected():
        logger.error("Não foi possível conectar ao dispositivo ADB %s", args.serial)
        sys.exit(1)

    if args.clean_only:
        logger.info("Limpando %s no dispositivo...", REMOTE_POC_DIR)
        adb.shell(f"rm -rf {REMOTE_POC_DIR}")
        logger.info("Limpeza concluída.")
        sys.exit(0)

    # Preparar diretório remoto e transferir binários se necessário
    adb.shell(f"mkdir -p {REMOTE_POC_DIR}")

    llama_bench_local = bin_dir / "llama-bench"
    llama_cli_local = bin_dir / "llama-cli"

    if not llama_bench_local.exists() or not llama_cli_local.exists():
        logger.error("Binários não encontrados em %s. Execute build-android.sh primeiro.", bin_dir)
        sys.exit(1)

    logger.info("Garantindo binários no dispositivo...")
    adb.push(llama_bench_local, f"{REMOTE_POC_DIR}/llama-bench")
    adb.push(llama_cli_local, f"{REMOTE_POC_DIR}/llama-cli")
    adb.shell(f"chmod +x {REMOTE_POC_DIR}/llama-bench {REMOTE_POC_DIR}/llama-cli")

    # Filtrar modelos desejados
    target_models = MODELS_SPEC
    if args.models:
        target_models = [m for m in MODELS_SPEC if m["alias"] in args.models]

    all_results = []
    summary_json_path = results_dir / "summary.json"
    summary_csv_path = results_dir / "summary.csv"

    for model_info in target_models:
        alias = model_info["alias"]
        model_result_file = results_dir / f"{alias}.json"

        if model_result_file.exists() and not args.force:
            logger.info("Carregando resultado existente para %s de %s...", alias, model_result_file)
            with open(model_result_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            all_results.append(res)
            continue

        res = run_benchmark_for_model(
            adb=adb,
            model_info=model_info,
            models_dir=models_dir,
            threads=args.threads,
            repetitions=args.repetitions,
            prompt_tokens=args.prompt_tokens,
            gen_tokens=args.gen_tokens
        )

        if "error" in res:
            logger.error("Falha no benchmark de %s: %s", alias, res["error"])
            continue

        # Salvar resultado individual imediatamente
        with open(model_result_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        all_results.append(res)

        # Pequena pausa para arrefecimento térmico entre modelos
        logger.info("Pausa de 5s para estabilização térmica...")
        time.sleep(5)

    # Consolidar JSON final
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Resumo JSON salvo em: %s", summary_json_path)

    # Consolidar CSV final
    csv_headers = [
        "alias", "name", "params", "file_size_mib", "prefill_tok_s", "decode_tok_s",
        "peak_ram_mib", "cold_load_s", "delta_ap_c", "delta_bat_c",
        "s24_total_lat_s", "s24_fits_10s", "s21_total_lat_proj_s", "s21_fits_10s_proj", "sanity_response"
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for r in all_results:
            writer.writerow([
                r["alias"],
                r["name"],
                r["params"],
                r["file_size_mib"],
                r["prefill_tok_per_sec"],
                r["decode_tok_per_sec"],
                r["peak_ram_mib"],
                r["cold_load_sec"],
                r["temperatures"]["delta_ap_c"],
                r["temperatures"]["delta_bat_c"],
                r["latency_budget"]["s24_total_sec"],
                r["latency_budget"]["s24_fits_budget"],
                r["latency_budget"]["s21_total_sec_proj"],
                r["latency_budget"]["s21_fits_budget_proj"],
                r["sanity"]["response"]
            ])
    logger.info("Resumo CSV salvo em: %s", summary_csv_path)

    # Limpeza completa no dispositivo
    logger.info("Limpando %s no dispositivo ao finalizar...", REMOTE_POC_DIR)
    adb.shell(f"rm -rf {REMOTE_POC_DIR}")
    logger.info("Dispositivo limpo com sucesso!")


if __name__ == "__main__":
    main()
