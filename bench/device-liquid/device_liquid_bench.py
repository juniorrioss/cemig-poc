#!/usr/bin/env python3
"""
bench/device-liquid/device_liquid_bench.py

Harness de benchmark comparativo para modelos LFM2.5 no Galaxy S24+ (Exynos 2400):
- LFM2.5-1.2B-Instruct Q4_K_M (baseline llama.cpp)
- LFM2.5-1.2B-Instruct QAD-Q4_0 (quantization-aware distillation da Liquid)
- LFM2.5-350M Q4_K_M (segundo ponto da família LFM2.5)

Mede:
1. Throughput de prefill e decode em 2 cenários:
   - pp1500 / tg128 (cenário pesado de RAG de campo)
   - pp800 / tg128 (cenário lean RAG / top-2 chunks)
2. Pico de memória RAM (VmHWM via /proc/<pid>/status)
3. Tempo de carregamento frio (cold load time)
4. Telemetria térmica (AP, BAT, SKIN via dumpsys thermalservice)
5. Qualidade e latência comparada nos 20 pares do smoke set (QAD-Q4_0 vs Q4_K_M)
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
logger = logging.getLogger("liquid_bench")

MODELS_CONFIG = [
    {
        "alias": "lfm2.5-1.2b-instruct-q4_k_m",
        "filename": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        "name": "Liquid LFM2.5 1.2B Instruct (Q4_K_M Vanilla)",
        "params": "1.2B",
        "quant": "Q4_K_M",
        "is_qad": False,
        "run_smoke": True,
    },
    {
        "alias": "lfm2.5-1.2b-instruct-qad-q4_0",
        "filename": "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf",
        "name": "Liquid LFM2.5 1.2B Instruct (QAD-Q4_0 Quant-Aware Distilled)",
        "params": "1.2B",
        "quant": "QAD-Q4_0",
        "is_qad": True,
        "run_smoke": True,
    },
    {
        "alias": "lfm2.5-350m-q4_k_m",
        "filename": "LFM2.5-350M-Q4_K_M.gguf",
        "name": "Liquid LFM2.5 350M (Q4_K_M)",
        "params": "0.35B",
        "quant": "Q4_K_M",
        "is_qad": False,
        "run_smoke": False,
    },
]

REMOTE_POC_DIR = "/data/local/tmp/poc"
SMOKE_SYSTEM_PROMPT = (
    "Você é um assistente especialista em Normas Regulamentadoras (NRs) de segurança do trabalho "
    "no setor elétrico brasileiro. Responda à pergunta de forma sucinta, precisa e estritamente "
    "técnica com base nas normas regulamentadoras aplicáveis."
)


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
        """Acorda a tela do dispositivo e garante tela ativa."""
        self.shell("input keyevent KEYCODE_WAKEUP")
        self.shell("svc power stayon true")
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


def run_llama_bench_config(
    adb: AdbDevice,
    remote_model_path: str,
    prompt_tokens: int,
    gen_tokens: int,
    threads: int = 6,
    repetitions: int = 3
) -> Dict[str, Any]:
    """Executa o llama-bench no dispositivo para uma dada configuração de contexto."""
    bench_remote_script = f"""
MODEL_PATH="{remote_model_path}"
BENCH_BIN="{REMOTE_POC_DIR}/llama-bench"
JSON_OUT="{REMOTE_POC_DIR}/bench_{prompt_tokens}.json"
LOG_OUT="{REMOTE_POC_DIR}/bench_{prompt_tokens}.log"
RAM_OUT="{REMOTE_POC_DIR}/ram_{prompt_tokens}.txt"

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
    t_start = time.time()
    code, out, err = adb.shell(bench_remote_script, timeout=600)
    wall_time = time.time() - t_start

    # Ler saídas
    _, json_raw, _ = adb.shell(f"cat {REMOTE_POC_DIR}/bench_{prompt_tokens}.json")
    _, log_raw, _ = adb.shell(f"cat {REMOTE_POC_DIR}/bench_{prompt_tokens}.log")
    _, ram_raw, _ = adb.shell(f"cat {REMOTE_POC_DIR}/ram_{prompt_tokens}.txt")

    # Limpeza dos arquivos temporários de bench
    adb.shell(f"rm -f {REMOTE_POC_DIR}/bench_{prompt_tokens}.json {REMOTE_POC_DIR}/bench_{prompt_tokens}.log {REMOTE_POC_DIR}/ram_{prompt_tokens}.txt")

    peak_ram_kb = 0
    for line in ram_raw.splitlines():
        if line.startswith("PEAK_RAM_KB="):
            try:
                peak_ram_kb = int(line.split("=")[-1].strip())
            except ValueError:
                pass
    peak_ram_mib = peak_ram_kb / 1024.0

    cold_load_s = 0.0
    load_matches = re.findall(r"load time\s*=\s*([0-9.]+)\s*ms", log_raw)
    if load_matches:
        cold_load_s = float(load_matches[0]) / 1000.0

    bench_data = []
    try:
        bench_data = json.loads(json_raw)
    except json.JSONDecodeError as e:
        logger.error("Erro decodificando JSON: %s\nSaída:\n%s", e, json_raw[:300])

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

    ttft_s = (prompt_tokens / pp_avg_ts) if pp_avg_ts > 0 else 0.0
    decode_100_s = (100.0 / tg_avg_ts) if tg_avg_ts > 0 else 0.0
    total_100_s = ttft_s + decode_100_s

    logger.info("Resultado pp%d: Prefill=%.2f t/s | Decode=%.2f t/s | TTFT=%.3fs | RAM=%.1f MiB",
                prompt_tokens, pp_avg_ts, tg_avg_ts, ttft_s, peak_ram_mib)

    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "prefill_tok_per_sec": round(pp_avg_ts, 2),
        "prefill_stddev": round(pp_stddev_ts, 2),
        "prefill_samples": pp_samples,
        "decode_tok_per_sec": round(tg_avg_ts, 2),
        "decode_stddev": round(tg_stddev_ts, 2),
        "decode_samples": tg_samples,
        "ttft_sec": round(ttft_s, 3),
        "total_latency_100t_sec": round(total_100_s, 3),
        "peak_ram_mib": round(peak_ram_mib, 2),
        "cold_load_sec": round(cold_load_s, 3),
        "bench_wall_time_sec": round(wall_time, 2),
    }


def run_smoke_qa_on_device(
    adb: AdbDevice,
    remote_model_path: str,
    qa_file: Path,
    threads: int = 6,
    max_tokens: int = 128
) -> List[Dict[str, Any]]:
    """Executa o benchmark de qualidade das 20 perguntas do smoke set no aparelho."""
    logger.info("Iniciando smoke QA de 20 perguntas via llama-cli...")
    questions = []
    with open(qa_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    results = []
    for idx, item in enumerate(questions, 1):
        q_id = item["id"]
        question = item["question"]
        golden = item["golden_answer"]
        category = item.get("category", "")
        doc = item.get("doc", "")

        prompt_chatml = (
            f"<|startoftext|><|im_start|>system\n{SMOKE_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Escapar aspas simples para o shell bash
        escaped_prompt = prompt_chatml.replace("'", "'\"'\"'")

        cli_cmd = (
            f"{REMOTE_POC_DIR}/llama-cli "
            f"-m '{remote_model_path}' "
            f"-p '{escaped_prompt}' "
            f"-n {max_tokens} --temp 0.1 --top-k 50 --repeat-penalty 1.05 "
            f"-t {threads} --single-turn < /dev/null"
        )

        t0 = time.time()
        adb.wake_up()
        code, stdout, stderr = adb.shell(cli_cmd, timeout=120)
        latency_s = round(time.time() - t0, 3)

        # Parsear resposta e métricas
        raw_output = ""
        prompt_ts = 0.0
        gen_ts = 0.0

        if stdout:
            # Localizar o bloco gerado após o prompt
            lines = stdout.splitlines()
            capture = False
            ans_lines = []
            for line in lines:
                if "<|im_start|>assistant" in line:
                    capture = True
                    remainder = line.split("<|im_start|>assistant", 1)[-1].strip()
                    if remainder:
                        ans_lines.append(remainder)
                    continue
                if capture:
                    if line.strip().startswith("[ Prompt:") or "Exiting..." in line:
                        break
                    ans_lines.append(line.strip())
            raw_output = " ".join([l for l in ans_lines if l]).strip()

            m_perf = re.search(r"\[ Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s \]", stdout)
            if m_perf:
                prompt_ts = float(m_perf.group(1))
                gen_ts = float(m_perf.group(2))

        logger.info("[%02d/20] %s (%.2fs, %.1f t/s): %s",
                    idx, q_id, latency_s, gen_ts, raw_output[:90])

        results.append({
            "id": q_id,
            "category": category,
            "doc": doc,
            "question": question,
            "golden_answer": golden,
            "response": raw_output,
            "latency_sec": latency_s,
            "prompt_tok_per_sec": prompt_ts,
            "gen_tok_per_sec": gen_ts,
        })

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Liquid On-Device Benchmark no S24+")
    parser.add_argument("--serial", default="192.168.0.6:41073", help="Número de série ADB")
    parser.add_argument("--adb-bin", default=str(Path.home() / "android-sdk/platform-tools/adb"), help="Caminho do binário adb")
    parser.add_argument("--models-dir", default=str(Path.home() / "models-poc"), help="Diretório dos modelos GGUF")
    parser.add_argument("--bin-dir", default=str(Path(__file__).parent.parent / "device/build-android/bin"), help="Diretório dos binários")
    parser.add_argument("--results-dir", default=str(Path(__file__).parent / "results"), help="Diretório de resultados")
    parser.add_argument("--smoke-file", default=str(Path(__file__).parent.parent / "data/smoke_qa_20.jsonl"), help="Arquivo smoke QA")
    parser.add_argument("--threads", type=int, default=6, help="Número de threads (6 = cluster de performance Exynos 2400)")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetições do llama-bench")
    args = parser.parse_args()

    adb = AdbDevice(serial=args.serial, adb_bin=args.adb_bin)
    if not adb.ensure_connected():
        logger.error("Falha ao conectar no dispositivo ADB %s", args.serial)
        return 1

    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    smoke_file = Path(args.smoke_file)
    bin_dir = Path(args.bin_dir)

    # 1. Instalar binários no aparelho
    logger.info("Configurando ambiente no dispositivo...")
    adb.shell(f"mkdir -p {REMOTE_POC_DIR}")
    adb.push(bin_dir / "llama-bench", f"{REMOTE_POC_DIR}/llama-bench")
    adb.push(bin_dir / "llama-cli", f"{REMOTE_POC_DIR}/llama-cli")
    adb.shell(f"chmod +x {REMOTE_POC_DIR}/llama-bench {REMOTE_POC_DIR}/llama-cli")

    all_results = []
    smoke_comparisons = {}

    for m_cfg in MODELS_CONFIG:
        alias = m_cfg["alias"]
        filename = m_cfg["filename"]
        local_model_path = models_dir / filename
        remote_model_path = f"{REMOTE_POC_DIR}/{filename}"

        if not local_model_path.exists():
            logger.error("Arquivo do modelo não encontrado: %s", local_model_path)
            continue

        file_size_mib = round(local_model_path.stat().st_size / (1024 * 1024), 2)
        logger.info("=================================================================")
        logger.info("Benchmarking %s (%s, %.1f MiB)", alias, m_cfg["name"], file_size_mib)
        logger.info("=================================================================")

        # Estado térmico inicial
        temps_before = adb.get_temperatures()
        ap_before = temps_before.get("AP", 0.0)
        bat_before = temps_before.get("BAT", 0.0)
        skin_before = temps_before.get("SKIN", 0.0)

        # Upload do modelo
        push_ok = adb.push(local_model_path, remote_model_path)
        if not push_ok:
            logger.error("Falha ao enviar modelo %s", filename)
            continue

        # Benchmark pp1500 / tg128
        res_pp1500 = run_llama_bench_config(
            adb=adb,
            remote_model_path=remote_model_path,
            prompt_tokens=1500,
            gen_tokens=128,
            threads=args.threads,
            repetitions=args.repetitions,
        )

        # Benchmark pp800 / tg128
        res_pp800 = run_llama_bench_config(
            adb=adb,
            remote_model_path=remote_model_path,
            prompt_tokens=800,
            gen_tokens=128,
            threads=args.threads,
            repetitions=args.repetitions,
        )

        # Smoke QA (apenas se configurado para o modelo)
        smoke_results = []
        if m_cfg["run_smoke"] and smoke_file.exists():
            smoke_results = run_smoke_qa_on_device(
                adb=adb,
                remote_model_path=remote_model_path,
                qa_file=smoke_file,
                threads=args.threads,
            )
            smoke_comparisons[alias] = smoke_results

        # Estado térmico final
        temps_after = adb.get_temperatures()
        ap_after = temps_after.get("AP", 0.0)
        bat_after = temps_after.get("BAT", 0.0)
        skin_after = temps_after.get("SKIN", 0.0)

        # Limpeza do modelo remoto para preservar espaço
        adb.remove_remote_file(remote_model_path)

        model_summary = {
            "alias": alias,
            "name": m_cfg["name"],
            "params": m_cfg["params"],
            "quant": m_cfg["quant"],
            "is_qad": m_cfg["is_qad"],
            "file_size_mib": file_size_mib,
            "threads": args.threads,
            "repetitions": args.repetitions,
            "metrics_pp1500": res_pp1500,
            "metrics_pp800": res_pp800,
            "temperatures": {
                "ap_before_c": ap_before,
                "ap_after_c": ap_after,
                "delta_ap_c": round(ap_after - ap_before, 2),
                "bat_before_c": bat_before,
                "bat_after_c": bat_after,
                "delta_bat_c": round(bat_after - bat_before, 2),
                "skin_before_c": skin_before,
                "skin_after_c": skin_after,
                "delta_skin_c": round(skin_after - skin_before, 2),
            },
            "smoke_qa_count": len(smoke_results),
        }

        # Salvar JSON individual
        ind_json_path = results_dir / f"{alias}.json"
        with open(ind_json_path, "w", encoding="utf-8") as f:
            json.dump({**model_summary, "smoke_qa": smoke_results}, f, indent=2, ensure_ascii=False)
        logger.info("Salvo: %s", ind_json_path)

        all_results.append(model_summary)

    # Salvar resumo consolidado JSON
    summary_json_path = results_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Salvo resumo consolidado: %s", summary_json_path)

    # Salvar comparação detalhada de qualidade do smoke QA
    if smoke_comparisons:
        smoke_cmp_path = results_dir / "smoke_qa_comparison.json"
        with open(smoke_cmp_path, "w", encoding="utf-8") as f:
            json.dump(smoke_comparisons, f, indent=2, ensure_ascii=False)
        logger.info("Salvo comparativo smoke QA: %s", smoke_cmp_path)

    # Salvar tabela CSV consolidada
    summary_csv_path = results_dir / "summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Modelo", "Quant", "Tamanho (MiB)",
            "pp1500 Prefill (t/s)", "pp1500 TTFT (s)", "pp1500 Total 100t (s)",
            "pp800 Prefill (t/s)", "pp800 TTFT (s)", "pp800 Total 100t (s)",
            "Decode (t/s)", "Pico RAM (MiB)", "Load Time (s)",
            "Delta AP (°C)", "Delta Bat (°C)"
        ])
        for m in all_results:
            p1500 = m["metrics_pp1500"]
            p800 = m["metrics_pp800"]
            t = m["temperatures"]
            writer.writerow([
                m["name"], m["quant"], m["file_size_mib"],
                p1500["prefill_tok_per_sec"], p1500["ttft_sec"], p1500["total_latency_100t_sec"],
                p800["prefill_tok_per_sec"], p800["ttft_sec"], p800["total_latency_100t_sec"],
                p1500["decode_tok_per_sec"], p1500["peak_ram_mib"], p1500["cold_load_sec"],
                t["delta_ap_c"], t["delta_bat_c"]
            ])
    logger.info("Salvo CSV consolidado: %s", summary_csv_path)

    # Limpeza no aparelho
    logger.info("Limpando diretório remoto %s...", REMOTE_POC_DIR)
    adb.shell(f"rm -rf {REMOTE_POC_DIR}")
    adb.shell("svc power stayon false")

    logger.info("Benchmark concluído com sucesso!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
