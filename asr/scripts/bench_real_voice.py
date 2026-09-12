#!/usr/bin/env python3
"""
Bancada de WER com VOZ HUMANA REAL — Whisper Base Q5_1 no Galaxy S24+.

Motivação (task poc-asr-fix): a medição histórica de 9,5% WER usou áudio SINTÉTICO
(edge-tts), que não captura sotaque, hesitações e respiração de campo. O capitão relatou
que a transcrição só acerta com fala lenta e pausada. Esta bancada mede o WER real e o
efeito da higiene de áudio + prompt de domínio + temperature fallback (a config NOVA do
JNI, embarcada em `android/whisper/src/main/cpp/whisper_jni.cpp`).

COMO USAR (quando os áudios reais chegarem):
  1. Grave os áudios conforme `asr/data/real_voice/RECORDING_INSTRUCTIONS.md`.
  2. Coloque os .wav em `asr/data/real_voice/audio/` e preencha o gabarito
     `asr/data/real_voice/manifest.jsonl` (uma linha por áudio: {"id","file","ref"}).
  3. Rode:  python3 asr/scripts/bench_real_voice.py
     (opções: --config new|old|both, default both; --model base|small, default base)

Compara duas configs, no MESMO áudio real, para isolar o ganho da nossa mudança:
  - "old": réplica do JNI ANTIGO — SEM temperature fallback (-nf -tpi 0.0), SEM prompt.
  - "new": réplica do JNI NOVO/embarcado — fallback + gates + prompt pt-BR de domínio.

Saída: tabela por-áudio + agregados (WER, KW acc) em stdout e
`asr/results/real_voice/real_voice_wer.json`.

NOTA sobre higiene de áudio: o recorte de silêncio + normalização de ganho do app
(`AudioPreprocessing.kt`) NÃO é replicado pelo whisper-cli. Para medir o pipeline COMPLETO
do app, grave falando naturalmente (com silêncio nas pontas e ganho de campo) — o app
aplica a higiene antes do Whisper. Este harness mede o motor+params; a higiene é coberta
pelos testes JVM (`AudioPreprocessingTest.kt`) e pela validação E2E no aparelho.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
import wave

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(BASE_DIR, "data", "real_voice")
AUDIO_DIR = os.path.join(REAL_DIR, "audio")
MANIFEST_PATH = os.path.join(REAL_DIR, "manifest.jsonl")
KEYWORDS_PATH = os.path.join(BASE_DIR, "data", "domain_keywords.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "real_voice")

ADB_BIN = os.path.expanduser("~/android-sdk/platform-tools/adb")
ADB_TARGET = os.environ.get("ADB_TARGET", "192.168.0.6:41073")
DEVICE_DIR = "/data/local/tmp/poc-asr"

# Prompt de domínio IDÊNTICO ao embarcado (WhisperCppEngine.DEFAULT_DOMAIN_PROMPT).
DOMAIN_PROMPT = (
    "Transcrição em português do Brasil de um eletricista em campo. "
    "Termos: NR-06, NR-10, NR-12, NR-18, NR-35, 13,8 kV, 1000 V, desenergização, religador, "
    "chave seccionadora, chave fusível, LOTO, bloqueio e etiquetagem, aterramento temporário, "
    "linha viva, bastão de manobra, EPI, EPC, tensão de segurança, talabarte, trava-quedas, "
    "cinto tipo paraquedista, luva isolante, capacete com jugular."
)

MODEL_FILES = {"base": "ggml-base-q5_1.bin", "small": "ggml-small-q5_1.bin"}


def adb(cmd: str, timeout: int = 120) -> str:
    full = [ADB_BIN, "-s", ADB_TARGET, "shell", cmd]
    try:
        r = subprocess.run(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__"


def adb_push(local: str, remote: str) -> bool:
    r = subprocess.run([ADB_BIN, "-s", ADB_TARGET, "push", local, remote],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode == 0


def normalize_pt(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(ref: str, hyp: str) -> float:
    r = normalize_pt(ref).split()
    h = normalize_pt(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def check_wav_format(path: str) -> str:
    """Retorna string de aviso se o WAV não for 16 kHz mono 16-bit, senão ''."""
    with wave.open(path, "rb") as w:
        issues = []
        if w.getframerate() != 16000:
            issues.append(f"{w.getframerate()}Hz (esperado 16000)")
        if w.getnchannels() != 1:
            issues.append(f"{w.getnchannels()}ch (esperado mono)")
        if w.getsampwidth() != 2:
            issues.append(f"{w.getsampwidth()*8}bit (esperado 16)")
    return "; ".join(issues)


def load_manifest() -> list:
    items = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    return items


def run_one(model_file: str, remote_wav: str, config: str) -> tuple:
    """Roda whisper-cli no aparelho; retorna (transcricao, latencia_s)."""
    out_base = f"{DEVICE_DIR}/rv_out"
    if config == "old":
        # Réplica do JNI ANTIGO: sem fallback, sem prompt.
        flags = "-nf -tpi 0.0"
        prompt = ""
    else:
        # Réplica do JNI NOVO/embarcado.
        flags = "-tpi 0.2 -et 2.4 -lpt -1.0 -nth 0.6 -sns"
        prompt = f' --prompt "{DOMAIN_PROMPT}"'
    # memtime (getrusage) mede wall-time e RSS; -np suprime prints do whisper-cli.
    cmd = (
        f"cd {DEVICE_DIR} && LD_LIBRARY_PATH={DEVICE_DIR} ./memtime ./whisper-cli "
        f"-m models/{model_file} -f {remote_wav} -l pt -nt -np "
        f"{flags}{prompt} -otxt -of {out_base} 2>&1"
    )
    raw = adb(cmd)
    txt = adb(f"cat {out_base}.txt 2>/dev/null || true").strip()
    transcription = txt.splitlines()[-1].strip() if txt else ""
    m = re.search(r"__MEMTIME_METRICS__: time_ms=([\d\.]+)", raw)
    latency = float(m.group(1)) / 1000.0 if m else 0.0
    return transcription, latency


def main():
    ap = argparse.ArgumentParser(description="WER com voz humana real (S24+)")
    ap.add_argument("--config", choices=["old", "new", "both"], default="both")
    ap.add_argument("--model", choices=["base", "small"], default="base")
    args = ap.parse_args()

    if not os.path.exists(MANIFEST_PATH):
        print(f"ERRO: manifesto não encontrado: {MANIFEST_PATH}")
        print("Grave os áudios e preencha o manifesto — ver "
              "asr/data/real_voice/RECORDING_INSTRUCTIONS.md")
        sys.exit(1)

    items = load_manifest()
    if not items:
        print("ERRO: manifesto vazio. Adicione linhas {\"id\",\"file\",\"ref\"}.")
        sys.exit(1)

    keywords = {}
    if os.path.exists(KEYWORDS_PATH):
        with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
            keywords = json.load(f)

    model_file = MODEL_FILES[args.model]
    configs = ["old", "new"] if args.config == "both" else [args.config]

    # Prepara aparelho: garante whisper-cli/libs/modelo staged (idempotente).
    adb(f"mkdir -p {DEVICE_DIR}/real_voice {DEVICE_DIR}/models")
    staged = adb(f"ls {DEVICE_DIR}/whisper-cli {DEVICE_DIR}/memtime")
    if "No such file" in staged or "__TIMEOUT__" in staged:
        print("ERRO: whisper-cli/memtime não estão no aparelho. Rode o stage do asr/README.md §6 "
              "(whisper-cli + libs ggml/whisper + libomp.so + memtime) primeiro.")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {"model": args.model, "target": ADB_TARGET, "items": [], "aggregate": {}}

    print(f"\n=== WER VOZ REAL — Whisper {args.model.upper()} Q5_1 @ {ADB_TARGET} ===")
    print(f"Áudios: {len(items)} | Configs: {configs}\n")

    # Push de cada áudio + verificação de formato.
    for it in items:
        local = os.path.join(AUDIO_DIR, it["file"])
        if not os.path.exists(local):
            print(f"  [!] áudio ausente: {local} — pulando {it['id']}")
            continue
        warn = check_wav_format(local)
        if warn:
            print(f"  [!] {it['id']} formato inesperado: {warn} "
                  f"(converta: ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav)")
        remote = f"{DEVICE_DIR}/real_voice/{it['file']}"
        adb_push(local, remote)

    agg = {c: {"wer": [], "kw_matched": 0, "kw_total": 0, "lat": []} for c in configs}

    for it in items:
        local = os.path.join(AUDIO_DIR, it["file"])
        if not os.path.exists(local):
            continue
        remote = f"{DEVICE_DIR}/real_voice/{it['file']}"
        dur = wav_duration(local)
        row = {"id": it["id"], "ref": it["ref"], "duration_sec": round(dur, 2), "configs": {}}
        print(f"[{it['id']}] ({dur:.1f}s) ref: {it['ref']}")
        for cfg in configs:
            hyp, lat = run_one(model_file, remote, cfg)
            wer = compute_wer(it["ref"], hyp)
            kw_syn = keywords.get(str(it.get("kw_id", it["id"])), [])
            kw_tot = len(kw_syn)
            kw_match = sum(1 for g in kw_syn if any(normalize_pt(s) in normalize_pt(hyp) for s in g))
            rtf = lat / dur if dur > 0 else 0.0
            row["configs"][cfg] = {
                "hyp": hyp, "wer_pct": round(wer * 100, 1),
                "kw_matched": kw_match, "kw_total": kw_tot,
                "latency_sec": round(lat, 2), "rtf": round(rtf, 2),
            }
            agg[cfg]["wer"].append(wer)
            agg[cfg]["kw_matched"] += kw_match
            agg[cfg]["kw_total"] += kw_tot
            agg[cfg]["lat"].append(lat)
            print(f"    {cfg:>3}: WER={wer*100:4.1f}% RTF={rtf:.2f} | {hyp}")
        results["items"].append(row)

    print("\n=== AGREGADO ===")
    print(f"{'Config':>5} | {'WER médio':>9} | {'KW acc':>7} | {'RTF médio':>9} | {'Lat média':>9}")
    for cfg in configs:
        n = len(agg[cfg]["wer"]) or 1
        wer_mean = sum(agg[cfg]["wer"]) / n * 100
        kw_acc = (agg[cfg]["kw_matched"] / agg[cfg]["kw_total"] * 100) if agg[cfg]["kw_total"] else 0.0
        lat_mean = sum(agg[cfg]["lat"]) / n
        durs = [wav_duration(os.path.join(AUDIO_DIR, it["file"]))
                for it in items if os.path.exists(os.path.join(AUDIO_DIR, it["file"]))]
        rtf_mean = lat_mean / (sum(durs) / len(durs)) if durs else 0.0
        results["aggregate"][cfg] = {
            "wer_mean_pct": round(wer_mean, 2), "kw_acc_pct": round(kw_acc, 2),
            "rtf_mean": round(rtf_mean, 2), "latency_mean_sec": round(lat_mean, 2),
            "n": len(agg[cfg]["wer"]),
        }
        print(f"{cfg:>5} | {wer_mean:8.1f}% | {kw_acc:6.1f}% | {rtf_mean:9.2f} | {lat_mean:8.2f}s")

    out = os.path.join(RESULTS_DIR, "real_voice_wer.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSalvo em {out}")


if __name__ == "__main__":
    main()
