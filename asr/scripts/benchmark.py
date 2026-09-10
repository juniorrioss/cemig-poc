#!/usr/bin/env python3
"""
Full ASR Benchmark on Device (Samsung Galaxy S24+)
Evaluates:
- whisper.cpp tiny-q5_1
- whisper.cpp base-q5_1
- whisper.cpp small-q5_1
- sherpa-onnx Nemotron 3.5 streaming 0.6B INT8 (official)
- sherpa-onnx Nemotron 3.5 streaming 0.6B INT8 (ottema pt-BR fine-tune)

Metrics:
- WER clean & noisy
- Domain Keyword Hit Rate (clean & noisy)
- RTF (Real Time Factor)
- Inference Latency
- Peak RAM (RSS) via memtime
- Package / Model Size
"""

import os
import sys
import json
import re
import unicodedata
import subprocess
import wave

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FRASES_PATH = os.path.join(DATA_DIR, "frases.txt")
KEYWORDS_PATH = os.path.join(DATA_DIR, "domain_keywords.json")

ADB_TARGET = "192.168.0.6:41073"
DEVICE_DIR = "/data/local/tmp/poc-asr"

def adb_cmd(cmd_str: str) -> str:
    adb_bin = os.path.expanduser("~/android-sdk/platform-tools/adb")
    full_cmd = [adb_bin, "-s", ADB_TARGET, "shell", cmd_str]
    res = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.stdout + "\n" + res.stderr

def normalize_pt(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[-/]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
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
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1   # substitution
                )
    return d[len(r)][len(h)] / len(r)

def check_domain_keywords(phrase_id: str, hyp: str, keywords_dict: dict) -> tuple:
    norm_hyp = normalize_pt(hyp)
    synonym_groups = keywords_dict.get(str(phrase_id), [])
    total = len(synonym_groups)
    if total == 0:
        return 1.0, 0, 0
    matched = 0
    for syn_group in synonym_groups:
        if any(normalize_pt(syn) in norm_hyp for syn in syn_group):
            matched += 1
    accuracy = matched / total
    return accuracy, matched, total

def get_audio_durations() -> dict:
    durations = {}
    for i in range(1, 26):
        wav_path = os.path.join(DATA_DIR, "clean", f"phrase_{i:02d}.wav")
        with wave.open(wav_path, "rb") as w:
            durations[i] = w.getnframes() / w.getframerate()
    return durations

def run_whisper_benchmark(model_name: str, model_file: str, model_size_mb: float, condition: str, phrases: dict, keywords: dict, durations: dict):
    print(f"\n--- Running Whisper ({model_name}) on {condition} ---")
    results = []
    
    # Run per-phrase for precise latency, RAM, and transcript
    for pid in range(1, 26):
        ref_text = phrases[pid]
        wav_device = f"{DEVICE_DIR}/{condition}/phrase_{pid:02d}.wav"
        out_txt_base = f"{DEVICE_DIR}/whisper_out_{pid}"
        
        cmd = (
            f"{DEVICE_DIR}/memtime {DEVICE_DIR}/whisper-cli "
            f"-m {DEVICE_DIR}/models/{model_file} "
            f"-f {wav_device} "
            f"-l pt -nt -np -otxt -of {out_txt_base}"
        )
        raw_output = adb_cmd(cmd)
        
        # Read text output
        cat_out = adb_cmd(f"cat {out_txt_base}.txt 2>/dev/null || true").strip()
        transcription = cat_out.splitlines()[-1] if cat_out else ""
        if not transcription:
            # fallback extract from raw_output
            for line in raw_output.splitlines():
                if "main: processing" in line or "whisper_" in line or "__MEMTIME" in line or "read_audio" in line:
                    continue
                if line.strip():
                    transcription = line.strip()
                    break
        
        # Extract timings & memory
        time_ms = 0.0
        rss_kb = 0
        mem_match = re.search(r"__MEMTIME_METRICS__: time_ms=([\d\.]+) max_rss_kb=(\d+)", raw_output)
        if mem_match:
            time_ms = float(mem_match.group(1))
            rss_kb = int(mem_match.group(2))
            
        dur = durations[pid]
        latency_sec = time_ms / 1000.0
        rtf = latency_sec / dur if dur > 0 else 0.0
        wer = compute_wer(ref_text, transcription)
        kw_acc, kw_match, kw_tot = check_domain_keywords(str(pid), transcription, keywords)
        
        results.append({
            "phrase_id": pid,
            "ref": ref_text,
            "hyp": transcription,
            "wer": wer,
            "kw_acc": kw_acc,
            "kw_match": kw_match,
            "kw_total": kw_tot,
            "duration_sec": dur,
            "latency_sec": latency_sec,
            "rtf": rtf,
            "rss_mb": rss_kb / 1024.0
        })
        print(f"[{pid:02d}/25] WER={wer*100:4.1f}% KW={kw_acc*100:5.1f}% RTF={rtf:4.2f} RSS={rss_kb/1024.0:5.1f}MB | Hyp: {transcription[:45]}")
        
    return results

def run_sherpa_benchmark(model_name: str, model_dir: str, model_size_mb: float, condition: str, phrases: dict, keywords: dict, durations: dict):
    print(f"\n--- Running Sherpa-ONNX Nemotron ({model_name}) on {condition} ---")
    results = []
    
    for pid in range(1, 26):
        ref_text = phrases[pid]
        wav_device = f"{DEVICE_DIR}/{condition}/phrase_{pid:02d}.wav"
        
        cmd = (
            f"LD_LIBRARY_PATH={DEVICE_DIR} {DEVICE_DIR}/memtime {DEVICE_DIR}/sherpa-onnx "
            f"--tokens={DEVICE_DIR}/models/{model_dir}/tokens.txt "
            f"--encoder={DEVICE_DIR}/models/{model_dir}/encoder.int8.onnx "
            f"--decoder={DEVICE_DIR}/models/{model_dir}/decoder.int8.onnx "
            f"--joiner={DEVICE_DIR}/models/{model_dir}/joiner.int8.onnx "
            f"--feat-dim=128 --num-threads=4 --decoding-method=greedy_search "
            f"{wav_device}"
        )
        raw_output = adb_cmd(cmd)
        
        # Extract json text
        transcription = ""
        json_match = re.search(r'\{ "text": "(.*?)"', raw_output)
        if json_match:
            transcription = json_match.group(1).strip()
            
        # Extract timings & memory
        time_ms = 0.0
        rss_kb = 0
        mem_match = re.search(r"__MEMTIME_METRICS__: time_ms=([\d\.]+) max_rss_kb=(\d+)", raw_output)
        if mem_match:
            time_ms = float(mem_match.group(1))
            rss_kb = int(mem_match.group(2))
            
        # Pure inference elapsed seconds from sherpa output
        inf_match = re.search(r"Elapsed seconds:\s*([\d\.]+)", raw_output)
        if inf_match:
            inf_sec = float(inf_match.group(1))
        else:
            inf_sec = time_ms / 1000.0
            
        dur = durations[pid]
        latency_sec = inf_sec
        rtf = latency_sec / dur if dur > 0 else 0.0
        wer = compute_wer(ref_text, transcription)
        kw_acc, kw_match, kw_tot = check_domain_keywords(str(pid), transcription, keywords)
        
        results.append({
            "phrase_id": pid,
            "ref": ref_text,
            "hyp": transcription,
            "wer": wer,
            "kw_acc": kw_acc,
            "kw_match": kw_match,
            "kw_total": kw_tot,
            "duration_sec": dur,
            "latency_sec": latency_sec,
            "wall_sec": time_ms / 1000.0,
            "rtf": rtf,
            "rss_mb": rss_kb / 1024.0
        })
        print(f"[{pid:02d}/25] WER={wer*100:4.1f}% KW={kw_acc*100:5.1f}% RTF={rtf:4.2f} RSS={rss_kb/1024.0:5.1f}MB | Hyp: {transcription[:45]}")
        
    return results

def aggregate_metrics(results: list) -> dict:
    wers = [r["wer"] for r in results]
    kw_matches = sum(r["kw_match"] for r in results)
    kw_totals = sum(r["kw_total"] for r in results)
    rtfs = [r["rtf"] for r in results]
    latencies = [r["latency_sec"] for r in results]
    rss_list = [r["rss_mb"] for r in results]
    
    return {
        "mean_wer": sum(wers) / len(wers) if wers else 0.0,
        "kw_acc": kw_matches / kw_totals if kw_totals > 0 else 0.0,
        "mean_rtf": sum(rtfs) / len(rtfs) if rtfs else 0.0,
        "mean_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
        "peak_rss_mb": max(rss_list) if rss_list else 0.0
    }

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load ground truth
    phrases = {}
    with open(FRASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pid, text = line.split("|", 1)
                phrases[int(pid)] = text
                
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        keywords = json.load(f)
        
    durations = get_audio_durations()
    
    models = [
        {"name": "Whisper Tiny Q5_1", "type": "whisper", "file": "ggml-tiny-q5_1.bin", "size_mb": 31.6},
        {"name": "Whisper Base Q5_1", "type": "whisper", "file": "ggml-base-q5_1.bin", "size_mb": 56.9},
        {"name": "Whisper Small Q5_1", "type": "whisper", "file": "ggml-small-q5_1.bin", "size_mb": 181.3},
        {"name": "Nemotron 3.5 INT8 (Official)", "type": "sherpa", "file": "nemotron-official", "size_mb": 651.5},
        {"name": "Nemotron 3.5 INT8 (Ottema PT-BR)", "type": "sherpa", "file": "nemotron-ottema", "size_mb": 651.5}
    ]
    
    all_results = {}
    summary_table = []
    
    for m in models:
        mname = m["name"]
        mtype = m["type"]
        mfile = m["file"]
        msize = m["size_mb"]
        
        all_results[mname] = {}
        for cond in ["clean", "noisy"]:
            if mtype == "whisper":
                res = run_whisper_benchmark(mname, mfile, msize, cond, phrases, keywords, durations)
            else:
                res = run_sherpa_benchmark(mname, mfile, msize, cond, phrases, keywords, durations)
            all_results[mname][cond] = res
            
        clean_agg = aggregate_metrics(all_results[mname]["clean"])
        noisy_agg = aggregate_metrics(all_results[mname]["noisy"])
        
        summary_table.append({
            "model": mname,
            "wer_clean_pct": clean_agg["mean_wer"] * 100.0,
            "wer_noisy_pct": noisy_agg["mean_wer"] * 100.0,
            "kw_acc_clean_pct": clean_agg["kw_acc"] * 100.0,
            "kw_acc_noisy_pct": noisy_agg["kw_acc"] * 100.0,
            "rtf_clean": clean_agg["mean_rtf"],
            "rtf_noisy": noisy_agg["mean_rtf"],
            "latency_clean_s": clean_agg["mean_latency_sec"],
            "latency_noisy_s": noisy_agg["mean_latency_sec"],
            "peak_rss_mb": max(clean_agg["peak_rss_mb"], noisy_agg["peak_rss_mb"]),
            "size_mb": msize
        })
        
    # Save detailed JSON
    with open(os.path.join(RESULTS_DIR, "benchmark_details.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    # Save summary JSON
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_table, f, ensure_ascii=False, indent=2)
        
    print("\n================ BENCHMARK COMPLETED ================")
    print(f"{'Modelo':<32} | {'WER Limpo':<10} | {'WER Ruído':<10} | {'KW Acc L':<9} | {'KW Acc R':<9} | {'RTF':<6} | {'RAM (MB)':<9} | {'Tamanho':<8}")
    print("-" * 110)
    for s in summary_table:
        print(f"{s['model']:<32} | {s['wer_clean_pct']:9.1f}% | {s['wer_noisy_pct']:9.1f}% | {s['kw_acc_clean_pct']:8.1f}% | {s['kw_acc_noisy_pct']:8.1f}% | {s['rtf_clean']:6.2f} | {s['peak_rss_mb']:8.1f}M | {s['size_mb']:6.1f}M")

if __name__ == "__main__":
    main()
