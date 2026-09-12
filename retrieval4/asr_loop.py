#!/usr/bin/env python3
"""
asr_loop.py — TRAVA 4 (ASR-in-the-loop): passa as 151 por TTS -> Whisper Base -> transcrição.

É o teste mais próximo do campo: em produção o operário FALA e o app transcreve com o
Whisper Base Q5_1 (engine embarcado). Aqui sintetizamos as 151 perguntas com edge-tts
(voz pt-BR), rodamos o MESMO whisper-cli/ggml-base-q5_1 do app no aparelho via ADB, e
salvamos a transcrição. O recall v4 é então medido SOBRE A TRANSCRIÇÃO, não o texto limpo.

Higiene: idempotente (pula áudios/transcrições já feitos), timeout por chamada, verboso.
NÃO usa GPU. Requer: edge-tts (classifier/.venv), ffmpeg, adb, whisper-cli android-static.

Uso (classifier/.venv):
  ../classifier/.venv/bin/python asr_loop.py --tts        # gera WAVs 16kHz
  ../classifier/.venv/bin/python asr_loop.py --transcribe # roda whisper no aparelho
  ../classifier/.venv/bin/python asr_loop.py --all

Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))

QA_V2 = _ROOT / "corpus" / "qa_pairs_v2.jsonl"
AUDIO_DIR = _HERE / "data" / "asr_audio"
OUT_JSONL = _HERE / "data" / "asr_transcriptions.jsonl"

ADB = str(Path.home() / "android-sdk" / "platform-tools" / "adb")
DEVICE = "192.168.0.6:41073"
WHISPER_BIN = str(Path.home() / "whisper.cpp" / "build-android-static" / "bin" / "whisper-cli")
WHISPER_MODEL = str(Path.home() / "models-asr" / "ggml-base-q5_1.bin")
DEV_DIR = "/data/local/tmp/asr_v4"
VOICE = "pt-BR-AntonioNeural"


def load_questions() -> List[Dict]:
    items = []
    for line in QA_V2.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        items.append({"id": r["id"], "question": r["question"]})
    return items


async def _tts_one(text: str, mp3: Path) -> None:
    import edge_tts
    comm = edge_tts.Communicate(text, VOICE)
    await comm.save(str(mp3))


def do_tts(items: List[Dict]) -> None:
    """Sintetiza as 151 em WAV 16kHz mono (formato do Whisper)."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items, 1):
        wav = AUDIO_DIR / f"{it['id']}.wav"
        if wav.exists():
            continue
        mp3 = AUDIO_DIR / f"{it['id']}.mp3"
        asyncio.run(_tts_one(it["question"], mp3))
        subprocess.run(["ffmpeg", "-y", "-i", str(mp3), "-ar", "16000", "-ac", "1",
                        "-f", "wav", str(wav)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mp3.unlink(missing_ok=True)
        if i % 20 == 0:
            print(f"  TTS {i}/{len(items)}", flush=True)
    print(f"TTS concluído: {len(list(AUDIO_DIR.glob('*.wav')))} WAVs em {AUDIO_DIR}")


def _adb(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([ADB, "-s", DEVICE, *args], capture_output=True, text=True, timeout=timeout)


def do_transcribe(items: List[Dict]) -> None:
    """Empurra binário+modelo+WAVs e roda whisper-cli no aparelho; salva transcrições."""
    # Setup no device (idempotente)
    _adb("shell", "mkdir", "-p", DEV_DIR)
    print("push whisper-cli + model...")
    _adb("push", WHISPER_BIN, f"{DEV_DIR}/whisper-cli", timeout=180)
    _adb("push", WHISPER_MODEL, f"{DEV_DIR}/model.bin", timeout=300)
    _adb("shell", "chmod", "755", f"{DEV_DIR}/whisper-cli")

    done: Dict[str, str] = {}
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                r = json.loads(line)
                done[r["id"]] = r["transcription"]

    fh = OUT_JSONL.open("a", encoding="utf-8")
    if not done:
        fh.write("# " + json.dumps({
            "artifact": "asr_transcriptions", "task": "poc-retrieval-v4",
            "engine": "whisper.cpp base-q5_1", "voice": VOICE, "device": DEVICE,
        }, ensure_ascii=False) + "\n")

    n_new = 0
    for i, it in enumerate(items, 1):
        if it["id"] in done:
            continue
        wav = AUDIO_DIR / f"{it['id']}.wav"
        if not wav.exists():
            print(f"  WAV ausente: {it['id']}")
            continue
        _adb("push", str(wav), f"{DEV_DIR}/in.wav", timeout=60)
        # 4 threads (mesma config do engine embarcado), saída limpa
        r = _adb("shell",
                 f"cd {DEV_DIR} && ./whisper-cli -m model.bin -f in.wav -l pt -t 4 -np -nt",
                 timeout=180)
        txt = " ".join(line.strip() for line in r.stdout.splitlines() if line.strip())
        txt = txt.strip()
        rec = {"id": it["id"], "question": it["question"], "transcription": txt}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        n_new += 1
        if i % 10 == 0:
            print(f"  ASR {i}/{len(items)} | último: {txt[:60]!r}", flush=True)
    fh.close()
    print(f"Transcrição concluída: +{n_new} novas em {OUT_JSONL}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    items = load_questions()
    print(f"{len(items)} perguntas do holdout 151")
    if args.tts or args.all:
        do_tts(items)
    if args.transcribe or args.all:
        do_transcribe(items)


if __name__ == "__main__":
    main()
