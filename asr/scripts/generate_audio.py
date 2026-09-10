#!/usr/bin/env python3
"""
Generate clean and noisy evaluation WAV files (16kHz mono PCM) for CEMIG ASR benchmark.
Clean: Generated via edge-tts (pt-BR-AntonioNeural, pt-BR-FranciscaNeural).
Noisy: Clean audio + 10 dB SNR realistic field noise (engine low-rumble + wind).
"""

import asyncio
import os
import subprocess
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
NOISY_DIR = os.path.join(DATA_DIR, "noisy")
FRASES_PATH = os.path.join(DATA_DIR, "frases.txt")

VOICES = ["pt-BR-AntonioNeural", "pt-BR-FranciscaNeural"]

def create_field_noise(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Synthesize realistic field noise:
    - Low-frequency diesel engine / vehicle hum (harmonics around 50, 100, 150 Hz + rumble)
    - Wind noise (filtered pink/brown noise between 80 Hz and 800 Hz)
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    # Engine tonal harmonics
    engine = (0.5 * np.sin(2 * np.pi * 50 * t) +
              0.3 * np.sin(2 * np.pi * 100 * t) +
              0.2 * np.sin(2 * np.pi * 150 * t))
    
    # Random broadband noise filtered for wind / turbulence
    white = np.random.randn(n_samples)
    # Bandpass 60 Hz to 900 Hz
    b, a = butter(4, [60 / (sample_rate / 2), 900 / (sample_rate / 2)], btype='band')
    wind = lfilter(b, a, white)
    
    # Combine engine and wind
    noise = 0.6 * engine + 0.4 * wind
    # Normalize noise std
    noise = noise / (np.std(noise) + 1e-9)
    return noise

def add_noise_at_snr(signal: np.ndarray, noise: np.ndarray, target_snr_db: float = 10.0) -> np.ndarray:
    """Add noise to signal at a target SNR (dB)."""
    # Signal power
    p_signal = np.mean(signal ** 2)
    p_noise = np.mean(noise ** 2)
    
    if p_signal == 0 or p_noise == 0:
        return signal
    
    # SNR = 10 * log10(p_signal / p_noise_scaled)
    target_p_noise = p_signal / (10 ** (target_snr_db / 10.0))
    scale = np.sqrt(target_p_noise / p_noise)
    
    noisy = signal + scale * noise
    # Prevent hard clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val * 0.99
    return noisy

async def generate_clean_audio(phrase_id: int, text: str, voice: str):
    import edge_tts
    temp_mp3 = os.path.join(CLEAN_DIR, f"temp_{phrase_id}.mp3")
    out_wav = os.path.join(CLEAN_DIR, f"phrase_{phrase_id:02d}.wav")
    
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(temp_mp3)
    
    # Convert to 16kHz mono 16-bit PCM WAV
    cmd = [
        "ffmpeg", "-y", "-i", temp_mp3,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        out_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    os.remove(temp_mp3)

def generate_noisy_audio(phrase_id: int):
    clean_wav = os.path.join(CLEAN_DIR, f"phrase_{phrase_id:02d}.wav")
    noisy_wav = os.path.join(NOISY_DIR, f"phrase_{phrase_id:02d}.wav")
    
    sr, data = wavfile.read(clean_wav)
    # Convert to float [-1.0, 1.0]
    signal = data.astype(np.float32) / 32768.0
    duration = len(signal) / sr
    
    noise = create_field_noise(duration, sr)
    noisy = add_noise_at_snr(signal, noise, target_snr_db=10.0)
    
    noisy_int16 = (noisy * 32767.0).astype(np.int16)
    wavfile.write(noisy_wav, sr, noisy_int16)

async def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(NOISY_DIR, exist_ok=True)
    
    with open(FRASES_PATH, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    print(f"Generating audio for {len(lines)} phrases...")
    for idx, line in enumerate(lines, 1):
        pid_str, text = line.split("|", 1)
        pid = int(pid_str)
        voice = VOICES[(pid - 1) % len(VOICES)]
        print(f"[{pid:02d}/25] Generating clean ({voice.split('-')[1]})...")
        await generate_clean_audio(pid, text, voice)
        print(f"[{pid:02d}/25] Generating noisy (10 dB SNR field noise)...")
        generate_noisy_audio(pid)
        
    print("Done! Audio files generated in:")
    print(f"  Clean: {CLEAN_DIR}")
    print(f"  Noisy: {NOISY_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
