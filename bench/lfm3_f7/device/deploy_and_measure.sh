#!/usr/bin/env bash
# deploy_and_measure.sh <adb-serial> — F7: instala o APK release no S24+, empurra os pesos
# externos (LFM3 Q4_0, EmbeddingGemma, Nemotron), roda o E2E das perguntas-exemplo
# (broadcast RUN_F7_E2E) e puxa o log bruto + RAM (dumpsys meminfo).
#
# Pesos (fora do git): ~/models-poc/lfm3_f7/f4-r32-ep2-{Q4_0 (md5 d26176d3…, default),Q4_K_M (md5 da25fdf5…, opcional)}.gguf,
# ~/models-poc/embed/embeddinggemma-300M-qat-Q4_0.gguf, ~/models-asr/nemotron-3.5-official-560ms-int8/
set -euo pipefail
S="$1"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${OUT:-$REPO/bench/lfm3_f7/data/device}"
PKG=br.org.ceia.cemigpoc
EXT=/sdcard/Android/data/$PKG/files
A="${ADB:-adb} -s $S"
mkdir -p "$OUT"

$A install -r -d "${APK:-$REPO/android/app/build/outputs/apk/release/app-release.apk}"
$A shell pm grant $PKG android.permission.RECORD_AUDIO || true
# cria o diretório externo do app (primeira abertura) e corrige permissão (lição poc-sft-v3)
$A shell am start -n $PKG/.MainActivity >/dev/null; sleep 3; $A shell am force-stop $PKG
$A shell mkdir -p $EXT
push_if_needed() {  # <local> <remote-name>
  local l="$1" r="$EXT/$2"
  local lsz; lsz=$(stat -c %s "$l")
  local rsz; rsz=$($A shell "stat -c %s '$r' 2>/dev/null" | tr -d '\r' || true)
  if [ "$lsz" != "$rsz" ]; then $A push "$l" "$r"; else echo "ok (já presente) $2"; fi
}
# quant do LFM3 = o do build (-Pcemig.lfm3Quant, default Q4_0; q4_k_m é a opção qualidade);
# LFM3_GGUF aponta o arquivo local (default ~/models-poc/lfm3_f7/f4-r32-ep2-$LFM3_QUANT.gguf)
LFM3_QUANT="${LFM3_QUANT:-Q4_0}"
# LFM3_MODEL = checkpoint do build (-Pcemig.lfm3Model: v3s400 default desde a F9; f4-r32-ep2 = APK v1.0)
LFM3_MODEL="${LFM3_MODEL:-v3s400}"
push_if_needed "${LFM3_GGUF:-$HOME/models-poc/lfm3_f7/$LFM3_MODEL-$LFM3_QUANT.gguf}" "lfm3-1.7b-$LFM3_MODEL-$LFM3_QUANT.gguf"
push_if_needed ~/models-poc/embed/embeddinggemma-300M-qat-Q4_0.gguf embeddinggemma-300M-qat-Q4_0.gguf
if ! $A shell "test -f $EXT/nemotron-3.5-official-560ms-int8/encoder.int8.onnx"; then
  $A push ~/models-asr/nemotron-3.5-official-560ms-int8 $EXT/
fi
$A shell chmod -R 775 $EXT || true

$A logcat -c
$A shell am start -n $PKG/.MainActivity >/dev/null
echo "[wait] carregando motores..."
for i in $(seq 1 120); do
  if $A logcat -d -s MainViewModel:I | grep -q "Inicialização dos motores concluída"; then break; fi; sleep 2
done
$A logcat -d -s MainViewModel:I | tail -5
$A shell am broadcast -a $PKG.RUN_F7_E2E -p $PKG ${2:-}  # -p: broadcast explícito (implícito não chega ao receiver de manifesto)
echo "[wait] E2E..."
for i in $(seq 1 360); do
  if $A logcat -d -s CEMIG_F7_E2E:I | grep -q "E2E F7 CONCLUÍDO"; then break; fi; sleep 5
done
$A shell dumpsys meminfo $PKG > "$OUT/meminfo_after.txt"
$A pull $EXT/f7_e2e_results.json "$OUT/f7_e2e_results${3:-}.json"
$A logcat -d > "$OUT/logcat_f7${3:-}.txt"
grep -E "TOTAL PSS|TOTAL RSS|TOTAL:" "$OUT/meminfo_after.txt" | head -3
echo "[ok] $OUT"
