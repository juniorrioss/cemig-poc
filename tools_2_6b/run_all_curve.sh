#!/bin/bash
# roda a curva de saturação (step750/1500/3000 r32) + varredura de rank (r32/r128 full) — Q4.
set -uo pipefail
declare -A M=(
  [tools_2_6b_step750_r32]=lfm2.5-2.6b-tools_2_6b_step750_r32-Q4_0.gguf
  [tools_2_6b_step1500_r32]=lfm2.5-2.6b-tools_2_6b_step1500_r32-Q4_0.gguf
  [tools_2_6b_step3000_r32]=lfm2.5-2.6b-tools_2_6b_step3000_r32-Q4_0.gguf
  [tools_2_6b_r32]=lfm2.5-2.6b-tools_2_6b_r32-Q4_0.gguf
  [tools_2_6b_r128]=lfm2.5-2.6b-tools_2_6b_r128-Q4_0.gguf
)
for L in tools_2_6b_step750_r32 tools_2_6b_step1500_r32 tools_2_6b_step3000_r32 tools_2_6b_r32 tools_2_6b_r128; do
  echo "===== CURVE EVAL $L ====="
  bash run_curve_eval.sh "~/cemig-poc/models/${M[$L]}" "$L" 8501
done
echo "ALL CURVE EVAL DONE"
