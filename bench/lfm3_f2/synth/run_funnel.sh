#!/bin/bash
# Orquestra o funil de geração sintética (fases b..f) após o s1_decompose.
# Front-loaded: tudo que precisa do 27B roda aqui.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../../classifier/.venv/bin/python

echo "== s2 expand (multi-persona x cenario x estilo) =="
$PY synth/s2_expand.py --workers 48

echo "== s3 socratic (unidades densas) =="
$PY synth/s3_socratic.py --workers 32

echo "== s5 paraphrase (registros do texto cru) =="
$PY synth/s5_paraphrase.py --workers 24

echo "== s4 critic (filtro de qualidade) =="
$PY synth/s4_critic.py --workers 48

echo "== s6 mix + relatorio de diversidade =="
$PY synth/s6_mix_report.py

echo "FUNIL COMPLETO"
