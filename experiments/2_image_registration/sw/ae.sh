#!/usr/bin/env bash
set -euo pipefail

# ---- config (modifica se serve) ----
BUILD_DIR="build"
LOG_DIR="logs"

PET="/scratch/gsorrentino/PET/"
CT="/scratch/gsorrentino/CT/"
OUT="./output"

VFPGA_ID=0
TX=10
TY=10
ANG=0
RUNS=10
GPU_ID=0
DEPTH=128

# Path dei sorgenti (rispetto a build/)
P2P_SRC="../p2p_registration_step.cpp"
NOP2P_SRC="../registration_step.cpp"

# Nome binario generato dentro build/
BIN_NAME="p2p_baseline"
# ------------------------------------

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1" >&2; exit 1; }; }
need cmake
need make
need grep
need tail
need sed
need awk
need tee
need stdbuf
need readlink

mkdir -p "$BUILD_DIR" "$LOG_DIR" "$OUT"

OUT_ABS="$(readlink -f "$OUT")"
BUILD_ABS="$(readlink -f "$BUILD_DIR")"
LOG_ABS="$(readlink -f "$LOG_DIR")"

# Estrae "Average execution time over ... runs: X s" da un log
extract_avg_time_from_log() {
  local logfile="$1"
  [[ -f "$logfile" ]] || { echo "ERROR: missing log file: $logfile" >&2; exit 1; }

  local t
  t="$(grep -E 'Average execution time over [0-9]+ runs: [0-9]+([.][0-9]+)? s' "$logfile" \
      | tail -n 1 \
      | sed -E 's/.*: ([0-9]+([.][0-9]+)?) s.*/\1/')"

  if [[ -z "${t:-}" ]] || ! [[ "$t" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: could not parse average time from: $logfile" >&2
    echo "Hint: last lines:" >&2
    tail -n 80 "$logfile" >&2
    exit 1
  fi

  echo "$t"
}

run_variant() {
  local src="$1"
  local label="$2"
  local logfile="$LOG_ABS/${label}.log"

  # Build (loggato, non stampato a schermo)
  (
    pushd "$BUILD_DIR" >/dev/null
    cmake .. -DEN_GPU=1 -DSRC="$src"
    make -j
    popd >/dev/null
  ) 2>&1 | tee "$logfile" >/dev/null

  # Run (loggato, non stampato a schermo)
  (
    cd "$LOG_ABS"  # così eventuali CSV del programma finiscono in logs/
    stdbuf -oL -eL "$BUILD_ABS/$BIN_NAME" \
      "$VFPGA_ID" "$PET" "$CT" "$OUT_ABS" "$TX" "$TY" "$ANG" "$RUNS" "$GPU_ID" "$DEPTH"
  ) 2>&1 | tee -a "$logfile" >/dev/null
}

# Esegui entrambe le varianti (nessun output in terminale)
run_variant "$P2P_SRC"  "P2P"
run_variant "$NOP2P_SRC" "NoP2P"

# Leggi i tempi dai log
P2P_TIME="$(extract_avg_time_from_log "$LOG_ABS/P2P.log")"
NOP2P_TIME="$(extract_avg_time_from_log "$LOG_ABS/NoP2P.log")"

# Calcola improvement
IMPROVEMENT="$(awk -v nop2p="$NOP2P_TIME" -v p2p="$P2P_TIME" 'BEGIN {
  if (nop2p <= 0) { printf "nan"; exit 0; }
  printf "%.2f", ((nop2p - p2p) / nop2p) * 100.0
}')"

# Stampa SOLO il summary finale
echo "==================== SUMMARY ===================="
echo
echo "latency P2P:   ${P2P_TIME} s"
echo "latency NoP2P: ${NOP2P_TIME} s"
echo "Improvement %: ${IMPROVEMENT}%"
echo
echo "GPU_ID:  ${GPU_ID}"
echo "DEPTH:   ${DEPTH}"
echo
echo "Logs saved in: $LOG_DIR/P2P.log and $LOG_DIR/NoP2P.log"
echo " ================================================"