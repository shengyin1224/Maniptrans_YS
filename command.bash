#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/data/maniptrans_data"
CONTACT_LEFT="/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/data/contacts/contacts_left/humoto"
CONTACT_RIGHT="/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/data/contacts/contacts_right/humoto"
RETARGET_ROOT="/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/data/retargeting/Humoto"
DEXHAND="inspire"
ITER=15000
CONDA_ENV="maniptrans"
GPUS=(3 4 6 7 8)

TMP_DIR="$(mktemp -d /tmp/mano2dex_queue.XXXXXX)"
FAIL_LOG="$TMP_DIR/fail.log"
: > "$FAIL_LOG"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

for gpu in "${GPUS[@]}"; do
  : > "$TMP_DIR/queue_${gpu}.tsv"
done

get_side_abbr() {
  local side="$1"
  if [[ "$side" == "left" ]]; then
    echo "lh"
  else
    echo "rh"
  fi
}

get_stage1_output() {
  local data_idx="$1"
  local side="$2"
  local side_abbr
  side_abbr="$(get_side_abbr "$side")"
  echo "${RETARGET_ROOT}/mano2${DEXHAND}_${side_abbr}/${data_idx}_stage1_nocontact.pkl"
}

get_stage2_output() {
  local data_idx="$1"
  local side="$2"
  local side_abbr
  side_abbr="$(get_side_abbr "$side")"
  echo "${RETARGET_ROOT}/mano2${DEXHAND}_${side_abbr}/${data_idx}.pkl"
}

# 任务分发：每个 data_idx 跑 left/right，并轮询分给不同 GPU
i=0
while IFS= read -r -d '' data_dir; do
  data_idx="$(basename "$data_dir")"
  for side in left right; do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    printf '%s\t%s\n' "$data_idx" "$side" >> "$TMP_DIR/queue_${gpu}.tsv"
    i=$((i + 1))
  done
done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

run_python_stage() {
  local gpu="$1"
  local data_idx="$2"
  local side="$3"
  local contact_data="$4"
  local stage="$5"
  local expected_output="$6"
  local cmd=(
    conda run --no-capture-output -n "$CONDA_ENV" python main/dataset/mano2dexhand_segmented.py
    --data_idx "$data_idx"
    --side "$side"
    --dexhand "$DEXHAND"
    --iter "$ITER"
    --contact_data "$contact_data"
    --draw_all_lines 0
    --stage "$stage"
    --headless
  )

  printf '[CMD][GPU %s] CUDA_VISIBLE_DEVICES=%s ' "$gpu" "$gpu"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"; then
    return 0
  fi

  if [[ -f "$expected_output" ]]; then
    echo "[WARN][GPU $gpu] data_idx=$data_idx side=$side stage=$stage nonzero-exit but output exists: $expected_output"
    return 0
  fi

  return 1
}

run_worker() {
  local gpu="$1"
  local queue_file="$2"
  local data_idx side contact_data stage1_output stage2_output

  while IFS=$'\t' read -r data_idx side; do
    [[ -n "${data_idx:-}" && -n "${side:-}" ]] || continue

    if [[ "$side" == "left" ]]; then
      contact_data="${CONTACT_LEFT}/contacts_${data_idx}.pkl"
    else
      contact_data="${CONTACT_RIGHT}/contacts_${data_idx}.pkl"
    fi

    if [[ ! -f "$contact_data" ]]; then
      echo "[MISS][GPU $gpu] data_idx=$data_idx side=$side path=$contact_data" | tee -a "$FAIL_LOG"
      continue
    fi

    stage1_output="$(get_stage1_output "$data_idx" "$side")"
    stage2_output="$(get_stage2_output "$data_idx" "$side")"

    if ! run_python_stage "$gpu" "$data_idx" "$side" "$contact_data" 1 "$stage1_output"; then
      echo "[FAIL][GPU $gpu] data_idx=$data_idx side=$side stage=1 missing_output=$stage1_output" | tee -a "$FAIL_LOG"
      continue
    fi

    if ! run_python_stage "$gpu" "$data_idx" "$side" "$contact_data" 2 "$stage2_output"; then
      echo "[FAIL][GPU $gpu] data_idx=$data_idx side=$side stage=2 missing_output=$stage2_output" | tee -a "$FAIL_LOG"
      continue
    fi

    echo "[DONE][GPU $gpu] data_idx=$data_idx side=$side"
  done < "$queue_file"
}

pids=()
for gpu in "${GPUS[@]}"; do
  run_worker "$gpu" "$TMP_DIR/queue_${gpu}.tsv" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

if [[ -s "$FAIL_LOG" ]]; then
  echo "===== FAILED / MISSING ====="
  cat "$FAIL_LOG"
  exit 1
fi

echo "All jobs finished successfully."
