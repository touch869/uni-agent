#!/usr/bin/env bash

set -uo pipefail

PROGRAM=${0##*/}

usage() {
  cat <<'EOF'
Wait for processes to exit and/or GPUs to become available, then run a command.

Usage:
  queue.sh [wait options] [run options] -- command [args...]

Wait options (at least one is required):
  -p, --pid PID           Wait for PID to exit. Repeatable.
  -N, --num-gpus COUNT    Automatically select COUNT available GPUs.
                          Cannot be combined with --gpu.
  -g, --gpu ID [ID...]    Wait for one or more GPUs. Repeatable.
  -m, --max-used-mib MIB  Maximum used GPU memory (default: 512).
  -i, --interval SEC      Polling interval (default: 10).
      --stable-interval S Candidate verification interval (default: 3).
  -s, --stable-checks N   Consecutive availability checks (default: 3).

Run options:
  -C, --workdir DIR       Command working directory (default: current directory).
  -l, --log FILE          Append command output to FILE and tee to stdout.
  -n, --dry-run           Validate and print without waiting or running.
  -h, --help              Show this help.

Examples:
  queue.sh --num-gpus 4 -- python train.py
  queue.sh -g 2 3 6 7 -C /path/to/project -l run.log -- bash run.sh
  queue.sh -p 12345 --num-gpus 2 -- env LR=1e-5 bash run.sh
EOF
}

die() {
  printf '%s: error: %s\n' "$PROGRAM" "$*" >&2
  exit 2
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_value() {
  [[ $# -ge 2 ]] || die "option $1 requires a value"
}

is_nonnegative_integer() {
  [[ $1 =~ ^[0-9]+$ ]]
}

is_positive_number() {
  local nonzero=${1//[0.]/}
  [[ $1 =~ ^[0-9]+([.][0-9]+)?$ && -n $nonzero ]]
}

pid_is_running() {
  local pid=$1 stat state

  [[ -d /proc/$pid ]] || return 1
  if [[ -r /proc/$pid/stat ]]; then
    IFS= read -r stat < "/proc/$pid/stat" || return 0
    stat=${stat##*) }
    state=${stat%% *}
    [[ $state != Z && $state != X ]]
    return
  fi
  return 0
}

gpu_used_mib() {
  local gpu=$1 value

  value=$(nvidia-smi --id="$gpu" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null) || return 1
  value=${value//[[:space:]]/}
  [[ $value =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

print_command() {
  local arg
  printf 'Command:'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

main() {
  local -a pids=() gpus=() all_gpus=() chosen_gpus=()
  local requested_gpu_count=0 max_used_mib=512 interval=10
  local stable_interval=3 stable_checks=3 workdir=$PWD command_log=
  local dry_run=false gpu_count gpu_index_output gpu pid log_dir
  local consecutive required_checks last_candidate= ready candidate_found
  local snapshot_ok gpu_snapshot gpu_id gpu_used used query_ok
  local -a monitored_gpus=() available_gpus=() candidate_gpus=() status_parts=()
  local candidate_key available_key next_interval phase selected_space selected_csv
  declare -A gpu_memory=()

  while [[ $# -gt 0 ]]; do
    case $1 in
      -p|--pid)
        require_value "$@"
        is_nonnegative_integer "$2" || die "invalid PID: $2"
        (( 10#$2 > 0 )) || die "PID must be greater than zero"
        pids+=("$((10#$2))")
        shift 2
        ;;
      -N|--num-gpus)
        require_value "$@"
        is_nonnegative_integer "$2" || die "invalid GPU count: $2"
        (( 10#$2 > 0 )) || die "GPU count must be greater than zero"
        requested_gpu_count=$((10#$2))
        shift 2
        ;;
      -g|--gpu)
        shift
        gpu_count=0
        while [[ $# -gt 0 ]] && is_nonnegative_integer "$1"; do
          gpus+=("$((10#$1))")
          ((gpu_count += 1))
          shift
        done
        (( gpu_count > 0 )) || die "--gpu requires at least one numeric GPU ID"
        ;;
      -m|--max-used-mib)
        require_value "$@"
        is_nonnegative_integer "$2" || die "invalid memory limit: $2"
        max_used_mib=$((10#$2))
        shift 2
        ;;
      -i|--interval)
        require_value "$@"
        is_positive_number "$2" || die "invalid polling interval: $2"
        interval=$2
        shift 2
        ;;
      --stable-interval)
        require_value "$@"
        is_positive_number "$2" || die "invalid stable interval: $2"
        stable_interval=$2
        shift 2
        ;;
      -s|--stable-checks)
        require_value "$@"
        is_nonnegative_integer "$2" || die "invalid stable check count: $2"
        (( 10#$2 > 0 )) || die "stable check count must be greater than zero"
        stable_checks=$((10#$2))
        shift 2
        ;;
      -C|--workdir)
        require_value "$@"
        workdir=$2
        shift 2
        ;;
      -l|--log)
        require_value "$@"
        command_log=$2
        shift 2
        ;;
      -n|--dry-run)
        dry_run=true
        shift
        ;;
      -h|--help)
        usage
        return 0
        ;;
      --)
        shift
        break
        ;;
      *)
        die "unknown option: $1 (place the command after --)"
        ;;
    esac
  done

  (( requested_gpu_count == 0 || ${#gpus[@]} == 0 )) || \
    die "--num-gpus cannot be combined with --gpu"
  [[ ${#pids[@]} -gt 0 || ${#gpus[@]} -gt 0 || $requested_gpu_count -gt 0 ]] || \
    die "specify at least one --pid, --gpu, or --num-gpus"
  [[ $# -gt 0 ]] || die "no command specified after --"
  [[ -d $workdir ]] || die "working directory does not exist: $workdir"

  if (( requested_gpu_count > 0 || ${#gpus[@]} > 0 )); then
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
  fi

  if (( requested_gpu_count > 0 )); then
    gpu_index_output=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null) || \
      die "cannot enumerate GPUs"
    while IFS= read -r gpu; do
      gpu=${gpu//[[:space:]]/}
      is_nonnegative_integer "$gpu" || die "invalid GPU index returned by nvidia-smi: $gpu"
      all_gpus+=("$((10#$gpu))")
    done <<< "$gpu_index_output"
    (( ${#all_gpus[@]} > 0 )) || die "no GPUs found"
    (( requested_gpu_count <= ${#all_gpus[@]} )) || \
      die "requested $requested_gpu_count GPUs, but only ${#all_gpus[@]} exist"
  else
    for gpu in "${gpus[@]}"; do
      gpu_used_mib "$gpu" >/dev/null || die "cannot query GPU $gpu"
    done
  fi

  if [[ -n $command_log ]]; then
    [[ $command_log == /* ]] || command_log=$PWD/$command_log
    log_dir=$(dirname -- "$command_log")
    [[ -d $log_dir ]] || die "log directory does not exist: $log_dir"
  fi

  if $dry_run; then
    printf 'PIDs: %s\n' "${pids[*]:-(none)}"
    if (( requested_gpu_count > 0 )); then
      printf 'GPU mode: auto-select %s of [%s]\n' "$requested_gpu_count" "${all_gpus[*]}"
    else
      printf 'GPUs: %s\n' "${gpus[*]:-(none)}"
    fi
    printf 'GPU threshold: %s MiB\n' "$max_used_mib"
    printf 'Interval: %s seconds\n' "$interval"
    printf 'Stable interval/checks: %s seconds / %s\n' "$stable_interval" "$stable_checks"
    printf 'Working directory: %s\n' "$workdir"
    printf 'Command log: %s\n' "${command_log:-(inherited stdout/stderr)}"
    print_command "$@"
    return 0
  fi

  trap 'log "Interrupted; command was not started"; exit 130' INT TERM
  log "Waiting for requested resources; press Ctrl-C to cancel"
  consecutive=0
  if (( requested_gpu_count > 0 || ${#gpus[@]} > 0 )); then
    required_checks=$stable_checks
  else
    required_checks=1
  fi
  if (( requested_gpu_count == 0 )); then
    chosen_gpus=("${gpus[@]}")
  fi

  while (( consecutive < required_checks )); do
    ready=true
    candidate_found=false
    status_parts=()

    for pid in "${pids[@]}"; do
      if pid_is_running "$pid"; then
        ready=false
        status_parts+=("pid $pid running")
      else
        status_parts+=("pid $pid done")
      fi
    done

    available_gpus=()
    if (( requested_gpu_count > 0 )); then
      monitored_gpus=("${all_gpus[@]}")
    else
      monitored_gpus=("${gpus[@]}")
    fi

    snapshot_ok=true
    if (( requested_gpu_count > 0 )); then
      gpu_memory=()
      if gpu_snapshot=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null); then
        while IFS=',' read -r gpu_id gpu_used; do
          gpu_id=${gpu_id//[[:space:]]/}
          gpu_used=${gpu_used//[[:space:]]/}
          if is_nonnegative_integer "$gpu_id" && is_nonnegative_integer "$gpu_used"; then
            gpu_memory["$((10#$gpu_id))"]=$((10#$gpu_used))
          else
            snapshot_ok=false
            break
          fi
        done <<< "$gpu_snapshot"
      else
        snapshot_ok=false
      fi
    fi

    for gpu in "${monitored_gpus[@]}"; do
      query_ok=false
      if (( requested_gpu_count > 0 )); then
        if $snapshot_ok && [[ -v gpu_memory[$gpu] ]]; then
          used=${gpu_memory[$gpu]}
          query_ok=true
        fi
      elif used=$(gpu_used_mib "$gpu"); then
        query_ok=true
      fi

      if $query_ok; then
        status_parts+=("gpu $gpu ${used}MiB")
        if (( used <= max_used_mib )); then
          available_gpus+=("$gpu")
        elif (( requested_gpu_count == 0 )); then
          ready=false
        fi
      else
        status_parts+=("gpu $gpu query-failed")
        (( requested_gpu_count > 0 )) || ready=false
      fi
    done

    if (( requested_gpu_count > 0 )); then
      if (( ${#available_gpus[@]} >= requested_gpu_count )); then
        candidate_gpus=()
        if [[ -n $last_candidate ]]; then
          read -r -a candidate_gpus <<< "$last_candidate"
          available_key=" ${available_gpus[*]} "
          for gpu in "${candidate_gpus[@]}"; do
            if [[ $available_key != *" $gpu "* ]]; then
              candidate_gpus=()
              break
            fi
          done
        fi
        if (( ${#candidate_gpus[@]} == 0 )); then
          candidate_gpus=("${available_gpus[@]:0:requested_gpu_count}")
        fi
        candidate_key=${candidate_gpus[*]}
        if [[ $candidate_key != "$last_candidate" ]]; then
          consecutive=0
          last_candidate=$candidate_key
        fi
        chosen_gpus=("${candidate_gpus[@]}")
        candidate_found=true
        status_parts+=("selected [${chosen_gpus[*]}]")
      else
        ready=false
        chosen_gpus=()
        last_candidate=
        status_parts+=("available ${#available_gpus[@]}/$requested_gpu_count")
      fi
    elif $ready; then
      candidate_found=true
    fi

    if $ready; then
      ((consecutive += 1))
    else
      consecutive=0
    fi
    if $candidate_found; then
      next_interval=$stable_interval
      phase=verifying
    else
      next_interval=$interval
      phase=scanning
    fi
    log "${status_parts[*]} (ready $consecutive/$required_checks, $phase)"
    (( consecutive >= required_checks )) || sleep "$next_interval"
  done

  if (( ${#chosen_gpus[@]} > 0 )); then
    selected_space=${chosen_gpus[*]}
    selected_csv=$(IFS=,; printf '%s' "${chosen_gpus[*]}")
    export GPU_IDS=$selected_space
    export CUDA_VISIBLE_DEVICES=$selected_csv
    export QUEUE_GPU_IDS=$selected_space
    log "Selected physical GPUs: $selected_space"
  fi

  log "Conditions satisfied; starting command"
  print_command "$@"
  cd -- "$workdir" || die "cannot enter working directory: $workdir"

  if [[ -n $command_log ]]; then
    printf '[%s] Starting command: ' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$command_log"
    printf '%q ' "$@" >> "$command_log"
    printf '\n' >> "$command_log"
    exec > >(tee -a "$command_log") 2>&1
  fi
  exec "$@"
}

main "$@"
