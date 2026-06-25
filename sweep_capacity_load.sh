#!/bin/bash
# Grid sweep over alpha and load_threshold for the capacity-based load scoring strategy,
# distributed across N hosts via round-robin (HOST_INDEX). All experiments run serially
# on each host. Each experiment patches YAML, runs infer_multi.sh (KVCAware router +
# mooncake + mock sandbox, 256 concurrency), then restores YAML.
#
# =====================================================================
# Updated 2026-06-26:
#   - round-robin split: 18 KVCAware groups (6 alpha × 3 threshold) sharded across
#     NUM_HOSTS by (alpha_idx*3 + threshold_idx) % NUM_HOSTS == HOST_INDEX
#       4 hosts → 5,5,4,4 per host (balanced). Set HOST_INDEX=0/1/2/3 per machine.
#   - control group: each host runs CONTROL_REPLICAS default-balancer runs at the end
#     (ROUTER_CONFIG='') to compare vs KVCAware and absorb per-machine perf variance.
#   - mooncake: ENABLE_MOONCAKE=1 + VLLM_HOST_IP=127.0.0.1 + tcp_tw_reuse=1; each group
#     runs with mooncake (MooncakeStoreConnector, kv_both) and cleans /tmp/mooncake_storage
#     beforehand to avoid cross-run KV-buffer/port residue (memory mooncake-tcp-port-exhaustion).
#   - 256-concurrency mock sandbox verified config (router-mock-256-launch-recipe).
# =====================================================================
#
# Usage (per host):
#   HOST_INDEX=0 bash sweep_capacity_load.sh [MODEL_PATH] [DATA_PATH] [LOG_DIR] [AGENT_CONFIG]
#   HOST_INDEX=1 bash sweep_capacity_load.sh ...
#   HOST_INDEX=2 bash sweep_capacity_load.sh ...
#   HOST_INDEX=3 bash sweep_capacity_load.sh ...
#
# Env overrides: HOST_INDEX, NUM_HOSTS, CONTROL_REPLICAS, ENABLE_MOONCAKE,
#   ALPHA_LIST, LOAD_THRESHOLD_LIST, CONCURRENCY, N, MAX_NUM_SEQS, NWORKERS, TP, NGPUS,
#   PROMPT_LEN, RESPONSE_LEN, MAX_SAMPLES, MAX_MODEL_LEN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"   # sweep lives at repo root

# ── User parameters ──
MODEL_PATH=${1:-/data1/models/Qwen/Qwen3-8B}
DATA_PATH=${2:-${PROJECT_ROOT}/scripts/swe_bench_verified_modal.parquet}
LOG_DIR=${3:-/tmp/sweep_capacity_load/$(hostname)}
AGENT_CONFIG=${4:-${PROJECT_ROOT}/scripts/agent_config_mock.yaml}   # type:mock (avoid uvloop fd)

# ── MUST export: PYTHONPATH so ray workers can import uni_agent (router pkg://) ──
export PYTHONPATH=${PYTHONPATH:-${PROJECT_ROOT}}

# ── 256-concurrency verified config ──
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
export CONCURRENCY=${CONCURRENCY:-256}
ROUTER_CONFIG="pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml"

# ── mooncake (cross-replica KV sharing, on for every group) ──
export ENABLE_MOONCAKE=${ENABLE_MOONCAKE:-1}
export VLLM_HOST_IP=${VLLM_HOST_IP:-127.0.0.1}   # avoid ephemeral-port exhaustion across replicas
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}        # MUST: consistent block hashes across DP ranks for prefix cache hit
# mooncake master + config (one master per host, shared by all dp replicas)
MOONCAKE_MASTER_PORT=${MOONCAKE_MASTER_PORT:-50051}
MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH:-${PROJECT_ROOT}/mooncake_config.json}
export MOONCAKE_CONFIG_PATH
MC_GLOBAL_SEGMENT=${MC_GLOBAL_SEGMENT:-4GB}
MC_LOCAL_BUFFER=${MC_LOCAL_BUFFER:-4GB}
# TIME_WAIT reuse — needs root; best-effort (ignore if no permission)
sysctl -w net.ipv4.tcp_tw_reuse=1 >/dev/null 2>&1 || true

# ── Round-robin host sharding ──
NUM_HOSTS=${NUM_HOSTS:-4}
HOST_INDEX=${HOST_INDEX:-0}

# ── Sweep space (override via ALPHAS / LOAD_THRESHOLDS env, space-separated) ──
# e.g. ALPHAS="0.5" LOAD_THRESHOLDS="0.9" for a quick 1-group validation run.
ALPHA_LIST=(0.3 0.4 0.5 0.6 0.7 0.8)
LOAD_THRESHOLD_LIST=(0.9 0.8 0.7)
[ -n "${ALPHAS:-}" ] && read -r -a ALPHA_LIST <<< "$ALPHAS"
[ -n "${LOAD_THRESHOLDS:-}" ] && read -r -a LOAD_THRESHOLD_LIST <<< "$LOAD_THRESHOLDS"

# ── Env defaults (256 mock router + mooncake verified) ──
NWORKERS=${NWORKERS:-8}
TP=${TP:-2}                      # tp2 → dp=4 replicas (8 GPU / 2)
NGPUS=${NGPUS:-8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
PROMPT_LEN=${PROMPT_LEN:-31744}  # B2: PROMPT_LEN+RESPONSE_LEN = 39936 ≤ 40960-1024
RESPONSE_LEN=${RESPONSE_LEN:-8192}
MAX_SAMPLES=${MAX_SAMPLES:-64}
N=${N:-4}                        # 64×4 = 256 rollout = full 256 concurrency
CONTROL_REPLICAS=${CONTROL_REPLICAS:-1}   # control (default balancer) per host

# ── Config file path ──
YAML_FILE="${PROJECT_ROOT}/uni_agent/llm_router/configs/strategies/kvc_aware_strategy.yaml"

# ── Functions ──

_gpu_has_compute_apps() {
    local apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    apps=$(echo "$apps" | tr -d '[:space:]')
    [ -z "$apps" ]
}

check_and_wait_gpu_idle() {
    local max_wait=60 poll_interval=5 waited=0
    echo "    Checking GPU occupancy..."
    while [ $waited -lt $max_wait ]; do
        if _gpu_has_compute_apps; then echo "    ✅ GPUs idle"; return 0; fi
        echo "    ⏳ GPU busy, waiting ${poll_interval}s..."
        sleep $poll_interval; waited=$((waited + poll_interval))
    done
    echo "    ⚠️  GPUs not idle after ${max_wait}s, killing stale..."
    local stale_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | grep -E '^[0-9]+' | sort -u | tr '\n' ' ')
    [ -n "$stale_pids" ] && { echo "    Killing: ${stale_pids}"; kill $stale_pids 2>/dev/null || true; sleep 5; kill -9 $stale_pids 2>/dev/null || true; sleep 3; }
    # Fully tear down ray (GCS/raylet/session), not just workers — a stale dead GCS
    # makes the next run's ray.init hang 60s then fail "Failed to connect to GCS".
    ray stop --force >/dev/null 2>&1 || true
    pkill -f "parallel_infer.py" 2>/dev/null || true
    pkill -f "vllm.*serve" 2>/dev/null || true
    pkill -f "raylet|gcs_server" 2>/dev/null || true
    sleep 5
    _gpu_has_compute_apps && { echo "    ✅ GPUs freed"; return 0; } || { echo "    ❌ still occupied — proceeding"; return 1; }
}

# Start/stop mooncake master (one per host, shared by all dp replicas in a run).
# Per vllm docs: master manages metadata + coordinates the distributed store; multiple
# vLLM instances share the same master. We restart it per group so KV state doesn't
# leak across (alpha/threshold) groups, and to reset ephemeral ports/state.
start_mooncake_master() {
    # stop any stale master on this port
    pkill -f "mooncake_master.*--port ${MOONCAKE_MASTER_PORT}" 2>/dev/null || true
    sleep 1
    # write the config JSON this master's workers will read (P2PHANDSHAKE = master-embedded metadata)
    cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{"metadata_server": "P2PHANDSHAKE", "master_server_address": "127.0.0.1:${MOONCAKE_MASTER_PORT}", "global_segment_size": "${MC_GLOBAL_SEGMENT}", "local_buffer_size": "${MC_LOCAL_BUFFER}", "protocol": "tcp", "device_name": ""}
EOF
    echo "    🚀 starting mooncake_master --port ${MOONCAKE_MASTER_PORT} (config → ${MOONCAKE_CONFIG_PATH})"
    nohup mooncake_master --port "${MOONCAKE_MASTER_PORT}" > /tmp/mooncake_master.log 2>&1 &
    # wait for master RPC to come up
    for i in $(seq 1 15); do
        if ss -tln 2>/dev/null | grep -q ":${MOONCAKE_MASTER_PORT}"; then
            echo "    ✅ mooncake master up (port ${MOONCAKE_MASTER_PORT})"; return 0
        fi
        sleep 1
    done
    echo "    ⚠️ mooncake master not listening after 15s (check /tmp/mooncake_master.log) — proceeding"
}

stop_mooncake_master() {
    pkill -f "mooncake_master" 2>/dev/null || true
}

patch_yaml() {
    local alpha_val=$1 load_thresh_val=$2
    sed -i.bak \
        -e "s/^alpha: .*/alpha: ${alpha_val}/" \
        -e "s/^load_threshold: .*/load_threshold: ${load_thresh_val}/" \
        "$YAML_FILE"
    rm -f "${YAML_FILE}.bak"
}

verify_yaml() {
    local expected_alpha=$1 expected_thresh=$2
    local actual_alpha=$(grep '^alpha:' "$YAML_FILE" | awk '{print $2}')
    local actual_thresh=$(grep '^load_threshold:' "$YAML_FILE" | awk '{print $2}')
    if [ "$actual_alpha" != "$expected_alpha" ] || [ "$actual_thresh" != "$expected_thresh" ]; then
        echo "    ❌ YAML verify FAILED: alpha=${actual_alpha}(exp ${expected_alpha}) thresh=${actual_thresh}(exp ${expected_thresh})"
        return 1
    fi
    echo "    ✅ YAML verified: alpha=${actual_alpha} load_threshold=${actual_thresh}"
}

# Run one infer_multi experiment with given ROUTER_CONFIG (empty = default balancer).
run_experiment() {
    local router_cfg=$1   # pkg://... or "" (default)
    local log_file=$2
    check_and_wait_gpu_idle
    start_mooncake_master
    echo "    Running infer_multi.sh → ${log_file}"
    MAX_SAMPLES=${MAX_SAMPLES} \
    N=${N} \
    NWORKERS=${NWORKERS} \
    TP=${TP} \
    NGPUS=${NGPUS} \
    MAX_NUM_SEQS=${MAX_NUM_SEQS} \
    PROMPT_LEN=${PROMPT_LEN} \
    RESPONSE_LEN=${RESPONSE_LEN} \
    ROUTER_CONFIG=${router_cfg} \
    bash "${PROJECT_ROOT}/scripts/infer_multi.sh" "$MODEL_PATH" "$DATA_PATH" "$AGENT_CONFIG" \
        > "$log_file" 2>&1 || true
    stop_mooncake_master
    echo "    done (exit=$?) → ${log_file}"
}

# ── Main ──

YAML_BACKUP=$(mktemp)
cp "$YAML_FILE" "$YAML_BACKUP"
echo "=== YAML backup → ${YAML_BACKUP} ==="

if [ ! -f "$YAML_FILE" ]; then echo "ERROR: YAML not found: $YAML_FILE" >&2; exit 1; fi
if [ ! -f "$AGENT_CONFIG" ]; then echo "ERROR: AGENT_CONFIG not found: $AGENT_CONFIG" >&2; exit 1; fi
# Verify uni_agent importable via PYTHONPATH (router pkg:// needs it)
if ! python -c "import uni_agent" 2>/dev/null; then
    echo "ERROR: cannot import uni_agent with PYTHONPATH=${PYTHONPATH}" >&2
    echo "       Fix: export PYTHONPATH=<repo root> or 'pip install -e .' in repo root" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"

# Count how many KVCAware groups THIS host runs (round-robin)
my_kv_groups=0
for ai in "${!ALPHA_LIST[@]}"; do
    for ti in "${!LOAD_THRESHOLD_LIST[@]}"; do
        exp_idx=$((ai * ${#LOAD_THRESHOLD_LIST[@]} + ti))
        if [ $((exp_idx % NUM_HOSTS)) -eq $HOST_INDEX ]; then my_kv_groups=$((my_kv_groups+1)); fi
    done
done

echo "=== sweep_capacity_load: host=$(hostname) HOST_INDEX=${HOST_INDEX}/${NUM_HOSTS} ==="
echo "=== this host: ${my_kv_groups} KVCAware + ${CONTROL_REPLICAS} control = $((my_kv_groups+CONTROL_REPLICAS)) runs, log_dir=${LOG_DIR} ==="
echo "=== alpha=[${ALPHA_LIST[*]}] load_threshold=[${LOAD_THRESHOLD_LIST[*]}] ==="
echo "=== CONCURRENCY=${CONCURRENCY} N=${N} TP=${TP}(→dp$((NGPUS/TP))) MAX_NUM_SEQS=${MAX_NUM_SEQS} MAX_MODEL_LEN=${MAX_MODEL_LEN} ==="
echo "=== ENABLE_MOONCAKE=${ENABLE_MOONCAKE} VLLM_HOST_IP=${VLLM_HOST_IP} MOONCAKE_MASTER_PORT=${MOONCAKE_MASTER_PORT} MOONCAKE_CONFIG=${MOONCAKE_CONFIG_PATH} PYTHONHASHSEED=${PYTHONHASHSEED} ==="
echo "=== MODEL=${MODEL_PATH} DATA=${DATA_PATH} AGENT_CONFIG=${AGENT_CONFIG} ==="
echo ""

# ── KVCAware groups (round-robin sharded) ──
for ai in "${!ALPHA_LIST[@]}"; do
    alpha=${ALPHA_LIST[$ai]}
    for ti in "${!LOAD_THRESHOLD_LIST[@]}"; do
        load_thresh=${LOAD_THRESHOLD_LIST[$ti]}
        exp_idx=$((ai * ${#LOAD_THRESHOLD_LIST[@]} + ti))
        # round-robin: only run groups assigned to this host
        if [ $((exp_idx % NUM_HOSTS)) -ne $HOST_INDEX ]; then continue; fi

        tag="a${alpha}_lt${load_thresh}"
        log_file="${LOG_DIR}/${tag}.log"
        echo "=== [KVCAware exp_idx=${exp_idx}] alpha=${alpha} load_threshold=${load_thresh} ==="
        patch_yaml "$alpha" "$load_thresh"
        if ! verify_yaml "$alpha" "$load_thresh"; then
            echo "    ❌ YAML patch failed — restoring, skipping"
            cp "$YAML_BACKUP" "$YAML_FILE"; continue
        fi
        run_experiment "${ROUTER_CONFIG}" "$log_file"
        cp "$YAML_BACKUP" "$YAML_FILE"
        echo "    YAML restored"
        echo ""
    done
done

# ── Control group: default balancer (no ROUTER_CONFIG), per host ──
# Baseline vs KVCAware + absorbs per-machine perf variance. ROUTER_CONFIG="" inline
# → infer_multi.sh skips --router-config-path → KVCAware/kv-events off, default balancer.
for rep in $(seq 1 "$CONTROL_REPLICAS"); do
    control_log="${LOG_DIR}/control_default_r${rep}.log"
    echo "=== [control ${rep}/${CONTROL_REPLICAS}] default balancer (ROUTER_CONFIG='') ==="
    run_experiment "" "$control_log"
    echo ""
done

# Cleanup
rm -f "$YAML_BACKUP"
echo "=== Host $(hostname) (HOST_INDEX=${HOST_INDEX}) done: ${my_kv_groups} KVCAware + ${CONTROL_REPLICAS} control. Logs in ${LOG_DIR} ==="
echo "=== YAML restored to original ==="
