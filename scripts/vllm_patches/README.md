# vllm patches (mooncake TP=2 fixes)

These patches are applied to the **vendored vllm-src** (editable install at
`/data1/hgq/vllm-src`), NOT the uni-agent repo. Kept here for reproducibility.

## mooncake_store_worker.py  (CPU staging patch)
Target: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`

Fixes: TP=2 mooncake `writeBody failed to copy from CUDA memory` on tp_rank:1
(the rank-1 worker's CUDA context isn't current when the mooncake transport
thread does the GPU→network memcpy → "invalid argument").

Mechanism: when `MOONCAKE_CPU_STAGING=1`, the vllm worker thread (which owns the
correct CUDA context) copies GPU KV → a pinned-CPU staging buffer first
(`_CpuStagingBuffer` / `_stage_put_batch`); mooncake then transports the CPU
buffer over TCP. Bypasses the cross-thread CUDA memcpy.

Apply:
```bash
cp mooncake_store_worker.py /data1/hgq/vllm-src/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
# then launch with MOONCAKE_CPU_STAGING=1 (set in sweep_capacity_load.sh for mc runs)
```

Validated on 150 (writeBody → 0). Pairs with `MC_TCP_ENABLE_CONNECTION_POOL=1`
(conn-pool, fixes TCP ephemeral-port exhaustion) — BOTH needed for mooncake
TP=2 to hit TRANSFER_FAIL≈0 + External hit>0. See work_log §4.
