"""Unit tests for layer-aware vLLM KV event decoding."""

from __future__ import annotations

import msgpack
import pytest

from uni_agent.llm_router.collectors.decoder.vllm.kv import VLLMKVDecoder
from uni_agent.llm_router.collectors.decoder.vllm.kv_event import KVCacheEvent
from uni_agent.llm_router.types import Layer
from uni_agent.llm_router.utils.hash import get_prefix_hashes

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


def test_gpu_block_stored_parses_medium_and_tokens() -> None:
    raw = [0.0, [[0, ["hash-1"], None, [1, 2, 3, 4], 4, None, "GPU", None]]]

    events = KVCacheEvent.from_raw(raw, default_node_id="replica-1")

    assert len(events) == 1
    event = events[0]
    assert event.medium == "GPU"
    assert event.block_size == 4
    assert event.token_ids is not None
    assert len(event.token_ids) == 1


def test_cpu_block_stored_allows_zero_block_size() -> None:
    raw = [0.0, [[0, ["hash-1"], None, [], 0, None, "CPU", None]]]

    events = KVCacheEvent.from_raw(raw, default_node_id="replica-1")

    assert len(events) == 1
    event = events[0]
    assert event.medium == "CPU"
    assert event.block_size == 0
    assert event.token_ids is None


def test_bytes_block_hash_normalizes_to_vllm_int_hash() -> None:
    raw_hash = bytes(range(32))
    vllm_int_hash = int.from_bytes(raw_hash, byteorder="big") & ((1 << 64) - 1)
    raw = [
        0.0,
        [
            [0, [vllm_int_hash], None, [1, 2, 3, 4], 4, None, "GPU", None],
            [0, [raw_hash], None, [], 0, None, "CPU", None],
        ],
    ]

    events = KVCacheEvent.from_raw(raw, default_node_id="replica-1")

    assert len(events) == 2
    assert events[0].block_hashes == [str(vllm_int_hash)]
    assert events[1].block_hashes == [str(vllm_int_hash)]


def test_decoder_resolves_tokenless_cpu_event_without_mutating_gpu_mapping() -> None:
    decoder = VLLMKVDecoder()
    raw_hash = bytes(range(32))
    vllm_block_hash = str(int.from_bytes(raw_hash, byteorder="big") & ((1 << 64) - 1))

    gpu_update = decoder.decode(
        msgpack.packb(
            [0.0, [[0, [int(vllm_block_hash)], None, [1, 2, 3, 4], 4, None, "GPU", None]]],
            use_bin_type=True,
        ),
        "replica-1",
    )
    assert gpu_update is not None
    local_hash = gpu_update.add_blocks[Layer.GPU][0]

    cpu_update = decoder.decode(
        msgpack.packb([0.0, [[0, [raw_hash], None, [], 0, None, "CPU", None]]], use_bin_type=True),
        "replica-1",
    )
    assert cpu_update is not None
    assert cpu_update.add_blocks == {Layer.CPU: [local_hash]}
    assert decoder.vllm_to_local_block_hash[(Layer.GPU, "replica-1", vllm_block_hash)] == local_hash
    assert decoder.vllm_to_local_block_hash[(Layer.CPU, "replica-1", vllm_block_hash)] == local_hash

    gpu_remove = decoder.decode(
        msgpack.packb([0.0, [[1, [int(vllm_block_hash)], "GPU", None]]], use_bin_type=True),
        "replica-1",
    )
    assert gpu_remove is not None
    assert gpu_remove.remove_blocks == {Layer.GPU: [local_hash]}
    assert (Layer.CPU, "replica-1", vllm_block_hash) in decoder.vllm_to_local_block_hash

    cpu_remove = decoder.decode(
        msgpack.packb([0.0, [[1, [raw_hash], "CPU", None]]], use_bin_type=True),
        "replica-1",
    )
    assert cpu_remove is not None
    assert cpu_remove.remove_blocks == {Layer.CPU: [local_hash]}


def test_decoder_resolves_tokenless_gpu_event_from_cpu_mapping() -> None:
    decoder = VLLMKVDecoder()
    vllm_block_hash = "shared-vllm-hash"

    cpu_update = decoder.decode(
        msgpack.packb(
            [0.0, [[0, [vllm_block_hash], None, [1, 2, 3, 4], 4, None, "CPU", None]]],
            use_bin_type=True,
        ),
        "replica-1",
    )
    assert cpu_update is not None
    local_hash = cpu_update.add_blocks[Layer.CPU][0]

    gpu_update = decoder.decode(
        msgpack.packb([0.0, [[0, [vllm_block_hash], None, [], 0, None, "GPU", None]]], use_bin_type=True),
        "replica-1",
    )

    assert gpu_update is not None
    assert gpu_update.add_blocks == {Layer.GPU: [local_hash]}
    assert decoder.vllm_to_local_block_hash[(Layer.CPU, "replica-1", vllm_block_hash)] == local_hash
    assert decoder.vllm_to_local_block_hash[(Layer.GPU, "replica-1", vllm_block_hash)] == local_hash


def test_decoder_skips_tokenless_event_without_known_mapping() -> None:
    decoder = VLLMKVDecoder()

    update = decoder.decode(
        msgpack.packb([0.0, [[0, ["unknown"], None, [], 0, None, "CPU", None]]], use_bin_type=True),
        "replica-1",
    )

    assert update is not None
    assert not update.add_blocks.get(Layer.CPU)
    assert decoder.vllm_to_local_block_hash == {}


def test_decoder_does_not_compute_hashes_with_mismatched_block_count() -> None:
    decoder = VLLMKVDecoder()

    update = decoder.decode(
        msgpack.packb(
            [0.0, [[0, ["vllm-1", "vllm-2"], None, [1, 2, 3, 4], 4, None, "GPU", None]]],
            use_bin_type=True,
        ),
        "replica-1",
    )

    assert update is not None
    assert not update.add_blocks.get(Layer.GPU)
    assert decoder.vllm_to_local_block_hash == {}


def test_decoder_does_not_compute_hashes_with_unknown_parent_mapping() -> None:
    decoder = VLLMKVDecoder()

    update = decoder.decode(
        msgpack.packb(
            [0.0, [[0, ["child"], "unknown-parent", [1, 2, 3, 4], 4, None, "GPU", None]]],
            use_bin_type=True,
        ),
        "replica-1",
    )

    assert update is not None
    assert not update.add_blocks.get(Layer.GPU)
    assert decoder.vllm_to_local_block_hash == {}


def test_decoder_resolves_parent_mapping_across_layers() -> None:
    decoder = VLLMKVDecoder()

    parent_update = decoder.decode(
        msgpack.packb(
            [0.0, [[0, ["parent"], None, [1, 2, 3, 4], 4, None, "CPU", None]]],
            use_bin_type=True,
        ),
        "replica-1",
    )
    assert parent_update is not None

    child_update = decoder.decode(
        msgpack.packb(
            [0.0, [[0, ["child"], "parent", [5, 6, 7, 8], 4, None, "GPU", None]]],
            use_bin_type=True,
        ),
        "replica-1",
    )

    assert child_update is not None
    expected_child_hash = str(get_prefix_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4)[1])
    assert child_update.add_blocks == {Layer.GPU: [expected_child_hash]}


def test_decoder_computes_cpu_hashes_from_token_ids() -> None:
    decoder = VLLMKVDecoder()
    update = decoder.decode(
        msgpack.packb(
            [0.0, [[0, ["cpu-vllm-hash-1"], None, [1, 2, 3, 4], 4, None, "CPU", None]]],
            use_bin_type=True,
        ),
        "replica-1",
    )

    assert update is not None
    local_hash = update.add_blocks[Layer.CPU][0]
    assert decoder.vllm_to_local_block_hash[(Layer.CPU, "replica-1", "cpu-vllm-hash-1")] == local_hash
    assert set(decoder.vllm_to_local_block_hash) == {(Layer.CPU, "replica-1", "cpu-vllm-hash-1")}


def test_decoder_hash_mapping_is_isolated_by_node() -> None:
    decoder = VLLMKVDecoder()
    vllm_block_hash = "shared-vllm-hash"
    payload = msgpack.packb(
        [0.0, [[0, [vllm_block_hash], None, [1, 2, 3, 4], 4, None, "GPU", None]]],
        use_bin_type=True,
    )

    replica_1_update = decoder.decode(payload, "replica-1")
    replica_2_update = decoder.decode(payload, "replica-2")
    assert replica_1_update is not None
    assert replica_2_update is not None
    local_hash = replica_1_update.add_blocks[Layer.GPU][0]
    assert replica_2_update.add_blocks[Layer.GPU] == [local_hash]

    remove_update = decoder.decode(
        msgpack.packb([0.0, [[1, [vllm_block_hash], "GPU", None]]], use_bin_type=True),
        "replica-1",
    )

    assert remove_update is not None
    assert remove_update.remove_blocks[Layer.GPU] == [local_hash]
    assert (Layer.GPU, "replica-1", vllm_block_hash) not in decoder.vllm_to_local_block_hash
    assert decoder.vllm_to_local_block_hash[(Layer.GPU, "replica-2", vllm_block_hash)] == local_hash
