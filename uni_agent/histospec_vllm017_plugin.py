"""HistoSpec cache adapter for vLLM 0.17's native suffix proposer."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_installed = False


def install() -> None:
    """Replace only the cache-facing methods of vLLM's suffix proposer."""
    global _installed
    if _installed:
        return

    import vllm

    if not vllm.__version__.startswith("0.17."):
        logger.warning("histospec_v017 requires vLLM 0.17.x; found %s", vllm.__version__)
        return

    from specrl.suffix_cache import SuffixCache
    from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

    def init(self, vllm_config):
        config = vllm_config.speculative_config
        assert config is not None
        self.num_speculative_tokens = min(config.num_speculative_tokens, 5)
        self.max_model_len = vllm_config.model_config.max_model_len
        self.suffix_cache = SuffixCache()
        self._histospec_requests: set[str] = set()
        self._histospec_calls = 0
        self._histospec_draft_tokens = 0
        logger.info(
            "HistoSpec vLLM 0.17 proposer initialized: max_draft_tokens=%d prefix_length=7 min_probability=0.1",
            self.num_speculative_tokens,
        )

    def propose(self, input_batch, sampled_token_ids, slot_mappings=None):
        del slot_mappings
        active_req_ids = set(input_batch.req_id_to_index)
        for req_id in self._histospec_requests - active_req_ids:
            self.suffix_cache.evict_responses(req_id)
        self._histospec_requests.intersection_update(active_req_ids)

        new_req_ids = []
        new_prompts = []
        for req_id in input_batch.req_ids:
            if req_id in self._histospec_requests:
                continue
            index = input_batch.req_id_to_index[req_id]
            prompt_len = input_batch.num_prompt_tokens[index]
            new_req_ids.append(req_id)
            new_prompts.append(input_batch.token_ids_cpu[index, :prompt_len].tolist())
            self._histospec_requests.add(req_id)
        if new_req_ids:
            self.suffix_cache.fetch_responses_by_prompts_batch(new_req_ids, new_prompts)

        req_ids = []
        patterns = []
        limits = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                req_ids.append("")
                patterns.append([])
                limits.append(0)
                continue
            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            self.suffix_cache.update_spec_len(req_id, len(sampled_ids))
            history_len = max(0, 7 - len(sampled_ids))
            history_start = max(0, num_tokens - history_len)
            pattern = input_batch.token_ids_cpu[i, history_start:num_tokens].tolist() + list(sampled_ids)
            req_ids.append(req_id)
            patterns.append(pattern[-7:])
            limits.append(min(self.num_speculative_tokens, self.max_model_len - num_tokens - 1))

        drafts = self.suffix_cache.speculate(req_ids, patterns, min_token_prob=0.1)
        drafts = [draft[: max(limit, 0)] for draft, limit in zip(drafts, limits)]
        self._histospec_calls += 1
        draft_count = sum(map(len, drafts))
        self._histospec_draft_tokens += draft_count
        if draft_count and self._histospec_draft_tokens == draft_count:
            logger.info("HistoSpec produced its first drafts: tokens=%d", draft_count)
        if self._histospec_calls % 1000 == 0:
            logger.info(
                "HistoSpec proposer counters: calls=%d draft_tokens=%d",
                self._histospec_calls,
                self._histospec_draft_tokens,
            )
        return drafts

    SuffixDecodingProposer.__init__ = init
    SuffixDecodingProposer.propose = propose
    _installed = True
    logger.info("Installed HistoSpec adapter for vLLM %s", vllm.__version__)
