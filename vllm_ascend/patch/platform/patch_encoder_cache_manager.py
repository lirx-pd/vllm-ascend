# SPDX-License-Identifier: Apache-2.0
"""Backport encoder-cache correctness fixes required by EC transfer."""

from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.request import Request


def _free_encoder_input(
    self: EncoderCacheManager,
    request: Request,
    input_id: int,
) -> None:
    req_id = request.request_id
    mm_hash = request.mm_features[input_id].identifier

    if req_id in self.request_cached_ids:
        self.request_cached_ids[req_id].discard(input_id)
        if not self.request_cached_ids[req_id]:
            del self.request_cached_ids[req_id]

    if not self.cached.get(mm_hash):
        return

    # A request has one hash reference even when the same item occurs more
    # than once. Keep that reference until its final occurrence is released.
    if any(request.mm_features[other_id].identifier == mm_hash for other_id in self.request_cached_ids.get(req_id, ())):
        return

    self.cached[mm_hash].discard(req_id)
    if not self.cached[mm_hash]:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.freeable[mm_hash] = num_encoder_embeds
        self.num_freeable_slots += num_encoder_embeds


def _get_freed_mm_hashes(self: EncoderCacheManager) -> list[str]:
    # An entry evicted early in a scheduling pass may be allocated again later
    # in that pass. Its worker-side tensor must remain cached in that case.
    freed = [mm_hash for mm_hash in self.freed if mm_hash not in self.cached]
    self.freed = []
    return freed


EncoderCacheManager.free_encoder_input = _free_encoder_input
EncoderCacheManager.get_freed_mm_hashes = _get_freed_mm_hashes
