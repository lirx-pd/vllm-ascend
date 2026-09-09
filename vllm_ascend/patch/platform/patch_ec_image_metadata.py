# SPDX-License-Identifier: Apache-2.0
"""Accept Qwen image grids when encoder embeddings arrive through EC transfer."""

from functools import wraps
from types import MethodType

from vllm.renderers.base import BaseRenderer


def _parse_ec_image_data(self, data):
    from vllm.model_executor.models.qwen2_vl import (
        Qwen2VLMultiModalDataParser,
        _create_qwen2vl_field_factory,
    )
    from vllm.multimodal.parse import DictEmbeddingItems

    if isinstance(data, dict) and "image_embeds" not in data:
        return DictEmbeddingItems(
            data,
            modality="image",
            required_fields={"image_grid_thw"},
            fields_factory=_create_qwen2vl_field_factory(self._spatial_merge_size),
        )
    return Qwen2VLMultiModalDataParser._parse_image_data(self, data)


_original_renderer_init = BaseRenderer.__init__


@wraps(_original_renderer_init)
def _renderer_init(self, config, tokenizer):
    _original_renderer_init(self, config, tokenizer)
    ec_config = config.ec_transfer_config
    mm_config = config.model_config.multimodal_config
    if ec_config is None or not ec_config.is_ec_consumer or mm_config is None or not mm_config.enable_mm_embeds:
        return

    from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser

    # Scope the relaxed requirement to this consumer's parsers. Producers and
    # ordinary renderers still require actual embeddings in embedding inputs.
    for processor in (self.mm_processor, self._readonly_mm_processor):
        if processor is None:
            continue
        # Rendering parses through info's cached parser; processor.data_parser
        # is a separate instance created by BaseMultiModalProcessor.__init__.
        for parser in (processor.data_parser, processor.info.data_parser):
            if isinstance(parser, Qwen2VLMultiModalDataParser):
                parser._parse_image_data = MethodType(_parse_ec_image_data, parser)


BaseRenderer.__init__ = _renderer_init
