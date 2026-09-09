# SPDX-License-Identifier: Apache-2.0
"""CPU checks against the installed vLLM parser and embedding merge contract."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.config import MultiModalConfig
from vllm.entrypoints.chat_utils import _merge_embeds
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser, _create_qwen2vl_field_factory
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing.context import BaseProcessingInfo

_PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_ec_image_metadata.py"
_spec = importlib.util.spec_from_file_location("ec_image_metadata_patch", _PATCH_PATH)
metadata_patch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metadata_patch)


class ECImageMetadataTest(unittest.TestCase):
    def renderer(self, *, consumer=True, enabled=True, parser=None):
        parser = parser or Qwen2VLMultiModalDataParser(spatial_merge_size=2)
        mm_config = MultiModalConfig(enable_mm_embeds=enabled)
        ctx = SimpleNamespace(
            model_config=SimpleNamespace(get_inputs_embeds_size=lambda: 8),
            get_mm_config=lambda: mm_config,
        )

        def make_processor(parser):
            info = BaseProcessingInfo(ctx)
            info.get_data_parser = lambda: Qwen2VLMultiModalDataParser(spatial_merge_size=2)
            info.get_supported_mm_limits = lambda: {"image": None}
            return SimpleNamespace(data_parser=parser, info=info)

        renderer = SimpleNamespace(
            mm_processor=make_processor(parser),
            _readonly_mm_processor=make_processor(Qwen2VLMultiModalDataParser(spatial_merge_size=2)),
        )
        config = SimpleNamespace(
            ec_transfer_config=None if consumer is None else SimpleNamespace(is_ec_consumer=consumer),
            model_config=SimpleNamespace(multimodal_config=mm_config),
        )
        with patch.object(metadata_patch, "_original_renderer_init"):
            metadata_patch._renderer_init(renderer, config, None)
        return renderer

    def test_consumer_grid_only_round_trip(self):
        renderer = self.renderer()
        factory = _create_qwen2vl_field_factory(2)
        renderer.mm_processor._get_mm_fields_config = lambda inputs, kwargs: factory(inputs)
        data = _merge_embeds(
            [{"image_grid_thw": torch.tensor([1, 32, 32])}, {"image_grid_thw": torch.tensor([1, 16, 32])}],
            renderer.mm_processor,
        )
        self.assertEqual(data["image_grid_thw"].shape, (2, 3))
        for processor in (renderer.mm_processor, renderer._readonly_mm_processor):
            self.assertIsNot(processor.data_parser, processor.info.data_parser)
            items = processor.info.parse_mm_data({"image": data})["image"]
            self.assertEqual(items.get_count(), 2)
            self.assertEqual(items.get_processor_data(), {})
            self.assertEqual(items.get(0)["image_grid_thw"].tolist(), [1, 32, 32])
            self.assertNotIn("image_embeds", items.get_passthrough_data())

    def test_other_renderers_still_require_embeddings(self):
        for consumer, enabled in ((False, True), (None, True), (True, False)):
            with self.subTest(consumer=consumer, enabled=enabled):
                info = self.renderer(consumer=consumer, enabled=enabled).mm_processor.info
                with self.assertRaises(ValueError):
                    info.parse_mm_data({"image": {"image_grid_thw": torch.tensor([[1, 32, 32]])}})
        generic = MultiModalDataParser()
        self.renderer(parser=generic)
        self.assertNotIn("_parse_image_data", generic.__dict__)

    def test_consumer_preserves_full_embeddings_and_requires_grid(self):
        info = self.renderer().mm_processor.info
        with self.assertRaises(ValueError):
            info.parse_mm_data({"image": {}})
        embeds = torch.zeros((256, 8))
        data = {"image_grid_thw": torch.tensor([[1, 32, 32]]), "image_embeds": embeds}
        items = info.parse_mm_data({"image": data})["image"]
        self.assertIs(items.get_passthrough_data()["image_embeds"], embeds)


if __name__ == "__main__":
    unittest.main()
