# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import json
from pathlib import Path

import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import DisaggEpdProxy, RemoteEPDServer
from tests.e2e.nightly.single_node.models.scripts.single_node_config import SingleNodeConfig
from tools.send_mm_request import send_image_request

MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
]
TENSOR_PARALLELS = [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("use_v2", [False, True], ids=["mrv1", "mrv2"])
async def test_models(model: str, tp_size: int, use_v2: bool, tmp_path: Path) -> None:
    encode_port = get_open_port()
    pd_port = get_open_port()
    ec_configs = []
    for role in ("ec_producer", "ec_consumer"):
        ec_config = {
            "ec_connector": "ECExampleConnector",
            "ec_role": role,
            "ec_connector_extra_config": {"shared_storage_path": str(tmp_path / "ec_cache")},
        }
        ec_configs.append(json.dumps(ec_config))
    vllm_server_args = [
        [
            "--port",
            str(encode_port),
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.01",
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--max-model-len",
            "10000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "1",
            "--ec-transfer-config",
            ec_configs[0],
        ],
        [
            "--port",
            str(pd_port),
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.95",
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--max-model-len",
            "10000",
            "--max-num-batched-tokens",
            "10000",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            ec_configs[1],
        ],
    ]
    proxy_port = get_open_port()
    proxy_args = [
        "--host",
        "127.0.0.1",
        "--port",
        str(proxy_port),
        "--encode-servers-urls",
        f"http://localhost:{encode_port}",
        "--decode-servers-urls",
        f"http://localhost:{pd_port}",
        "--prefill-servers-urls",
        "disable",
    ]

    with (
        RemoteEPDServer(
            vllm_serve_args=vllm_server_args,
            env_dict={"VLLM_USE_V2_MODEL_RUNNER": str(int(use_v2))},
        ),
        DisaggEpdProxy(proxy_args=proxy_args) as proxy,
    ):
        config = SingleNodeConfig(
            name=model,
            model=model,
            mm_request={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is the content of this image?"},
                            {"type": "image_url", "image_url": {"url": "{IMAGE_0}"}},
                        ],
                    }
                ],
                "api_args": {
                    "eos_token_id": [1, 106],
                    "pad_token_id": 0,
                    "top_k": 64,
                    "top_p": 0.95,
                    "max_tokens": 8192,
                    "stream": False,
                },
            },
        )
        send_image_request(config, proxy)
