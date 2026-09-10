# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run under `npu run --gpus 2 -- pytest .../test_ec_mooncake_v2.py`."""

import base64
import io
import json
import os
from contextlib import ExitStack

import pytest
import requests
from PIL import Image
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import DisaggEpdProxy, RemoteOpenAIServer


@pytest.mark.parametrize("runner_version", ["0", "1"], ids=["v1", "v2"])
def test_ec_mooncake_v2_repeated_image(runner_version):
    devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    assert len(devices) >= 2, "Reserve two NPUs before running this test."
    model = os.environ.get("QWEN35_DENSE_MODEL", "Qwen/Qwen3.5-9B")
    reservation_port = get_open_port()
    common = [
        "--served-model-name",
        "ec-test",
        "--enforce-eager",
        "--max-model-len",
        "4096",
        "--max-num-batched-tokens",
        "4096",
        "--max-num-seqs",
        "4",
        "--no-enable-prefix-caching",
        "--limit-mm-per-prompt",
        '{"image":1,"video":0}',
        "--dtype",
        "bfloat16",
        "--mamba-ssm-cache-dtype",
        "bfloat16",
    ]
    image = io.BytesIO()
    Image.new("RGB", (224, 224), "red").save(image, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()

    with ExitStack() as stack:
        servers = []
        for index, role in enumerate(("ec_producer", "ec_consumer")):
            port = get_open_port()
            extra = {"mooncake_protocol": "ascend"}
            if role == "ec_consumer":
                extra["reservation_zmq_port"] = reservation_port
            connector = {
                "ec_connector": "ECMooncakeConnector",
                "ec_connector_module_path": "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake_ec_connector",
                "ec_role": role,
                "ec_buffer_device": "npu",
                "ec_buffer_size": 64 * 1024 * 1024,
                "ec_connector_extra_config": extra,
            }
            role_args = ["--port", str(port), "--gpu-memory-utilization", "0.05" if index == 0 else "0.8"]
            if role == "ec_consumer":
                role_args += ["--enable-mm-embeds"]
            servers.append(
                stack.enter_context(
                    RemoteOpenAIServer(
                        model,
                        common + role_args + ["--ec-transfer-config", json.dumps(connector)],
                        server_host="127.0.0.1",
                        server_port=port,
                        auto_port=False,
                        max_wait_seconds=600,
                        env_dict={
                            "VLLM_USE_V2_MODEL_RUNNER": runner_version,
                            "ASCEND_RT_VISIBLE_DEVICES": devices[index],
                            "OMP_NUM_THREADS": "8" if index == 0 else "1",
                        },
                    )
                )
            )
        proxy = stack.enter_context(
            DisaggEpdProxy(
                proxy_args=[
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(get_open_port()),
                    "--encode-servers-urls",
                    servers[0].url_root,
                    "--prefill-servers-urls",
                    "disable",
                    "--decode-servers-urls",
                    servers[1].url_root,
                    "--ec-consumer-zmq-addrs",
                    f"tcp://127.0.0.1:{reservation_port}",
                    "--decode-retries",
                    "0",
                ],
                server_host="127.0.0.1",
            )
        )
        # Repeat the image to exercise cache reuse across distinct requests.
        for _ in range(2):
            response = requests.post(
                proxy.url_for("v1", "chat", "completions"),
                json={
                    "model": "ec-test",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url}},
                                {"type": "text", "text": "What color is this image? Answer with one word."},
                            ],
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 16,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=120,
            )
            response.raise_for_status()
            assert "red" in response.json()["choices"][0]["message"]["content"].lower()
