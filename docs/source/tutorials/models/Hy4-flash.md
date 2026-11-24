# Hy4 Flash on Ascend 910C

The Atlas A3 implementation serves the [Tencent Hy4 Flash BF16 checkpoint](https://huggingface.co/tencent/Hy4-flash)
and ModelSlim `W8A8_DYNAMIC` checkpoints with model runner V2. Both weight formats support eager execution
and `FULL_DECODE_ONLY` ACLGraph with BF16 paged KV, sliding-window and IndexPool attention, learned attention
sinks, and TP-sharded Engram tables. The Tencent checkpoint includes a vision tower for image and video inputs.

## Prepare vLLM

Use the Atlas A3 software environment for this vLLM Ascend branch. Apply the companion patch before
installing vLLM from the `v0.32.0` tag. The patch supplies the checkpoint configuration,
shared model and weight loading code, and multimodal processor. This patch can be dropped once the
supported upstream vLLM release provides these interfaces.

```bash
ASCEND_SRC="$(pwd)"  # Run from the vllm-ascend checkout.
VLLM_SRC="../vllm-hy4flash"
git clone --branch v0.32.0 --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_SRC"
git -C "$VLLM_SRC" apply "$ASCEND_SRC/patches/hy4flash-vllm-v0.32.0.patch"
VLLM_TARGET_DEVICE=empty python3 -m pip install -e "$VLLM_SRC"
python3 -m pip install -e "$ASCEND_SRC"
```

## Start the service

Set `MODEL_PATH` to a local ModelSlim `W8A8_DYNAMIC` checkpoint with its `quant_model_description.json`.
Keep Engram and the IndexPool `key_value_projection` and `key_gate_projection` tensors in their original
precision. Use symmetric INT8 weights with one weight scale per output channel. Weight offsets can be omitted.
Any supplied offset must be zero. A packed projection omitted from the quantization description runs in BF16.
When describing split projections, include every shard with the same quantization type.
For the BF16 checkpoint, omit `--quantization ascend`. The command uses 16 visible Atlas A3 devices
and stores the Engram tables in host memory. The BF16 checkpoint's shard index records about 441 GB of weights.

```bash
export VLLM_VERSION=0.32.0
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve "$MODEL_PATH" \
  --served-model-name hy4-flash \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --quantization ascend \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4
```

For eager execution, replace `--compilation-config` and its value with `--enforce-eager`. ACLGraph captures
eligible batches with one decode token per request. Prefill, mixed batches, and uncaptured sizes run in
eager mode. Both paths cache each completed IndexPool group's key in its final token's cache row.
Graph replay projects at most one group per request and scores cached group keys at the captured context bound.

Use greedy decoding to compare eager and graph outputs from the same checkpoint. Include short prompts,
text with images, and prompts longer than 2048 tokens to exercise IndexPool recall. Check the first
generated token and subsequent decode tokens. The MRV2 log emits `ACL graph replay is active for the V2 target model`
when it first replays a graph.

This path requires PP, PCP and DCP of 1 and speculative decoding, KV transfer, FlashComm and fused MC2 disabled.
