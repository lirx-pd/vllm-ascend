# Hy4 Flash on Ascend 910C

The Atlas A3 implementation serves the BF16 [Tencent Hy4 Flash checkpoint](https://huggingface.co/tencent/Hy4-flash)
with model runner V2 in eager mode. It uses BF16 paged KV, sliding-window and IndexPool attention,
the learned attention sink, and TP-sharded Engram tables. The released checkpoint includes a vision
tower for image and video inputs.

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

Set `MODEL_PATH` to the local BF16 checkpoint directory. The command uses a single Atlas A3
node with 16 visible devices. The checkpoint's shard index records about 441 GB of weights.
Engram tables use host memory with the configuration below.

```bash
export VLLM_VERSION=0.32.0
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve "$MODEL_PATH" \
  --served-model-name hy4-flash \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --enforce-eager \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4
```

Use greedy decoding to compare short prompts, text with images, and prompts longer than
2048 tokens against the reference implementation. Longer prompts exercise IndexPool recall and
the smaller batch token budget exercises chunked prefill. Check both the first generated token
and subsequent decode tokens.

This path requires PP, PCP and DCP of 1. Graph capture, FP8 weights or KV, speculative decoding,
PD disaggregation, and FlashComm are outside the 910C eager implementation.
