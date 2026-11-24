# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _indexpool_fp8_kernel(source, destination, WIDTH: tl.constexpr, STAGES: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, WIDTH)
    x = tl.load(source + row * WIDTH + lane).to(tl.float32)
    for stage in tl.static_range(STAGES):
        bit = 1 << stage
        pair = tl.gather(x, lane ^ bit, axis=0)
        x = tl.where((lane & bit) == 0, x + pair, pair - x)
    x = x * (WIDTH**-0.5)

    # Match the power-of-two scale and round-to-nearest-even E4M3 reference.
    row_max = tl.maximum(tl.max(tl.abs(x), 0), 1e-4) / 448.0
    scale = tl.exp2(tl.ceil(tl.log2(row_max)))
    magnitude = tl.minimum(tl.abs(x / scale), 448.0)
    unit = tl.exp2(tl.floor(tl.log2(tl.maximum(magnitude, 2.0**-6))) - 3.0)
    units = magnitude / unit
    lower = tl.floor(units)
    fraction = units - lower
    round_up = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    value = (lower + round_up.to(tl.float32)) * unit * scale
    tl.store(destination + row * WIDTH + lane, tl.where(x < 0, -value, value))


def fused_indexpool_fp8(vectors: torch.Tensor) -> torch.Tensor:
    width = vectors.shape[-1]
    result = torch.empty_like(vectors, dtype=torch.float32)
    _indexpool_fp8_kernel[(vectors.numel() // width,)](vectors, result, width, width.bit_length() - 1)
    return result


@triton.jit
def _gather_paged_rows_kernel(
    cache,
    block_table,
    positions,
    output,
    ROWS: tl.constexpr,
    CACHE_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    COLUMN: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLUMNS: tl.constexpr,
):
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    col = tl.arange(0, TILE_COLUMNS)
    position = tl.load(positions + row, row < ROWS, other=-1).to(tl.int32)
    live = position >= 0
    position = tl.maximum(position, 0)
    page = tl.load(block_table + position // PAGE_SIZE, live, other=0).to(tl.int64)
    slot = page * PAGE_SIZE + position % PAGE_SIZE
    values = tl.load(
        cache + slot[:, None] * CACHE_WIDTH + COLUMN + col[None, :],
        live[:, None] & (col[None, :] < WIDTH),
        other=0,
    )
    tl.store(
        output + row[:, None] * WIDTH + col[None, :],
        values.to(output.dtype.element_ty),
        (row[:, None] < ROWS) & (col[None, :] < WIDTH),
    )


def gather_paged_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    column: int,
    width: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read just the requested cache field and zero masked positions."""
    rows = positions.numel()
    tile_rows = 32 if width <= 128 else 16
    result = torch.empty((*positions.shape, width), dtype=dtype, device=cache.device)
    _gather_paged_rows_kernel[(triton.cdiv(rows, tile_rows),)](
        cache,
        block_table,
        positions,
        result,
        rows,
        cache.shape[-1],
        cache.shape[2],
        column,
        width,
        tile_rows,
        triton.next_power_of_2(width),
    )
    return result
