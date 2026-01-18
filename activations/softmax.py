import torch
import triton
import triton.language as tl
import triton.testing

# Tiled Online Softmax (Two-Pass) with Software Pipelining & Memory Coalescing


@triton.autotune(
    configs=[
        triton.Config({"TILE_SIZE": 1024}, num_warps=4),
        triton.Config({"TILE_SIZE": 2048}, num_warps=8),
        triton.Config({"TILE_SIZE": 4096}, num_warps=8),
        triton.Config({"TILE_SIZE": 8192}, num_warps=16),
    ],
    key=["n_cols"],
)
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    row_start = input_ptr + pid * input_row_stride
    output_row_start = output_ptr + pid * output_row_stride

    running_max = tl.zeros([], dtype=tl.float32) - float("inf")
    running_sum = tl.zeros([], dtype=tl.float32)
    for tile_start in tl.range(0, n_cols, TILE_SIZE, num_stages=3):
        col_offset = tile_start + tl.arange(0, TILE_SIZE)
        mask = col_offset < n_cols
        tile = tl.load(row_start + col_offset, mask=mask, other=-float("inf"))
        tile = tile.to(tl.float32)
        tile_max = tl.max(tile, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.exp(tile - new_max), axis=0)
        running_max = new_max

    for tile_start in tl.range(0, n_cols, TILE_SIZE, num_stages=3):
        col_offset = tile_start + tl.arange(0, TILE_SIZE)
        mask = col_offset < n_cols
        tile = tl.load(row_start + col_offset, mask=mask, other=-float("inf"))
        tile = tile.to(tl.float32)
        output = tl.exp(tile - running_max) / running_sum
        tl.store(output_row_start + col_offset, output.to(tl.float16), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    # wrapper
    original_shape = x.shape
    x = x.contiguous().view(-1, x.shape[-1])
    n_rows, n_cols = x.shape
    input_row_stride = x.stride(0)

    output = torch.empty_like(x)
    output_row_stride = output.stride(0)

    grid = (n_rows,)

    softmax_kernel[grid](
        output,
        x,
        input_row_stride,
        output_row_stride,
        n_cols,
    )

    return output.view(original_shape)
