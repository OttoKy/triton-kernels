import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

# General purpose LayerNorm kernel with 2 paths, single and tiled (Welford)


@triton.autotune(
    configs=[
        triton.Config({"TILE_SIZE": 64}, num_warps=1, num_stages=2),
        triton.Config({"TILE_SIZE": 128}, num_warps=2, num_stages=2),
        triton.Config({"TILE_SIZE": 256}, num_warps=2, num_stages=2),
        triton.Config({"TILE_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"TILE_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"TILE_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"TILE_SIZE": 4096}, num_warps=8, num_stages=4),
    ],
    key=["n_cols"],
)
@triton.jit
def layerNorm_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    gamma_ptr,
    beta_ptr,
    n_cols,
    eps,
    TILE_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    row_start = input_ptr + pid * input_row_stride
    output_row_start = output_ptr + pid * output_row_stride

    if TILE_SIZE >= n_cols:
        col_offset = tl.arange(0, TILE_SIZE)
        mask = col_offset < n_cols

        x = tl.load(row_start + col_offset, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + col_offset, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + col_offset, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x) / n_cols
        diff = tl.where(mask, x - mean, 0.0)
        var = tl.sum(diff * diff) / n_cols

        out = (x - mean) * tl.rsqrt(var + eps) * gamma + beta
        tl.store(output_row_start + col_offset, out.to(tl.float16), mask=mask)
        return

    global_count = tl.full((), 0.0, tl.float32)
    global_mean = tl.full((), 0.0, tl.float32)
    global_m2 = tl.full((), 0.0, tl.float32)

    for tile_start in tl.range(0, n_cols, TILE_SIZE):
        col_offset = tile_start + tl.arange(0, TILE_SIZE)
        mask = col_offset < n_cols
        tile = tl.load(row_start + col_offset, mask=mask, other=0.0).to(tl.float32)

        tile_count = tl.minimum(TILE_SIZE, n_cols - tile_start).to(tl.float32)
        tile_mean = tl.sum(tile) / tile_count
        diff = tl.where(mask, tile - tile_mean, 0.0)
        tile_m2 = tl.sum(diff * diff)

        delta = tile_mean - global_mean
        new_count = global_count + tile_count
        global_mean = global_mean + delta * (tile_count / new_count)
        global_m2 = (
            global_m2 + tile_m2 + delta * delta * global_count * tile_count / new_count
        )
        global_count = new_count

    variance = global_m2 / global_count
    inv_std = tl.rsqrt(variance + eps)

    for tile_start in tl.range(0, n_cols, TILE_SIZE):
        col_offset = tile_start + tl.arange(0, TILE_SIZE)
        mask = col_offset < n_cols
        tile = tl.load(row_start + col_offset, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + col_offset, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + col_offset, mask=mask, other=0.0).to(tl.float32)

        out = (tile - global_mean) * inv_std * gamma + beta
        tl.store(output_row_start + col_offset, out.to(tl.float16), mask=mask)


# wrapper
def layernorm(x, gamma, beta, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, orig_shape[-1])
    n_rows, n_cols = x_2d.shape

    output = torch.empty_like(x_2d)

    layerNorm_kernel[(n_rows,)](
        output,
        x_2d,
        x_2d.stride(0),
        output.stride(0),
        gamma,
        beta,
        n_cols,
        eps,
    )

    return output.view(orig_shape)


def layernorm_torch_flat(x, gamma, beta, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, orig_shape[-1])
    y_2d = F.layer_norm(x_2d, (x_2d.shape[-1],), gamma, beta, eps=eps)
    return y_2d.view(orig_shape)
