import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import triton.testing

from activations.softmax import softmax

SHAPES = [
    (1024, 4096),
    (512, 65536),
    (32, 128, 512),
    (8, 16, 512, 512),
]

print("| Shape | Max Error | Triton (ms) | PyTorch (ms) | Speedup |")
print("|-------|-----------|-------------|--------------|---------|")

for shape in SHAPES:
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    triton_out = softmax(x)
    torch_out = torch.softmax(x, dim=-1)
    err = (triton_out - torch_out).abs().max().item()
    x2d = x.contiguous().view(-1, x.shape[-1])
    t_triton = triton.testing.do_bench(lambda: softmax(x2d))
    t_torch = triton.testing.do_bench(lambda: torch.softmax(x2d, dim=-1))
    speedup = t_torch / t_triton

    print(
        f"| {str(shape):20} | {err:.2e} | {t_triton:.4f} | {t_torch:.4f} | {speedup:.2f}x |"
    )
