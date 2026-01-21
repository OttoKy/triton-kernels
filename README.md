
# Triton Kernels

Triton kernels for deep learning primitives. Learning GPU kernel development through Triton.
## Kernels

| Kernel | Status | Notes |
|--------|--------|-------|
| Softmax | ✅ | Tiled online softmax with autotuning |
| LayerNorm | ✅ | Single-pass for small rows, tiled Welford for large |

## Benchmarks

Tested on RTX 5070 Ti, CUDA 13.0
```bash
PYTHONPATH=. python benchmarks/run_benchmarks.py
```

### Softmax (fp16)

| Shape | Max Error | Triton (ms) | PyTorch (ms) |
|-------|-----------|-------------|--------------|
| (1024, 4096) | 1.91e-06 | 0.0267 | 0.0437 |
| (512, 65536) | 2.38e-07 | 0.2635 | 0.2040 |
| (32, 128, 512) | 1.53e-05 | 0.0164 | 0.0199 |
| (8, 16, 512, 512) | 1.53e-05 | 0.2196 | 0.1861 |

### LayerNorm (fp16)

| Shape | Max Error | Triton (ms) | PyTorch (ms) |
|-------|-----------|-------------|--------------|
| (1024, 4096) | 1.95e-03 | 0.0251 | 0.0363 |
| (512, 65536) | 1.95e-03 | 0.2666 | 0.3015 |
| (32, 128, 512) | 1.95e-03 | 0.0142 | 0.0192 |
| (8, 16, 512, 512) | 1.95e-03 | 0.1942 | 0.2069 |