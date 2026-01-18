
# GPU Kernels


Custom Triton kernels for deep learning primitives


## Kernels


| Kernel | Status | Notes |

|--------|--------|-------|

| Softmax | ✅ | Tiled online softmax (two-pass) with autotuning |




## Benchmarks


Tested on RTX 5070 Ti, CUDA 13.0


## Benchmarks

Tested on RTX 5070 Ti, CUDA 13.0
```bash
PYTHONPATH=. python benchmarks/run_benchmarks.py
```

### Softmax (fp16)

| Shape | Max Error | Triton (ms) | PyTorch (ms) | Speedup |
|-------|-----------|-------------|--------------|---------|
| (1024, 4096) | 1.91e-06 | 0.0267 | 0.0437 | 1.64x |
| (512, 65536) | 2.38e-07 | 0.2635 | 0.2040 | 0.77x |
| (32, 128, 512) | 1.53e-05 | 0.0164 | 0.0199 | 1.21x |
| (8, 16, 512, 512) | 1.53e-05 | 0.2196 | 0.1861 | 0.85x |

*Note: PyTorch may outperform on very wide rows (65k+). This kernel is a tiled two-pass baseline.*