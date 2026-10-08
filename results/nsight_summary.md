# Nsight summary

GPU Tesla T4, driver 580.178.04
580.178.04, CUDA 12.8, TensorRT 10.16.1.11, torch 2.11.0+cu128. Produced on Kaggle by `kaggle/run_trt.py`; trace files are not committed.

### Custom median3x3 kernel (tiled), batch 1024, 50 launches (Nsight Systems)

| Time % | Total ms | Calls | Avg us | Kernel |
|---|---|---|---|---|
| 100.0 | 30.09 | 60 | 501.5 | `median3x3_tiled_kernel(const float *, float *, int, int)` |

### TensorRT FP16 engine, batch 1024, 50 runs (Nsight Systems)

| Time % | Total ms | Calls | Avg us | Kernel |
|---|---|---|---|---|
| 100.0 | 0.01 | 3 | 3.0 | `void cask_trt::computeOffsetsKernel<(bool)0, (bool)0>(cask_trt::ComputeOffsetsParams)` |

### Nsight Compute: median3x3_tiled_kernel (one launch)

(no metrics parsed)
