# Nsight summary

GPU Tesla T4, driver 580.178.04
580.178.04, CUDA 12.8, TensorRT 10.16.1.11, torch 2.11.0+cu128. Produced on Kaggle by `kaggle/run_trt.py` in a separate run of the same code and environment as `tensorrt_bench.json` (the first run's TensorRT trace used a batch outside the engine's profile); trace files are not committed.

### Custom median3x3 kernel (tiled), batch 1024, 50 launches (Nsight Systems)

| Time % | Total ms | Calls | Avg us | Kernel |
|---|---|---|---|---|
| 100.0 | 30.04 | 60 | 500.7 | `median3x3_tiled_kernel(const float *, float *, int, int)` |

### TensorRT FP16 engine, batch 128, 50 runs (Nsight Systems)

| Time % | Total ms | Calls | Avg us | Kernel |
|---|---|---|---|---|
| 20.7 | 20.03 | 60 | 333.9 | `sm75_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x...` |
| 18.7 | 18.13 | 180 | 100.7 | `sm50_xmma_pooling_coalescedC_NHWC_kMAX_2_False_execute_kernel_trt` |
| 17.9 | 17.33 | 60 | 288.9 | `trt_turing_h1688cudnn_256x64_ldg8_relu_exp_small_nhwc_tn_v1` |
| 17.4 | 16.81 | 60 | 280.2 | `trt_turing_h1688cudnn_256x128_ldg8_relu_exp_small_nhwc_tn_v1` |
| 15.4 | 14.88 | 60 | 248.0 | `trt_turing_h1688cudnn_128x128_ldg8_relu_exp_small_nhwc_tn_v1` |
| 5.4 | 5.25 | 60 | 87.5 | `void genericReformat::copyPackedKernel<float, __half, (bool)1, (bool)1, genericReformat::A...` |
| 3.1 | 2.99 | 60 | 49.9 | `sm50_xmma_pooling_coalescedC_NHWC_kMAX_3_False_execute_kernel_trt` |
| 0.7 | 0.72 | 60 | 12.1 | `sm75_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage1_warpsize2x2x1_tensor16x8x8_a...` |

### Nsight Compute: median3x3_tiled_kernel (one launch)

(no metrics parsed)

ncu output tail (exit 0):
```
==WARNING== No metrics to collect found in sections.
==PROF== Connected to process 1484 (/usr/bin/python3.13)
==PROF== Profiling "median3x3_tiled_kernel": 0%....50%....100% - 1 pass
==PROF== Disconnected from process 1484
[1484] python3.13@127.0.0.1
  median3x3_tiled_kernel(const float *, float *, int, int) (4, 4, 1024)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 7.5

```
