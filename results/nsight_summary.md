# Nsight summary

GPU Tesla T4, driver 580.178.04
580.178.04, CUDA 12.8, TensorRT 10.16.1.11, torch 2.11.0+cu128. Produced on Kaggle by `kaggle/run_nsight.py` (a profiling-only run of the same code and environment as `tensorrt_bench.json`); trace files are not committed. Nsight Compute cannot run on Kaggle: the container is not allowed to read the GPU performance counters (ERR_NVGPUCTRPERM below), so no occupancy or memory-throughput metrics could be collected.

### Custom median3x3 kernel (tiled), batch 1024, 50 launches (Nsight Systems)

| Time % | Total ms | Calls | Avg us | Kernel |
|---|---|---|---|---|
| 100.0 | 30.00 | 60 | 500.1 | `median3x3_tiled_kernel(const float *, float *, int, int)` |

### TensorRT FP16 engine, batch 128, 50 runs (Nsight Systems)

| Time % | Total ms | Calls | Avg us | Kernel |
|---|---|---|---|---|
| 20.1 | 19.94 | 60 | 332.4 | `sm75_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x...` |
| 18.3 | 18.11 | 60 | 301.8 | `trt_turing_h1688cudnn_256x64_ldg8_relu_exp_small_nhwc_tn_v1` |
| 17.6 | 17.45 | 60 | 290.8 | `trt_turing_h1688cudnn_256x128_ldg8_relu_exp_small_nhwc_tn_v1` |
| 17.0 | 16.87 | 120 | 140.5 | `sm50_xmma_pooling_coalescedC_NHWC_kMAX_3_False_execute_kernel_trt` |
| 15.7 | 15.54 | 60 | 259.1 | `trt_turing_h1688cudnn_128x128_ldg8_relu_exp_small_nhwc_tn_v1` |
| 5.5 | 5.42 | 60 | 90.4 | `void genericReformat::copyPackedKernel<float, __half, (bool)1, (bool)1, genericReformat::A...` |
| 4.4 | 4.33 | 120 | 36.1 | `sm50_xmma_pooling_coalescedC_NHWC_kMAX_2_False_execute_kernel_trt` |
| 0.8 | 0.75 | 60 | 12.4 | `sm75_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage1_warpsize2x2x1_tensor16x8x8_a...` |

### Nsight Compute: median3x3_tiled_kernel (one launch)

(no metrics parsed)

ncu output tail (exit 1):
```
==PROF== Connected to process 1258 (/usr/bin/python3.13)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
==PROF== Disconnected from process 1258

```
