# Interview notes: how it works and what I measured

Plain-words answers to the questions this project invites. Numbers come from `results/*.json`
(see the README tables; nothing here is rounded differently from those files).

## The problem in one breath

A wafer map is a grid of dies, each marked pass or fail after electrical test. The *pattern* of
failures tells a process engineer what went wrong: a ring at the edge points to an edge-bead or
etch problem, a scratch to handling, a centre blob to a deposition non-uniformity. WM-811K is
811,457 real wafer maps from a fab; 172,950 are labelled with one of nine patterns, and 85% of
those are "none". The task is to classify the pattern so yield engineers can triage lots without
looking at every map.

## How I split the data, and why by lot

Wafers from the same lot went through the same process window and look alike. A random
wafer-level split would put near-duplicates on both sides and inflate the test score. So I assign
whole lots to train, validation or test, stratified by each lot's majority label so every class is
represented in every split. The test set has 34,769 wafers; the rarest class (Near-full) has 30
of them, which is why per-class numbers for it are noisy and I say so.

## The custom CUDA kernels: thread and block layout

All three kernels use the same layout: a 2-D block of 16x16 threads computes a 16x16 output tile,
`grid.x` and `grid.y` tile the image, and `grid.z` is the batch index, so one launch processes a
whole batch of 1,024 wafers. Inside a block, consecutive threads in a warp own consecutive
columns of one row, so their global loads and stores touch consecutive addresses: the memory
accesses coalesce into a few 128-byte transactions per warp instead of 32 separate ones.

**Preprocess** takes variable-size wafer maps (26x26 up to 300x202) packed into one flat uint8
buffer with per-wafer height, width and offset tables. The resize kernel maps each output pixel
back to a source pixel with the "nearest-exact" rule (`src = floor((dst + 0.5) * in / out)`), the
same rule PyTorch uses, which is what made the exact comparison possible. A second kernel then
standardises each wafer: one block per wafer, each thread accumulates partial sums of x and x²
over a strided range of pixels, a shared-memory tree reduction produces the mean and variance,
and the same threads write `(x - mean) / (std + eps)` back. The whole batch is two launches; the
PyTorch path has to loop over wafers because `F.interpolate` wants one size per call.

**conv3x3 and median3x3** come in two versions each. The *naive* kernel reads its 3x3
neighbourhood straight from global memory, nine loads per output pixel, with the boundary handled
per pixel (zero padding for the convolution, replicate padding for the median). The *tiled*
kernel first has every thread load one pixel of an 18x18 shared-memory tile (the 16x16 output
tile plus a one-pixel halo; the threads on the edge load the halo too), calls `__syncthreads()`,
and then reads its nine neighbours from shared memory. Each input pixel is read from global
memory once per tile instead of up to nine times.

The median itself is a fixed 19-comparison sorting network on the nine values: no branches, no
local-memory array, every thread does identical work, which keeps the warp converged.

## Why tiling did not win here, and when it would

The honest result in the benchmark is that for a 3x3 window on 64x64 maps the naive kernels are
as fast or faster than the tiled ones. The reason is cache: a 64x64 float image is 16 KB, the T4
has 64 KB of L1 per SM, and the nine reads per pixel of the naive kernel mostly hit L1 after the
first. The tiled kernel pays for the halo loads, the `__syncthreads()` and the shared-memory
bank traffic, and gets back a reuse the cache was already giving it. Tiling pays when the
reuse per pixel is large relative to what L1 can hold: bigger windows (7x7 reads 49 pixels), bigger
images, or when several filters share the tile. I kept both kernels because the comparison is the
point; guessing that tiling helps is exactly what the benchmark is there to check.

cuDNN also beats both of my 3x3 convolutions. It uses implicit-GEMM and Winograd kernels tuned
per architecture; a hand-written direct convolution is a teaching exercise next to that. Where
the custom kernels win by orders of magnitude is where PyTorch has no fused operator: the median
filter (PyTorch's route is unfold to a 9xHxW tensor and a median reduction, which materialises
nine copies of the image) and the packed variable-size preprocess (PyTorch needs a Python loop
with one launch per wafer, so part of that gap is launch and interpreter overhead, and I say so
in the README).

## How I checked correctness

Every kernel has a PyTorch reference (`src/wafer/reference.py`), and the benchmark reports the
maximum absolute difference against it on real wafer batches before it reports any timing. The
CUDA pytest file runs the same checks on Kaggle (`results/cuda_tests.txt`). The first run caught a
real bug: the tiled median was off by exactly 1.0 on some edge pixels. In the halo load I wrote
`blockIdx.y * TILE + ly - 1` inside a clamp, and `blockIdx.y` is unsigned, so at the top edge
"row minus one" wrapped to a huge number and clamped to the *last* row instead of the first. A
cast to `int` before the arithmetic fixed it. The convolution had the same expression assigned to
an `int` first, which is why it never showed the bug. That is the kind of thing the reference
comparison exists for.

## Class imbalance and the sampler

With 85% "none", a model trained on the natural distribution learns to say "none" and scores 85%
accuracy while missing most defects. I sample with probability inverse to class frequency, so an
epoch sees roughly equal counts of every class, and the same sampler partitions the draw across
DDP ranks: each rank gets a disjoint slice of one balanced draw per epoch, reseeded per epoch.
The metric I optimise and report is macro-F1 (every class counts the same), with per-class recall
and the confusion matrix, not accuracy.

## DDP: what I did and what the scaling number means

`torchrun --nproc_per_node=2` starts one process per GPU. Each process holds a copy of the model,
trains on its own slice of the balanced draw, and `DistributedDataParallel` all-reduces the
gradients with NCCL after each backward pass, so both copies take the same step. The per-rank
batch is the same in both runs (256), so the global batch doubles on two GPUs. Throughput is
wafers processed by all ranks per second, averaged over epochs after the first (the first
includes cuDNN autotuning and allocator warm-up).

Scaling efficiency is (throughput with 2 GPUs / throughput with 1 GPU) / 2. I measured 90.3%,
i.e. 1.807x (10,060.9 vs 18,175.7 wafers/s). The missing 10% is the all-reduce of the gradients and the fact that with a 0.4M
parameter model and 64x64 inputs each step is short, so communication is a visible fraction of
it; a bigger model would amortise it better. Test macro-F1 was 0.860 on 1 GPU and 0.853 on 2
GPUs, within the run-to-run noise for six epochs (I ran each once; three seeds would be the
next step before claiming either is better).

## Things I would change at scale

- A mixed-precision data pipeline that keeps wafers as uint8 on the GPU and standardises in the
  kernel per batch, instead of pre-standardising the whole training set to float32 in host RAM.
- Gradient accumulation or a larger per-rank batch to raise the compute-to-communication ratio.
- A 7x7 or separable filter benchmark where shared-memory tiling should start to pay, and an
  Nsight Compute profile to show achieved occupancy and memory throughput rather than inferring
  them from wall time.
- Three seeds per configuration and confidence intervals on macro-F1.

## TensorRT: what I did and what the numbers say

I exported the trained model to ONNX (opset 17) because that is the format both ONNX Runtime and TensorRT consume; TensorRT then fuses layers, picks kernels for the T4 and, for INT8, inserts the quantisation that PyTorch eager does not do for inference. I ran a parity check before benchmarking: maximum absolute logit difference on the test set against PyTorch eager is 0.00005 for ONNX Runtime and 0.00004 for TensorRT FP32, both acceptable. An optimization profile in TensorRT tells the builder which batch sizes to expect and binds the engine to that range. I built one profile covering batch 1 to 128, so the same engine handles all sizes without rebuild, trading a small throughput cost for a single compact engine. INT8 quantization starts with entropy calibration: the builder runs the model on 512 training wafers, records a histogram of activations in each layer, then chooses the quantization scale that minimises KL divergence between the quantized and float distributions. Calibration data must come from the training split only, never validation or test, because the builder is learning the scale, not evaluating the model. The honest answer: at batch 32 TensorRT INT8 delivers 1.6x throughput over FP16 and 7.2x over PyTorch eager; at batch 128 it is 1.9x and 8.0x. At batch 1, INT8 at 0.140 ms vs FP16 at 0.164 ms is a modest edge because kernel launch overhead dominates the time. Accuracy is stable: INT8 macro-F1 0.8618 versus PyTorch 0.8601, a delta of 0.0017 indistinguishable from run-to-run noise, with 99.70% top-1 agreement on predictions. FP16 achieves 3x speedup over FP32 on a T4 because Turing has dedicated tensor cores for FP16x2 operations; INT8 adds another 1.5 to 1.9x on top of that by halving the data size and enabling int8 tensor-core paths. Batch 1 barely moves because even with zero computation the GPU must still launch the kernel, transfer control, and synchronize, a fixed cost that dominates a tiny batch. Next step: TensorRT 11 removed implicit quantization, so explicit Q/DQ (quantize/dequantize) node insertion is the path forward for INT8 with arbitrary scales.

## Limits, stated plainly

T4s, not A100s or H100s; one node; six epochs; one seed per run; a sleep-free benchmark but a
laptop-sized dataset; the preprocess comparison includes Python loop overhead on the PyTorch side.
