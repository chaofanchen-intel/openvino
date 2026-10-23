# Qwen3 (0.6B / 1.7B) prefill-TTFT optimizations for the Intel GPU plugin

Standalone Intel GPU plugin patches that reduce **cold-start prefill / TTFT** of
dense Qwen3 models on Intel integrated GPUs. opt-1..3 started from INT4 weight
compression (`group_size=128` first, per-output-channel second); opt-4..6 were
added for the FP16 and INT8 variants of Qwen3-0.6B and apply to every precision.

This is the dense-Qwen3 counterpart to [`../qwen3.5_patch/`](../qwen3.5_patch/),
which targets the hybrid SSM Qwen3.5 models. The two sets are disjoint: the
Qwen3.5 patches address Gated-DeltaNet / causal-conv1d op patterns that a dense
Qwen3 graph does not contain.

## Base

Patches are generated against OpenVINO 2026.4.1 release tag
`e213a14725769a6a2bf86fc86650aa78a9c05d3e` and apply with `git apply` to a
pristine `2026.4_base` tree.

Compatibility checks are recorded in
[VALIDATION_2026.4.1.md](https://github.com/chaofanchen-intel/openvino/blob/2026.4_base/benchmark/openvino/VALIDATION_2026.4.1.md).

```bash
git apply qwen3_patch/opt1_rms_block_access.patch
git apply qwen3_patch/opt2_dynamic_quantize_reuse.patch
git apply qwen3_patch/opt3_fuse_rope_into_rms.patch   # after opt1: shares rms_gpu_bfyx_opt.cl
git apply qwen3_patch/opt4_pa_kv_cache_vector_update.patch
git apply qwen3_patch/opt5_sdpa_micro_xe3_igpu_tile.patch
git apply qwen3_patch/opt6_fuse_uncompressed_qkv_fcs.patch
cmake --build <build-dir> --target openvino_intel_gpu_plugin -j
```

Apply them in order. opt-3 touches the same kernel as opt-1; opt-2 and opt-4..6
are independent of each other (opt-4..6 touch files none of opt-1..3 touch).
Each carries no runtime switch except opt-3, which can be turned off without a
rebuild via `OV_DISABLE_ROPE_RMS_FUSION=1` (used for A/B).

opt-1..4 and opt-6 are bit-exact. **opt-5 is not**: it re-tiles the attention
kernel, which changes rounding (details and a bit-exact variant below).

## Test setup

| | |
|---|---|
| GPU | Intel Graphics iGPU, Pantherlake-P, 32 XVE x 10 threads, 2.45 GHz, XMX present |
| Runtime | OpenVINO 2026.4.1 + GenAI 2026.4.1.0, `ATTENTION_BACKEND=PA` |
| Workload | Current rebenchmark: 1000 input / 512 output tokens, batch 1, greedy; historical TTFT results use `-ic 1` |
| Models | Current rebenchmark: both sizes, FP16 / INT8 / INT4-g128 / INT4-channel-wise; FP32 excluded |
| Driver | Current rebenchmark: OpenVINO GenAI Python and C++ `model_bench`; prompt permutation ON |

Prompt permutation must stay ON. PA's latency scheduler enables prefix caching,
so a repeated identical prompt is answered from cached KV blocks and reports a
TTFT that never ran a prefill — this silently understated every previously
recorded TTFT by 3-5x. The historical baseline-first source comparison is recorded
in
[`Qwen3-Optimization/docs/results.md`](../../../Qwen3-Optimization/docs/results.md).

## Where prefill time actually goes

Measured with Intel VTune `gpu-hotspots` on the real PA pipeline (not the
stateful SDPA graph, which `LLMPipeline` never runs). Qwen3-0.6B INT4-g128,
1000-token prefill. VTune's total GPU compute-task time is **142.3 ms/prefill**
against a measured TTFT of **140.6 ms**, i.e. prefill on this device is
essentially 100% GPU-bound with no meaningful host-side gap.

| Kernel family | ms/prefill | % | XVE active / stalled |
|---|---:|---:|---|
| `gemm_kernel` (oneDNN FC, incl. lm_head) | 78.39 | 55.1% | 32-42% / 55-65% |
| `sdpa_micro` prefill | 17.05 | 12.0% | 28% / 67% |
| `rms_gpu_bfyx_opt` | 16.32 | 11.5% | 37% / 58% |
| `dynamic_quantize_gpu_opt` | 12.44 | 8.7% | 31-43% / 53-66% |
| `rope_opt` | 9.73 | 6.8% | 30-36% / 60-65% |
| `pa_kv_cache_update_ref` | 7.22 | 5.1% | 24% / 72% |

Every kernel is 53-72% stalled; this part is memory-latency-bound throughout.

## The optimizations

| # | Patch | Hotspot | Files | Mechanism |
|---|---|---|---|---|
| opt-1 | `opt1_rms_block_access.patch` | `rms_gpu_bfyx_opt` scalar memory access | `kernel_selector/cl_kernels/rms_gpu_bfyx_opt.cl` | Two independent access-path defects that force scalar loads/stores instead of sub-group block transfers. Bit-exact. |
| opt-2 | `opt2_dynamic_quantize_reuse.patch` | 28 duplicate `DynamicQuantize` nodes | `graph/dynamic_quantize.cpp`, `plugin/transformations/dynamic_quantize_fully_connected.cpp` | `gate_proj` and `up_proj` consume the same `post_attention_layernorm` output but each got its own activation quantization. Reuse one when the quantization config matches field-for-field. Bit-exact. |
| opt-3 | `opt3_fuse_rope_into_rms.patch` | `rms_gpu_bfyx_opt` + `rope_opt` streaming the same tensor twice | `primitives/rms.hpp`, `graph/graph_optimizer/prepare_primitive_fusing.cpp`, `graph/impls/ocl/rms.cpp`, `kernel_selector/kernels/rms/*`, `kernel_selector/cl_kernels/rms_gpu_bfyx_opt.cl` | Fold the RotateHalf RoPE into the RMS epilogue, removing the intermediate `[tokens, heads, 128]` tensor entirely. Bit-exact. |
| opt-4 | `opt4_pa_kv_cache_vector_update.patch` | `pa_kv_cache_update_ref` scalar 2-byte stores into the transposed key cache | `graph/impls/ocl_v2/pa_kv_cache_update_ref.cl`, `graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp` | Full 16-token blocks of an uncompressed cache: transpose the block in registers and write each key channel's 16 tokens as one 32-byte store, copy value rows with sub-group block writes, and split key and value over two sub-groups. Bit-exact. FP16 only (INT8/INT4 caches are compressed and keep the old path). |
| opt-5 | `opt5_sdpa_micro_xe3_igpu_tile.patch` | `sdpa_micro` prefill tile tuned for a 128-XVE-core GPU, 128 coarse work groups on this 32-XVE one | `graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp` | For integrated Xe3, paged-attention prefill, head size 128: 32 queries x 128 keys on 8 sub-groups instead of 128 x 256 on 32. **Not bit-exact** (see opt-5 section). |
| opt-6 | `opt6_fuse_uncompressed_qkv_fcs.patch` | Q/K/V of an f16 model run as three small GEMMs | `plugin/transformations/fc_horizontal_fusion.{cpp,hpp}`, `plugin/transformations_pipeline.cpp`, unit tests | The existing horizontal fusion only knows compressed FCs. New pass: three bias-free constant-weight FCs on one activation become one FC + `VariadicSplit` (in-place crops). Bit-exact; FP16 only. |

### opt-1: restore vectorized access in the RMS kernel

Two separate bugs, both in guard expressions:

**(a) The block-write path was dead code for every shape.**

```c
#define USE_BLOCK_WRITE ((OUTPUT_TYPE_SIZE * OUTPUT_FEATURE_PITCH) & 0xF == 0)
```

In C, `==` binds tighter than `&`, so this parses as
`(OUTPUT_TYPE_SIZE * OUTPUT_FEATURE_PITCH) & (0xF == 0)`, i.e. `X & 0`, i.e.
**always 0**. The intended 16-byte alignment test never ran, and the
`BLOCK_WRITE` branch it guards was unreachable for *all* tensor shapes, not just
Qwen3's. Every RMS normalization in the plugin has been writing its output with
scalar stores.

**(b) A single sub-group was excluded from the block-read path.**

```c
if (workers_per_data > SUB_GROUP_SIZE)   // vectorized BLOCK_READ
...                                      // else: scalar strided loads
```

`SetDefault` picks the local size so each work-item handles ~8 elements, so the
normalized-axis width decides `workers_per_data`:

| Tensor | axis width | `workers_per_data` | path taken |
|---|---:|---:|---|
| `input_layernorm` / `post_attention_layernorm` `[1000, 1024]` | 1024 | 128 | vectorized |
| **`q_norm` `[1000, 16, 128]`** | 128 | **16** | **scalar** |
| **`k_norm` `[1000, 8, 128]`** | 128 | **16** | **scalar** |

Qwen3 applies RMSNorm per attention head (`head_dim = 128`), which lands exactly
on `workers_per_data == SUB_GROUP_SIZE` and falls off the fast path. The strict
`>` is unnecessary: with one sub-group `subgroup_offset` is 0 and
`BLOCK_READ<N>` covers `N * SUB_GROUP_SIZE` contiguous elements — exactly one
row. `>=` is correct.

The measured consequence, same kernel, same device:

| Tensor | before | after |
|---|---:|---:|
| `q_norm` / `k_norm` | 28 GB/s | 36 GB/s |
| `input_layernorm` | 56 GB/s | 64 GB/s |

**Correctness.** Bit-exact, not merely numerically close. Lane `l` accumulates
elements `l, l+16, l+32, ...` in that order on *both* paths — `BLOCK_READ<N>`
returns `vec[j] = input[base + j*SUB_GROUP_SIZE + lane]`, which is the same
element in the same loop position as the scalar load it replaces. Only the
instruction used to fetch it changes. All 39 GPU RMS unit tests
(`ov_gpu_unit_tests --gtest_filter='*rms*:*Rms*:*RMS*'`) pass.

Per-kernel effect (VTune, Qwen3-0.6B INT4-g128, ms/prefill):

| RMS instance | before | after | delta |
|---|---:|---:|---:|
| `q_norm` (GWS 16 x 16000) | 8.08 | 6.37 | −1.71 |
| `k_norm` (GWS 16 x 8000) | 4.06 | 3.25 | −0.81 |
| layernorms (GWS 128 x 1000) | 4.19 | 3.64 | −0.54 |
| **RMS total** | **16.32** | **13.26** | **−3.06** |
| whole prefill | 142.33 | 139.70 | −2.63 |

### opt-2: reuse one DynamicQuantize across FCs that share an activation

The plugin inserts an activation `DynamicQuantize` per compressed
FullyConnected. Qwen3's `q/k/v` are already horizontally fused into a single FC,
but `gate_proj` and `up_proj` are two FCs reading the *same*
`post_attention_layernorm` output, so the identical quantization ran twice --
28 redundant nodes, each a full read+write pass over a `[tokens, hidden]`
activation.

The fusion reuses one DQ when the quantization config matches field-for-field.
It also relaxes an assertion that assumed `users == outputs`: with several FCs
sharing one DQ the ratio becomes `outputs * num_consumers`, and the decode-time
"skip to f16 passthrough" is still valid because every consumer reads the
passthrough identically.

Measured on the PA graph: DynamicQuantize node count 141 -> 113. FP32 is
unaffected (no DQ at all), which is a useful sanity check in the table below.

### opt-3: fold the RoPE into the RMS epilogue

Qwen3 normalizes per attention head and then immediately rotates the same
`[tokens, heads, 128]` tensor:

```
q_proj -> reshape -> q_norm (RMS) -> reshape -> RoPE -> PagedAttention
```

Both kernels are pure memory traffic at the DRAM ceiling, so their cost is the
bytes they move: RMS reads `x` and writes `normalized`, then RoPE reads
`normalized` and writes `rotated`. The middle tensor exists only to hand the
value from one kernel to the next.

**Why it folds cleanly.** `rms_gpu_bfyx_opt` sizes its local work group so each
work item handles ~8 elements, so Qwen3's `head_dim = 128` gives `LWS == 16`
(one sub-group) with `SUBGROUP_BLOCK_SIZE == items_num == 8`. Under exactly that
shape `BLOCK_READ8` leaves `data[j]` holding the element at
`r = j*SUB_GROUP_SIZE + lane`, and since `HALF_ROTARY_NDIMS == 4*SUB_GROUP_SIZE`,
RotateHalf's partner of element `j` is element `j + 4` **in the same lane**. The
rotation therefore needs no cross-lane shuffle and no extra barrier, and cos/sin
arrive with one block read each. Any other `head_size` is left alone.

The pass walks through the rank-fixing `Reshape` between the two ops, and
derives the token index as `row_index / head_count` where the head count is the
product of every axis between the batch and the normalized one -- Qwen3 puts the
heads in **Y** (`[tokens, 1, 16, 128]`), not in F, which is the one detail that
has to be right or every token reads the wrong cos/sin row.

**Correctness.** The epilogue rounds to `OUTPUT_TYPE` before rotating -- that is
exactly what the standalone RoPE used to read back from memory -- and evaluates
the rotation in the input type, reproducing `rope_opt.cl` operation for
operation. Verified end to end: 4 prompts x 128 greedy tokens produce a
**byte-identical md5** to the unfused plugin on both Qwen3-0.6B INT4-g128
(`0f7fb0eaec35ce0f5a094e93cdaaad73`) and Qwen3-1.7B INT4 channel-wise
(`22e5ea17ec95a94b11975930c397d189`). All 39 RMS and 12 RoPE GPU unit tests
pass.

Measured on the PA graph: the 56 `rope_opt` dispatches disappear entirely; RMS
absorbs part of their cost (6.14 -> 7.08 ms by the OV counters) while 4.94 ms of
RoPE goes to zero.

### opt-4: write the paged key/value cache with vector stores

After opt-1..3 the VTune breakdown of Qwen3-0.6B **FP16** (1000-token prefill, 135.7 ms of GPU time) is:

| Kernel family | ms/prefill | % |
|---|---:|---:|
| `gemm_kernel` (oneDNN FC, incl. 7.4 ms `lm_head`) | 99.12 | 73% |
| `sdpa_micro` prefill | 17.25 | 13% |
| `rms_gpu_bfyx_opt` (with the fused RoPE) | 11.79 | 9% |
| `pa_kv_cache_update_ref` | 7.30 | 5% |

(INT8, 115.4 ms: GEMM 66.5, sdpa 17.1, RMS 14.2, `dynamic_quantize` 11.9, KV update 5.5.)

`pa_kv_cache_update_ref` stores the key cache transposed, `[block][kv_head][channel][16 tokens]`, so
the reference kernel issued one 2-byte scalar store per lane per token: 16 x 128 partial-line
stores per (block, head), 20 GB/s, 71% stalled. For a **full block of an uncompressed cache** the
patch holds 16 tokens x 4 channel chunks in registers, so every lane writes the 16 tokens of its
channel as a single 32-byte vector. Value rows (contiguous) become sub-group block writes. Key and
value are independent, so they run in two sub-groups (the dispatch is doubled along the head
axis). Partial blocks, the decode stage and compressed (INT8/INT4) caches keep the old code; the
host and kernel share one predicate (`use_vector_block_update` / `PA_KV_VEC_BLOCK`).

7.30 -> 5.50 ms of kernel time, FP16 TTFT -1.5 to -1.9 ms. Bit-exact (md5 of 6 prompts x 96 greedy
tokens identical on every model, see below). It does nothing for INT8/INT4 models: their default
KV cache is i8/u4 (by-channel keys), whose update path is unchanged and already sits at the DRAM floor
(5.5 ms; skipping either the key or the value half of the work only moves it 187 -> 103/112 us per
layer, i.e. it is bound by the memory traffic, not by the stores).

### opt-5: a sdpa_micro tile for a 4-Xe-core GPU

`sdpa_micro` is the second largest kernel (17 ms). Its tile configuration for paged-attention
prefill (`xehpc_h128 = {16, 64, 32, 16, 16, 2, 4, 8}`) comes from the Ponte Vecchio tuning: 128
queries x 256 keys per work group of 32 sub-groups. On this 32-XVE part that is 8 query tiles x 16
heads = 128 big work groups. A sweep of ~100 valid tile configurations
([`sdpa_config_sweep.py`](../../Qwen3-Optimization/scripts/tools/sdpa_config_sweep.py), run with a
temporary `OV_SDPA_CFG` hook, [`sdpa_config_env_override.patch`](../../Qwen3-Optimization/scripts/tools/sdpa_config_env_override.patch))
found a clear optimum at **32 queries x 128 keys on 8 sub-groups** (`{16, 32, 32, 16, 8, 1, 4, 2}`),
4x more and smaller work groups. It is applied only for `gpu_arch::xe3`, integrated, paged-attention
prefill, head size 128, non-quantized; every other device keeps its table entry.

| | 128x256 tile (before) | 32x128 tile | 32 x 256 bit-exact variant |
|---|---:|---:|---:|
| `sdpa_micro` 0.6B FP16 (VTune) | 17.25 ms | 12.77 ms | |
| FP16 TTFT (interleaved) | 132.2 | **128.2** | 129.6 |
| INT8 TTFT (interleaved) | 114.4 | **110.2** | 111.5 |

TTFT vs prompt length (FP16, same plugins): 128 tokens -0.5 ms, 512 -0.6, 2000 -11.5, 4000 -31.5. It
was neutral-to-better on every precision and on Qwen3-1.7B (-1.1% to -2.5%).

**It is not bit-exact.** The 128-key tile changes the order in which the online softmax accumulates
over the key axis, so results differ in the last bits. FP16 output is byte-identical on the 6-prompt
gate, but greedy decoding flips on near-ties for some long prompts of the quantized models (INT8 2/6
prompts, INT4 3/6, e.g. *"let's tackle this query"* vs *"let me try to figure out"*; 225 paged-attention/sdpa
unit tests pass). If bit-identical output matters more than ~1.3 ms, change the one table entry to
`{16, 64, 32, 16, 16, 1, 4, 4}`: that keeps the 256-key tile, was byte-identical to the unpatched
build on every model checked, and keeps ~60% of the gain (FP16 -2.6 ms, INT8 -2.9 ms).

### opt-6: fuse Q/K/V of uncompressed models

For compressed weights `FullyConnectedHorizontalFusion` already merges q/k/v into one GEMM
(that is why INT8/INT4 graphs show a single `..._fused_3FCs`). A plain f16 model never got that:
Q, K and V ran as three GEMMs of N = 2048, 1024, 1024. The new pass applies the same rewrite to bias-free
`FullyConnected` nodes with constant weights when exactly three share one activation (the
two-FC gate/up case is deliberately excluded: its split is on the innermost axis and costs a copy,
see the negative results). The concatenated weights fold at compile time and the `VariadicSplit`
lowers to in-place crops. Isolated GEMMs gain 12% (728 -> 648 us per layer); in the model the gain is
**-1.8 ms on 0.6B FP16** (interleaved A/B, 4 rounds, every round faster) and -1.1 ms on 1.7B FP16,
byte-identical output. Compressed and bias-carrying FCs are untouched; 5 new unit tests.

## Results

The per-patch TTFT figures below are historical 2026.3.1 measurements. The
current 2026.4.1 full-generation rebenchmark, including each cumulative patch
stage and both benchmark implementations, is documented in
[`Qwen3-Optimization/docs/performance_2026.4.1.md`](../../../Qwen3-Optimization/docs/performance_2026.4.1.md).

Each patch was measured with the same provenance-backed harness: it measures the
unpatched arm first, applies the patch, rebuilds the GPU plugin, and measures
again. Values are arithmetic means over 10 measured iterations after one warm-up,
in milliseconds. Each run records the source commit, patch sha256, patch state,
and plugin hashes in its `metadata.json`.

| Run | Patch measured | Baseline is |
|---|---|---|
| [`cold_ttft_patch_compare_20260913`](../../../Qwen3-Optimization/results/cold_ttft_patch_compare_20260913/) | opt-1 | historical pristine 2026.3.1 |
| [`opt2_dq_reuse_20260913`](../../../Qwen3-Optimization/results/opt2_dq_reuse_20260913/) | opt-2 | opt-1 applied |
| [`opt3_rope_rms_20260915`](../../../Qwen3-Optimization/results/opt3_rope_rms_20260915/) | opt-3 | opt-1 + opt-2 applied |

Per-patch delta (each measured against the previous stage):

| Model | Variant | opt-1 | opt-2 | opt-3 |
|---|---|---:|---:|---:|
| Qwen3-0.6B | FP32 | −2.80% | +0.26% | −5.73% |
| Qwen3-0.6B | INT8 | −3.96% | −0.93% | −7.49% |
| Qwen3-0.6B | **INT4 g128** | **−2.91%** | **−1.46%** | **−6.02%** |
| Qwen3-0.6B | INT4 channel-wise | −4.04% | −1.93% | −7.88% |
| Qwen3-1.7B | FP32* | — | — | — |
| Qwen3-1.7B | INT8 | * | −2.22% | −2.88% |
| Qwen3-1.7B | **INT4 g128** | **−1.37%** | **−0.86%** | **−2.98%** |
| Qwen3-1.7B | INT4 channel-wise | −2.49% | −2.64% | −4.01% |

`*` Qwen3-1.7B FP32 is excluded throughout: its run-to-run std is 20-25 ms (it
does not fit comfortably in the shared iGPU memory path), so no claim can be
made from it. The 1.7B INT8 opt-1 cell was similarly noisy in that first run.
Every other cell has std ≤ 1.5 ms in both arms, so the deltas are well outside
noise.

FP32 is unaffected by opt-2 (+0.26%, noise) exactly as expected — an FP32 graph
has no `DynamicQuantize` at all — which is a useful confirmation that the patch
does what it claims.

### Cumulative

The formal all-precision stack run is
[`opt123_all_precisions_fp16_20260915`](../../../Qwen3-Optimization/results/opt123_all_precisions_fp16_20260915/).
The historical run measured pristine 2026.3.1 first, then applied opt-1, opt-2, and opt-3 as
one stack:

| Model | Variant | pristine mean | opt-1+2+3 mean | total |
|---|---|---:|---:|---:|
| Qwen3-0.6B | FP32 | 146.32 | 134.80 | −7.9% |
| Qwen3-0.6B | INT8 | 129.00 | 115.28 | −10.6% |
| Qwen3-0.6B | **INT4 g128** | **139.96** | **127.09** | **−9.2%** |
| Qwen3-0.6B | INT4 channel-wise | 114.35 | 100.78 | −11.9% |
| Qwen3-1.7B | FP32* | 572.31 | 384.86 | −32.8% |
| Qwen3-1.7B | INT8 | 272.08 | 256.26 | −5.8% |
| Qwen3-1.7B | **INT4 g128** | **277.17** | **259.71** | **−6.3%** |
| Qwen3-1.7B | INT4 channel-wise | 219.19 | 202.84 | −7.5% |

`*` 1.7B FP32 is excluded from patch-effect claims because both arms are noisy.

The formal DQ=0 run is
[`opt123_dq0_0p6b_fp16_20260915`](../../../Qwen3-Optimization/results/opt123_dq0_0p6b_fp16_20260915/):

| Model | Variant | DQ=0 pristine | DQ=0 + opt-1+2+3 | Patch delta | DQ=0 vs default after patches |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | FP32 | 149.23 | 138.58 | −7.13% | +2.80% |
| Qwen3-0.6B | INT8 | 134.74 | 124.37 | −7.69% | +7.89% |
| Qwen3-0.6B | **INT4 g128** | **127.97** | **116.24** | **−9.17%** | **−8.54%** |
| Qwen3-0.6B | INT4 channel-wise | 125.40 | 113.80 | −9.24% | +12.92% |

Only INT4 g128 benefits from DQ=0. Its final 116.24 ms is **−16.95%** versus
the default pristine 139.96 ms; DQ=0 is a per-model runtime setting, not a
source patch.

opt-1 and opt-3 both remove activation traffic that is independent of the weight
quantization, so their absolute gain is roughly constant (≈4 ms and ≈8 ms per
prefill) and the *percentage* is therefore largest on the fastest configurations.

### 2026.4.1 FP16/INT8 follow-up: opt-4, opt-5, opt-6

Cold prefill, 1000 tokens, one generated token, 10 measured iterations after one warm-up, prompt
permutation ON, arithmetic mean +/- std in milliseconds. Each stage is the plugin built from the
cumulative source state; raw reports, plugin binaries and `manifest.json` are in
[`opt456_stage_compare_2026.4.1_20261023`](../../Qwen3-Optimization/results/runs/opt456_stage_compare_2026.4.1_20261023/).
The opt-1+2+3 and opt-1..5 stages were measured twice (before/after the other stages) to show the
drift of the machine, which is up to ~1.5 ms over a few minutes; **the S6 column of the INT rows is
unchanged code** (opt-6 only fuses uncompressed FCs) and shows that drift. The reliable opt-6 number is the
interleaved A/B: FP16 127.7 -> 125.9 ms.

Qwen3-0.6B:

| Variant | Backend | opt-1+2+3 (2 runs) | + opt-4 | + opt-5 (2 runs) | + opt-6 | opt-4 / opt-5 / opt-6 | Total vs opt-1+2+3 |
|---|---|---:|---:|---:|---:|---:|---:|
| FP16 | Python | 133.56 ± 0.36 / 133.09 ± 0.25 | 131.39 ± 0.21 | 127.06 ± 0.27 / 128.66 ± 0.16 | 126.44 ± 0.21 | -1.4% / -2.7% / -1.1% | -5.2% |
| FP16 | C++ | 133.67 ± 0.19 / 133.49 ± 0.20 | 132.11 ± 0.17 | 127.62 ± 0.15 / 130.03 ± 4.33 | 127.52 ± 1.55 | -1.1% / -2.5% / -1.0% | -4.5% |
| INT8 | Python | 114.46 ± 0.30 / 114.50 ± 0.23 | 114.11 ± 0.12 | 109.73 ± 0.14 / 110.20 ± 0.17 | 110.68 ± 0.30 | -0.3% / -3.6% / +0.6% | -3.3% |
| INT8 | C++ | 114.09 ± 0.15 / 114.20 ± 0.18 | 113.97 ± 0.17 | 110.00 ± 0.23 / 110.93 ± 0.30 | 109.76 ± 0.14 | -0.2% / -3.1% / -0.6% | -3.8% |
| INT4 g128 | Python | 126.84 ± 1.08 / 126.44 ± 0.34 | 126.77 ± 0.20 | 122.89 ± 0.22 / 122.78 ± 0.36 | 122.79 ± 0.15 | +0.1% / -3.1% / -0.0% | -3.0% |
| INT4 g128 | C++ | 126.19 ± 0.27 / 126.79 ± 0.33 | 126.76 ± 0.27 | 122.36 ± 0.29 / 122.69 ± 0.20 | 123.11 ± 0.28 | +0.2% / -3.3% / +0.5% | -2.7% |
| INT4 channel-wise | Python | 100.52 ± 0.26 / 99.55 ± 0.21 | 100.49 ± 0.24 | 95.90 ± 0.22 / 96.17 ± 0.20 | 96.24 ± 0.20 | +0.5% / -4.4% / +0.2% | -3.8% |
| INT4 channel-wise | C++ | 100.36 ± 0.30 / 99.83 ± 0.34 | 99.92 ± 0.33 | 95.85 ± 0.24 / 95.98 ± 0.27 | 96.47 ± 0.20 | -0.2% / -4.0% / +0.6% | -3.6% |

Qwen3-1.7B (opt-1+2+3 measured twice vs the opt-1..5 stage):

| Variant | Backend | opt-1+2+3 (2 runs) | + opt-4/5 | Change |
|---|---|---:|---:|---:|
| FP16 | Python | 350.87 ± 1.76 / 350.19 ± 0.38 | 343.12 ± 0.41 | -2.1% |
| FP16 | C++ | 351.00 ± 2.32 / 350.38 ± 1.41 | 342.75 ± 0.40 | -2.3% |
| INT8 | Python | 252.02 ± 0.40 / 253.72 ± 0.34 | 248.96 ± 0.27 | -1.5% |
| INT8 | C++ | 252.72 ± 0.28 / 254.07 ± 0.33 | 247.84 ± 0.28 | -2.2% |
| INT4 g128 | Python | 257.73 ± 0.25 / 259.71 ± 0.42 | 255.86 ± 0.62 | -1.1% |
| INT4 g128 | C++ | 258.41 ± 0.33 / 260.53 ± 0.60 | 255.93 ± 0.76 | -1.4% |
| INT4 channel-wise | Python | 201.67 ± 0.24 / 202.65 ± 0.61 | 197.14 ± 0.59 | -2.5% |
| INT4 channel-wise | C++ | 201.87 ± 0.43 / 201.37 ± 0.76 | 198.52 ± 0.35 | -1.5% |

Kernel attribution (VTune `gpu-hotspots`, ms per prefill, before = opt-1+2+3, after = opt-1..5; sum of
GPU compute tasks is within 1.5% of TTFT):

| | FP16 before | FP16 after | INT8 before | INT8 after |
|---|---:|---:|---:|---:|
| `gemm_kernel` (incl. `lm_head`) | 99.12 | 98.92 | 66.54 | 66.76 |
| `sdpa_micro` prefill | 17.25 | 12.77 | 17.08 | 12.44 |
| `rms_gpu_bfyx_opt` (incl. fused RoPE) | 11.79 | 11.96 | 14.20 | 14.22 |
| `dynamic_quantize_gpu_opt` | - | - | 11.87 | 11.92 |
| `pa_kv_cache_update_ref` | 7.30 | 5.50 | 5.48 | 5.49 |
| **total** | **135.67** | **129.37** | **115.44** | **111.09** |

Raw VTune results: `Qwen3-Optimization/results/hotspot/q3_0p6b_{FP16,INT8}_{opt123,opt12345}_overview`
(+ `*_kernels.csv`).

Correctness gate ([`scripts/md5_check.py`](../../Qwen3-Optimization/scripts/md5_check.py): 6 prompts, three of them long,
x 96 greedy tokens, one md5 each; each plugin build is swapped into `bin/` for its run):

| | Result |
|---|---|
| opt-4 | byte-identical on 0.6B FP16/INT8/INT4-g128/INT4-channel-wise and 1.7B INT8/INT4-g128 |
| opt-5 | identical on 0.6B FP16; differs on 2/6 prompts (INT8), 3/6 (INT4 g128, INT4 channel-wise), as described above. Bit-exact variant: identical on 0.6B INT8/INT4/INT4-cw and 1.7B INT8 |
| opt-6 | byte-identical to opt-1..5 on 0.6B FP16/FP32/INT8/INT4-g128/INT4-cw and 1.7B FP16/INT8/INT4-g128 |
| unit tests | `ov_gpu_unit_tests`: 225 paged-attention / sdpa / kv-reorder tests (221 pass, 4 skipped `cm`, 0 fail); 16 horizontal-fusion tests incl. 5 new |

## Not shipped: measured negative and dead-end results

Recorded so they are not re-attempted. All measured on the same setup.

| Idea | Result | Why |
|---|---|---|
| **Horizontally fuse `gate_proj`+`up_proj`** (lower `FullyConnectedHorizontalFusion`'s 2-FC threshold onto the oneDNN path) | **+9.07 ms (+6.4%)** regression, 3/3 reps | The fused FC needs a `VariadicSplit` on the innermost axis, which cannot be an in-place crop — it costs a 12 MB copy per layer (~11 ms), more than the fusion saves. Only pays off with the cldnn-only SwiGLU post-fusion, which `fuse_swiglu` disables below 128 EUs anyway. The upstream guard is correct. |
| **Route grouped-INT4 FCs to the cldnn OCL FC kernel** instead of oneDNN | **45 515 ms** (≈320x slower) | The OCL dynamic-shape FC path is not viable for this shape/type combination. |
| **`KV_CACHE_PRECISION=f16` or `u8`** instead of the default INT4 by-channel KV cache | ±1-2 ms, within noise, on all 4 INT4 variants | `pa_kv_cache_update` is layout-bound, not precision-bound; prefill writes the cache once and reads it once, so compression is close to free either way. |
| **`DYNAMIC_QUANTIZATION_GROUP_SIZE` = 32 / 64 / 128 / 256 / 1024** | +4 ms to +55 ms | 32 and 64 are badly mismatched to the g128 weight groups; ≥128 is uniformly worse than the per-token default. |
| **Reverse `rope_opt`'s GWS so the contiguous rotary index is the fastest axis** (upstream already does this, but only for `vec_size == 1`, which excludes every f16 LLM) | +0.4 to +1.0 ms, consistent across 3 models | RoPE is not coalescing-limited. Judged neutral by p50 and only revealed as a regression by the mean — measure with the mean. |
| **Right-size the RMS kernel's SLM buffer** (it is sized in work items, over-allocating by `SUB_GROUP_SIZE`; for a dynamic shape that is ~4 KB per work group to reduce one value) | VTune: peak occupancy 40% → 100%, SLM limit removed — **runtime unchanged** (6.33 → 6.32 ms) | A real 16x over-allocation and a real resource cap, but these kernels are bandwidth-bound, not occupancy-bound, so lifting it buys nothing here. Worth reporting upstream; not shipped because it is not measurable. |
| **Pack several 128-wide rows per RMS work group** (16 000 single-sub-group work groups, 512 B each) | −0.2 ms, noise | Same conclusion as above: the limiter is bandwidth, not work-group count. |
| **Enable `FuseGatedMLP`** (oneDNN gated-MLP primitive; the pass exists but `disable_gated_mlp_fusion` is on in release builds) | FP16 133.4 -> 152.2 ms (**+14%**), INT8 114.5 -> 164.4 ms (**+44%**) | The fused primitive is much slower than three tuned GEMMs on this GPU. |
| **Split the KV-cache update over 4 sub-groups instead of 2** (K halves, V halves) | 127.4 -> 127.5 ms | The 2-way split already saturates what the memory system gives. |
| **Vector path for the INT8 by-channel KV cache update** (same idea as opt-4 for the compressed cache) | kernel 5.49 -> 5.29 ms, TTFT unchanged | 4 MB read + 2 MB written per layer at 42 / 27 GB/s *is* ~170 us; skipping the key half or the value half alone gives 103 / 112 us instead of 187 us, i.e. it is traffic-bound. |
| **Sweep oneDNN GEMM kernel strategies** (~30 candidates per problem through the dev-mode `GEMM_KERNEL` override, random data) | The selector's pick was the fastest for q-proj (11.9 vs 11.0 TFLOPS) and gate+swiglu (9.0 vs 8.8) | Tile choice is not the limit, see "Hardware limits". |
| **Force a C-prefetch strategy (`+m@N`) onto the GEMM to hide the binary post-op read** | kernel creation hung / crashed | `strategy.prefetchC` exists in the generator but no catalog strategy uses it and it only covers the C matrix, not binary post-op operands. |
| **`DYNAMIC_QUANTIZATION_GROUP_SIZE` 256 / 512 / 1024 / per-token for the INT8 model** | 128 (default) 110.67, 256 109.32, 512 108.22, 1024 107.55, per-token 107.70 ms (-2.8%) | The plugin deliberately clamps INT8 weights to activation groups of 128 (`use_gs128_for_int8_per_token`), presumably for accuracy; coarser groups make the K=1024 INT8 GEMMs ~10% faster. Not shipped because it trades accuracy (not evaluated here); usable as a runtime setting. |

### Partially positive but not generalizable: disabling activation dynamic quantization

`DYNAMIC_QUANTIZATION_GROUP_SIZE=0` is **not** the same as leaving the property
unset. `execution_config.cpp` rewrites an unset `0` to per-token dynamic
quantization on any immad-capable platform; setting `0` explicitly marks it
user-set and genuinely disables it.

The current all-three-patch comparison is:

| Model | Variant | default after patches | `=0` after patches | delta |
|---|---|---:|---:|---:|
| Qwen3-0.6B | FP32 | 134.80 | 138.58 | +3.78 |
| Qwen3-0.6B | INT8 | 115.28 | 124.37 | +9.09 |
| Qwen3-0.6B | **INT4 g128** | **127.09** | **116.24** | **−10.85** |
| Qwen3-0.6B | INT4-channel-wise | 100.78 | 113.80 | +13.02 |

Earlier VTune attribution explains the 0.6B-g128 case — activation quantization
costs a full extra pass but barely speeds up a GEMM that is bound by *weight*
dequant:

| | default | `=0` |
|---|---:|---:|
| `gemm_kernel` | 78.39 ms | 80.29 ms (+1.90) |
| `dynamic_quantize` | 12.44 ms | **0** |
| total | 141.49 ms | 131.27 ms |

It is shipped as a **documented per-model runtime setting, not a patch**: the
sign of the effect flips with quantization scheme even within 0.6B. For
Qwen3-0.6B INT4-g128 specifically it is worth −8.54% after all three patches
and *improves* accuracy because activations stay f16. The other three 0.6B
formats become slower.

The principled fix is to stop paying for `DynamicQuantize` as a standalone pass
by fusing it into its producer. See open items below.

## Hardware limits found on the way (Pantherlake-P, 32 XVE, 2.45 GHz)

Measured with the probes in
[`Qwen3-Optimization/scripts/tools/gemm_probe/`](../../Qwen3-Optimization/scripts/tools/gemm_probe/).
They explain why the remaining ~80% of the prefill (GEMM + attention) did not move and why the streaming
kernels sit where they do.

| Finding | Number | Consequence |
|---|---|---|
| DRAM bandwidth seen by the GPU | read **42 GB/s** (34 GB/s for 2-6 MB buffers, i.e. no useful cache between kernels), write ~27 GB/s, copy 60 GB/s | RMS (218 us for 8 MB moved), KV update, `dynamic_quantize`, `lm_head` (311 MB in 7.4 ms) all sit at this floor; only removing traffic helps (that is what opt-3 did) |
| XMX throughput depends on the data | the same f16 GEMM runs at **17-19 TFLOPS with constant-ish data and 9.3-13 TFLOPS with random data**, also for 4096^3; clocks stay at 2450 MHz, `throttle/*` all 0 | The model's GEMMs run at ~9.6 TFLOPS and `sdpa_micro` at ~9 TFLOPS, i.e. at this limit: ~85% of the FP16 TTFT is XMX-power-bound, and benchmarking GEMMs on constant data (as `benchdnn` fills) overstates them by ~1.7x |
| oneDNN GEMM tiling | the selector's choice beat all of its ~30 candidates | no tile-size win in the GEMMs |
| Binary GEMM post-ops | INT8 1024x1024x3072: none 360 us, swish 377 us, **binary mul/add/sum 543 us**; 2 MB residual adds cost +55-65 us each | A post-op that reads a second [M, N] tensor is not prefetched; its latency is exposed at the end of every tile. Per layer that is ~180 us (gate x up), ~65 (down + residual), ~55 (o_proj + residual), about 4-8 ms per prefill |
| INT8 GEMM efficiency | 15-20 TOPS vs ~40 TOPS int8 peak; symmetric weights change nothing | the u8 + zero-point path is not the limiter |

## Open items

### FP16 / INT8 after opt-1..6

What is left of the 0.6B FP16 (127 ms) and INT8 (110 ms) prefill, ranked by what a fix could plausibly return:

1. **Prefetch the binary post-op operand inside the oneDNN GEMM generator - ~4-8 ms (3-6%), every precision.**
   Residual adds and the gate x up product read a second [M, N] tensor whose latency is exposed (hardware
   limits above). The fix is in `thirdparty/onednn_gpu` (gemmstone: issue a block-2D prefetch of the binary
   tile before the k-loop, next to the existing `gemmPrefetchC`). Forcing the existing C-prefetch strategy
   through `GEMM_KERNEL` hung kernel creation, so this is real generator work, not a switch.
2. **Residual add folded into the following RMS - up to ~2 ms.** A two-output RMS (`sum` for the residual
   stream, `norm` for the next GEMM) would replace the o_proj/down_proj `binary_add` post-ops. Needs a
   multi-output RMS primitive.
3. **INT8 only: fuse SwiGLU into the `down_proj` DynamicQuantize - up to ~2.5 ms.** Run gate/up as plain GEMMs
   (saves the 180 us binary post-op) and have `dynamic_quantize_gpu_opt` read both halves; the extra DQ read
   traffic eats most of the gain, so the net is small. For INT8 weights the plugin uses 128-element
   activation groups (`use_gs128_for_int8_per_token`), so `../qwen3.5_patch/opt7_*` (RMS into DQ) could fire
   here, but the hidden-size DQs it would remove cost only ~1.5 ms in total.
4. **INT8 only: coarser activation quantization groups - 3.1 ms (2.8%)** as a *runtime* setting
   (`DYNAMIC_QUANTIZATION_GROUP_SIZE=1024`); an accuracy question, not an engineering one.
5. **The rest is at a hardware floor:** GEMM and `sdpa_micro` at the XMX data-dependent throughput limit,
   `lm_head`, RMS, KV update and `dynamic_quantize` at 42 GB/s read / 27 GB/s write.

### INT4-g128 (unchanged since opt-1/2/3)

Ranked by measured headroom on Qwen3-0.6B INT4-g128, after opt-1/2/3.

1. **The grouped-INT4 GEMM itself — 26.5 ms (≈20% of TTFT).** The entire
   g128-vs-channel-wise prefill gap is inside the FC GEMMs (78.39 vs 51.92 ms);
   every other kernel differs by ≤1.7 ms in total. VTune instruction counts show
   oneDNN's grouped path issuing **2.2x the ALU0 instructions** (3633M vs 1658M)
   and 23% more XMX instructions for identical GEMM math, and choosing a
   completely different work decomposition (`GWS = N*8 x 2` vs `256 x 8`). This
   is inside `thirdparty/onednn_gpu`, not the plugin, and is by far the largest
   remaining item for the g128 case specifically.
2. **Fuse the two remaining `DynamicQuantize` producers — ~10 ms.** opt-3's
   approach does not reach them: `down_proj`'s DQ (6.43 ms, the largest single
   one) consumes the SwiGLU epilogue of a oneDNN GEMM, and `o_proj`'s (3.83 ms)
   consumes PagedAttention's output. Neither producer is a cldnn kernel whose
   epilogue we can extend, and a GEMM epilogue cannot compute a per-token
   quantization scale because the row is split across work groups.
   `../qwen3.5_patch/opt7_fuse_rms_into_dynamic_quantize.patch` covers only the
   RMS-fed third and **cannot fire here**: it requires 128-element activation
   groups, while this systolic-capable GPU defaults to per-token quantization
   (verified — it applies cleanly after opt3 but folds nothing).
3. **`sdpa_micro` prefill — 17.2 ms (13%).** Addressed by opt-5 (-4.4 ms on 0.6B).
4. **`pa_kv_cache_update_ref` — 7.4 ms at ~20 GB/s**, 76% stalled, 51%
   occupancy, still the reference kernel. It writes an INT4 by-channel KV cache;
   `KV_CACHE_PRECISION` makes no difference, so it is layout-bound. (opt-4 fixes the uncompressed FP16 cache,
   not this compressed path, which is at its traffic floor.)
5. **`q_norm` still runs at ~36 GB/s vs the layernorm's ~64 GB/s** even after
   opt-1+opt-3. Two follow-ups were tried and both were neutral — see the
   negative results above — so the remaining gap is per-work-group launch cost
   for 128-wide rows, not occupancy.

## Reproducing

```bash
cd /home/chaofan/model_benchmark
source Qwen3-Optimization/scripts/env.sh

# baseline-first comparison for the complete default patch stack
python3 Qwen3-Optimization/scripts/run_matrix.py \
   --patch-compare --run-id my_run --iterations 10

# 0.6B runtime-setting comparison with the same patch stack
python3 Qwen3-Optimization/scripts/run_matrix.py \
   --patch-compare \
   --runtime-config Qwen3-Optimization/configs/qwen3_int4_g128_dq0.json \
   --models qwen3_0p6b_fp32 qwen3_0p6b_int8 qwen3_0p6b_int4_g128 qwen3_0p6b_int4_channelwise \
   --run-id my_dq0_run --iterations 10

# VTune GPU kernel breakdown on the real PA pipeline
./Qwen3-Optimization/scripts/vtune_gpu.sh \
    Qwen3-Optimization/models/converted/Qwen3-0.6B-INT4-g128 my_tag overview 5 1000

# numerical-equivalence gate for a plugin build (6 prompts x 96 greedy tokens, one md5 each)
"$Q3_PYTHON" Qwen3-Optimization/scripts/md5_check.py Qwen3-Optimization/models/converted/Qwen3-0.6B-FP16

# interleaved A/B of plugin builds (the runtime loads the plugin from bin/intel64/Release, so builds are swapped in)
Qwen3-Optimization/scripts/tools/plugin_ab.sh Qwen3-Optimization/models/converted/Qwen3-0.6B-FP16 4 a.so b.so

# per-op profile of the PA graph from the C++ driver
./model_bench/build/model_bench llm -m <model> -d GPU -n 1 -ic 4 \
    --input-tokens 1000 --no-monitor --profile --profile-graph pa
```
