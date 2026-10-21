# Qwen3 (0.6B / 1.7B) prefill-TTFT optimizations for the Intel GPU plugin

Standalone Intel GPU plugin patches that reduce **cold-start prefill / TTFT** of
dense Qwen3 models on Intel integrated GPUs, with the emphasis on INT4 weight
compression (`group_size=128` first, per-output-channel second).

This is the dense-Qwen3 counterpart to [`../qwen3.5_patch/`](../qwen3.5_patch/),
which targets the hybrid SSM Qwen3.5 models. The two sets are disjoint: the
Qwen3.5 patches address Gated-DeltaNet / causal-conv1d op patterns that a dense
Qwen3 graph does not contain.

## Base

Patches are generated against OpenVINO 2026.4.1 release tag
`e213a14725769a6a2bf86fc86650aa78a9c05d3e` and apply with `git apply` to a
pristine `2026.4_base` tree.

Compatibility checks are recorded in
[VALIDATION_2026.4.1.md](../VALIDATION_2026.4.1.md).

```bash
git apply qwen3_patch/opt1_rms_block_access.patch
git apply qwen3_patch/opt2_dynamic_quantize_reuse.patch
git apply qwen3_patch/opt3_fuse_rope_into_rms.patch   # after opt1: shares rms_gpu_bfyx_opt.cl
cmake --build <build-dir> --target openvino_intel_gpu_plugin -j
```

Apply them in order. opt-3 touches the same kernel as opt-1; opt-2 is independent.
Each carries no runtime switch except opt-3, which can be turned off without a
rebuild via `OV_DISABLE_ROPE_RMS_FUSION=1` (used for A/B).

## Test setup

| | |
|---|---|
| GPU | Intel Graphics iGPU, Pantherlake-P, 32 XVE x 10 threads, 2.45 GHz, XMX present |
| Runtime | OpenVINO + GenAI 2026.4, `ATTENTION_BACKEND=PA` (the `LLMPipeline` default) |
| Workload | 1000-token cold prefill, batch 1, greedy, `-ic 1` |
| Models | Qwen3-0.6B and Qwen3-1.7B, FP32 / INT8 / INT4-g128 / INT4-channel-wise |
| Driver | `model_bench llm` (C++), prompt permutation ON |

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

## Results

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

## Open items

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
3. **`sdpa_micro` prefill — 17.2 ms (13%).** oneDNN micro-kernel, untouched.
4. **`pa_kv_cache_update_ref` — 7.4 ms at ~20 GB/s**, 76% stalled, 51%
   occupancy, still the reference kernel. It writes an INT4 by-channel KV cache;
   `KV_CACHE_PRECISION` makes no difference, so it is layout-bound.
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

# per-op profile of the PA graph from the C++ driver
./model_bench/build/model_bench llm -m <model> -d GPU -n 1 -ic 4 \
    --input-tokens 1000 --no-monitor --profile --profile-graph pa
```
