# Qwen3.5 (hybrid/SSM) TTFT optimizations for the Intel GPU plugin

This directory contains a set of standalone, numerically-lossless Intel GPU-plugin
source patches that reduce the **prefill / time-to-first-token (TTFT)** of Qwen3.5
(hybrid state-space / Gated-DeltaNet) models, plus the dense Qwen3 control case, on
Intel integrated GPUs.

Optimizations **opt-1 … opt-5** were validated to be **byte-identical** in output (128-token
greedy-decode `result_md5` multiset unchanged vs. the unpatched baseline). They change
data layout, GPU work-group dispatch, or graph structure only — never the arithmetic.
**opt-6 and opt-7 are exceptions** and are held to a weaker (explicitly stated) correctness
bar — see [Correctness](#correctness).
The patches are generated against this repository's OpenVINO baseline and apply with
`git apply`.

## TL;DR — which patches to use

| Attention backend | Recommended patches | Why |
|---|---|---|
| **PA** (`ATTENTION_BACKEND=PA`, OpenVINO default) | `opt3` + `opt4` + `opt5` + `opt6` + `opt7` | opt-1/opt-2 do not fire under PA; opt-4 raises the causal-conv1d occupancy; opt-5 fixes a remaining dynamic-shape `activation_ref` hotspot; opt-6 removes standalone Gated-DeltaNet SiLU passes; opt-7 removes the RMSNorm-to-DynamicQuantize f16 hand-off |
| **PA, byte-exact multi-step decode required** | `opt3` + `opt4` + `opt5` | opt-6 and opt-7 change long greedy decode output, although both are bit-exact through prefill/first token |
| **PA, byte-exact prefill/first token required** | `opt3` + `opt4` + `opt5` + `opt6` + `opt7` | the full stack is bit-exact through prefill and the first token |
| **SDPA** (`ATTENTION_BACKEND=SDPA`, legacy) | `opt1` + `opt2` + `opt3` | opt-4 targets the PA-only `paged_causal_conv1d` kernel and does not apply |

## Applying

```bash
# PA (default backend) — recommended stack
git apply qwen3.5_patch/opt3_dynamic_quantize_reuse.patch
git apply qwen3.5_patch/opt4_conv1d_token_tiled.patch
git apply qwen3.5_patch/opt5_dynamic_activation_opt.patch
git apply qwen3.5_patch/opt6_fuse_silu_into_producer.patch   # apply after opt4: shares the conv1d files
git apply qwen3.5_patch/opt7_fuse_rms_into_dynamic_quantize.patch   # apply after opt3 and opt6

# SDPA (legacy backend)
git apply qwen3.5_patch/opt1_depthwise_conv_planar.patch
git apply qwen3.5_patch/opt2_permute_lws_f_axis.patch
git apply qwen3.5_patch/opt3_dynamic_quantize_reuse.patch

# then rebuild only the GPU plugin, e.g.
#   cmake --build <build-dir> --target openvino_intel_gpu_plugin -j
```

`opt4` picks the number of token tiles `G` **adaptively** at dispatch time from the
device's hardware-thread budget (`execution_units_count × num_threads_per_eu`) and the
sequence length, so it needs no per-device tuning: a larger device gets a larger `G`, a
smaller device gets less, and single-token decode falls back to `G=1` (the original
serial kernel). Set `OV_CONV1D_TOKEN_GROUPS=<n>` to **force** a specific `G`
(unset/`0` = auto), or `OV_CONV1D_DEBUG=1` to print the selected `G` once.

## The optimizations

| # | Patch | Backend | Hotspot addressed | Files | Mechanism |
|---|---|---|---|---|---|
| opt-1 | `opt1_depthwise_conv_planar.patch` | SDPA | `Reorder` around the SSM depthwise conv1d | `graph/layout_optimizer.cpp` | Keep the fully-depthwise `GroupConvolution` (groups == in_ch == out_ch) in planar `bfyx` instead of blocked `fsv16`, eliminating the bfyx↔fsv16 reorders inserted before and after it (a depthwise conv has zero channel-block reuse, so the blocked layout only adds two reorders) |
| opt-2 | `opt2_permute_lws_f_axis.patch` | SDPA | `Transpose` (`permute_ref`) | `kernel_selector/.../permute/permute_kernel_ref.cpp` | Fix a work-group dispatch defect: when the F·B axis dominates, force the local work-group onto the memory-coalesced F axis so output writes coalesce (removes a large slowdown that depended on sequence-length divisibility). Pure LWS choice — no indexing/math change |
| opt-3 | `opt3_dynamic_quantize_reuse.patch` | PA **and** SDPA | Redundant `DynamicQuantize` | `graph/dynamic_quantize.cpp`, `plugin/transformations/dynamic_quantize_fully_connected.cpp` | When several FullyConnected share the same activation (q/k/v/z share `input_layernorm`; gate/up share `post_attention_layernorm`), reuse a single `DynamicQuantize` node instead of inserting one per FC, but only when the quantization config matches field-for-field. Also relaxes the decode "skip" assertion so the shared DQ can still be skipped to f16 passthrough during decode (keeps decode bit-exact) |
| opt-4 | `opt4_conv1d_token_tiled.patch` | PA | `paged_causal_conv1d_ref` low occupancy | `graph/impls/ocl_v2/paged_causal_conv1d_ref.{cl,cpp}` | Token-**tiled** dispatch (`GWS = {seq, hidden, G}`). Each work-item processes one contiguous token tile serially with a sliding window (no redundant reads), reaching full occupancy without extra global-memory traffic. `G` is chosen **adaptively** from the device thread budget (`EU × threads/EU`) and sequence length, so a single patch is portable across device sizes. Bit-exact for any `G ≥ 1` (`G=1` reduces to the original serial kernel) |
| opt-5 | `opt5_dynamic_activation_opt.patch` | PA **and any dynamic activation using this selector** | Large dynamic-shape `activation_ref` kernels | `kernel_selector/cl_kernels/activation_opt.cl`, `kernel_selector/kernels/activation/activation_kernel_opt.cpp` | Allow simple dynamic-shape activations to use the existing vectorized `activation_opt` float4 kernel instead of the scalar/index-heavy `activation_ref`. The dynamic path is restricted to same-layout activations with no fused ops and no runtime activation-parameter input |
| opt-6 | `opt6_fuse_silu_into_producer.patch` | PA | Two standalone SiLU (`activation_opt`) passes over the Gated-DeltaNet tensors | `plugin/transformations/move_swish_up_through_reshape.{cpp,hpp}` (new), `plugin/transformations_pipeline.cpp`, `graph/graph_optimizer/prepare_primitive_fusing.cpp`, `graph/impls/ocl_v2/paged_causal_conv1d_ref.{cl,cpp}` | Qwen3.5's two SiLUs each sit one Reshape away from their producer (`PagedCausalConv1D` and the `in_proj_z` FC), and `prepare_primitive_fusing` only fuses into a *direct* dependency — so neither was ever fused, and each ran as a full extra read+write pass over its tensor. A new pass hoists `Swish(Reshape(x))` → `Reshape(Swish(x))` (value-identical), after which the conv1d impl absorbs its SiLU into the kernel epilogue and the FC absorbs its SiLU as a oneDNN `eltwise_swish` post-op |
| opt-7 | `opt7_fuse_rms_into_dynamic_quantize.patch` | PA and supported DynamicQuantize paths | RMSNorm-to-DynamicQuantize activation traffic | `graph/graph_optimizer/prepare_primitive_fusing.cpp`, `graph/dynamic_quantize.cpp`, `graph/primitive_inst.cpp`, `graph/impls/ocl/dynamic_quantize.cpp`, `kernel_selector/cl_kernels/dynamic_quantize_gpu_opt.cl` | Folds RMSNorm into the DynamicQuantize kernel. The normalized f16 hand-off is kept in registers, rounded to f16 at the original boundary, and then quantized to s8 with scales. |

### Why opt-1/opt-2 are SDPA-only

Under PA, the SSM path runs a set of fused kernels
(`paged_gated_delta_net_opt`, `paged_causal_conv1d_ref`) that do **not** emit the
standalone `Reorder`/`Transpose` operators opt-1 and opt-2 target, so those two patches
have no effect on the PA graph. opt-3, by contrast, targets `DynamicQuantize` in front
of the shared-activation FCs, which is present on both backends — so opt-3 helps on PA
as well.

### opt-4 design rationale: token tiling vs. full token-parallelism

The reference `paged_causal_conv1d` kernel dispatched `GWS = {seq_count, hidden}` and
walked each sequence serially. At prefill `seq_count == 1`, so only `hidden` work-items
run — the dispatch is work-size-limited and cannot fill the device's hardware-thread
budget, capping occupancy (on a small integrated GPU with a 320-thread budget,
`hidden=6144` at SIMD32 fills only ~192 threads ≈ 60% occupancy).

Two ways to raise occupancy were evaluated:

- **Fully token-parallel** (one work-item per output token, each rebuilding its own
  K-element causal window from global memory). This reaches full occupancy but
  multiplies global reads by ~K and becomes memory-bound. On a smaller, bandwidth-limited
  integrated GPU it was a **net regression** (the conv1d kernel went ~14.1 → ~24.7
  ms/iter), even though it helped on a larger GPU. It is **not shipped**.
- **Token-tiled** (opt-4). Each sequence's tokens are split into `G` contiguous tiles;
  each work-item streams its tile with a sliding window, so the only extra reads are the
  `K-1` history elements to prime the window per tile (negligible). This recovers full
  occupancy without the memory blow-up — on the same small GPU the conv1d kernel improved
  to ~10.5 ms/iter — and, because `G` is selected adaptively from the device thread
  budget, the one patch is portable across device sizes (small and large) without
  re-tuning.

### opt-6 design rationale: why the SiLUs were never fused, and why it pays

Qwen3.5's Gated-DeltaNet block applies SiLU in two places, and in both the activation is
the producer's only consumer:

```
paged_causal_conv1d (1 user) → reshape (1 user) → swish      # over [tokens, 6144]
in_proj_z FullyConnected     → reshape (1 user) → swish      # over [tokens, 2048]
```

Nothing about either is intrinsically unfusable — `fully_connected` is already on the
`prepare_primitive_fusing` activation whitelist. The blocker is purely the **interposed
Reshape**: `PagedCausalConv1DFusion` deliberately ends its pattern at the `Slice` and
inserts a Reshape to restore the 3D shape, and the fusing pass only ever looks at a node's
*direct* dependency. So both SiLUs survive as standalone `activation_opt` kernels.

The generic `ov::pass::MoveEltwiseUpThroughDataMov` does not rescue this either: it matches
`UnaryElementwiseArithmetic` / `BinaryElementwiseArithmetic`, while `ov::op::v4::Swish`
derives from `ov::op::Op` directly (it carries an optional beta input), so it is never
matched. Hence the new, narrowly-scoped hoisting pass.

Why this is worth doing at all: a standalone elementwise kernel is *pure* memory traffic —
it re-reads and re-writes the entire tensor (~24 MB per layer for the 6144-wide one at a
1k-token prefill) to do one multiply-ish op, at a point where the producing kernel still
holds the value in a register. On a bandwidth-limited integrated GPU those passes already
run at the device's memory-bandwidth ceiling, so they **cannot be made faster — only
eliminated**. That also explains why opt-6's payoff is device-dependent in the opposite
direction from opt-4: the tighter the memory subsystem, the more opt-6 is worth.

### opt-7 design rationale: fold RMSNorm into DynamicQuantize

After opt-3 through opt-6, the largest remaining removable boundary was the pair of
kernels formed by RMSNorm followed immediately by DynamicQuantize. For one `[tokens,
hidden]` f16 activation, the unfused path performs:

```
rms:  read x + write normalized f16
dq:   read normalized f16 + write s8 output and scales
```

The normalized f16 tensor is only a hand-off. Opt-7 performs the reduction, normalization,
gamma multiplication, f16 rounding, and quantization in the DynamicQuantize kernel, so the
intermediate f16 write and read are removed. In the 2B model, 47 of 48 candidate layernorm
sites fold; layer 0 input layernorm remains unfused because its normalized value has a
second consumer.

The fused form is intentionally narrow. It requires the supported large-group kernel
configuration: symmetric i8 quantization, a 128-element group on the last axis, identity
scale order, no precomputed reduction, a static reduction extent with whole groups, f16
input/output constraints, and a gamma tensor covering the normalized axis. Unsupported
configurations stay on the original RMSNorm plus DynamicQuantize path.

The fused reduction follows the standalone RMSNorm reduction structure and rounds the
normalized value to f16 before quantization. This preserves the prefill/first-token bytes.
At decode batch 1, the original DynamicQuantize may skip to an f16 pass-through, but a
folded node cannot skip because its input is not normalized. The folded sites therefore
quantize during decode, which can change a long greedy sequence even though prefill and the
first token remain bit-exact.

## Measured effect

Setup: local source build, GPU, real ~1000-token prompt (1k input / 512 output),
greedy decoding, several measured iterations after warmup; A/B done with hot-swapped
plugin binaries in one session to avoid thermal drift. All numbers are byte-identical in
output vs. baseline. Relative gains differ by model and device because the causal-conv1d
and DynamicQuantize kernels are a larger fraction of prefill on smaller models, and the
two integrated GPUs measured have very different EU/thread budgets:

- **iGPU-A** — Intel Core Ultra X7 358H (Arc B390 iGPU, larger thread budget)
- **iGPU-B** — Intel Core Ultra 7 356H (smaller iGPU, 320-thread budget)
- **iGPU-C** — Intel Core Ultra 7 356H, 32 EU / 320-thread budget, but a **memory-frequency-limited**
  part: no kernel of any type exceeds ~48 GB/s of measured DRAM bandwidth on it. Same
  nominal class as iGPU-B, materially different memory behaviour — which is why the
  opt-4/opt-5/opt-6 ranking differs between them.

### PA backend (recommended stack: opt-3 + opt-4 + opt-5, measured on iGPU-B)

`infer` = pure-prefill first-token latency (`first_token_infer_mean_ms`); `e2e` =
end-to-end first-token latency including host overhead.

| Model | pristine infer | opt-3+4 infer | **opt-3+4+5 infer** | final vs pristine | opt-5 incr. | pristine e2e | opt-3+4 e2e | **opt-3+4+5 e2e** | final e2e |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 151.3 | 145.3 | **137.3** | **−9.2%** | −5.5% | 196.7 | 190.6 | **185.7** | **−5.6%** |
| Qwen3.5-2B-INT4 | 214.3 | 204.8 | **196.9** | **−8.1%** | −3.8% | 260.7 | 250.6 | **244.8** | **−6.1%** |
| Qwen3.5-4B-INT4 | 505.0 | 496.2 | **481.2** | **−4.7%** | −3.0% | 551.9 | 542.1 | **529.5** | **−4.1%** |

The conv1d step (opt-4) contribution is shown separately below, comparing the shipped
token-tiled kernel against the rejected fully-token-parallel variant on iGPU-B:

| Model | opt-3 only | **opt-3 + opt-4 (tiled)** | fully-token-parallel variant (rejected) |
|---|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | −1.5% | **−4.0%** | +7.0% (regression) |
| Qwen3.5-2B-INT4 | −3.2% | **−4.4%** | +4.7% (regression) |
| Qwen3.5-4B-INT4 | −2.1% | −1.7% | +3.9% (regression) |

### PA backend, per-stage on iGPU-C (full stack incl. opt-6)

Cumulative stages, each a separately-built plugin binary hot-swapped **interleaved** within
one session (this part drifts thermally by ~1 ms per run, which is the same order as the
opt-4/opt-5 increments, so one-pass-per-stage ordering would bias the later stages).
`model_bench vlm`, PA, real 1002-token prompt, `n=5`, `ic=1`, 3 reps, medians reported.
`infer` = `first_token_infer`, `e2e` = end-to-end TTFT.

**Qwen3.5-2B-INT4** (`group_size=-1, sym=true` → i4, per-output-channel scale):

| Stage | infer (ms) | vs pristine | increment | e2e (ms) | vs pristine |
|---|---:|---:|---:|---:|---:|
| pristine | 293.45 | — | — | 299.15 | — |
| + opt-3 | 277.69 | −5.37% | −5.37 pp | 283.33 | −5.29% |
| + opt-4 | 275.66 | −6.06% | −0.69 pp | 281.30 | −5.97% |
| + opt-5 | 275.22 | −6.21% | −0.15 pp | 280.93 | −6.09% |
| **+ opt-6** | **259.19** | **−11.68%** | **−5.46 pp** | **264.88** | **−11.45%** |

**Qwen3.5-2B-INT4-g128** (`group_size=128, sym=false` → u4, grouped scale + zero-point):

| Stage | infer (ms) | vs pristine | increment |
|---|---:|---:|---:|
| pristine | 350.66 | — | — |
| + opt-3 | 335.54 | −4.31% | −4.31 pp |
| + opt-4 | 334.19 | −4.70% | −0.39 pp |
| + opt-5 | 332.10 | −5.29% | −0.59 pp |
| **+ opt-6** | **318.08** | **−9.29%** | **−4.00 pp** |

Reading these:

- **opt-3 and opt-6 are the two real contributors on this part**; together they are
  essentially the whole stack. opt-6 alone is worth about as much as opt-3.
- **opt-4 and opt-5 are near-noise here (≤0.8 pp each)**, against −4.4% and −3.8% on
  iGPU-B. This is not a contradiction: both buy *occupancy*, and on a
  memory-frequency-limited part occupancy is not the binding constraint. opt-4 in
  particular pays extra history re-reads for its tiling, which a starved memory subsystem
  charges for. They remain positive-or-neutral, so they stay in the recommended stack, but
  do not expect iGPU-B's numbers from them on bandwidth-limited hardware.
- opt-6's benefit is **largely independent of the weight quantization** (−5.46 pp vs
  −4.00 pp), as expected for an optimization that removes memory passes over *activations*.
  The g128 model is ~55 ms slower overall purely because its grouped-scale + grouped-zp
  INT4 dequant makes every GEMM slower; that is a quantization-scheme cost, not something
  a plugin patch can reach.

### opt-7 measured increment on iGPU-C

The following fresh paired A/B runs use the same 1002-token prompt, `n=5`, `ic=1`, and
four interleaved repetitions. Each value is the median p50 across the repeated runs. The
three columns distinguish pure model inference, pipeline TTFT, and end-to-end TTFT.

**Per-output-channel INT4 (`group_size=-1`, symmetric):**

| Stage | Infer TTFT (ms) | Pipeline TTFT (ms) | E2E TTFT (ms) |
|---|---:|---:|---:|
| opt-3+4+5+6 base | 260.31 | 265.89 | 268.13 |
| opt-3+4+5+6+opt-7 | 248.97 | 254.60 | 256.77 |
| **opt-7 delta** | **−11.34 (−4.36%)** | **−11.29 (−4.24%)** | **−11.36 (−4.24%)** |

Per-repetition infer/pipeline/e2e deltas were `−9.82/−9.84/−9.83`,
`−11.07/−11.05/−11.13`, `−11.44/−11.51/−11.57`, and
`−11.61/−11.52/−11.58` ms.

**Grouped INT4 (`group_size=128`, asymmetric):**

| Stage | Infer TTFT (ms) | Pipeline TTFT (ms) | E2E TTFT (ms) |
|---|---:|---:|---:|
| opt-3+4+5+6 base | 317.37 | 323.05 | 325.25 |
| opt-3+4+5+6+opt-7 | 317.34 | 322.95 | 325.11 |
| **opt-7 delta** | **−0.04 (−0.01%)** | **−0.09 (−0.03%)** | **−0.14 (−0.04%)** |

The grouped result is neutral within measurement noise. The per-output-channel opt-7 gain
must not be generalized to the grouped quantization scheme.

### SDPA backend (legacy stack: opt-1 + opt-2 + opt-3)

Pure-prefill first-token latency (ms), baseline → +opt-1 → +opt-1+2 → +opt-1+2+3:

**iGPU-A (larger):**

| Model | baseline | +opt-1 | +opt-1+2 | +opt-1+2+3 | total |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 132.4 | 120.2 | 108.2 | 104.9 | **−20.8%** |
| Qwen3.5-2B-INT4 | 157.7 | 146.1 | 134.6 | 132.3 | **−16.1%** |
| Qwen3.5-4B-INT4 | 333.0 | 315.1 | 293.8 | 290.9 | **−12.6%** |
| Qwen3-1.7B-INT4 (dense, control) | 63.1 | 62.3 | 62.6 | 62.6 | −0.7% (does not trigger) |

**iGPU-B (smaller):**

| Model | baseline | +opt-1 | +opt-1+2 | +opt-1+2+3 | total |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 292.3 | 289.8 | 255.9 | 251.1 | **−14.1%** |
| Qwen3.5-2B-INT4 | 348.9 | 344.4 | 315.3 | 309.7 | **−11.3%** |
| Qwen3.5-4B-INT4 | 800.6 | 788.1 | 731.5 | 720.4 | **−10.0%** |

opt-2 is the largest contributor on both devices. Dense Qwen3 is unaffected — the
optimizations target SSM-specific op patterns and shared-activation FC groups.

## Correctness

Correctness is judged by the MD5 of the decoded 128-token greedy output: for **opt-1 …
opt-5** and every model the MD5 multiset under the patched plugin equals the baseline
multiset (prefill and 128-token decode both checked). opt-3 additionally preserves the
decode "skip-to-f16" path, which keeps decode bit-exact after the FCs are made to share one
DynamicQuantize. Opt-6 and opt-7 are bit-exact through prefill/first token but are not
guaranteed to preserve a long greedy decode byte-for-byte.

### opt-6 does not meet that bar — read this before shipping it

| Check | Result |
|---|---|
| Prefill / first-token output (`ic=1`, n=8) | **bit-identical** — md5 `dd7536794b63bf90eccfd37f9b147d7f`, same as baseline |
| 128-token greedy decode md5 (`ic=128`) | **differs** from baseline |

The decode difference is real, not measurement noise: baseline and patched each reproduce
their own md5 multiset exactly across repeated runs. The cause is understood and is a
rounding difference of at most 1 ULP in f16, amplified by greedy decoding:

- **conv1d epilogue.** The fused epilogue deliberately reproduces the un-fused rounding
  order — round the f32 accumulator to f16 first, then evaluate SWISH in f16, because the
  standalone `activation_opt` gets `input_dt = f16`. What cannot be reproduced is that the
  standalone kernel evaluates `exp` on a **`half4` vector** while the epilogue evaluates it
  on a **scalar `half`**; the two kernels are also compiled as separate OpenCL programs.
  Matching a vectorized f16 transcendental bit-for-bit from inside another kernel is not a
  reliably achievable target.
- **`in_proj_z` FC.** Here the SiLU becomes a oneDNN `eltwise_swish` post-op evaluated on
  the **f32 GEMM accumulator** before the f16 store. This is strictly *more* accurate than
  the f16 standalone kernel — but of course not bit-identical.

So opt-6 is best described as **numerically equivalent-or-better per element, but not
byte-reproducible over a long greedy decode**. Divergence of a 128-step greedy sequence
from a ≤1-ULP perturbation is expected behaviour on near-tied logits, not evidence of a
defect — but it does mean opt-6 cannot be validated by the md5 gate the other patches use.

**A quantitative accuracy comparison (cosine / max-abs against the unpatched plugin over
fixed inputs) has not yet been done for opt-6.** Until it is, treat opt-6 as: safe to
benchmark, safe where first-token output is what matters, and **not** to be shipped into a
deployment whose acceptance gate is "128-token md5 unchanged" — use `opt3`+`opt4`+`opt5`
there. `OV_DISABLE_SWISH_HOIST=1` turns opt-6 off at runtime without rebuilding.

### opt-7: bit-exact prefill, changed decode

| Check | Result |
|---|---|
| Prefill / first-token output (`ic=1`) | **bit-identical** to the opt-3+4+5+6 base |
| 128-token greedy decode md5 (`ic=128`) | **can differ** from the opt-3+4+5+6 base |

The prefill result follows from matching the standalone RMSNorm reduction structure and
rounding to f16 before quantization. The decode difference has a different cause from opt-6:
at batch 1, the unfused DynamicQuantize can skip and pass f16 directly to the FC, while a
folded node must execute normalization and quantization because its input is unnormalized.
This changes where the existing quantization approximation is applied. The opt-7 paired
performance result is therefore appropriate for prefill/first-token workloads, not for a
deployment whose acceptance gate requires an unchanged 128-token decode MD5.

## Scope / safety

- **opt-1** only fires for dynamic, fully-depthwise convolutions; blocked layout is
  never a correctness requirement for a depthwise conv, and the plugin re-inserts a
  reorder on any format mismatch, so the worst case is a performance trade-off, never a
  wrong result or a compile failure.
- **opt-2** only enlarges the local work-group (never shrinks it) and changes no
  indexing/math — bit-exact by construction.
- **opt-3** only merges DQ nodes whose activation *and* quantization config are
  field-for-field equal; otherwise it does not merge (at worst a missed reuse).
- **opt-4** only changes GPU work-item partitioning plus an equivalent window
  reconstruction; the per-output-element FMA order is identical to the serial reference
  for any `G`.
- **opt-5** only changes kernel selection for simple same-layout activations with no
  fused ops and no runtime activation-parameter input. The selected kernel is an existing
  vectorized implementation; the patch merely makes the safe dynamic subset eligible and
  supplies the optional shape-info argument expected by shape-agnostic kernels.
- **opt-6 and opt-7** are the patches here that are not bit-exact over long decode (see
  [Correctness](#correctness)); both are narrowly guarded:
  - the hoisting pass only fires when the Reshape has **exactly one consumer**, so no other
    consumer can ever observe pre-activation values, and `Swish(Reshape(x))` →
    `Reshape(Swish(x))` is a value-for-value identity for an elementwise op;
  - by default it only hoists over a `PagedCausalConv1D` producer. Hoisting over the
    FullyConnected producer (the `in_proj_z` gate, worth roughly a further −1.6% on
    iGPU-C) is **opt-in** via `OV_SWISH_HOIST_ALL=1`, precisely because it moves that SiLU
    into oneDNN's f32 accumulator;
  - the fusing whitelist entry accepts **only** `activation_func::swish`, which is the one
    function the conv1d epilogue implements, and the impl's JIT independently re-checks
    that it received exactly one SWISH fused desc. An activation the kernel cannot express
    therefore fails to fuse and stays a separate node — it can never be silently dropped.
- **opt-7** is guarded by the RMSNorm and DynamicQuantize preconditions described above.
  The fusing pass requires a single-consumer hand-off and a supported large-group kernel;
  unsupported configurations remain unfused rather than being silently changed.

## Runtime switches

| Variable | Effect |
|---|---|
| `OV_CONV1D_TOKEN_GROUPS=<n>` | opt-4: force the conv1d token-tile count (`0`/unset = adaptive; `1` = original serial kernel) |
| `OV_CONV1D_DEBUG=1` | opt-4: print the selected `G` once |
| `OV_DISABLE_SWISH_HOIST=1` | opt-6: disable entirely, restoring the un-hoisted graph (useful for interleaved A/B without swapping binaries) |
| `OV_SWISH_HOIST_ALL=1` | opt-6: additionally fuse the `in_proj_z` SiLU into the oneDNN FC post-op (extra gain, f32-accumulator evaluation) |

## Additional Local Performance Report

The historical platform measurements in this README are preserved. New local measurements,
including opt-7 and both quantization variants on PA and SDPA, are recorded separately in
[PA_SDPA_OPTIMIZATION_PERFORMANCE_REPORT.md](PA_SDPA_OPTIMIZATION_PERFORMANCE_REPORT.md).