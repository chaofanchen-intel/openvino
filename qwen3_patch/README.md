# Qwen3 prefill optimizations for the Intel GPU plugin

Four standalone patches reduce cold-prefill latency for dense Qwen3 models on
Intel integrated GPUs. The current four-patch set consolidates an earlier
six-patch draft; this document and its measurements use only the current
`opt-1` through `opt-4` numbering.

The patches target different kernels or passes, modify disjoint files, and can
be applied independently in any order. `opt-1` through `opt-3` are
bit-exact; `opt-4` is an optional attention-tile change that is not bit-exact.

## Base and application

The patches target OpenVINO 2026.4.1 release tag
`e213a14725769a6a2bf86fc86650aa78a9c05d3e` and apply to a pristine
`2026.4_base` checkout.

```bash
git apply qwen3_patch/opt1_rms_rope_qk_norm.patch
git apply qwen3_patch/opt2_pa_kv_cache_vector_update.patch
git apply qwen3_patch/opt3_fuse_uncompressed_qkv_fcs.patch
git apply qwen3_patch/opt4_sdpa_micro_xe3_igpu_tile.patch  # optional, not bit-exact
cmake --build build --target openvino_intel_gpu_plugin -j 10
```

The GPU plugin is loaded from `bin/intel64/Release`. Do not run overlapping
builds or plugin A/B tests against the same OpenVINO tree.

## Hotspots and changes

| Patch | Hotspot | Scope | Change | Measured stage effect |
|---|---|---|---|---:|
| opt-1 | `rms_gpu_bfyx_opt` and `rope_opt`: per-head Q/K norm followed by RoPE | All tested precisions | Enable the RMS block-read/write path for 128-wide heads and fold RoPE into the RMS epilogue, removing the intermediate tensor and RoPE dispatch | −11.5 to −12.0 ms |
| opt-2 | `pa_kv_cache_update_ref`: scalar stores into the transposed key cache | FP16 cache, full 16-token blocks | Transpose in registers and issue vector key stores and block value stores | −1.4 ms on FP16 |
| opt-3 | `gemm_kernel`: separate uncompressed Q, K, and V projections | FP16 weights | Fuse the three FCs into one projection and split the result with in-place crops | −1.3 ms on FP16 |
| opt-4 | `sdpa_micro` prefill tile | All tested precisions on the target integrated GPU | Use a smaller query/key tile and fewer sub-groups for head size 128 | −4.0 to −4.5 ms; not bit-exact |

`opt-1` fixes two RMS access guards: the alignment test for block writes had
incorrect operator precedence, and the block-read guard excluded the
single-sub-group case used by Qwen3's `head_dim = 128`. Its RoPE epilogue uses
the same element order and output-type rounding as the standalone operations.

`opt-2` changes only the full-block update path for uncompressed FP16 KV
caches. Partial blocks, decode, and compressed caches retain the existing
path. `opt-3` applies to three bias-free constant-weight FCs sharing an
activation; compressed projections already use their own fusion. `opt-4`
changes the online-softmax accumulation order through its tile shape.

## Test protocol

The current stage comparison measures Qwen3-0.6B FP16, INT8, INT4-g128, and
INT4-channel-wise on an Intel integrated GPU using OpenVINO 2026.4.1 and
OpenVINO GenAI 2026.4.1.0.

- Cold prefill: 1000 input tokens, one generated token, batch size 1.
- C++ `model_bench`, `GPU`, PagedAttention, prompt permutation enabled.
- Five interleaved rounds, six measured iterations per stage, one discarded
  run after each plugin swap.
- Reported values are the arithmetic mean of the five round means, with the
  standard deviation of those means.

## Current results

Latency in milliseconds. Each column is the cumulative patch stage.

| Precision | pristine | + opt-1 | + opt-2 | + opt-3 | + opt-4 | Total |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 145.74 ± 0.41 | 133.71 ± 0.19 | 132.34 ± 0.28 | 131.07 ± 0.17 | 127.04 ± 0.48 | −12.8% |
| INT8 | 126.36 ± 0.26 | 114.61 ± 0.28 | 114.58 ± 0.35 | 114.52 ± 0.32 | 110.03 ± 0.31 | −12.9% |
| INT4 g128 | 138.04 ± 0.36 | 126.59 ± 0.11 | 126.55 ± 0.22 | 126.77 ± 0.16 | 122.26 ± 0.28 | −11.4% |
| INT4 channel-wise | 111.76 ± 0.13 | 100.24 ± 0.29 | 100.14 ± 0.26 | 100.27 ± 0.38 | 95.76 ± 0.25 | −14.3% |

The FP16-only patches, `opt-2` and `opt-3`, have no intended effect on the
compressed INT8/INT4 paths; those columns are controls for build and runtime
consistency.

The raw logs, source and patch hashes, stage plugin hashes, and correctness
digests are in
[`Qwen3-Optimization/results/runs/stage_compare_4patch_2026.4.1_20261026/`](../../../Qwen3-Optimization/results/runs/stage_compare_4patch_2026.4.1_20261026/).
The manifest records all five plugin hashes. The run bundle retains `s0.so`
and `s4.so`; intermediate stage binaries can be reproduced with
[`build_stage_plugins.sh`](../../../Qwen3-Optimization/scripts/tools/build_stage_plugins.sh).

## Correctness

The 6-prompt × 96-token greedy-output gate compares pristine `s0`, the
bit-exact core `s3` (`opt-1..opt-3`), and the complete `s4` stack:

| Precision | `s3` vs pristine | `s4` vs pristine |
|---|---|---|
| FP16 | Identical | Identical |
| INT8 | Identical | 2 of 6 texts differ |
| INT4 g128 | Identical | 3 of 6 texts differ |
| INT4 channel-wise | Identical | 3 of 6 texts differ |

The `opt-4` differences occur on near-tie tokens with long prompts. The patch
is separate so it can be omitted when bit-exact output is required. Patch
unit-test evidence is recorded alongside each patch in the archived
[patch index](../../../Qwen3-Optimization/patches/README.md).

## Rebuild integrity

The GPU plugin embeds OpenCL kernels, while host dispatch code is compiled into
separate objects. A stale object can therefore combine host code and kernels
from different source states. Use
[`build_stage_plugins.sh`](../../../Qwen3-Optimization/scripts/tools/build_stage_plugins.sh)
to touch all patched files, build each cumulative stage, and verify patch
markers in each binary. Use
[`plugin_ab.sh`](../../../Qwen3-Optimization/scripts/tools/plugin_ab.sh) to
swap plugins with copy-then-rename, discard the first run after each swap, and
retry a failed measured run once.

Run
[`sync_from_canonical.sh`](../../../Qwen3-Optimization/patches/sync_from_canonical.sh)
to verify that the archived patch copies and engineering notes match this
directory.
