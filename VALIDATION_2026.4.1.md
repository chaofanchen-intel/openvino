# OpenVINO 2026.4.1 Patch Compatibility Validation

## Base

- Release tag: `2026.4.1`
- Base commit: `e213a14725769a6a2bf86fc86650aa78a9c05d3e`
- Patch branch: `2026.4_base`

## Current Patch Archive Replay

The current patch archives apply and reverse cleanly against a clean worktree
at the release tag:

| Patch family | Validated chain(s) |
|---|---|
| GME | `all_gme_int4_gpu_opts_round8.patch` |
| Qwen3 | current `opt-1` → `opt-2` → `opt-3` → `opt-4` |
| Qwen3.5 PA | current patches `0001` → `0002` → `0003` → `0004` |
| Qwen3.5 SDPA | retained `opt1_depthwise_conv_planar.patch` → `opt2_permute_lws_f_axis.patch` |

The current Qwen3 and Qwen3.5 patch details and benchmark evidence are
maintained in their optimization workspaces:
[`Qwen3-Optimization`](../../Qwen3-Optimization/README.md) and
[`Qwen3.5-Optimization`](../../Qwen3.5-Optimization/README.md).

## Historical Compile Checks

The following compile and runtime results cover the earlier patch series and
names that preceded the current Qwen3/Qwen3.5 consolidation. They are retained
as historical evidence and do not establish a full plugin build for the
current four-patch Qwen3 or Qwen3.5 PA series.

CMake configured the source tree as OpenVINO 2026.4.1 (Build 22986). The
modified C++ translation units compiled through their generated object targets:

| Patch family | Compiled translation units | Result |
|---|---:|---|
| GME round-8 | 16 | PASS |
| Qwen3 | 7 | PASS |
| Qwen3.5 SDPA | 4 | PASS |
| Qwen3.5 PA | 12 | PASS |

## Historical Full GPU Plugin Link

The complete `openvino_intel_gpu_plugin` target linked successfully for each
patch stack:

| Patch family | Stack | Result |
|---|---|---|
| GME | round-8 combined | PASS |
| Qwen3 | opt-1 → opt-2 → opt-3 | PASS |
| Qwen3.5 PA | opt-3 → opt-4 → opt-5 → opt-6 → opt-7 | PASS |
| Qwen3.5 SDPA | opt-1 → opt-2 → opt-3 | PASS |

The source-built GPU plugin reported
`2026.4.1-22987-ff0bcafebcc-2026.4_base` and was loaded from
`bin/intel64/Release/libopenvino_intel_gpu_plugin.so`.

## Historical GPU Model Smoke Tests

Each generation smoke test ran one warm-up and one measured iteration, with
four generated tokens. These are functional checks, not performance claims.

| Patch family | Model / workload | Device / backend | Result |
|---|---|---|---|
| GME round-8 | GME Qwen2-VL-2B INT4 causal-vopt; text, query, image, fused | GPU | PASS |
| Qwen3 | Qwen3-1.7B-INT4; 1000-token text prefill | GPU / PA | PASS |
| Qwen3.5 PA | Qwen3.5-2B-INT4; 224×224 image prompt | GPU / PA | PASS |
| Qwen3.5 SDPA | Qwen3.5-2B-INT4; 224×224 image prompt | GPU / SDPA | PASS |

The Qwen3 and Qwen3.5 runs used the existing OpenVINO GenAI 2026.4.0.0
Python package with the source-built OpenVINO 2026.4.1 core and GPU plugin.
The plugin version was checked at runtime.

## Not Covered by This Validation Record

No current-stack full plugin build, comparative benchmark, or
numeric-equivalence test against an unpatched build was run as part of this
patch-archive replay. Current performance and correctness results remain tied
to the runtime and protocols documented in the optimization workspaces.
