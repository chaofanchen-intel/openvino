# OpenVINO 2026.4.1 Patch Compatibility Validation

## Base

- Release tag: `2026.4.1`
- Base commit: `e213a14725769a6a2bf86fc86650aa78a9c05d3e`
- Patch branch: `2026.4_base`

## Patch Replay

All listed patch chains applied and reversed cleanly against a clean worktree at
the release tag:

| Patch family | Validated chain(s) |
|---|---|
| GME | Independent opt-1, opt-2, opt-3, opt-4; round-8 combined patch; staged base + opt-4 + opt-5 + opt-6 |
| Qwen3 | opt-1 → opt-2 → opt-3 |
| Qwen3.5 SDPA | opt-1 → opt-2 → opt-3 |
| Qwen3.5 PA | opt-3 → opt-4 → opt-5 → opt-6 → opt-7 |

## Compile Checks

CMake configured the source tree as OpenVINO 2026.4.1 (Build 22986). The
modified C++ translation units compiled through their generated object targets:

| Patch family | Compiled translation units | Result |
|---|---:|---|
| GME round-8 | 16 | PASS |
| Qwen3 | 7 | PASS |
| Qwen3.5 SDPA | 4 | PASS |
| Qwen3.5 PA | 12 | PASS |

## Not Covered

The aggregate `openvino_intel_gpu_plugin` target did not complete. No final
plugin link, GPU runtime/JIT validation, model smoke test, or performance
benchmark is claimed by this report. Historical benchmark results in the patch
READMEs remain tied to their original runtime versions.
