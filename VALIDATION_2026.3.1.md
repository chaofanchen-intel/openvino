# OpenVINO 2026.3.1 Optimization Validation

## Source

- Base tag: `2026.3.1`
- Base commit: `759c5a6ab8c066af5f4bc5ebd04643706012a37d`
- Branch: `model-optimizations`
- Qwen3.5 stack: `opt3 + opt4 + opt5 + opt6 + opt7`
- GME stack: `gme_patch/all_gme_int4_gpu_opts_round8.patch`

## Build

Build configuration used for the three OpenVINO variants:

```text
CMAKE_BUILD_TYPE=Release
ENABLE_TESTS=OFF
ENABLE_FUNCTIONAL_TESTS=OFF
ENABLE_JS=OFF
ENABLE_SAMPLES=ON
ENABLE_PYTHON=ON
```

The base, Qwen3.5, and GME variants use separate build and install prefixes. All three
production builds completed with the GPU plugin, frontends, Python bindings, and
samples enabled. Tests were disabled because the environment did not provide the
`gmock/gmock.h` dependency required by the test targets.

## Patch Replay

The Qwen3.5 PA stack and GME round-8 stack both apply cleanly to a pristine
`2026.3.1` checkout. They are validated in separate source trees and are never applied
together.

## Runtime Smoke Tests

| Variant | Model | Device | Status | Notes |
|---|---|---|---|---|
| pristine 3.1 | OpenVINO Core | GPU | PASS | Python Core reports 2026.3.1 and `['CPU', 'GPU']` |
| Qwen3.5 opt3-opt7 | Qwen3.5-2B-INT4 | GPU | PASS | 512-token generation completed |
| Qwen3.5 opt3-opt7 | Qwen3.5-2B-INT4-g128 | GPU | PASS | 512-token generation completed |
| GME round8 | gme-Qwen2-VL-2B-Instruct | GPU | PASS | INT8/INT4/gs64 text/query/image/fused completed |

## Benchmark Results

The raw benchmark JSON/CSV/log files were used for the measurements below but are
intentionally excluded from the source submission. The arithmetic-mean results and
comparison conclusions are preserved here.

| Variant | Workload | Samples | Mean latency | Result |
|---|---|---:|---:|---|
| Qwen3.5 opt3-opt7 | PA, per-OC, 3 measured iterations | 3 | 261.77 ms first infer | PASS; base 302.74 ms, 1.157x |
| Qwen3.5 opt3-opt7 | PA, gs128, 3 measured iterations | 3 | 325.76 ms first infer | PASS; base 354.59 ms, 1.088x |
| GME round8 | INT8 text/query/image/fused, 5 measured calls | 5 | 44.01/43.34/2015.48/2023.66 ms | PASS |
| GME round8 | INT4 per-OC text/query/image/fused, 5 measured calls | 5 | 31.85/32.05/2009.22/1977.86 ms | PASS |
| GME round8 | INT4 gs64 text/query/image/fused, 5 measured calls | 5 | 36.26/36.34/2000.08/1997.31 ms | PASS |

The GME round8 speedups versus the clean 3.1 runtime were 1.251x/1.229x/1.066x/1.058x
for INT8 text/query/image/fused, 1.363x/1.359x/1.055x/1.076x for INT4 per-OC, and
1.027x/1.035x/1.073x/1.075x for INT4 gs64. Values are arithmetic means of the raw
per-call measurements.

## 3.0 Versus 3.1: Same GME Configuration

The 3.1 GME run below uses the historical 3.0 text/query configuration exactly:
effective sequence length 1024, GPU, Python backend, `repeat=20`, and the same GME
package. The 3.0 runtime was not still installed on this machine, so the 3.0 values
are the recorded arithmetic means from the historical 3.0 vopt report; the 3.1 values
are fresh runs whose aggregate results are recorded below. Raw generated files are not
included in the source submission.

| Precision | 3.0 stock text/query (ms) | 3.1 base text/query (ms) | Base change vs 3.0 | 3.0 optimized text/query (ms) | 3.1 round8 text/query (ms) | Optimized change vs 3.0 |
|---|---:|---:|---:|---:|---:|---:|
| INT8 causal vopt | 296.57 / 296.01 | 276.33 / 275.76 | -6.82% / -6.84% | 253.36 / 252.94 | 257.34 / 256.90 | +1.57% / +1.57% |
| INT4 causal vopt | 291.443 / 280.598 | 266.04 / 267.31 | -8.72% / -4.74% | 216.73 / 217.05 | 221.43 / 220.77 | +2.17% / +1.71% |
| INT4 gs64 causal vopt | not recorded | 267.79 / 267.38 | n/a | 231.18 / 231.32 | 235.58 / 234.56 | +1.90% / +1.40% |

Interpretation:

- 3.1 base is faster than the historical 3.0 stock result by about 4.7-8.7% for the
  available precision rows. This is a version-baseline improvement, not an optimization
  effect; the historical 3.0 runtime was not re-run in the same process on this machine.
- 3.1 round8 is close to the historical 3.0 optimized result: approximately 1.4-2.2%
  slower for the available vopt precision rows. The optimization direction and scale
  remain consistent.
- Within the fresh 3.1 run, round8 improves base first-inference latency by 1.074x/1.073x
  for INT8 text/query, 1.201x/1.211x for INT4 causal text/query, and 1.137x/1.140x for
  INT4 gs64 text/query.

Fresh same-configuration raw outputs were used to calculate the table but are not
included in the source submission.

## Notes

- A build failure caused only by unavailable optional test dependencies is recorded
  separately from a production target failure.
- GenAI is validated against the OpenVINO 2026.3.1 installation and is not mixed with
  an older OpenVINO runtime.
