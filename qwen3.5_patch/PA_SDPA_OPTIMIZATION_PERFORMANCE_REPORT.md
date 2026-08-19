# Qwen3.5 Intel GPU Optimization Performance Report

## 1. Scope

This report records the performance contribution of opt-1 through opt-7 in this patch set.
It is self-contained: all numbers needed to compare the optimization stages are included
below.

The measurements cover both attention backends and both INT4 quantization variants:

- PA backend: opt-3, opt-4, opt-5, opt-6, and opt-7.
- SDPA backend: opt-1, opt-2, and opt-3.
- Per-OC INT4: `group_size=-1`, symmetric quantization, signed 4-bit weights, per-output-channel scales.
- Grouped INT4: `group_size=128`, asymmetric quantization, unsigned 4-bit weights, grouped scales and zero-points.

Opt-7 is implemented for both PA and SDPA, but the measurements in this report use PA.
No SDPA opt-4 through opt-7 result is inferred from the PA measurements.

## 2. Hardware and Workload

| Item | Value |
|---|---|
| CPU/GPU platform | Intel Core Ultra 7 356H, 32 EU integrated GPU |
| Memory behavior | Memory-frequency-limited; measured DRAM ceiling approximately 48 GB/s |
| Device | GPU |
| Model | Qwen3.5-2B-INT4, per-OC or group_size=128 variant |
| Input | 1002 input tokens from the same fixed prompt |
| Output mode | One generated token (`ic=1`), so the measured path is prefill/first-token TTFT |
| Batch | 1 |
| Performance hint | `LATENCY` |
| Prompt permutation | Enabled |
| Measured iterations | 5 per process, one warm-up iteration excluded |
| Stage repetitions | Three interleaved repetitions for cumulative stages |
| Opt-7 repetitions | Four interleaved repetitions for the paired A/B comparison |
| PA stage plugin build | `2026.3.0-22452-c2638477a62-qwen-ttft-optimizations` |
| Opt-7 paired plugin build | `2026.3.0-22453-32f59cd8191-qwen-ttft-optimizations` |
| GenAI build | `2026.3.0.0-3278-0c38b54d7e7-qwen-ttft-llm-bench` |

Each stage was measured by hot-swapping a separately built GPU plugin binary. The stage
order was interleaved to reduce thermal drift bias. Each table cell is the median of the
per-process p50 values across the interleaved repetitions.

## 3. Metric Definitions

- **Infer TTFT**: `first_token_infer_p50_ms`, the pure first-token model inference latency.
- **Pipeline TTFT**: `ttft_p50_ms`, the pipeline first-token latency including the pipeline-side overhead.
- **E2E TTFT**: `e2e_p50_ms`, the end-to-end first-token latency including host-side overhead.
- **Cumulative improvement**: `(pristine - stage) / pristine`.
- **Incremental improvement**: `(previous stage - current stage) / previous stage`.

A positive percentage means lower latency after the optimization.

## 4. Optimization Coverage

| Optimization | PA | SDPA | Main change |
|---|---:|---:|---|
| opt-1 | No | Yes | Keep fully depthwise convolution in a planar layout and remove unnecessary layout conversions. |
| opt-2 | No | Yes | Correct the `permute_ref` local work-group placement for the memory-coalesced axis. |
| opt-3 | Yes | Yes | Reuse one DynamicQuantize node when multiple FullyConnected nodes share the same activation and quantization configuration. |
| opt-4 | Yes | No | Tile prefill tokens in the causal-convolution dispatch to raise useful occupancy without full token-parallel redundant reads. |
| opt-5 | Yes | Yes* | Allow the vectorized activation kernel for the safe dynamic-shape activation subset. |
| opt-6 | Yes | No | Hoist SiLU through the producer reshape and fuse it into the causal-convolution or FullyConnected producer. |
| opt-7 | Yes | Yes* | Fold RMSNorm into DynamicQuantize so the intermediate normalized f16 tensor is not materialized. |

`*` Opt-5 and opt-7 are supported by code paths that can be used by SDPA, but the detailed
opt-5/opt-7 measurements in this report are PA measurements. SDPA measurements cover
opt-1, opt-2, and opt-3 only.

## 5. PA: Per-OC INT4

### 5.1 Cumulative stages

| Stage | Infer TTFT (ms) | Pipeline TTFT (ms) | E2E TTFT (ms) | Infer vs pristine | Pipeline vs pristine | E2E vs pristine |
|---|---:|---:|---:|---:|---:|---:|
| Pristine | 293.44 | 299.08 | 301.24 | 0.00% | 0.00% | 0.00% |
| + opt-3 | 277.54 | 283.31 | 285.55 | 5.42% | 5.27% | 5.21% |
| + opt-3 + opt-4 | 275.43 | 281.21 | 283.44 | 6.14% | 5.98% | 5.91% |
| + opt-3 + opt-4 + opt-5 | 275.31 | 281.01 | 283.32 | 6.18% | 6.04% | 5.95% |
| + opt-3 + opt-4 + opt-5 + opt-6 | 259.34 | 265.05 | 267.31 | 11.62% | 11.38% | 11.27% |

### 5.2 Incremental contribution of each optimization

| Optimization | Stage transition | Delta infer (ms) | Delta pipeline (ms) | Delta E2E (ms) | Infer improvement | Pipeline improvement | E2E improvement |
|---|---|---:|---:|---:|---:|---:|---:|
| opt-3 | Pristine -> opt-3 | -15.90 | -15.77 | -15.69 | 5.42% | 5.27% | 5.21% |
| opt-4 | opt-3 -> opt-3+4 | -2.11 | -2.10 | -2.11 | 0.76% | 0.74% | 0.74% |
| opt-5 | opt-3+4 -> opt-3+4+5 | -0.12 | -0.20 | -0.12 | 0.04% | 0.07% | 0.04% |
| opt-6 | opt-3+4+5 -> opt-3+4+5+6 | -15.97 | -15.96 | -16.01 | 5.80% | 5.68% | 5.65% |

### 5.3 Opt-7 paired A/B

The paired base is measured in the same interleaved session as opt-7. It is therefore the
correct reference for the opt-7 incremental result; it is not substituted into the
cumulative table above.

| Paired stage | Infer TTFT (ms) | Pipeline TTFT (ms) | E2E TTFT (ms) |
|---|---:|---:|---:|
| opt-3+4+5+6 base | 260.31 | 265.89 | 268.13 |
| opt-3+4+5+6+opt-7 | 248.97 | 254.60 | 256.77 |
| Opt-7 delta | -11.34 (-4.36%) | -11.29 (-4.24%) | -11.36 (-4.24%) |

Per-repetition opt-7 paired deltas:

| Repetition | Infer delta (ms) | Pipeline delta (ms) | E2E delta (ms) |
|---:|---:|---:|---:|
| 1 | -9.82 | -9.84 | -9.83 |
| 2 | -11.07 | -11.05 | -11.13 |
| 3 | -11.44 | -11.51 | -11.57 |
| 4 | -11.61 | -11.52 | -11.58 |
| Median | -11.34 | -11.29 | -11.36 |

## 6. PA: Group_size=128 INT4

### 6.1 Cumulative stages

| Stage | Infer TTFT (ms) | Pipeline TTFT (ms) | E2E TTFT (ms) | Infer vs pristine | Pipeline vs pristine | E2E vs pristine |
|---|---:|---:|---:|---:|---:|---:|
| Pristine | 348.73 | 354.19 | 356.48 | 0.00% | 0.00% | 0.00% |
| + opt-3 | 334.12 | 339.93 | 342.02 | 4.19% | 4.03% | 4.06% |
| + opt-3 + opt-4 | 333.02 | 338.71 | 340.98 | 4.50% | 4.37% | 4.35% |
| + opt-3 + opt-4 + opt-5 | 332.07 | 337.81 | 340.03 | 4.78% | 4.62% | 4.61% |
| + opt-3 + opt-4 + opt-5 + opt-6 | 317.23 | 323.03 | 325.23 | 9.03% | 8.80% | 8.77% |

### 6.2 Incremental contribution of each optimization

| Optimization | Stage transition | Delta infer (ms) | Delta pipeline (ms) | Delta E2E (ms) | Infer improvement | Pipeline improvement | E2E improvement |
|---|---|---:|---:|---:|---:|---:|---:|
| opt-3 | Pristine -> opt-3 | -14.60 | -14.26 | -14.46 | 4.19% | 4.03% | 4.06% |
| opt-4 | opt-3 -> opt-3+4 | -1.10 | -1.22 | -1.04 | 0.33% | 0.36% | 0.30% |
| opt-5 | opt-3+4 -> opt-3+4+5 | -0.96 | -0.90 | -0.95 | 0.29% | 0.26% | 0.28% |
| opt-6 | opt-3+4+5 -> opt-3+4+5+6 | -14.83 | -14.78 | -14.79 | 4.47% | 4.38% | 4.35% |

### 6.3 Opt-7 paired A/B

For group_size=128, opt-7 is effectively neutral on this machine. The paired run is still
reported because it prevents incorrectly transferring the per-OC opt-7 gain to the grouped
quantization scheme.

| Paired stage | Infer TTFT (ms) | Pipeline TTFT (ms) | E2E TTFT (ms) |
|---|---:|---:|---:|
| opt-3+4+5+6 base | 317.37 | 323.05 | 325.25 |
| opt-3+4+5+6+opt-7 | 317.34 | 322.95 | 325.11 |
| Opt-7 delta | -0.04 (-0.01%) | -0.09 (-0.03%) | -0.14 (-0.04%) |

Per-repetition deltas for the group_size=128 paired run were:

| Repetition | Infer delta (ms) | Pipeline delta (ms) | E2E delta (ms) |
|---:|---:|---:|---:|
| 1 | +1.36 | +1.37 | +1.28 |
| 2 | -0.02 | -0.13 | -0.11 |
| 3 | +0.04 | +0.06 | +0.12 |
| 4 | -0.38 | -0.41 | -0.44 |
| Median | -0.04 | -0.09 | -0.14 |

## 7. SDPA: Per-OC INT4

### 7.1 Cumulative stages

| Stage | Infer TTFT (ms) | Pipeline TTFT (ms) | E2E TTFT (ms) | Infer vs pristine | Pipeline vs pristine | E2E vs pristine |
|---|---:|---:|---:|---:|---:|---:|
| Pristine | 470.24 | 473.07 | 474.59 | 0.00% | 0.00% | 0.00% |
| + opt-1 | 459.82 | 462.60 | 464.11 | 2.22% | 2.21% | 2.21% |
| + opt-1 + opt-2 | 431.88 | 434.63 | 436.11 | 8.16% | 8.12% | 8.11% |
| + opt-1 + opt-2 + opt-3 | 418.11 | 420.98 | 422.54 | 11.09% | 11.01% | 10.97% |

### 7.2 Incremental contribution of each optimization

| Optimization | Stage transition | Delta infer (ms) | Delta pipeline (ms) | Delta E2E (ms) | Infer improvement | Pipeline improvement | E2E improvement |
|---|---|---:|---:|---:|---:|---:|---:|
| opt-1 | Pristine -> opt-1 | -10.42 | -10.47 | -10.48 | 2.22% | 2.21% | 2.21% |
| opt-2 | opt-1 -> opt-1+2 | -27.94 | -27.96 | -27.99 | 6.08% | 6.05% | 6.03% |
| opt-3 | opt-1+2 -> opt-1+2+opt-3 | -13.77 | -13.65 | -13.57 | 3.19% | 3.14% | 3.11% |

## 8. SDPA: Group_size=128 INT4

### 8.1 Cumulative stages

| Stage | Infer TTFT (ms) | Pipeline TTFT (ms) | E2E TTFT (ms) | Infer vs pristine | Pipeline vs pristine | E2E vs pristine |
|---|---:|---:|---:|---:|---:|---:|
| Pristine | 522.37 | 525.09 | 526.66 | 0.00% | 0.00% | 0.00% |
| + opt-1 | 518.89 | 521.72 | 523.33 | 0.67% | 0.64% | 0.63% |
| + opt-1 + opt-2 | 490.67 | 493.46 | 495.01 | 6.07% | 6.02% | 6.01% |
| + opt-1 + opt-2 + opt-3 | 467.01 | 469.84 | 471.35 | 10.60% | 10.52% | 10.50% |

### 8.2 Incremental contribution of each optimization

| Optimization | Stage transition | Delta infer (ms) | Delta pipeline (ms) | Delta E2E (ms) | Infer improvement | Pipeline improvement | E2E improvement |
|---|---|---:|---:|---:|---:|---:|---:|
| opt-1 | Pristine -> opt-1 | -3.48 | -3.37 | -3.33 | 0.67% | 0.64% | 0.63% |
| opt-2 | opt-1 -> opt-1+2 | -28.22 | -28.26 | -28.32 | 5.44% | 5.42% | 5.41% |
| opt-3 | opt-1+2 -> opt-1+2+opt-3 | -23.66 | -23.62 | -23.67 | 4.82% | 4.79% | 4.78% |

## 9. Cross-Configuration Summary

### Final cumulative latency

| Backend | Quantization | Pristine infer | Final infer | Infer improvement | Pristine pipeline | Final pipeline | Pipeline improvement | Pristine E2E | Final E2E | E2E improvement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PA | Per-OC | 293.44 | 259.34 (+opt-6) | 11.62% | 299.08 | 265.05 | 11.38% | 301.24 | 267.31 | 11.27% |
| PA | Group_size=128 | 348.73 | 317.23 (+opt-6) | 9.03% | 354.19 | 323.03 | 8.80% | 356.48 | 325.23 | 8.77% |
| SDPA | Per-OC | 470.24 | 418.11 (+opt-3) | 11.09% | 473.07 | 420.98 | 11.01% | 474.59 | 422.54 | 10.97% |
| SDPA | Group_size=128 | 522.37 | 467.01 (+opt-3) | 10.60% | 525.09 | 469.84 | 10.52% | 526.66 | 471.35 | 10.50% |

### Main findings

1. PA per-OC: opt-3 and opt-6 are the major contributors. Opt-4 and opt-5 are small but positive on this memory-limited device. Opt-7 adds a stable 4.36% paired improvement.
2. PA group_size=128: opt-3 and opt-6 remain the major contributors, but their gains are smaller than for per-OC. Opt-7 is neutral within measurement noise, so its per-OC gain must not be generalized to grouped quantization.
3. SDPA per-OC: opt-2 is the largest single contributor, followed by opt-3 and then opt-1.
4. SDPA group_size=128: opt-2 and opt-3 remain the main contributors. Opt-1 is small but positive.
5. The grouped quantization model is slower than the per-OC model in both backends because grouped scale and zero-point dequantization costs more. This is a quantization-scheme effect, not an optimization regression.
6. Absolute values across separate sessions should not be compared more strongly than the paired incremental deltas. The reported stage order and repetition strategy are designed to reduce thermal drift, not eliminate all background variation.

## 10. Correctness Boundaries

- Opt-1 through opt-5 are intended to preserve the byte-level output contract, subject to the standard validation of the target model and workload.
- Opt-6 is bit-exact through prefill/first token but can change a long greedy decode because fused SiLU evaluation can differ by a small floating-point rounding amount.
- Opt-7 is bit-exact through prefill/first token but can change long greedy decode because folded DynamicQuantize nodes cannot use the original batch-1 f16 pass-through behavior.
- A deployment that requires byte-identical long decode should use opt-1 through opt-5 only. A deployment that only requires prefill/first-token equivalence can use the full measured PA stack.

## 11. Interpretation

The measurements show that the largest gains on this machine come from removing memory passes
or redundant graph work, not from increasing arithmetic throughput:

- PA opt-3 removes redundant DynamicQuantize work.
- PA opt-6 fuses standalone SiLU passes into their producers.
- PA opt-7 removes the intermediate RMSNorm-to-DynamicQuantize f16 hand-off for the per-OC case.
- SDPA opt-2 improves a memory-coalescing dispatch choice.
- SDPA opt-1 removes depthwise layout conversions.

The optimization ranking is hardware- and quantization-dependent. The tables above are the
measurement record for this machine and workload, not a universal prediction for every device.
