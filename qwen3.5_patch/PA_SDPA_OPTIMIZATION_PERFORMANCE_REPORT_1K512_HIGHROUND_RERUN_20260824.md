# Qwen3.5 1k-512 Optimization Performance Rerun - High-Round model_bench and Corrected GenAI

Date: 2026-08-24

This report supersedes the earlier single-process report
`PA_SDPA_OPTIMIZATION_PERFORMANCE_REPORT_1K512_RERUN_20260824.md`. All primary numbers
here are recomputed from the high-round raw results in
[results_1k512_highround_rerun_20260824](results_1k512_highround_rerun_20260824).
No latency or throughput value was copied from the earlier low-round result directory.

## Executive summary

`model_bench` is the primary method. The main metric is the mean first-token model
inference latency (`1st infer`). `TTFT` is the pipeline first-token latency. For
`model_bench`, each stage has five repetitions, and each repetition has five measured
iterations plus one warm-up; the primary tables use the mean of 25 measured samples.
The corrected GenAI cross-check uses one invocation per stage/model with `-n 5`, so its
tables use five measured samples plus one warm-up per stage/model.

| Backend | Quantization | Pristine 1st infer mean (ms) | Final 1st infer mean (ms) | Improvement | Pristine TTFT mean (ms) | Final TTFT mean (ms) | Improvement |
|---|---|---:|---:|---:|---:|---:|---:|
| PA | per-OC | 300.205 | 256.601 (+opt7) | 14.52% | 309.412 | 265.784 (+opt7) | 14.10% |
| PA | gs128 | 355.340 | 322.175 (+opt7) | 9.33% | 364.450 | 331.174 (+opt7) | 9.13% |
| SDPA | per-OC | 470.112 | 418.477 (+opt3) | 10.98% | 476.409 | 424.823 (+opt3) | 10.83% |
| SDPA | gs128 | 525.482 | 476.518 (+opt3) | 9.32% | 532.107 | 483.270 (+opt3) | 9.18% |

The previous five-iteration single-process result made PA per-OC opt7 look unstable
because it included a large cold/background outlier in one process. With five repeated
processes, the high-round mean shows opt6 and opt7 clearly again:

- PA per-OC opt6: `281.115 -> 265.473 ms` first-infer mean, `+5.38%` increment.
- PA per-OC opt7: `265.473 -> 256.601 ms`, `+3.34%` increment.
- PA gs128 opt6: `337.759 -> 322.169 ms`, `+4.63%` increment.
- PA gs128 opt7: effectively neutral, `322.169 -> 322.175 ms`.

The corrected GenAI run now agrees with the primary result direction. Its final
first-infer improvements are `+14.40%` (PA per-OC), `+9.49%` (PA gs128), `+10.31%`
(SDPA per-OC), and `+9.36%` (SDPA gs128). The earlier GenAI files were invalid for
stage comparison because the Python OpenVINO binding loaded its package-local runtime;
the corrected run forces the fresh install core library with `LD_PRELOAD`.

## Hardware and workload

| Item | Value |
|---|---|
| CPU | Intel(R) Core(TM) Ultra 7 356H |
| CPU topology | 16 online CPUs, 16 cores, 1 thread per core |
| GPU | Intel Graphics, xpu-smi device 0, state normal |
| Device | GPU |
| Models | `benchmark/models/Qwen3.5-2B-INT4` and `benchmark/models/Qwen3.5-2B-INT4-g128` |
| Quantization | per-OC: group size -1, symmetric; gs128: group size 128, grouped scale and zero-point |
| Prompt | `benchmark/tests/prompts/qwen3.5_1k.jsonl` |
| Effective input | 1002 tokens |
| Batch | 1 |
| Generation | `infer_count=512`, 512 output tokens observed |
| Prompt permutation | Enabled |
| Performance hint | `LATENCY` |
| model_bench samples | 5 repetitions x 5 measured iterations = 25 per stage/model |
| GenAI samples | 1 invocation x 5 measured iterations = 5 per stage/model |
| Warm-up | One per process, excluded |
| model_bench monitor | Disabled with `--no-monitor` |
| PA extra fusion | `OV_SWISH_HOIST_ALL=1`, matching the previous PA stage runner |
| OpenVINO runtime | `2026.3.0-22454-09c4cc4117e-qwen-ttft-optimizations` |
| GenAI runtime | `2026.3.0.0-3278-0c38b54d7e7-qwen-ttft-llm-bench` |

## Stage construction and test method

The tests were performed directly in `benchmark/openvino`, starting from
`09c4cc4117e220cb6771a626d30a45ed027901cd` and using freshly built plugin binaries.
For model_bench, each process hot-swapped the selected stage plugin into the OpenVINO
install directory before running the benchmark; five repetitions visited every stage in
order. GenAI was rerun once per stage/model after correcting the Python runtime loading
path, with `-n 5` and the same plugin hot-swap method.

| Backend | Cumulative stages |
|---|---|
| PA | pristine -> opt3 -> opt3+opt4 -> opt3+opt4+opt5 -> opt3+opt4+opt5+opt6 -> opt3+opt4+opt5+opt6+opt7 |
| SDPA | pristine -> opt1 -> opt1+opt2 -> opt1+opt2+opt3 |

| Optimization | Main change | Backend |
|---|---|---|
| opt1 | Keep the fully depthwise SSM convolution planar and remove unnecessary layout conversions. | SDPA |
| opt2 | Correct the `permute_ref` local work-group placement for the memory-coalesced axis. | SDPA |
| opt3 | Reuse compatible DynamicQuantize nodes shared by FullyConnected operators. | PA and SDPA |
| opt4 | Tile prefill tokens in the causal-convolution dispatch. | PA |
| opt5 | Enable the vectorized activation kernel for the safe dynamic-shape subset. | PA and supported dynamic activation paths |
| opt6 | Hoist and fuse SiLU into its producer. | PA |
| opt7 | Fold RMSNorm into DynamicQuantize. | PA and supported DynamicQuantize paths |

The build used `ENABLE_PYTHON=OFF`, `ENABLE_TESTS=OFF`, and
`ENABLE_FUNCTIONAL_TESTS=OFF` because only the GPU plugin was rebuilt. The full source
and plugin build logs are preserved in the high-round result directory.

The model_bench command used `-n 5`, `-ic 512`, the 1002-token prompt, and the
corresponding PA or SDPA config. The corrected GenAI command used the same workload and
`-n 5`, once per stage/model. It also set
`LD_PRELOAD=benchmark/openvino/install/runtime/lib/intel64/libopenvino.so.2630` so the
Python binding's package-local RPATH could not load the stale
`build_python3.12/.../openvino/libs` core runtime. Without this preload, replacing the
install-directory plugin does not reliably change the plugin used by the Python binding;
those earlier GenAI files were discarded as invalid stage comparisons.

## Model_bench primary results: means

All values in these tables are means over 25 measured samples per stage/model. `Compile`
is the mean compile time recorded by the five processes; it is informational and is not
included in the optimization percentages.

### PA per-OC

| Stage | Compile (s) | TTFT (ms) | 1st infer (ms) | TPOT (ms) | Total TPS | Decode TPS | Generation (s) | Decode (ms) | E2E (ms) | Tokenize (ms) | Detokenize (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 5.118 | 309.412 | 300.205 | 32.461 | 30.296 | 30.806 | 16.89999 | 16590.577 | 16899.989 | 1.0442 | 0.4096 |
| opt3 | 4.894 | 291.898 | 282.798 | 32.362 | 30.419 | 30.901 | 16.83165 | 16539.755 | 16831.653 | 1.0294 | 0.4190 |
| opt3+opt4 | 4.926 | 290.174 | 281.114 | 32.394 | 30.392 | 30.870 | 16.84643 | 16556.260 | 16846.434 | 1.0328 | 0.3971 |
| opt3+opt4+opt5 | 4.871 | 289.674 | 280.579 | 32.404 | 30.384 | 30.861 | 16.85097 | 16561.290 | 16850.965 | 1.0401 | 0.4395 |
| opt3+opt4+opt5+opt6 | 4.862 | 274.663 | 265.473 | 32.323 | 30.486 | 30.938 | 16.79448 | 16519.812 | 16794.475 | 1.0521 | 0.4255 |
| opt3+opt4+opt5+opt6+opt7 | 4.860 | 265.784 | 256.601 | 32.201 | 30.616 | 31.055 | 16.72336 | 16457.576 | 16723.360 | 1.0412 | 0.4152 |

### PA gs128

| Stage | Compile (s) | TTFT (ms) | 1st infer (ms) | TPOT (ms) | Total TPS | Decode TPS | Generation (s) | Decode (ms) | E2E (ms) | Tokenize (ms) | Detokenize (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 5.737 | 364.450 | 355.340 | 33.527 | 29.258 | 29.827 | 17.49948 | 17135.031 | 17499.481 | 1.0406 | 0.4078 |
| opt3 | 5.716 | 348.302 | 339.394 | 33.444 | 29.356 | 29.901 | 17.44127 | 17092.970 | 17441.272 | 1.0469 | 0.4197 |
| opt3+opt4 | 5.803 | 346.736 | 337.759 | 33.451 | 29.352 | 29.894 | 17.44330 | 17096.560 | 17443.295 | 1.0378 | 0.4409 |
| opt3+opt4+opt5 | 5.687 | 346.809 | 337.808 | 33.441 | 29.361 | 29.904 | 17.43802 | 17091.215 | 17438.024 | 1.0466 | 0.4322 |
| opt3+opt4+opt5+opt6 | 5.640 | 331.253 | 322.169 | 33.417 | 29.408 | 29.925 | 17.41033 | 17079.073 | 17410.327 | 1.0388 | 0.4380 |
| opt3+opt4+opt5+opt6+opt7 | 5.732 | 331.174 | 322.175 | 33.397 | 29.425 | 29.943 | 17.40018 | 17069.010 | 17400.184 | 1.0438 | 0.4124 |

### SDPA per-OC

| Stage | Compile (s) | TTFT (ms) | 1st infer (ms) | TPOT (ms) | Total TPS | Decode TPS | Generation (s) | Decode (ms) | E2E (ms) | Tokenize (ms) | Detokenize (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 5.505 | 476.409 | 470.112 | 38.529 | 25.404 | 25.972 | 20.16651 | 19690.099 | 20166.508 | 1.1205 | 0.4704 |
| opt1 | 5.361 | 467.344 | 460.851 | 37.950 | 25.804 | 26.378 | 19.86145 | 19394.106 | 19861.450 | 1.1125 | 0.5418 |
| opt1+opt2 | 4.988 | 442.807 | 436.141 | 38.911 | 25.187 | 25.700 | 20.32806 | 19885.257 | 20328.064 | 1.1109 | 0.5225 |
| opt1+opt2+opt3 | 4.958 | 424.823 | 418.477 | 38.169 | 25.707 | 26.219 | 19.93101 | 19506.188 | 19931.011 | 1.1075 | 0.4928 |

### SDPA gs128

| Stage | Compile (s) | TTFT (ms) | 1st infer (ms) | TPOT (ms) | Total TPS | Decode TPS | Generation (s) | Decode (ms) | E2E (ms) | Tokenize (ms) | Detokenize (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 5.939 | 532.107 | 525.482 | 40.219 | 24.284 | 24.866 | 21.08614 | 20554.031 | 21086.139 | 1.1398 | 0.5650 |
| opt1 | 5.966 | 527.763 | 520.847 | 40.148 | 24.329 | 24.908 | 21.04539 | 20517.624 | 21045.387 | 1.1454 | 0.5354 |
| opt1+opt2 | 5.790 | 500.886 | 493.877 | 39.986 | 24.462 | 25.016 | 20.93565 | 20434.760 | 20935.646 | 1.1463 | 0.5650 |
| opt1+opt2+opt3 | 5.686 | 483.270 | 476.518 | 39.071 | 25.054 | 25.613 | 20.45055 | 19967.281 | 20450.551 | 1.1346 | 0.5232 |

## Model_bench cumulative and incremental changes

The cumulative columns use pristine as the denominator. The incremental columns use the
immediately preceding stage as the denominator. Positive means lower latency; negative
means regression.

### PA per-OC

| Stage | TTFT mean (ms) | vs pristine | incremental | 1st infer mean (ms) | vs pristine | incremental | E2E mean (ms) | vs pristine | incremental |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 309.412 | 0.00% | - | 300.205 | 0.00% | - | 16899.989 | 0.00% | - |
| opt3 | 291.898 | +5.66% | +5.66% | 282.798 | +5.80% | +5.80% | 16831.653 | +0.40% | +0.40% |
| opt3+opt4 | 290.174 | +6.22% | +0.59% | 281.114 | +6.36% | +0.60% | 16846.434 | +0.32% | -0.09% |
| opt3+opt4+opt5 | 289.674 | +6.38% | +0.17% | 280.579 | +6.54% | +0.19% | 16850.965 | +0.29% | -0.03% |
| opt3+opt4+opt5+opt6 | 274.663 | +11.23% | +5.18% | 265.473 | +11.57% | +5.38% | 16794.475 | +0.62% | +0.34% |
| opt3+opt4+opt5+opt6+opt7 | 265.784 | +14.10% | +3.23% | 256.601 | +14.52% | +3.34% | 16723.360 | +1.05% | +0.42% |

### PA gs128

| Stage | TTFT mean (ms) | vs pristine | incremental | 1st infer mean (ms) | vs pristine | incremental | E2E mean (ms) | vs pristine | incremental |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 364.450 | 0.00% | - | 355.340 | 0.00% | - | 17499.481 | 0.00% | - |
| opt3 | 348.302 | +4.43% | +4.43% | 339.394 | +4.49% | +4.49% | 17441.272 | +0.33% | +0.33% |
| opt3+opt4 | 346.736 | +4.86% | +0.45% | 337.759 | +4.95% | +0.48% | 17443.295 | +0.32% | -0.01% |
| opt3+opt4+opt5 | 346.809 | +4.84% | -0.02% | 337.808 | +4.93% | -0.01% | 17438.024 | +0.35% | +0.03% |
| opt3+opt4+opt5+opt6 | 331.253 | +9.11% | +4.49% | 322.169 | +9.34% | +4.63% | 17410.327 | +0.51% | +0.16% |
| opt3+opt4+opt5+opt6+opt7 | 331.174 | +9.13% | +0.02% | 322.175 | +9.33% | -0.00% | 17400.184 | +0.57% | +0.06% |

### SDPA per-OC

| Stage | TTFT mean (ms) | vs pristine | incremental | 1st infer mean (ms) | vs pristine | incremental | E2E mean (ms) | vs pristine | incremental |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 476.409 | 0.00% | - | 470.112 | 0.00% | - | 20166.508 | 0.00% | - |
| opt1 | 467.344 | +1.90% | +1.90% | 460.851 | +1.97% | +1.97% | 19861.450 | +1.51% | +1.51% |
| opt1+opt2 | 442.807 | +7.05% | +5.25% | 436.141 | +7.23% | +5.36% | 20328.064 | -0.80% | -2.35% |
| opt1+opt2+opt3 | 424.823 | +10.83% | +4.06% | 418.477 | +10.98% | +4.05% | 19931.011 | -1.17% | +1.95% |

### SDPA gs128

| Stage | TTFT mean (ms) | vs pristine | incremental | 1st infer mean (ms) | vs pristine | incremental | E2E mean (ms) | vs pristine | incremental |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 532.107 | 0.00% | - | 525.482 | 0.00% | - | 21086.139 | 0.00% | - |
| opt1 | 527.763 | +0.82% | +0.82% | 520.847 | +0.88% | +0.88% | 21045.387 | +0.19% | +0.19% |
| opt1+opt2 | 500.886 | +5.87% | +5.09% | 493.877 | +6.01% | +5.18% | 20935.646 | +0.71% | +0.52% |
| opt1+opt2+opt3 | 483.270 | +9.18% | +3.52% | 476.518 | +9.32% | +3.51% | 20450.551 | +3.01% | +2.32% |

## GenAI benchmark.py cross-check means

These values come from one corrected `benchmark.py -n 5` invocation per stage/model.
Each raw GenAI JSON contains one warm-up record, five measured records, and
`results_averaged`; GenAI is reported separately from model_bench and is not pooled with
it. The previous 100 per-repetition GenAI result sets were removed because the Python
binding loaded a stale package-local OpenVINO core/plugin instead of the hot-swapped
stage plugin.

### PA per-OC

| Stage | Generation (s) | Latency (ms/token) | 1st token (ms) | Other token (ms) | 1st infer (ms) | Other infer (ms) | 2nd tok/s | Total tok/s | Tokenize (ms) | Detokenize (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 17.02657 | 33.25501 | 308.498 | 32.71003 | 299.479 | 32.70886 | 30.572 | 30.071 | 1.1272 | 0.4474 |
| opt3 | 16.95166 | 33.10872 | 292.376 | 32.59504 | 283.184 | 32.59381 | 30.680 | 30.204 | 1.1338 | 0.4090 |
| opt3+opt4 | 16.96271 | 33.13030 | 291.263 | 32.61860 | 282.080 | 32.61734 | 30.657 | 30.184 | 1.1316 | 0.5290 |
| opt3+opt4+opt5 | 16.94031 | 33.08655 | 289.435 | 32.57845 | 280.276 | 32.57729 | 30.695 | 30.224 | 1.1352 | 0.4810 |
| opt3+opt4+opt5+opt6 | 16.86326 | 32.93606 | 274.197 | 32.45770 | 264.863 | 32.45665 | 30.809 | 30.362 | 1.1224 | 0.4460 |
| opt3+opt4+opt5+opt6+opt7 | 16.85859 | 32.92694 | 265.592 | 32.46517 | 256.366 | 32.46389 | 30.802 | 30.370 | 1.1322 | 0.5118 |

### PA gs128

| Stage | Generation (s) | Latency (ms/token) | 1st token (ms) | Other token (ms) | 1st infer (ms) | Other infer (ms) | 2nd tok/s | Total tok/s | Tokenize (ms) | Detokenize (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 17.57097 | 34.31830 | 363.963 | 33.66679 | 354.957 | 33.66565 | 29.703 | 29.139 | 1.1480 | 0.4974 |
| opt3 | 17.75212 | 34.67210 | 349.689 | 34.04902 | 340.485 | 34.04784 | 29.369 | 28.846 | 1.1462 | 0.4674 |
| opt3+opt4 | 17.52193 | 34.22252 | 346.902 | 33.60426 | 337.567 | 33.60304 | 29.758 | 29.221 | 1.1516 | 0.4628 |
| opt3+opt4+opt5 | 17.48414 | 34.14872 | 345.251 | 33.53358 | 336.129 | 33.53249 | 29.821 | 29.284 | 1.1388 | 0.4512 |
| opt3+opt4+opt5+opt6 | 17.46964 | 34.12039 | 331.217 | 33.53256 | 321.971 | 33.53131 | 29.822 | 29.308 | 1.1620 | 0.4276 |
| opt3+opt4+opt5+opt6+opt7 | 17.49465 | 34.16925 | 330.241 | 33.58356 | 321.264 | 33.58239 | 29.776 | 29.266 | 1.1636 | 0.4344 |

### SDPA per-OC

| Stage | Generation (s) | Latency (ms/token) | 1st token (ms) | Other token (ms) | 1st infer (ms) | Other infer (ms) | 2nd tok/s | Total tok/s | Tokenize (ms) | Detokenize (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 20.10098 | 39.25972 | 474.210 | 38.40427 | 467.606 | 37.65408 | 26.039 | 25.494 | 1.1362 | 0.6282 |
| opt1 | 19.27893 | 37.65416 | 468.271 | 36.80744 | 461.879 | 36.05542 | 27.168 | 26.558 | 1.1548 | 0.5314 |
| opt1+opt2 | 20.25155 | 39.55381 | 437.632 | 38.77057 | 431.064 | 37.99361 | 25.793 | 25.299 | 1.1774 | 0.5852 |
| opt1+opt2+opt3 | 19.89582 | 38.85903 | 425.807 | 38.09772 | 419.383 | 37.31170 | 26.248 | 25.754 | 1.1508 | 0.5502 |

### SDPA gs128

| Stage | Generation (s) | Latency (ms/token) | 1st token (ms) | Other token (ms) | 1st infer (ms) | Other infer (ms) | 2nd tok/s | Total tok/s | Tokenize (ms) | Detokenize (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine | 20.07408 | 39.20719 | 531.177 | 38.24014 | 524.419 | 37.47566 | 26.151 | 25.506 | 1.2134 | 0.5786 |
| opt1 | 21.21148 | 41.42866 | 525.158 | 40.47764 | 518.396 | 39.69870 | 24.705 | 24.138 | 1.1744 | 0.6400 |
| opt1+opt2 | 21.01289 | 41.04080 | 501.786 | 40.13483 | 495.009 | 39.39653 | 24.916 | 24.367 | 1.1724 | 0.6030 |
| opt1+opt2+opt3 | 20.37683 | 39.79850 | 481.827 | 38.92908 | 475.311 | 38.15368 | 25.688 | 25.156 | 1.1610 | 0.6836 |

## GenAI cumulative and incremental changes

| Backend | Quantization | Final stage | 1st token vs pristine | 1st infer vs pristine | Main reading |
|---|---|---|---:|---:|---|
| PA | per-OC | opt3+4+5+6+7 | +13.91% | +14.40% | Agrees with model_bench |
| PA | gs128 | opt3+4+5+6+7 | +9.27% | +9.49% | Agrees with model_bench |
| SDPA | per-OC | opt1+2+3 | +10.21% | +10.31% | Agrees with model_bench |
| SDPA | gs128 | opt1+2+3 | +9.29% | +9.36% | Agrees with model_bench |

The corrected GenAI cross-check and model_bench are not numerically identical, but their
optimization directions now agree. The primary ranking still follows model_bench, while
GenAI remains available as an independent method with its own raw records.

## Raw data inventory

The high-round directory contains:

- 100 `*_model_bench.json` files and 100 matching model_bench CSV files.
- 20 corrected GenAI JSON files and 20 matching GenAI CSV files. Each is one `-n 5` run.
- 120 benchmark logs, plus the two plugin-build logs.
- Ten stage plugin binaries under `plugins/`, plus the PA and SDPA config files.

Each model_bench JSON/CSV retains the complete per-process aggregate fields, including
mean, standard deviation, p50, p90, p95, minimum, and maximum for TTFT, TPOT, first-token
inference, other-token inference, throughput, decode throughput, generation time, prefill,
decode, E2E, tokenization, and detokenization. Each GenAI JSON retains the warm-up,
five measured iterations, averaged fields, resource data, and result MD5 values.

Representative files:

- [PA pristine per-OC model_bench r1](results_1k512_highround_rerun_20260824/pa_pristine_per_oc_r1_model_bench.json)
- [PA opt6 per-OC model_bench r5](results_1k512_highround_rerun_20260824/pa_opt3_opt4_opt5_opt6_per_oc_r5_model_bench.json)
- [PA opt7 per-OC model_bench r5](results_1k512_highround_rerun_20260824/pa_opt3_opt4_opt5_opt6_opt7_per_oc_r5_model_bench.json)
- [SDPA pristine per-OC GenAI](results_1k512_highround_rerun_20260824/sdpa_pristine_per_oc.json)
- [SDPA opt3 gs128 GenAI](results_1k512_highround_rerun_20260824/sdpa_opt1_opt2_opt3_gs128.json)

## Interpretation

1. The high-round model_bench means restore the expected PA opt6 gain on both quantization schemes.
2. PA opt7 is a clear additional per-OC improvement, while it is neutral for gs128.
3. PA opt3 is the first major improvement; opt4 is a small positive increment; opt5 is near neutral on this machine.
4. SDPA opt1, opt2, and opt3 all improve the model_bench first-token mean, with opt2 the largest single SDPA increment.
5. The corrected GenAI cross-check agrees with the model_bench optimization direction. It remains a separate measurement method and is not pooled with the C++ results.
6. The mean is the primary statistic in this report. Per-process p50/p90/p95/min/max remain in every raw file for diagnosing thermal or background outliers.
7. The model_bench stage sequence is repeated five times but is not a strict paired A/B measurement within one process. GenAI uses one `-n 5` run per stage/model as requested. Both methods hot-swap stage binaries; future paired work should retain the alternating-order design and report pairwise differences.
