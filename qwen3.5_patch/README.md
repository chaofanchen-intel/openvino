# Qwen3.5 OpenVINO GPU-plugin patch archive

These patches target the OpenVINO 2026.4.1 release tag
`e213a14725769a6a2bf86fc86650aa78a9c05d3e`.

## Current PA series

The four current Qwen3.5 patches are copied from
[`Qwen3.5-Optimization/patches/`](../../../Qwen3.5-Optimization/patches/).
Apply them in order; patches 0001 and 0002 both modify the paged causal
convolution path.

| Patch | Change |
|---|---|
| [`0001-GPU-Fuse-the-Gated-DeltaNet-SiLUs-into-their-produce.patch`](0001-GPU-Fuse-the-Gated-DeltaNet-SiLUs-into-their-produce.patch) | Fuse Gated-DeltaNet SiLU operations into their producers |
| [`0002-GPU-Token-tiled-dispatch-for-paged_causal_conv1d.patch`](0002-GPU-Token-tiled-dispatch-for-paged_causal_conv1d.patch) | Tile the paged causal-convolution prefill dispatch |
| [`0003-GPU-Work-group-cooperative-scaled-state-PagedGatedDe.patch`](0003-GPU-Work-group-cooperative-scaled-state-PagedGatedDe.patch) | Add the cooperative scaled-state Gated-DeltaNet prefill kernel |
| [`0004-GPU-Fold-RMSNorm-into-DynamicQuantize.patch`](0004-GPU-Fold-RMSNorm-into-DynamicQuantize.patch) | Fold eligible RMSNorm work into DynamicQuantize |

```bash
# From a clean OpenVINO 2026.4.1 source checkout:
git apply /path/to/qwen3.5_patch/0001-GPU-Fuse-the-Gated-DeltaNet-SiLUs-into-their-produce.patch
git apply /path/to/qwen3.5_patch/0002-GPU-Token-tiled-dispatch-for-paged_causal_conv1d.patch
git apply /path/to/qwen3.5_patch/0003-GPU-Work-group-cooperative-scaled-state-PagedGatedDe.patch
git apply /path/to/qwen3.5_patch/0004-GPU-Fold-RMSNorm-into-DynamicQuantize.patch
cmake --build <build-dir> --target openvino_intel_gpu_plugin --parallel 8
```

## Retained SDPA-only patches

These two earlier patches are retained because they optimize the legacy SDPA
path and are not part of the current PA series in `Qwen3.5-Optimization`:

| Patch | Change |
|---|---|
| [`opt1_depthwise_conv_planar.patch`](opt1_depthwise_conv_planar.patch) | Keep fully-depthwise SSM convolutions in planar layout |
| [`opt2_permute_lws_f_axis.patch`](opt2_permute_lws_f_axis.patch) | Improve SDPA-path transpose work-group dispatch |

For an SDPA build, apply both after the current PA series (or independently
when only the SDPA changes are needed):

```bash
git apply /path/to/qwen3.5_patch/opt1_depthwise_conv_planar.patch
git apply /path/to/qwen3.5_patch/opt2_permute_lws_f_axis.patch
```

Do not use the historical `opt3`–`opt7` instructions in
[`README_legacy.md`](README_legacy.md); those patch files were superseded.

## Results and validation

Current PA-series performance, correctness, and focused tests are documented
in [`Qwen3.5-Optimization/docs/results.md`](../../../Qwen3.5-Optimization/docs/results.md)
and [`Qwen3.5-Optimization/patches/README.md`](../../../Qwen3.5-Optimization/patches/README.md).
Patch replay and the scope of historical compile checks are summarized in
[`../VALIDATION_2026.4.1.md`](../VALIDATION_2026.4.1.md).
