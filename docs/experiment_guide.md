# Experiment Guide

## Prerequisites

Run all commands from the repository root. Install dependencies first:

```bash
python -m pip install -r requirements.txt
```

For paper-scale training, download
`radial_velocity_direct_inversion_training_dataset_v1.mat` from Mendeley Data
and put it in `datasets/`. The MAT file contains `velocity2` (`N x 256`) and
`param2` (`N x 10`). MATLAB is not required to run these Python experiments.

## Step 1: build derived datasets

This is required before the sampling, quality, structured-output, and encoder
experiments. It uses the full MAT dataset and writes files locally:

```bash
python src/build_paper_sampling_datasets.py
```

To set paths explicitly:

```bash
python src/build_paper_sampling_datasets.py --source-data datasets/radial_velocity_direct_inversion_training_dataset_v1.mat --output-dir datasets/derived --target-size 51047
```

The expected derived MAT files are D1, D2, D3, and P10/P20/P30 under
`datasets/derived/mat/`. Dataset audits and source indices are saved below
`datasets/derived/reports/` and `datasets/derived/indices/`.

The repository also retains the later contrast design that isolates missing
quality control, incomplete 1200-3000 m/s parameter support, and full-range
coverage-constrained sampling. Build it with:

```bash
python src/build_sampling_mode_contrast_datasets.py
python src/run_sampling_strategy_benchmark.py --config ../configs/sampling_mode_contrast_config.json --output-name sampling_mode_contrast
```

## Step 2: run the paper experiments

Each full command uses three training seeds and up to 80 epochs unless noted.
A CUDA-capable TensorFlow installation is recommended. Use a different output
name for every run.

| Paper analysis | Code and configuration | Command |
| --- | --- | --- |
| Main Transformer-SP noise experiment | `src/run_architecture_benchmark.py`, `configs/paper_common_transformer_sp.json` | `python src/run_architecture_benchmark.py --config ../configs/paper_common_transformer_sp.json --output-name paper_main_noise` |
| D1/D2/D3 sampling comparison | `src/run_sampling_strategy_benchmark.py`, `configs/paper_sampling_d1_d2_d3.json` | `python src/run_sampling_strategy_benchmark.py --config ../configs/paper_sampling_d1_d2_d3.json --output-name paper_sampling` |
| P10/P20/P30 data-quality comparison | `src/run_sampling_strategy_benchmark.py`, `configs/paper_data_quality_p10_p30.json` | `python src/run_sampling_strategy_benchmark.py --config ../configs/paper_data_quality_p10_p30.json --output-name paper_quality` |
| Training-set scale, 10k to 200k | `src/run_data_scale_benchmark.py`, `configs/paper_data_scale_10k_200k.json` | `python src/run_data_scale_benchmark.py --config ../configs/paper_data_scale_10k_200k.json --output-name paper_scale` |
| Free / soft / structured output | `src/run_architecture_benchmark.py`, `configs/structured_output_threeway_paper_config.json` | `python src/run_architecture_benchmark.py --config ../configs/structured_output_threeway_paper_config.json --output-name paper_output_modes` |
| Five encoder families | `src/run_architecture_benchmark.py`, `configs/paper_encoder_five_architectures.json` | `python src/run_architecture_benchmark.py --config ../configs/paper_encoder_five_architectures.json --output-name paper_encoders` |
| MLP depth and data scale | `src/run_complexity_scale_benchmark.py`, `configs/mlp_depth_scale_config.json` | `python src/run_complexity_scale_benchmark.py --config ../configs/mlp_depth_scale_config.json --output-name paper_mlp_depth` |

## Outputs and manuscript tables

Every run creates `runs/<output-name>/`, containing a copy of the used
configuration, CSV metrics, training histories, and a Markdown report. The
final aggregate tables cited in the manuscript are retained in `data/` and are
mapped in `data/README.md`.

Small numerical changes can occur with different GPU kernels, TensorFlow
versions, and early stopping. Keep the supplied seeds, sample counts, bounds,
and noise settings unchanged for a quantitative reproduction.

## Protocol notes

- D1 is a weakly organized, middle-biased subset; D2 is physics-filtered
  random sampling; D3 adds coverage balancing over reference velocity, total
  rise, and curvature.
- P10/P20/P30 retain D2 as the base data and inject the documented
  label-corruption mechanisms.
- The scale experiment uses 10k, 30k, 50k, 70k, 100k, 120k, 150k, and 200k.
- The `src/` and `configs/` directories contain the complete public entry
  points and settings required for reproduction.

## What the examples do not do

`examples/quick_test.py` does not call the dipole-dispersion forward solver or
train TensorFlow. It generates format-compatible synthetic data and checks
constraints. The smoke benchmark trains for only two epochs on that synthetic
data. Both examples verify execution, not manuscript figures or quantitative
results.
