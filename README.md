# Collaborative Design Framework for Geophysical Direct Inversion

Code, configurations, example data, and aggregate results for the manuscript
*Collaborative Design Framework for Geophysical Direct Inversion and Its
Application in Radial Velocity Inversion*.

The study considers direct inversion of dipole-acoustic dispersion data for a
10-layer radial shear-wave velocity profile. Rather than treating inversion as
only a network-selection problem, the workflow jointly examines training-data
construction, physically feasible output parameterization, and task-adapted
encoder selection.

## Research Scope

The public code supports the following analyses reported in the manuscript:

1. Noise robustness of the structured Transformer-based model under clean,
   30 dB, 20 dB, and 10 dB dispersion inputs.
2. Sampling-organization comparisons: weakly organized random sampling,
   physics-filtered sampling, and coverage-constrained sampling.
3. Controlled P10, P20, and P30 training-data contamination experiments.
4. Training-set scale experiments from 10,000 to 200,000 samples.
5. Free-output, soft-constrained-output, and structured-output ablations.
6. MLP, 1D-CNN, CNN-BiLSTM, CNN-BiLSTM-Attention, and Transformer comparisons.
7. MLP depth and data-scale sensitivity experiments.

## Repository Layout

```text
src/        Python scripts for data construction, training, and evaluation
configs/    JSON settings for each manuscript experiment
data/       Final aggregate CSV tables reported in the manuscript
docs/       Data-format description and full reproduction instructions
examples/   Directly runnable checks and a 10,000-sample training subset
```

| Task | Script |
| --- | --- |
| Build D1/D2/D3 and P10/P20/P30 data | `src/build_paper_sampling_datasets.py` |
| Build random/range-biased/coverage contrast data | `src/build_sampling_mode_contrast_datasets.py` |
| Output-mode and encoder comparisons | `src/run_architecture_benchmark.py` |
| Sampling and data-quality comparisons | `src/run_sampling_strategy_benchmark.py` |
| Data-scale experiment | `src/run_data_scale_benchmark.py` |
| MLP-depth experiment | `src/run_complexity_scale_benchmark.py` |

## Requirements and Installation

The code was developed with Python 3.10 and TensorFlow 2.16. A CUDA-capable
TensorFlow environment is recommended for the full protocol.

```bash
git clone https://github.com/dinghuaishuo-0921/Collaborative-Design-Framework-for-Geophysical-Direct-Inversion-.git
cd Collaborative-Design-Framework-for-Geophysical-Direct-Inversion-
python -m pip install -r requirements.txt
```

## Quick Start

### 1. Format and constraint check

This command runs without the full Mendeley dataset. It generates a 128-sample
synthetic MAT file with the same variable names and dimensions used by the
training scripts, verifies the structured-output mapping, and writes
`results/quick_test_summary.json`.

```bash
python examples/quick_test.py
```

The synthetic arrays have `velocity2: 128 x 256` and `param2: 128 x 10`.
This is a software and data-format check, not a reproduction of paper metrics.

### 2. End-to-end smoke training

After installing TensorFlow, run the two-epoch smoke benchmark:

```bash
python src/run_architecture_benchmark.py --config ../configs/quick_benchmark_example.json --output-name quick_benchmark --smoke
```

It trains MLP and Transformer-SP on the synthetic example and writes metrics,
histories, and the configuration copy under `runs/quick_benchmark/`. It only
confirms that the public training pipeline executes on a new machine.

`examples/training_subset_10000.mat` is a real forward-modelled 10,000-sample
subset for user-developed training demonstrations.

## Full Reproduction

The full training pool is released separately through Mendeley Data because it
is approximately 493 MB. Download
`radial_velocity_direct_inversion_training_dataset_v1.mat` and place it at:

```text
datasets/radial_velocity_direct_inversion_training_dataset_v1.mat
```

The MAT file contains `velocity2` (`N x 256` paired dispersion inputs) and
`param2` (`N x 10` radial shear-wave velocity targets in m/s). Build the
derived datasets before the sampling, data-quality, output-mode, or encoder
experiments:

```bash
python src/build_paper_sampling_datasets.py
```

Use the exact commands and expected output tables in
[`docs/experiment_guide.md`](docs/experiment_guide.md). The JSON files retain
the reported sample counts, seeds, velocity bounds, noise settings, and
training parameters. Full runs use three seeds and up to 80 epochs; GPU
acceleration is recommended.

## Reported Results

The `data/` folder stores final compact result tables. For example,
`transformer_scale_consistent_10k_200k.csv` contains the final eight-point
sample-size experiment and `encoder_replaceability.csv` contains the
five-encoder comparison. See [`data/README.md`](data/README.md) for the full
manuscript-to-table mapping.

## Data and Code Availability

- **Code, configurations, examples, and aggregate results:** this repository.
- **Full synthetic training data:** associated Mendeley Data record; the DOI
  will be added after publication.
- **Field logging data:** not redistributed because of data-use restrictions.

## License

This repository is distributed under the [MIT License](LICENSE).
