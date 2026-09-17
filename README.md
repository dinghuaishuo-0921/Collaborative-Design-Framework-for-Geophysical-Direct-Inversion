# Collaborative Design Framework for Geophysical Direct Inversion

This repository contains the executable experimental code for the manuscript
*Collaborative Design Framework for Geophysical Direct Inversion and Its
Application in Radial Velocity Inversion*. The case study estimates a 10-layer
radial shear-wave velocity profile from a 256-point dipole-acoustic dispersion
response.

This package contains the training, dataset-construction, evaluation, and aggregate-result
files needed to reproduce the reported protocol. The complete forward-modelled
training dataset is distributed separately through Mendeley Data because it is
approximately 493 MB.

## Contents

- `src/`: maintained Python entry points for dataset construction, training,
  and evaluation.
- `configs/`: portable JSON configurations for every paper experiment.
- `data/`: final aggregate CSV tables reported in the manuscript.
- `examples/`: directly runnable quick checks and a 10,000-sample subset.
- `docs/`: data format, result-table mapping, and full run instructions.

## Installation

The code was developed with Python 3.10 and TensorFlow 2.16.

```bash
python -m pip install -r requirements.txt
```

## Examples: what runs immediately

**Format and constraint check**. This runs immediately after installing NumPy
and SciPy. It creates a 128-sample synthetic MAT file, verifies the data format
and structured-output mapping, and writes a JSON summary. It does not train a
neural network or reproduce manuscript values.

```bash
python examples/quick_test.py
```

**End-to-end smoke training**. After the first command and TensorFlow
installation, this trains MLP and Transformer-SP for two epochs on the
synthetic example. It proves the public pipeline executes, but is deliberately
too small to reproduce manuscript metrics.

```bash
python src/run_architecture_benchmark.py --config ../configs/quick_benchmark_example.json --output-name quick_benchmark --smoke
```

`examples/training_subset_10000.mat` is a real forward-modelled 10,000-sample
subset for user-developed training demonstrations. It is not a replacement for
the full release or the derived D1/D2/D3 datasets.

## Full reproduction

Download `radial_velocity_direct_inversion_training_dataset_v1.mat` from the
associated Mendeley Data record and put it here:

```text
datasets/radial_velocity_direct_inversion_training_dataset_v1.mat
```

Then build the reproducible D1/D2/D3 and P10/P20/P30 datasets:

```bash
python src/build_paper_sampling_datasets.py
```

See [`docs/experiment_guide.md`](docs/experiment_guide.md) for every paper
experiment, exact commands, expected outputs, and the mapping to final CSV
tables. Full runs use three seeds for up to 80 epochs and can require a GPU.

## Data availability

- **Code and aggregate results:** this GitHub repository.
- **Full synthetic training data:** the associated Mendeley Data record; add
  the final DOI link here after publication.
- **Field data:** not redistributed because of data-use restrictions.

The MAT dataset includes `velocity2` (`N x 256`, dispersion inputs) and
`param2` (`N x 10`, radial shear-wave velocity targets in m/s). See
[`docs/data_format.md`](docs/data_format.md).

## License

The code is released under the MIT License. See [`LICENSE`](LICENSE).
