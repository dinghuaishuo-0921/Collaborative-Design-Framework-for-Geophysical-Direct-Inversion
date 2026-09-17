"""Run a lightweight end-to-end check for the public radial-inversion code.

The script creates a small synthetic MAT file using the same variable names as
the main experiments, validates the structured output parameterization, and
fits a linear baseline. It does not reproduce the paper's forward solver or
reported metrics; it verifies that a third party can run the shared workflow.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
DATA_PATH = HERE / "example_radial_velocity_data.mat"
SUMMARY_PATH = REPOSITORY / "results" / "quick_test_summary.json"
V_MIN = 1200.0
V_MAX = 3500.0


def make_synthetic_dataset(n_samples: int = 128, seed: int = 20260917):
    """Create format-compatible synthetic profiles and dispersion-like inputs."""
    rng = np.random.default_rng(seed)
    reference = rng.uniform(1900.0, 3200.0, size=(n_samples, 1))
    total_rise = rng.uniform(120.0, 650.0, size=(n_samples, 1))
    increments = rng.dirichlet(np.full(9, 2.0), size=n_samples) * total_rise
    inner = reference - total_rise
    profiles = np.concatenate([inner + np.cumsum(increments, axis=1), reference], axis=1)

    frequency = np.linspace(0.5, 5.0, 256, dtype=np.float32)
    layer_coordinate = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    sensitivity = np.exp(-frequency[:, None] * (0.25 + layer_coordinate[None, :]))
    sensitivity /= sensitivity.sum(axis=1, keepdims=True)
    effective_velocity = profiles @ sensitivity.T
    dispersion = 1.0e6 / effective_velocity
    dispersion += rng.normal(0.0, 0.2, size=dispersion.shape)
    return dispersion.astype(np.float32), profiles.astype(np.float32)


def structured_mapping(raw: np.ndarray) -> np.ndarray:
    """Map 11 unconstrained values to a bounded monotonic 10-layer profile."""
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    reference = V_MIN + (V_MAX - V_MIN) * sigmoid(raw[:, :1])
    total_rise = np.minimum(reference - V_MIN, V_MAX - V_MIN) * sigmoid(raw[:, 1:2])
    logits = raw[:, 2:]
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    gaps = total_rise * weights
    inner = reference - total_rise
    layer_values = inner + np.c_[np.zeros((len(raw), 1)), np.cumsum(gaps[:, :-1], axis=1)]
    profile = np.concatenate([layer_values, reference], axis=1)
    return profile.astype(np.float32)


def linear_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit and evaluate a least-squares baseline on a deterministic split."""
    split = int(0.75 * len(x))
    x_train = np.c_[np.ones(split), x[:split]]
    x_test = np.c_[np.ones(len(x) - split), x[split:]]
    weights, *_ = np.linalg.lstsq(x_train, y[:split], rcond=None)
    return x_test @ weights


def main() -> None:
    x, y = make_synthetic_dataset()
    savemat(DATA_PATH, {"velocity2": x, "param2": y})
    loaded = loadmat(DATA_PATH)
    x_loaded = loaded["velocity2"].astype(np.float32)
    y_loaded = loaded["param2"].astype(np.float32)
    if x_loaded.shape != (128, 256) or y_loaded.shape != (128, 10):
        raise RuntimeError("Unexpected MAT data shape.")

    raw = np.random.default_rng(7).normal(size=(64, 11)).astype(np.float32)
    structured = structured_mapping(raw)
    if structured.min() < V_MIN - 1.0e-5 or structured.max() > V_MAX + 1.0e-5:
        raise RuntimeError("Structured mapping violates the velocity bounds.")
    if np.any(np.diff(structured, axis=1) < -1.0e-5):
        raise RuntimeError("Structured mapping violates monotonicity.")

    prediction = linear_baseline(x_loaded, y_loaded)
    mae = float(np.mean(np.abs(prediction - y_loaded[int(0.75 * len(y_loaded)) :])))
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": str(DATA_PATH.relative_to(REPOSITORY)),
        "input_shape": list(x_loaded.shape),
        "target_shape": list(y_loaded.shape),
        "linear_baseline_mae_m_per_s": mae,
        "structured_mapping_violation_rate_percent": 0.0,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
