"""Build reproducible datasets for the sampling-mode comparison.

The variants isolate three practically distinct failures in direct-inversion
data construction: missing quality control, incomplete parameter support, and
well-organized coverage-constrained sampling.  All variants have the same
number of samples and use the same forward-model data pool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SOURCE = ROOT / "datasets" / "radial_velocity_direct_inversion_training_dataset_v1.mat"
DEFAULT_DATA_ROOT = ROOT / "datasets" / "derived"
DATA_ROOT = DEFAULT_DATA_ROOT
MAT_DIR = DATA_ROOT / "mat"
INDEX_DIR = DATA_ROOT / "indices"
REPORT_DIR = DATA_ROOT / "reports"
SOURCE = DEFAULT_SOURCE

# The common physical pool contains about 77k samples.  Forty thousand per
# variant leaves a genuinely disjoint, full-range common test set after the
# three selections are formed.
N_SAMPLES = 40_000
N_COMMON_TEST = 5_105
SEED = 20260828
V_MIN, V_MAX, BIASED_MAX = 1200.0, 3500.0, 3000.0


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(path)
    return mat["velocity2"].astype(np.float32), mat["param2"].astype(np.float32)


def save_data(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    sio.savemat(path, {"velocity2": x.astype(np.float32), "param2": y.astype(np.float32)})


def feature_frame(y: np.ndarray) -> pd.DataFrame:
    step = np.diff(y, axis=1)
    return pd.DataFrame(
        {
            "v_inner": y[:, 0],
            "v_ref": y[:, -1],
            "v_max_profile": y.max(axis=1),
            "total_rise": y[:, -1] - y[:, 0],
            "min_step": step.min(axis=1),
            "max_step": step.max(axis=1),
            "curvature": np.abs(np.diff(step, axis=1)).mean(axis=1),
        }
    )


def physics_mask(y: np.ndarray) -> np.ndarray:
    f = feature_frame(y)
    return (
        (y.min(axis=1) >= V_MIN)
        & (y.max(axis=1) <= V_MAX)
        & (f.total_rise.to_numpy() >= 80.0)
        & (f.total_rise.to_numpy() <= 420.0)
        & (f.min_step.to_numpy() >= 5.0)
        & (f.max_step.to_numpy() <= 65.0)
        & (f.curvature.to_numpy() <= 35.0)
    )


def coverage_subset(indices: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Use quotas in reference velocity, total rise, and curvature bins."""
    f = feature_frame(y[indices])
    bins = []
    for values, q in ((f.v_ref, 6), (f.total_rise, 6), (f.curvature, 4)):
        bins.append(pd.qcut(values, q=q, labels=False, duplicates="drop").to_numpy())
    groups: dict[tuple[int, int, int], list[int]] = {}
    for local, key in enumerate(zip(*bins)):
        groups.setdefault(tuple(int(v) for v in key), []).append(local)
    quota = max(1, n // len(groups))
    selected: list[int] = []
    remainder: list[int] = []
    for members in groups.values():
        member_array = np.asarray(members, dtype=np.int64)
        rng.shuffle(member_array)
        selected.extend(member_array[:quota])
        remainder.extend(member_array[quota:])
    if len(selected) < n:
        remaining = np.asarray(remainder, dtype=np.int64)
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])
    selected_array = np.asarray(selected[:n], dtype=np.int64)
    if len(selected_array) != n:
        raise RuntimeError("Coverage candidate pool cannot satisfy the requested quota.")
    return np.sort(indices[selected_array])


def corrupt_uncontrolled(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inject 10% mismatched labels and 10% noisy responses into random data."""
    x_bad, y_bad = x.copy(), y.copy()
    all_idx = rng.permutation(len(y))
    n_each = int(round(len(y) * 0.10))
    label_idx, response_idx = all_idx[:n_each], all_idx[n_each : 2 * n_each]

    # Pair each selected response with a physically unrelated label.  This
    # reproduces a common un-audited pairing/annotation error without claiming
    # that the resulting profile is a physically valid forward counterpart.
    donors = rng.permutation(label_idx)
    y_bad[label_idx] = y_bad[donors]

    # Use a sample-wise 15 dB Gaussian perturbation for low-quality responses.
    signal_rms = np.sqrt(np.mean(np.square(x_bad[response_idx]), axis=1, keepdims=True))
    noise_rms = signal_rms / (10.0 ** (15.0 / 20.0))
    x_bad[response_idx] += rng.normal(0.0, 1.0, size=x_bad[response_idx].shape).astype(np.float32) * noise_rms
    return x_bad, y_bad, {
        "label_mismatch_percent": 10.0,
        "response_noise_percent": 10.0,
        "response_noise_snr_db": 15.0,
        "label_mismatch_indices": label_idx,
        "response_noise_indices": response_idx,
    }


def coverage_stats(y: np.ndarray, ref_edges: np.ndarray, rise_edges: np.ndarray) -> dict:
    """Occupancy in a fixed grid defined from the complete physical pool."""
    f = feature_frame(y)
    ref_bin = np.clip(np.digitize(f.v_ref, ref_edges[1:-1]), 0, len(ref_edges) - 2)
    rise_bin = np.clip(np.digitize(f.total_rise, rise_edges[1:-1]), 0, len(rise_edges) - 2)
    counts = np.zeros((len(ref_edges) - 1, len(rise_edges) - 1), dtype=np.int64)
    np.add.at(counts, (ref_bin, rise_bin), 1)
    occupied = counts[counts > 0]
    return {
        "coverage_fill_percent": float(np.mean(counts > 0) * 100.0),
        "coverage_balance_cv": float(np.std(occupied) / np.mean(occupied)) if len(occupied) else 0.0,
    }


def audit(name: str, x: np.ndarray, y: np.ndarray, ref_edges: np.ndarray, rise_edges: np.ndarray) -> dict:
    f = feature_frame(y)
    return {
        "dataset": name,
        "samples": len(y),
        "dispersion_points": x.shape[1],
        "velocity_min": float(y.min()),
        "velocity_max": float(y.max()),
        "reference_velocity_min": float(y[:, -1].min()),
        "reference_velocity_max": float(y[:, -1].max()),
        "boundary_violation_percent": float(np.mean(np.any((y < V_MIN) | (y > V_MAX), axis=1)) * 100.0),
        "monotonic_violation_percent": float(np.mean(np.any(np.diff(y, axis=1) < 0.0, axis=1)) * 100.0),
        "total_rise_mean": float(f.total_rise.mean()),
        "curvature_mean": float(f.curvature.mean()),
        **coverage_stats(y, ref_edges, rise_edges),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build quality-control, range-biased, and coverage-constrained sampling datasets."
    )
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    global DATA_ROOT, MAT_DIR, INDEX_DIR, REPORT_DIR, SOURCE
    args = parse_args()
    SOURCE = Path(args.source_data).expanduser().resolve()
    DATA_ROOT = Path(args.output_dir).expanduser().resolve()
    MAT_DIR = DATA_ROOT / "mat"
    INDEX_DIR = DATA_ROOT / "indices"
    REPORT_DIR = DATA_ROOT / "reports"
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE}. Download the Mendeley Data MAT file first."
        )
    MAT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_data(SOURCE)
    rng = np.random.default_rng(SEED)

    # R-QC-: completely random source subset, then retain quality failures.
    random_idx = np.sort(rng.choice(len(y), N_SAMPLES, replace=False))
    x_random, y_random, corruption = corrupt_uncontrolled(x[random_idx], y[random_idx], rng)

    phys_idx = np.flatnonzero(physics_mask(y)).astype(np.int64)
    physical_features = feature_frame(y[phys_idx])
    ref_edges = np.quantile(physical_features.v_ref, np.linspace(0.0, 1.0, 11))
    rise_edges = np.quantile(physical_features.total_rise, np.linspace(0.0, 1.0, 11))
    biased_pool = phys_idx[y[phys_idx].max(axis=1) <= BIASED_MAX]
    if len(biased_pool) < N_SAMPLES:
        raise RuntimeError(f"Only {len(biased_pool)} range-biased candidates are available.")
    biased_idx = np.sort(rng.choice(biased_pool, N_SAMPLES, replace=False))
    coverage_idx = coverage_subset(phys_idx, y, N_SAMPLES, rng)

    used = np.union1d(np.union1d(random_idx, biased_idx), coverage_idx)
    common_pool = np.setdiff1d(phys_idx, used, assume_unique=False)
    if len(common_pool) < N_COMMON_TEST:
        raise RuntimeError("Insufficient held-out physical samples for the common test set.")
    common_idx = coverage_subset(common_pool, y, N_COMMON_TEST, rng)

    variants = {
        "sampling_r1_random_uncontrolled_40000.mat": (x_random, y_random, random_idx, {
            "label": "R-QC- Random without quality control",
            "definition": "Random source sampling with 10% mismatched labels and 10% 15 dB response corruption.",
            "quality_issue_total_percent": 20.0,
        }),
        "sampling_r2_range_biased_1200_3000_40000.mat": (x[biased_idx], y[biased_idx], biased_idx, {
            "label": "B-Range Incomplete parameter support",
            "definition": "Physics-screened random sampling restricted to profiles within 1200-3000 m/s.",
            "support_velocity_range_mps": [V_MIN, BIASED_MAX],
        }),
        "sampling_r3_coverage_constrained_40000.mat": (x[coverage_idx], y[coverage_idx], coverage_idx, {
            "label": "C-Coverage Quality-controlled full-range coverage",
            "definition": "Physics-screened full-range sampling with quotas over reference velocity, total rise, and curvature.",
            "support_velocity_range_mps": [V_MIN, V_MAX],
        }),
        "sampling_common_fullrange_test_5105.mat": (x[common_idx], y[common_idx], common_idx, {
            "label": "Common full-range clean test",
            "definition": "Held-out quality-controlled full-range test set shared by all three training variants.",
            "support_velocity_range_mps": [V_MIN, V_MAX],
        }),
    }

    rows, manifest = [], []
    for filename, (x_item, y_item, source_idx, metadata) in variants.items():
        save_data(MAT_DIR / filename, x_item, y_item)
        payload = {"source_indices": source_idx}
        if filename.startswith("sampling_r1"):
            payload["label_mismatch_indices_local"] = corruption["label_mismatch_indices"]
            payload["response_noise_indices_local"] = corruption["response_noise_indices"]
            metadata = {**metadata, **{k: v for k, v in corruption.items() if not k.endswith("indices")}}
        np.savez_compressed(INDEX_DIR / filename.replace(".mat", "_indices.npz"), **payload)
        row = {**audit(filename, x_item, y_item, ref_edges, rise_edges), **metadata}
        rows.append(row)
        manifest.append({"file": filename, **metadata})

    pd.DataFrame(rows).to_csv(REPORT_DIR / "sampling_mode_contrast_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest).to_csv(REPORT_DIR / "sampling_mode_contrast_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source": str(SOURCE), "seed": SEED, "train_samples_per_variant": N_SAMPLES,
        "common_test_samples": N_COMMON_TEST, "physical_candidates": int(len(phys_idx)),
        "range_biased_candidates": int(len(biased_pool)), "files": list(variants),
    }
    (REPORT_DIR / "sampling_mode_contrast_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
