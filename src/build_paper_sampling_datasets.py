from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SOURCE_FILE = ROOT / "datasets" / "radial_velocity_direct_inversion_training_dataset_v1.mat"
DEFAULT_OUT_DIR = ROOT / "datasets" / "derived"
OUT_DIR = DEFAULT_OUT_DIR
OUT_DATA = OUT_DIR / "mat"
OUT_INDEX = OUT_DIR / "indices"
OUT_REPORT = OUT_DIR / "reports"
SOURCE_FILE = DEFAULT_SOURCE_FILE
TARGET_SIZE = 51047
RNG_SEED = 20260706
V_MIN = 1200.0
V_MAX = 3500.0


@dataclass(frozen=True)
class DatasetOutput:
    name: str
    x: np.ndarray
    y: np.ndarray
    metadata: dict


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_mat(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(path), {"velocity2": x.astype(np.float32), "param2": y.astype(np.float32)})


def load_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(str(path))
    return mat["velocity2"].astype(np.float32), mat["param2"].astype(np.float32)


def dataset_features(y: np.ndarray) -> pd.DataFrame:
    diff = np.diff(y, axis=1)
    curvature = np.mean(np.abs(np.diff(diff, axis=1)), axis=1)
    return pd.DataFrame(
        {
            "v_inner": y[:, 0],
            "v_ref": y[:, -1],
            "total_drop": y[:, -1] - y[:, 0],
            "min_step": np.min(diff, axis=1),
            "max_step": np.max(diff, axis=1),
            "curvature": curvature,
        }
    )


def qcut_codes(values: np.ndarray, q: int) -> np.ndarray:
    return pd.qcut(values, q=q, labels=False, duplicates="drop").to_numpy(dtype=np.int64)


def balanced_subset_indices(y: np.ndarray, target_size: int, seed: int) -> np.ndarray:
    feats = dataset_features(y)
    rng = np.random.default_rng(seed)

    ref_bins = qcut_codes(feats["v_ref"], q=6)
    drop_bins = qcut_codes(feats["total_drop"], q=6)
    curv_bins = qcut_codes(feats["curvature"], q=4)

    groups: dict[tuple[int, int, int], list[int]] = {}
    for idx, key in enumerate(zip(ref_bins, drop_bins, curv_bins)):
        groups.setdefault(tuple(int(v) for v in key), []).append(idx)

    chosen = []
    leftovers = []
    quota = max(1, target_size // max(len(groups), 1))
    for key, idxs in groups.items():
        arr = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(arr)
        take = min(len(arr), quota)
        chosen.append(arr[:take])
        if len(arr) > take:
            leftovers.append(arr[take:])

    chosen_idx = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)
    if len(chosen_idx) < target_size and leftovers:
        rest = np.concatenate(leftovers)
        rng.shuffle(rest)
        chosen_idx = np.concatenate([chosen_idx, rest[: target_size - len(chosen_idx)]])
    elif len(chosen_idx) > target_size:
        rng.shuffle(chosen_idx)
        chosen_idx = chosen_idx[:target_size]

    return np.sort(chosen_idx.astype(np.int64))


def weighted_midbiased_indices(y: np.ndarray, target_size: int, seed: int) -> np.ndarray:
    feats = dataset_features(y)
    rng = np.random.default_rng(seed)
    ref = feats["v_ref"].to_numpy(dtype=np.float64)
    drop = feats["total_drop"].to_numpy(dtype=np.float64)
    ref_lo, ref_hi = np.quantile(ref, [0.12, 0.88])
    drop_lo, drop_hi = np.quantile(drop, [0.12, 0.88])
    mid_mask = (ref >= ref_lo) & (ref <= ref_hi) & (drop >= drop_lo) & (drop <= drop_hi)
    candidate = np.where(mid_mask)[0].astype(np.int64)
    if len(candidate) < target_size:
        raise RuntimeError("Mid-biased candidate pool is smaller than target size.")
    idx = rng.choice(candidate, size=target_size, replace=False)
    return np.sort(idx.astype(np.int64))


def weak_pool_mask(y: np.ndarray) -> np.ndarray:
    return (np.min(y, axis=1) >= V_MIN) & (np.max(y, axis=1) <= V_MAX)


def physics_pool_mask(y: np.ndarray) -> np.ndarray:
    feats = dataset_features(y)
    return (
        weak_pool_mask(y)
        & (feats["total_drop"].to_numpy() >= 80.0)
        & (feats["total_drop"].to_numpy() <= 420.0)
        & (feats["min_step"].to_numpy() >= 5.0)
        & (feats["max_step"].to_numpy() <= 65.0)
        & (feats["curvature"].to_numpy() <= 35.0)
    )


def apply_reversal(profile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = profile.copy()
    j = int(rng.integers(1, len(out) - 1))
    out[j] = out[j - 1] - float(rng.integers(20, 85))
    return out


def apply_spike(profile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = profile.copy()
    j = int(rng.integers(1, len(out) - 2))
    out[j] = out[j - 1] + float(rng.integers(120, 230))
    out[j + 1] = out[j] - float(rng.integers(80, 180))
    return out


def apply_out_of_bound(profile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = profile.copy()
    if rng.random() < 0.7:
        out[0] = float(rng.integers(980, 1180))
        for i in range(1, len(out)):
            out[i] = max(out[i], out[i - 1] + float(rng.integers(3, 25)))
    else:
        out[-1] = float(rng.integers(3520, 3660))
        for i in range(len(out) - 2, -1, -1):
            out[i] = min(out[i], out[i + 1] - float(rng.integers(3, 25)))
    return out


def apply_excessive_drop(profile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = profile.copy()
    extra = float(rng.integers(120, 260))
    out[0] = max(1050.0, out[0] - extra)
    increments = np.maximum(np.diff(profile), float(rng.integers(2, 12)))
    out[1:] = out[0] + np.cumsum(increments)
    out[-1] = max(out[-1], profile[-1])
    out = np.maximum.accumulate(out)
    return out


def make_polluted_variant(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    x_out = x.copy()
    y_out = y.copy()
    n_corrupt = int(round(len(y) * ratio))
    chosen = np.sort(rng.choice(len(y), size=n_corrupt, replace=False).astype(np.int64))

    types = np.array(["reversal", "spike", "out_of_bound", "excessive_drop"])
    probs = np.array([0.40, 0.30, 0.20, 0.10], dtype=np.float64)
    codes = rng.choice(types, size=n_corrupt, p=probs)

    for idx, code in zip(chosen, codes):
        profile = y_out[idx]
        if code == "reversal":
            y_out[idx] = apply_reversal(profile, rng)
        elif code == "spike":
            y_out[idx] = apply_spike(profile, rng)
        elif code == "out_of_bound":
            y_out[idx] = apply_out_of_bound(profile, rng)
        else:
            y_out[idx] = apply_excessive_drop(profile, rng)

    metadata = {
        "source_dataset": "sampling_d2_physics_random_51047.mat",
        "pollution_ratio": float(ratio),
        "n_corrupt": int(n_corrupt),
        "corruption_counts": {k: int(np.sum(codes == k)) for k in types},
        "corrupted_indices_file": f"polluted_{int(ratio * 100):02d}_indices.npz",
    }
    return x_out, y_out, metadata


def occupancy_stats(feats: pd.DataFrame, ref_edges: np.ndarray, drop_edges: np.ndarray) -> dict:
    ref_bin = np.clip(np.digitize(feats["v_ref"], ref_edges[1:-1], right=False), 0, len(ref_edges) - 2)
    drop_bin = np.clip(np.digitize(feats["total_drop"], drop_edges[1:-1], right=False), 0, len(drop_edges) - 2)
    counts_df = pd.crosstab(ref_bin, drop_bin)
    full_index = pd.RangeIndex(len(ref_edges) - 1)
    full_columns = pd.RangeIndex(len(drop_edges) - 1)
    counts_df = counts_df.reindex(index=full_index, columns=full_columns, fill_value=0.0)
    counts = counts_df.to_numpy(dtype=np.float64)
    flat = counts.reshape(-1)
    nonzero = flat[flat > 0]
    return {
        "coverage_bins_total": int(flat.size),
        "coverage_bins_filled": int(np.sum(flat > 0)),
        "coverage_fill_percent": float(np.mean(flat > 0) * 100.0),
        "coverage_balance_cv": float(np.std(nonzero) / max(np.mean(nonzero), 1.0e-12)) if len(nonzero) else 0.0,
    }


def audit_dataset(name: str, x: np.ndarray, y: np.ndarray, ref_edges: np.ndarray, drop_edges: np.ndarray, metadata: dict) -> dict:
    feats = dataset_features(y)
    boundary_violation = (y < V_MIN) | (y > V_MAX)
    mono_violation = np.diff(y, axis=1) < -1.0e-6
    stats = occupancy_stats(feats, ref_edges, drop_edges)
    row = {
        "dataset": name,
        "samples": int(len(y)),
        "dispersion_points": int(x.shape[1]),
        "v_min_actual": float(np.min(y)),
        "v_max_actual": float(np.max(y)),
        "v_inner_mean": float(np.mean(y[:, 0])),
        "v_ref_mean": float(np.mean(y[:, -1])),
        "total_drop_mean": float(np.mean(feats["total_drop"])),
        "total_drop_std": float(np.std(feats["total_drop"])),
        "curvature_mean": float(np.mean(feats["curvature"])),
        "boundary_violation_percent": float(np.mean(np.any(boundary_violation, axis=1)) * 100.0),
        "monotonic_violation_percent": float(np.mean(np.any(mono_violation, axis=1)) * 100.0),
        "middle_drop_mass_percent": float(
            np.mean((feats["total_drop"] >= drop_edges[2]) & (feats["total_drop"] <= drop_edges[4]))
            * 100.0
        ),
        **stats,
        "notes": metadata.get("notes", ""),
    }
    for key, value in metadata.items():
        if key == "notes":
            continue
        row[f"meta_{key}"] = value
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct the D1/D2/D3 and P10/P20/P30 paper datasets from the public MAT dataset."
    )
    parser.add_argument(
        "--source-data",
        default=str(DEFAULT_SOURCE_FILE),
        help="MAT file containing velocity2 (N x 256) and param2 (N x 10).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for derived MAT datasets, indices, and audits.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=TARGET_SIZE,
        help="Number of samples in each derived training set.",
    )
    return parser.parse_args()


def main() -> None:
    global OUT_DIR, OUT_DATA, OUT_INDEX, OUT_REPORT, SOURCE_FILE, TARGET_SIZE
    args = parse_args()
    SOURCE_FILE = Path(args.source_data).expanduser().resolve()
    OUT_DIR = Path(args.output_dir).expanduser().resolve()
    OUT_DATA = OUT_DIR / "mat"
    OUT_INDEX = OUT_DIR / "indices"
    OUT_REPORT = OUT_DIR / "reports"
    TARGET_SIZE = int(args.target_size)
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_FILE}. Download the Mendeley Data MAT file first."
        )
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_INDEX.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)

    x_wide, y_wide = load_mat(SOURCE_FILE)
    feats_wide = dataset_features(y_wide)
    ref_edges = np.quantile(feats_wide["v_ref"], np.linspace(0.0, 1.0, 11))
    drop_edges = np.quantile(feats_wide["total_drop"], np.linspace(0.0, 1.0, 11))

    weak_idx_all = np.where(weak_pool_mask(y_wide))[0].astype(np.int64)
    physics_idx_all = np.where(physics_pool_mask(y_wide))[0].astype(np.int64)
    if len(weak_idx_all) < TARGET_SIZE or len(physics_idx_all) < TARGET_SIZE:
        raise RuntimeError("Candidate pool is smaller than TARGET_SIZE; relax constraints or reduce target size.")

    d1_local = weighted_midbiased_indices(y_wide[weak_idx_all], TARGET_SIZE, RNG_SEED)
    d1_idx = weak_idx_all[d1_local]
    d2_local = np.random.default_rng(RNG_SEED + 11).choice(len(physics_idx_all), size=TARGET_SIZE, replace=False)
    d2_idx = np.sort(physics_idx_all[d2_local].astype(np.int64))
    d3_local = balanced_subset_indices(y_wide[physics_idx_all], TARGET_SIZE, RNG_SEED + 29)
    d3_idx = np.sort(physics_idx_all[d3_local].astype(np.int64))

    outputs = [
        DatasetOutput(
            name="sampling_d1_wide_random_51047.mat",
            x=x_wide[d1_idx],
            y=y_wide[d1_idx],
            metadata={
                "strategy": "D1",
                "source_dataset": SOURCE_FILE.name,
                "selection_rule": "weak_pool_midbiased_random",
                "candidate_pool_size": int(len(weak_idx_all)),
                "selected_size": int(TARGET_SIZE),
                "notes": "仅施加宽松边界约束，并通过中值偏置抽样削弱边界与异常模式覆盖。",
            },
        ),
        DatasetOutput(
            name="sampling_d2_physics_random_51047.mat",
            x=x_wide[d2_idx],
            y=y_wide[d2_idx],
            metadata={
                "strategy": "D2",
                "source_dataset": SOURCE_FILE.name,
                "selection_rule": "physics_filtered_uniform_random",
                "candidate_pool_size": int(len(physics_idx_all)),
                "selected_size": int(TARGET_SIZE),
                "notes": "加入边界、总降幅、最小层间增量、最大层间增量与曲率约束后的随机采样。",
            },
        ),
        DatasetOutput(
            name="sampling_d3_coverage_constrained_51047.mat",
            x=x_wide[d3_idx],
            y=y_wide[d3_idx],
            metadata={
                "strategy": "D3",
                "source_dataset": SOURCE_FILE.name,
                "selection_rule": "physics_filtered_coverage_balanced",
                "candidate_pool_size": int(len(physics_idx_all)),
                "selected_size": int(TARGET_SIZE),
                "notes": "在 D2 约束基础上进一步按外层速度、总降幅和曲率进行覆盖均衡。",
            },
        ),
    ]

    d2_x = outputs[1].x
    d2_y = outputs[1].y
    for pct in [10, 20, 30]:
        ratio = pct / 100.0
        x_bad, y_bad, meta = make_polluted_variant(d2_x, d2_y, ratio, RNG_SEED + 100 + pct)
        outputs.append(
            DatasetOutput(
                name=f"sampling_polluted_p{pct}_51047.mat",
                x=x_bad,
                y=y_bad,
                metadata={
                    "strategy": f"P{pct}",
                    "notes": "在 D2 数据集上注入弱非物理或错误标注样本，用于检验数据质量影响。",
                    **meta,
                },
            )
        )

    audit_rows = []
    manifest_rows = []
    for output in outputs:
        save_mat(OUT_DATA / output.name, output.x, output.y)
        if output.name.startswith("sampling_d1"):
            idx = d1_idx
        elif output.name.startswith("sampling_d2"):
            idx = d2_idx
        elif output.name.startswith("sampling_d3"):
            idx = d3_idx
        else:
            idx = d2_idx

        index_payload = {"source_indices": idx.astype(np.int64)}
        if "corrupted_indices_file" in output.metadata:
            changed = np.where(np.any(np.abs(output.y - d2_y) > 1.0e-6, axis=1))[0].astype(np.int64)
            index_payload["corrupted_indices"] = changed
        np.savez_compressed(OUT_INDEX / output.name.replace(".mat", "_indices.npz"), **index_payload)

        audit_rows.append(audit_dataset(output.name, output.x, output.y, ref_edges, drop_edges, output.metadata))
        manifest_rows.append(
            {
                "dataset": output.name,
                "samples": int(len(output.y)),
                "usage": output.metadata.get("notes", ""),
                "selection_rule": output.metadata.get("selection_rule", output.metadata.get("strategy", "")),
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    audit_df.to_csv(OUT_REPORT / "paper_sampling_dataset_audit.csv", index=False, encoding="utf-8-sig")
    manifest_df.to_csv(OUT_REPORT / "paper_sampling_dataset_manifest.csv", index=False, encoding="utf-8-sig")

    save_json(
        OUT_REPORT / "paper_sampling_dataset_builder_summary.json",
        {
            "source_file": str(SOURCE_FILE),
            "target_size": TARGET_SIZE,
            "seed": RNG_SEED,
            "weak_pool_size": int(len(weak_idx_all)),
            "physics_pool_size": int(len(physics_idx_all)),
            "output_files": [row["dataset"] for row in manifest_rows],
        },
    )

    notes = [
        "# Paper Sampling Dataset Suite",
        "",
        "## Generated datasets",
    ]
    for row in manifest_rows:
        notes.append(f"- `{row['dataset']}`: {row['usage']}")
    notes.extend(
        [
            "",
            "## Three-level sampling logic",
            "- `D1`: weak prior + middle-value biased random subset, used to simulate poor coverage under loose organization.",
            "- `D2`: physics-filtered random subset, used to isolate the value of task-reasonable sampling.",
            "- `D3`: physics-filtered and coverage-balanced subset, used as the strongest organized dataset.",
            "",
            "## Pollution logic",
            "- `P10/P20/P30` are built from `D2` by injecting label corruption while keeping the observed dispersion responses unchanged.",
            "- Corruption types include local reversals, jagged spikes, mild out-of-bound profiles and excessive total drop cases.",
            "",
            "## Main report files",
            f"- Audit CSV: `{OUT_REPORT / 'paper_sampling_dataset_audit.csv'}`",
            f"- Manifest CSV: `{OUT_REPORT / 'paper_sampling_dataset_manifest.csv'}`",
        ]
    )
    (OUT_REPORT / "PAPER_SAMPLING_DATASETS.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Generated {len(outputs)} paper sampling datasets under: {OUT_DATA}")


if __name__ == "__main__":
    main()
