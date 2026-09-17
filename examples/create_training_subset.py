"""Create a deterministic small training subset from a full MAT data pool."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Full MAT file containing param2 and velocity2")
    parser.add_argument("output", type=Path, help="Output MAT file")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data = loadmat(args.source, variable_names=["param2", "velocity2"])
    n = data["param2"].shape[0]
    if data["velocity2"].shape[0] != n:
        raise ValueError("param2 and velocity2 must have the same sample count.")
    if not 1 <= args.samples <= n:
        raise ValueError(f"samples must be between 1 and {n}.")

    indices = np.sort(np.random.default_rng(args.seed).choice(n, args.samples, replace=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        args.output,
        {"param2": data["param2"][indices], "velocity2": data["velocity2"][indices]},
        do_compression=True,
    )
    print(f"Wrote {args.output} with {args.samples} samples.")


if __name__ == "__main__":
    main()
