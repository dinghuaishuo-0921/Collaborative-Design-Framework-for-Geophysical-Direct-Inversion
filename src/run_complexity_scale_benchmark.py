from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import run_architecture_benchmark as bench  # noqa: E402
import run_strict_standard_rerun as strict  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {}
    for name in ["models", "predictions", "histories", "tables", "figures", "reports", "splits"]:
        path = run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def savefig(fig_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    plt.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def aggregate(df: pd.DataFrame, group_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(group))
        for col in numeric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_error_curves(agg: pd.DataFrame, fig_dir: Path, noise: str, name: str, title: str) -> None:
    if agg.empty:
        return
    plt.figure(figsize=(7.8, 5.0))
    for arch, group in agg[agg["noise"].eq(noise)].groupby("architecture_label"):
        g = group[group["dataset"].eq("internal_test")].sort_values("sample_size")
        if g.empty:
            continue
        plt.errorbar(
            g["sample_size"],
            g["mae_mean"],
            yerr=g["mae_std"],
            marker="o",
            linewidth=2,
            capsize=3,
            label=arch,
        )
    plt.xlabel("Training Sample Size")
    plt.ylabel("MAE (m/s)")
    plt.title(title)
    plt.legend()
    savefig(fig_dir, name)


def plot_training_time(agg: pd.DataFrame, fig_dir: Path) -> None:
    if agg.empty:
        return
    plt.figure(figsize=(7.8, 5.0))
    for arch, group in agg.groupby("architecture_label"):
        g = group.sort_values("sample_size")
        plt.errorbar(
            g["sample_size"],
            g["train_seconds_mean"] / 60.0,
            yerr=g["train_seconds_std"] / 60.0,
            marker="o",
            linewidth=2,
            capsize=3,
            label=arch,
        )
    plt.xlabel("Training Sample Size")
    plt.ylabel("Training Time (min)")
    plt.title("Training Cost Under Different Encoder Complexities")
    plt.legend()
    savefig(fig_dir, "fig_c03_complexity_scale_training_time")


def plot_param_tradeoff(agg: pd.DataFrame, fig_dir: Path) -> None:
    clean = agg[(agg["dataset"].eq("internal_test")) & (agg["noise"].eq("clean"))].copy()
    if clean.empty:
        return
    plt.figure(figsize=(7.2, 5.0))
    sizes = {int(v): s for v, s in zip(sorted(clean["sample_size"].unique()), [70, 110, 150, 190, 230])}
    for _, row in clean.iterrows():
        plt.scatter(
            row["parameter_count_mean"],
            row["mae_mean"],
            s=sizes.get(int(row["sample_size"]), 120),
            alpha=0.8,
            label=f"{row['architecture_label']}-{int(row['sample_size'])}",
        )
        plt.text(row["parameter_count_mean"], row["mae_mean"], row["architecture_label"], fontsize=8)
    plt.xlabel("Parameter Count")
    plt.ylabel("Clean MAE (m/s)")
    plt.title("Complexity-Accuracy Trade-off")
    savefig(fig_dir, "fig_c04_complexity_scale_param_tradeoff")


def write_report(run_dir: Path, cfg: dict, point_agg: pd.DataFrame, time_agg: pd.DataFrame) -> None:
    lines = ["# Complexity-Scale Benchmark", ""]
    lines.append(f"- Data file: `{cfg['data_file']}`")
    lines.append(f"- Architectures: `{cfg['architectures']}`")
    lines.append(f"- Sample sizes: `{cfg['sample_sizes']}`")
    lines.append(f"- Seeds: `{cfg['train_seeds']}`")
    lines.append("")

    clean = point_agg[(point_agg["dataset"].eq("internal_test")) & (point_agg["noise"].eq("clean"))]
    if not clean.empty:
        lines.append("## Internal Clean Accuracy")
        cols = ["architecture_label", "sample_size", "mae_mean", "mae_std", "rmse_mean", "violation_rate_percent_mean"]
        lines.append(clean[cols].sort_values(["architecture_label", "sample_size"]).to_markdown(index=False))
        lines.append("")

    noisy = point_agg[(point_agg["dataset"].eq("internal_test")) & (point_agg["noise"].eq("10dB"))]
    if not noisy.empty:
        lines.append("## Internal 10 dB Accuracy")
        cols = ["architecture_label", "sample_size", "mae_mean", "mae_std", "rmse_mean", "violation_rate_percent_mean"]
        lines.append(noisy[cols].sort_values(["architecture_label", "sample_size"]).to_markdown(index=False))
        lines.append("")

    external_clean = point_agg[(~point_agg["dataset"].eq("internal_test")) & (point_agg["noise"].eq("clean"))]
    if not external_clean.empty:
        lines.append("## External Clean Accuracy")
        cols = ["dataset", "architecture_label", "sample_size", "mae_mean", "rmse_mean"]
        lines.append(external_clean[cols].sort_values(["dataset", "architecture_label", "sample_size"]).to_markdown(index=False))
        lines.append("")

    if not time_agg.empty:
        lines.append("## Training Cost")
        cols = ["architecture_label", "sample_size", "train_seconds_mean", "ms_per_sample_mean", "parameter_count_mean"]
        lines.append(time_agg[cols].sort_values(["architecture_label", "sample_size"]).to_markdown(index=False))
        lines.append("")

    lines.append("## Suggested Paper Use")
    lines.append("- Use the clean and 10 dB MAE curves to show that model complexity and data volume must be considered jointly.")
    lines.append("- Use the parameter-time trade-off figure to support the statement that higher complexity is not automatically better.")
    (run_dir / "reports" / "COMPLEXITY_SCALE_BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict, run_dir: Path, smoke: bool = False) -> None:
    bench.style_plots()
    dirs = ensure_dirs(run_dir)
    bench.configure_strict_globals(cfg)

    config_source = Path(cfg.get("_config_source", HERE / "complexity_vs_scale_config.json"))
    if config_source.exists():
        shutil.copy2(config_source, dirs["reports"] / "complexity_vs_scale_config_used.json")

    if smoke:
        cfg = dict(cfg)
        cfg["train_seeds"] = [int(cfg["train_seeds"][0])]
        cfg["sample_sizes"] = [2000, 5000]
        cfg["architectures"] = cfg["architectures"][:2]
        cfg["epochs"] = 2
        cfg["external_eval_max_samples"] = 1000
        print("Running complexity-scale smoke test: two sizes, two architectures, one seed, two epochs.")

    data_file = (HERE / cfg["data_file"]).resolve()
    x_raw, y_raw = bench.load_raw_data(data_file)
    split_counts = cfg.get("split_counts")
    if split_counts:
        split_data = strict.prepare_split(
            x_raw,
            y_raw,
            int(cfg["split_seed"]),
            train_count=int(split_counts["train"]),
            val_count=int(split_counts["validation"]),
            test_count=int(split_counts["test"]),
        )
    else:
        split_data = strict.prepare_split(x_raw, y_raw, int(cfg["split_seed"]))
    (x_train_raw, x_train_scaled, y_train), (x_val_raw, x_val_scaled, y_val), (x_test_raw, _, y_test), scaler_x, split = split_data
    strict.configure_initializers(y_train)
    np.savez(dirs["splits"] / "split_indices.npz", **split)

    save_json(
        dirs["reports"] / "dataset_summary.json",
        {
            "data_file": str(data_file),
            "x_shape": list(x_raw.shape),
            "y_shape": list(y_raw.shape),
            "train_available": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
            "constraints": {
                "v_min": strict.V_MIN,
                "v_max": strict.V_MAX,
                "structured_v_ref_min": strict.V_REF_MIN,
                "structured_v_ref_max": strict.V_REF_MAX,
                "structured_max_total_drop": strict.MAX_TOTAL_DROP,
            },
        },
    )

    eval_sets = bench.make_eval_sets(x_test_raw, scaler_x, cfg, seed_offset=81000)
    external_sets = bench.load_external_eval_sets(cfg, scaler_x, smoke=smoke)
    args = bench.make_training_args(cfg)
    specs = bench.selected_architectures(cfg)

    point_rows = []
    layer_rows = []
    time_rows = []
    available = len(y_train)
    strict_sample_size_check = bool(cfg.get("strict_sample_size_check", True))
    nested_subsets = bool(cfg.get("nested_subsets", True))
    same_training_seed_across_scales = bool(cfg.get("same_training_seed_across_scales", True))

    requested_sizes = sorted({int(size) for size in cfg["sample_sizes"]})
    unavailable_sizes = [size for size in requested_sizes if size > available]
    if unavailable_sizes and strict_sample_size_check:
        raise ValueError(
            "Requested training sample size exceeds the available training pool after split: "
            f"available={available}, requested={unavailable_sizes}. "
            "The previous implementation silently truncated these sizes; strict mode refuses to do so."
        )

    for seed in cfg["train_seeds"]:
        seed_rng = np.random.default_rng(int(seed) + 20260706)
        subset_indices_by_size = {}
        if nested_subsets:
            # Use nested prefixes of one permutation so that adjacent scales differ
            # only by the newly added training samples.
            permutation = seed_rng.permutation(available)
            for requested_size in requested_sizes:
                subset_indices_by_size[requested_size] = np.sort(
                    permutation[:requested_size].astype(np.int64)
                )
        else:
            for requested_size in requested_sizes:
                subset_indices_by_size[requested_size] = np.sort(
                    seed_rng.choice(available, size=requested_size, replace=False).astype(np.int64)
                )

        for sample_size, subset_idx in subset_indices_by_size.items():
            subset_data = (
                (x_train_raw[subset_idx], x_train_scaled[subset_idx], y_train[subset_idx]),
                (x_val_raw, x_val_scaled, y_val),
                (x_test_raw, eval_sets["clean"], y_test),
                scaler_x,
                split,
            )
            np.savez_compressed(dirs["splits"] / f"subset_seed{seed}_n{sample_size}.npz", subset_indices=subset_idx)

            for spec in specs:
                spec_key = bench.compact_name(spec.key)
                print("\n" + "=" * 88)
                print(f"[seed {seed}] Training {spec.key} with {sample_size} samples")
                print("=" * 88)
                training_seed = int(seed) if same_training_seed_across_scales else int(seed) + sample_size
                model, history = bench.train_one_architecture(spec, subset_data, args, training_seed)
                save_json(dirs["histories"] / f"hist_s{seed}_n{sample_size}_{spec_key}.json", history)

                model_dir = dirs["models"] / f"seed_{seed}"
                model_dir.mkdir(exist_ok=True)
                model.save_weights(str(model_dir / f"{spec_key}_n{sample_size}.weights.h5"))

                eval_batch_size = bench.architecture_batch_size(spec, args.batch_size)
                inference = bench.measure_inference(model, eval_sets["clean"], eval_batch_size)
                time_rows.append(
                    {
                        "seed": int(seed),
                        "sample_size": int(sample_size),
                        "architecture": spec.key,
                        "architecture_label": spec.label,
                        "train_seconds": float(history["train_seconds"]),
                        "parameter_count": int(model.count_params()),
                        **inference,
                    }
                )

                eval_groups = {"internal_test": (y_test, eval_sets)}
                eval_groups.update(external_sets)
                for dataset_name, (y_eval, dataset_eval_sets) in eval_groups.items():
                    dataset_key = bench.compact_name(dataset_name)
                    for noise, x_eval in dataset_eval_sets.items():
                        noise_key = bench.compact_name(noise)
                        pred_mu, _ = bench.predict_model(model, spec, x_eval, eval_batch_size)
                        np.savez_compressed(
                            dirs["predictions"] / f"s{seed}_n{sample_size}_{spec_key}_{dataset_key}_{noise_key}.npz",
                            y_true=y_eval,
                            pred_mu=pred_mu,
                        )
                        pm = bench.point_metrics(y_eval, pred_mu)
                        point_rows.append(
                            {
                                "seed": int(seed),
                                "sample_size": int(sample_size),
                                "architecture": spec.key,
                                "architecture_label": spec.label,
                                "dataset": dataset_name,
                                "noise": noise,
                                "parameter_count": int(model.count_params()),
                                "train_seconds": float(history["train_seconds"]),
                                "ms_per_sample": inference["ms_per_sample"],
                                **{k: v for k, v in pm.items() if k != "layer_mae"},
                            }
                        )
                        for layer_idx, layer_mae in enumerate(pm["layer_mae"], start=1):
                            layer_rows.append(
                                {
                                    "seed": int(seed),
                                    "sample_size": int(sample_size),
                                    "architecture": spec.key,
                                    "architecture_label": spec.label,
                                    "dataset": dataset_name,
                                    "noise": noise,
                                    "layer": layer_idx,
                                    "layer_mae": float(layer_mae),
                                }
                            )

    point_df = pd.DataFrame(point_rows)
    layer_df = pd.DataFrame(layer_rows)
    time_df = pd.DataFrame(time_rows)
    point_df.to_csv(dirs["tables"] / "seed_architecture_scale_point_metrics.csv", index=False, encoding="utf-8-sig")
    layer_df.to_csv(dirs["tables"] / "seed_architecture_scale_layer_mae.csv", index=False, encoding="utf-8-sig")
    time_df.to_csv(dirs["tables"] / "seed_architecture_scale_efficiency.csv", index=False, encoding="utf-8-sig")

    point_agg = aggregate(
        point_df,
        ["architecture", "architecture_label", "sample_size", "dataset", "noise"],
        ["mae", "rmse", "mape_percent", "r2", "violation_rate_percent", "parameter_count", "train_seconds", "ms_per_sample"],
    )
    time_agg = aggregate(
        time_df,
        ["architecture", "architecture_label", "sample_size"],
        ["parameter_count", "train_seconds", "ms_per_sample", "samples_per_second"],
    )
    point_agg.to_csv(dirs["tables"] / "aggregate_architecture_scale_point_metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    time_agg.to_csv(dirs["tables"] / "aggregate_architecture_scale_efficiency_mean_std.csv", index=False, encoding="utf-8-sig")

    plot_error_curves(point_agg, dirs["figures"], "clean", "fig_c01_complexity_scale_clean_mae", "Clean Accuracy Under Different Data Scales")
    plot_error_curves(point_agg, dirs["figures"], "10dB", "fig_c02_complexity_scale_10db_mae", "10 dB Accuracy Under Different Data Scales")
    plot_training_time(time_agg, dirs["figures"])
    plot_param_tradeoff(point_agg, dirs["figures"])
    write_report(run_dir, cfg, point_agg, time_agg)
    print(f"\nComplexity-scale benchmark finished: {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run encoder-complexity versus data-scale benchmark.")
    parser.add_argument("--config", default="complexity_vs_scale_config.json")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = (HERE / args.config).resolve()
    cfg = load_config(cfg_path)
    cfg["_config_source"] = str(cfg_path)
    suffix = "_smoke" if args.smoke else ""
    run_name = args.output_name or f"{cfg['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    run_dir = Path.cwd() / "runs" / run_name if not Path(run_name).is_absolute() else Path(run_name)
    run(cfg, run_dir, smoke=args.smoke)
