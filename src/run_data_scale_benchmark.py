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
import run_strict_standard_rerun as strict  # noqa: E402
import run_architecture_benchmark as bench  # noqa: E402


# The reported scale table uses the same structured Transformer encoder as the
# main benchmark. Keep this explicit so run labels match the manuscript.
PROPOSED = bench.ArchitectureSpec("Transformer_SP", "Transformer-SP", True, False, False)


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


def style_plots() -> None:
    bench.style_plots()


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
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else len(group)
        for col in numeric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scale_error(agg: pd.DataFrame, fig_dir: Path) -> None:
    if agg.empty:
        return
    order_noise = ["clean", "30dB", "20dB", "10dB"]
    plt.figure(figsize=(7.8, 5.0))
    for noise in order_noise:
        g = agg[agg["noise"].eq(noise)].sort_values("sample_size")
        if g.empty:
            continue
        plt.errorbar(
            g["sample_size"],
            g["mae_mean"],
            yerr=g["mae_std"],
            marker="o",
            linewidth=2,
            capsize=3,
            label=noise,
        )
    plt.xscale("log")
    plt.xlabel("Training Sample Size")
    plt.ylabel("MAE (m/s)")
    plt.title("Data Scale Sensitivity of the Proposed Model")
    plt.legend()
    savefig(fig_dir, "fig_s01_data_scale_mae")


def plot_scale_time(time_agg: pd.DataFrame, fig_dir: Path) -> None:
    if time_agg.empty:
        return
    g = time_agg.sort_values("sample_size")
    plt.figure(figsize=(7.2, 4.6))
    plt.errorbar(
        g["sample_size"],
        g["train_seconds_mean"] / 60.0,
        yerr=g["train_seconds_std"] / 60.0,
        marker="o",
        linewidth=2,
        capsize=3,
        color="#F58518",
    )
    plt.xscale("log")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Training Time (min)")
    plt.title("Training Cost Versus Data Scale")
    savefig(fig_dir, "fig_s02_data_scale_training_time")


def plot_scale_layer_heatmap(layer_df: pd.DataFrame, fig_dir: Path) -> None:
    clean = layer_df[layer_df["noise"].eq("clean")]
    if clean.empty:
        return
    pivot = clean.groupby(["sample_size", "layer"])["layer_mae"].mean().unstack("layer")
    pivot = pivot.sort_index()
    plt.figure(figsize=(7.8, 4.8))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Layer MAE (m/s)")
    plt.yticks(np.arange(len(pivot.index)), [str(int(v)) for v in pivot.index])
    plt.xticks(np.arange(10), np.arange(1, 11))
    plt.xlabel("Radial Layer")
    plt.ylabel("Training Sample Size")
    plt.title("Layer-Wise Error Under Different Data Scales")
    savefig(fig_dir, "fig_s03_data_scale_layer_mae")


def write_report(run_dir: Path, cfg: dict, point_agg: pd.DataFrame, time_agg: pd.DataFrame, uq_agg: pd.DataFrame) -> None:
    lines = ["# Data Scale Benchmark Report", ""]
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Data file: `{cfg['data_file']}`")
    lines.append(f"- Sample sizes: `{cfg['sample_sizes']}`")
    lines.append(f"- Seeds: `{cfg['train_seeds']}`")
    lines.append(f"- Epochs per run: `{cfg['epochs']}`")
    lines.append("")
    lines.append("## Point Metrics")
    if not point_agg.empty:
        cols = ["sample_size", "noise", "mae_mean", "mae_std", "rmse_mean", "r2_mean", "violation_rate_percent_mean"]
        lines.append(point_agg[cols].sort_values(["noise", "sample_size"]).to_markdown(index=False))
    lines.append("")
    lines.append("## Training Cost")
    if not time_agg.empty:
        cols = ["sample_size", "train_seconds_mean", "train_seconds_std", "ms_per_sample_mean"]
        lines.append(time_agg[cols].sort_values("sample_size").to_markdown(index=False))
    if not uq_agg.empty:
        lines.append("")
        lines.append("## Calibrated Uncertainty Metrics")
        cols = ["sample_size", "noise", "corr_std_abs_error_mean", "picp_95_mean", "mpiw_95_mean", "ece_mean"]
        lines.append(uq_agg[uq_agg["calibration"].eq("calibrated")][cols].sort_values(["noise", "sample_size"]).to_markdown(index=False))
    lines.append("")
    lines.append("## Suggested Paper Use")
    lines.append("- Use clean and 10 dB MAE curves to show whether the training sample size is sufficient.")
    lines.append("- Use training time curve to justify the selected practical sample size.")
    lines.append("- Use layer-wise heatmap to show whether deeper layers need more samples.")
    (run_dir / "reports" / "DATA_SCALE_BENCHMARK_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict, run_dir: Path, smoke: bool = False) -> None:
    style_plots()
    dirs = ensure_dirs(run_dir)
    bench.configure_strict_globals(cfg)
    config_source = Path(cfg.get("_config_source", HERE / "data_scale_config.json"))
    if config_source.exists():
        shutil.copy2(config_source, dirs["reports"] / "data_scale_config_used.json")

    if smoke:
        cfg = dict(cfg)
        cfg["train_seeds"] = [int(cfg["train_seeds"][0])]
        cfg["sample_sizes"] = [1000, 2000]
        cfg["epochs"] = 2
        print("Running data-scale smoke test: sizes=1000/2000, seed=1, epochs=2")

    data_file = (HERE / cfg["data_file"]).resolve()
    x_raw, y_raw = bench.load_raw_data(data_file)
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
            "y_min_per_layer": np.min(y_raw, axis=0).astype(float).tolist(),
            "y_max_per_layer": np.max(y_raw, axis=0).astype(float).tolist(),
            "v_min_constraint": strict.V_MIN,
            "v_max_constraint": strict.V_MAX,
            "structured_v_ref_min": strict.V_REF_MIN,
            "structured_v_ref_max": strict.V_REF_MAX,
            "structured_max_total_drop": strict.MAX_TOTAL_DROP,
        },
    )

    args = bench.make_training_args(cfg)
    eval_sets = bench.make_eval_sets(x_test_raw, scaler_x, cfg, seed_offset=80000)
    coverages = [float(v) for v in cfg.get("uq_nominal_coverages", [0.5, 0.68, 0.8, 0.9, 0.95])]

    point_rows = []
    uq_rows = []
    coverage_rows = []
    bin_rows = []
    layer_rows = []
    time_rows = []

    available = len(y_train)
    for seed in cfg["train_seeds"]:
        rng = np.random.default_rng(int(seed) + 20260608)
        for requested_size in cfg["sample_sizes"]:
            sample_size = min(int(requested_size), available)
            subset_idx = rng.choice(available, size=sample_size, replace=False)
            subset_data = (
                (x_train_raw[subset_idx], x_train_scaled[subset_idx], y_train[subset_idx]),
                (x_val_raw, x_val_scaled, y_val),
                (x_test_raw, eval_sets["clean"], y_test),
                scaler_x,
                split,
            )
            print("\n" + "=" * 88)
            print(f"[seed {seed}] Training Transformer-SP with {sample_size} samples")
            print("=" * 88)
            model, history = bench.train_one_architecture(PROPOSED, subset_data, args, int(seed) + sample_size)
            save_json(dirs["histories"] / f"history_seed{seed}_n{sample_size}.json", history)
            model.save_weights(str(dirs["models"] / f"seed{seed}_n{sample_size}_Transformer_SP.weights.h5"))

            inference = bench.measure_inference(model, eval_sets["clean"], args.batch_size)
            time_rows.append(
                {
                    "seed": int(seed),
                    "sample_size": int(sample_size),
                    "train_seconds": float(history["train_seconds"]),
                    "parameter_count": int(model.count_params()),
                    **inference,
                }
            )

            calibration_shifts = {}
            val_sets = bench.make_eval_sets(x_val_raw, scaler_x, cfg, seed_offset=70000)
            for noise, x_val_eval in val_sets.items():
                val_mu, val_log = bench.predict_model(model, PROPOSED, x_val_eval, args.batch_size)
                calibration_shifts[noise] = bench.fit_logvar_shift(y_val, val_mu, val_log)
            save_json(dirs["reports"] / f"calibration_seed{seed}_n{sample_size}.json", calibration_shifts)

            for noise, x_eval in eval_sets.items():
                pred_mu, pred_log = bench.predict_model(model, PROPOSED, x_eval, args.batch_size)
                np.savez_compressed(
                    dirs["predictions"] / f"seed{seed}_n{sample_size}_{noise}.npz",
                    y_true=y_test,
                    pred_mu=pred_mu,
                    pred_log_var=pred_log,
                )
                pm = bench.point_metrics(y_test, pred_mu)
                point_rows.append(
                    {
                        "seed": int(seed),
                        "sample_size": int(sample_size),
                        "noise": noise,
                        **{k: v for k, v in pm.items() if k != "layer_mae"},
                    }
                )
                for layer_idx, layer_mae in enumerate(pm["layer_mae"], start=1):
                    layer_rows.append(
                        {
                            "seed": int(seed),
                            "sample_size": int(sample_size),
                            "noise": noise,
                            "layer": layer_idx,
                            "layer_mae": float(layer_mae),
                        }
                    )
                for calibration in ["raw", "calibrated"]:
                    log_for_eval = pred_log
                    shift = 0.0
                    if calibration == "calibrated":
                        shift = calibration_shifts.get(noise, calibration_shifts.get("clean", 0.0))
                        log_for_eval = pred_log + shift
                    um = bench.uncertainty_metrics(y_test, pred_mu, log_for_eval, coverages)
                    uq_rows.append(
                        {
                            "seed": int(seed),
                            "sample_size": int(sample_size),
                            "noise": noise,
                            "calibration": calibration,
                            "logvar_shift": float(shift),
                            **{k: v for k, v in um.items() if k != "coverage_curve"},
                        }
                    )
                    for item in um["coverage_curve"]:
                        coverage_rows.append(
                            {
                                "seed": int(seed),
                                "sample_size": int(sample_size),
                                "noise": noise,
                                "calibration": calibration,
                                **item,
                            }
                        )
                    for item in bench.uncertainty_bins(y_test, pred_mu, log_for_eval):
                        bin_rows.append(
                            {
                                "seed": int(seed),
                                "sample_size": int(sample_size),
                                "noise": noise,
                                "calibration": calibration,
                                **item,
                            }
                        )

    point_df = pd.DataFrame(point_rows)
    uq_df = pd.DataFrame(uq_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    bin_df = pd.DataFrame(bin_rows)
    layer_df = pd.DataFrame(layer_rows)
    time_df = pd.DataFrame(time_rows)

    point_df.to_csv(dirs["tables"] / "seed_data_scale_point_metrics.csv", index=False, encoding="utf-8-sig")
    uq_df.to_csv(dirs["tables"] / "seed_data_scale_uncertainty_metrics.csv", index=False, encoding="utf-8-sig")
    coverage_df.to_csv(dirs["tables"] / "seed_data_scale_coverage_curves.csv", index=False, encoding="utf-8-sig")
    bin_df.to_csv(dirs["tables"] / "seed_data_scale_uncertainty_bins.csv", index=False, encoding="utf-8-sig")
    layer_df.to_csv(dirs["tables"] / "seed_data_scale_layer_mae.csv", index=False, encoding="utf-8-sig")
    time_df.to_csv(dirs["tables"] / "seed_data_scale_efficiency.csv", index=False, encoding="utf-8-sig")

    point_agg = aggregate(point_df, ["sample_size", "noise"], ["mae", "rmse", "mape_percent", "r2", "violation_rate_percent"])
    time_agg = aggregate(time_df, ["sample_size"], ["train_seconds", "parameter_count", "ms_per_sample", "samples_per_second"])
    uq_agg = aggregate(
        uq_df,
        ["sample_size", "noise", "calibration"],
        ["nll", "corr_std_abs_error", "spearman_std_abs_error", "picp_95", "mpiw_95", "ece", "mean_pred_std"],
    )
    point_agg.to_csv(dirs["tables"] / "aggregate_data_scale_point_metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    time_agg.to_csv(dirs["tables"] / "aggregate_data_scale_efficiency_mean_std.csv", index=False, encoding="utf-8-sig")
    uq_agg.to_csv(dirs["tables"] / "aggregate_data_scale_uncertainty_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    plot_scale_error(point_agg, dirs["figures"])
    plot_scale_time(time_agg, dirs["figures"])
    plot_scale_layer_heatmap(layer_df, dirs["figures"])
    write_report(run_dir, cfg, point_agg, time_agg, uq_agg)
    print(f"\nData scale benchmark finished: {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run data-scale benchmark for the proposed model.")
    parser.add_argument("--config", default="data_scale_config.json")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    cfg_path = (HERE / parsed.config).resolve()
    cfg = load_config(cfg_path)
    cfg["_config_source"] = str(cfg_path)
    suffix = "_smoke" if parsed.smoke else ""
    run_name = parsed.output_name or f"{cfg['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    run_dir = Path.cwd() / "runs" / run_name if not Path(run_name).is_absolute() else Path(run_name)
    run(cfg, run_dir, smoke=parsed.smoke)
