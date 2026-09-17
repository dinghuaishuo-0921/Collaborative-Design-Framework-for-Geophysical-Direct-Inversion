from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import run_architecture_benchmark as bench


HERE = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_local(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (HERE / path).resolve()


def apply_updates(base: dict, updates: dict | None) -> dict:
    result = dict(base)
    if not updates:
        return result
    for key, value in updates.items():
        result[key] = value
    return result


def summarize(root_dir: Path, strategies: list[dict]) -> None:
    table_dir = root_dir / "tables"
    report_dir = root_dir / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for strategy in strategies:
        agg_path = root_dir / strategy["key"] / "tables" / "aggregate_architecture_point_metrics_mean_std.csv"
        if not agg_path.exists():
            continue
        df = pd.read_csv(agg_path)
        df.insert(0, "strategy_key", strategy["key"])
        df.insert(1, "strategy_label", strategy.get("label", strategy["key"]))
        df.insert(2, "strategy_notes", strategy.get("notes", ""))
        all_rows.append(df)

    if not all_rows:
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(table_dir / "sampling_strategy_all_metrics.csv", index=False, encoding="utf-8-sig")

    focus_mask = (
        ((combined["dataset"] == "internal_test") & (combined["noise"].isin(["clean", "10dB"])))
        | ((combined["dataset"] != "internal_test") & (combined["noise"] == "clean"))
    )
    focus = combined.loc[focus_mask].copy()
    focus.to_csv(table_dir / "sampling_strategy_focus_metrics.csv", index=False, encoding="utf-8-sig")

    lines = ["# Dataset Variant Benchmark", ""]
    lines.append("## Variants")
    for strategy in strategies:
        lines.append(
            f"- `{strategy['key']}`: {strategy.get('label', strategy['key'])}. "
            f"{strategy.get('notes', '').strip()}"
        )
    lines.append("")
    lines.append("## Suggested Paper Use")
    lines.append("- Use `sampling_strategy_focus_metrics.csv` for the manuscript table.")
    lines.append("- Compare the same case network under different dataset variants while holding all other settings fixed.")
    lines.append("- Highlight clean MAE, 10 dB MAE, external clean MAE and physical violation rate.")
    lines.append("")
    lines.append("## Output Files")
    lines.append(f"- Combined metrics: `{table_dir / 'sampling_strategy_all_metrics.csv'}`")
    lines.append(f"- Focus metrics: `{table_dir / 'sampling_strategy_focus_metrics.csv'}`")
    (report_dir / "DATASET_VARIANT_BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_name: str | None, smoke: bool) -> Path:
    cfg = load_json(config_path)
    base_cfg_path = resolve_local(cfg["benchmark_config"])
    base_cfg = load_json(base_cfg_path)
    base_cfg = apply_updates(base_cfg, cfg.get("benchmark_overrides"))

    suffix = "_smoke" if smoke else ""
    run_name = output_name or f"{cfg['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    root_dir = Path.cwd() / "runs" / run_name if not Path(run_name).is_absolute() else Path(run_name)
    root_dir.mkdir(parents=True, exist_ok=True)

    generated_cfg_dir = root_dir / "generated_configs"
    save_json(generated_cfg_dir / "sampling_strategy_config_used.json", cfg)

    strategies = cfg["sampling_datasets"]
    for strategy in strategies:
        strategy_cfg = dict(base_cfg)
        strategy_cfg["experiment_name"] = f"{cfg['experiment_name']}_{strategy['key']}"
        strategy_cfg["train_data_file"] = strategy["train_data_file"]
        strategy_cfg = apply_updates(strategy_cfg, strategy.get("benchmark_overrides"))

        generated_cfg_path = generated_cfg_dir / f"{strategy['key']}.json"
        save_json(generated_cfg_path, strategy_cfg)

        strategy_dir = root_dir / strategy["key"]
        bench.run(strategy_cfg, strategy_dir, smoke=smoke, config_source=generated_cfg_path)

    summarize(root_dir, strategies)
    return root_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark D1/D2/D3 sampling strategies under a shared framework model.")
    parser.add_argument("--config", default="sampling_strategy_benchmark_template.json")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(resolve_local(args.config), args.output_name, args.smoke)
