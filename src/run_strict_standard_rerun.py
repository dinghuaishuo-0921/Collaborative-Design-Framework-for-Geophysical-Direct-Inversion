"""
Strict standard rerun for the radial shear velocity inversion experiment.

This script keeps the original CNN-LSTM-attention backbone, but makes the
experiment more reproducible and better calibrated:

1. fixed train/validation/test split indices are saved;
2. all models use the same split, noise realizations, epochs and optimizer;
3. structured-prior models use data-informed initialization;
4. the uncertainty model optimizes MSE + weighted heteroscedastic NLL;
5. uncertainty variance is calibrated on the validation set and both raw and
   calibrated metrics are saved;
6. model weights, histories, metrics, predictions, tables and figures are saved
   under a timestamped output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.preprocessing as preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers, optimizers

import run_complete_experiment_suite as base


plt.switch_backend("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent
DATA_FILE = SOURCE_DIR / "51047_ok.mat"

N_LAYERS = 10
V_MIN = 1200.0
V_MAX = 2600.0
V_REF_MIN = 1450.0
V_REF_MAX = 2450.0
MAX_TOTAL_DROP = 700.0

LOG_VAR_MIN = -4.0
LOG_VAR_MAX = 16.0
VAR_EPS = 1.0e-6

BOUNDARY_WEIGHT = 1.0
MONO_WEIGHT = 1.0
SMOOTH_WEIGHT = 5.0e-4
NOISE_LEVELS = [None, 30, 20, 10]

DIRECT_BIAS = None
STRUCTURED_BIAS = None
LOG_VAR_BIAS = math.log(30.0 ** 2)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {}
    for name in ["models", "tables", "figures", "results", "predictions", "logs"]:
        path = run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_data(path: Path):
    mat = sio.loadmat(str(path))
    x_raw = mat["velocity2"].astype("float32")
    y_raw = mat["param2"].astype("float32")
    return x_raw, y_raw


def prepare_split(
    x_raw,
    y_raw,
    seed: int,
    test_size=0.1,
    val_size=0.1,
    train_count=None,
    val_count=None,
    test_count=None,
):
    indices = np.arange(len(x_raw))
    explicit_counts = [train_count, val_count, test_count]
    if any(value is not None for value in explicit_counts):
        if not all(value is not None for value in explicit_counts):
            raise ValueError("train_count, val_count and test_count must be provided together.")
        counts = [int(train_count), int(val_count), int(test_count)]
        if min(counts) <= 0 or sum(counts) != len(indices):
            raise ValueError(
                "Explicit split counts must be positive and sum to the number of available samples: "
                f"counts={counts}, available={len(indices)}"
            )
        rng = np.random.default_rng(int(seed))
        shuffled = rng.permutation(indices)
        train_end = counts[0]
        val_end = train_end + counts[1]
        train_idx = shuffled[:train_end]
        val_idx = shuffled[train_end:val_end]
        test_idx = shuffled[val_end:]
    else:
        train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=seed, shuffle=True)

    scaler_x = preprocessing.StandardScaler()
    scaler_x.fit(x_raw[train_idx])

    def scaled(idx):
        return scaler_x.transform(x_raw[idx]).reshape(-1, 256, 1).astype(np.float32)

    data = (
        (x_raw[train_idx], scaled(train_idx), y_raw[train_idx]),
        (x_raw[val_idx], scaled(val_idx), y_raw[val_idx]),
        (x_raw[test_idx], scaled(test_idx), y_raw[test_idx]),
        scaler_x,
        {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx},
    )
    return data


def logit(x: float) -> float:
    x = min(max(x, 1.0e-5), 1.0 - 1.0e-5)
    return math.log(x / (1.0 - x))


def configure_initializers(y_train: np.ndarray) -> None:
    global DIRECT_BIAS, STRUCTURED_BIAS, LOG_VAR_BIAS
    DIRECT_BIAS = y_train.mean(axis=0).astype("float32")
    ref_mean = float(y_train[:, -1].mean())
    drop = y_train[:, -1:] - y_train[:, :1]
    drop_mean = float(np.mean(drop))
    gaps = np.diff(y_train, axis=1)
    gap_mean = np.maximum(gaps.mean(axis=0), 1.0e-3)
    prop = gap_mean / np.sum(gap_mean)

    ref_q = (ref_mean - V_REF_MIN) / (V_REF_MAX - V_REF_MIN)
    drop_q = drop_mean / MAX_TOTAL_DROP
    STRUCTURED_BIAS = np.concatenate(
        [
            np.array([logit(ref_q), logit(drop_q)], dtype=np.float32),
            np.log(prop.astype(np.float32)),
        ]
    ).astype("float32")
    LOG_VAR_BIAS = math.log(35.0 ** 2)


def structured_velocity_mapping(raw):
    raw_ref = raw[:, :1]
    raw_drop = raw[:, 1:2]
    raw_prop = raw[:, 2:11]

    v_ref = V_REF_MIN + (V_REF_MAX - V_REF_MIN) * tf.sigmoid(raw_ref)
    drop_cap = tf.minimum(v_ref - V_MIN, MAX_TOTAL_DROP)
    total_drop = drop_cap * tf.sigmoid(raw_drop)
    proportions = tf.nn.softmax(raw_prop, axis=1)
    gaps = total_drop * proportions
    tail_sum = tf.reverse(tf.cumsum(tf.reverse(gaps, axis=[1]), axis=1), axis=[1])
    inner_to_layer9 = v_ref - tail_sum
    return tf.concat([inner_to_layer9, v_ref], axis=1)


def attention_pool(sequence):
    score = layers.Dense(64, activation="tanh", name="attention_tanh")(sequence)
    score = layers.Dense(1, name="attention_score")(score)
    weights = layers.Softmax(axis=1, name="attention_weights")(score)
    return layers.Lambda(
        lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=1),
        name="attention_context",
    )([sequence, weights])


def build_model(config, input_shape=(256, 1)):
    inputs = layers.Input(shape=input_shape, name="dispersion_input")
    convs = [
        layers.Conv1D(32, k, activation="relu", padding="same", name=f"conv_k{k}")(inputs)
        for k in [3, 5, 7, 9]
    ]
    x = layers.Concatenate(name="multi_scale_concat")(convs)
    x = layers.BatchNormalization(name="multi_scale_bn")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_1")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_2")(x)
    x = attention_pool(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.15, name="dropout_128")(x)
    shared = layers.Dense(64, activation="relu", name="shared_feature")(x)

    if config.use_structured_prior:
        raw = layers.Dense(
            11,
            name="structured_raw",
            bias_initializer=tf.keras.initializers.Constant(STRUCTURED_BIAS),
        )(shared)
        mu = layers.Lambda(structured_velocity_mapping, name="mu_output")(raw)
    else:
        mu = layers.Dense(
            N_LAYERS,
            name="mu_output",
            bias_initializer=tf.keras.initializers.Constant(DIRECT_BIAS),
        )(shared)

    if config.use_uncertainty:
        log_var_hidden = layers.Dense(64, activation="relu", name="log_var_hidden")(shared)
        log_var = layers.Dense(
            N_LAYERS,
            name="log_var_output",
            bias_initializer=tf.keras.initializers.Constant(LOG_VAR_BIAS),
        )(log_var_hidden)
        return Model(inputs=inputs, outputs=[mu, log_var], name=config.key)
    return Model(inputs=inputs, outputs=mu, name=config.key)


def add_gaussian_noise_by_snr(x, snr_db, rng):
    signal_power = np.mean(np.square(x), axis=1, keepdims=True)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(np.maximum(noise_power, 1e-12)).astype(np.float32)
    return x + rng.normal(0, 1, x.shape).astype(np.float32) * noise_std


def supervised_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def uncertainty_nll(y_true, y_pred_mu, y_pred_log_var):
    clipped = tf.clip_by_value(y_pred_log_var, LOG_VAR_MIN, LOG_VAR_MAX)
    sq_error = tf.square(y_true - y_pred_mu)
    nll = 0.5 * (sq_error * tf.exp(-clipped) + clipped)
    return tf.reduce_mean(nll), clipped


def physics_loss_components(y_pred):
    boundary_min = tf.maximum(0.0, V_MIN - y_pred)
    boundary_max = tf.maximum(0.0, y_pred - V_MAX)
    boundary = tf.reduce_mean(tf.square(boundary_min) + tf.square(boundary_max))
    diff = y_pred[:, 1:] - y_pred[:, :-1]
    mono = tf.reduce_mean(tf.square(tf.maximum(0.0, -diff)))
    smooth = tf.reduce_mean(tf.square(diff))
    prior = BOUNDARY_WEIGHT * boundary + MONO_WEIGHT * mono + SMOOTH_WEIGHT * smooth
    return prior, boundary, mono, smooth


def transform_batch(x_raw, scaler_x, rng, noise_prob: float):
    if noise_prob > 0 and rng.random() < noise_prob:
        snr = int(rng.choice([30, 20, 10]))
        x_raw = add_gaussian_noise_by_snr(x_raw, snr, rng)
    return scaler_x.transform(x_raw).reshape(-1, 256, 1).astype(np.float32)


def train_one_model(config, data, args, seed: int):
    (x_train_raw, x_train_scaled, y_train), (x_val_raw, x_val_scaled, y_val), _, scaler_x, _ = data
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_model(config)
    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    n_batches = max(1, int(np.ceil(len(x_train_scaled) / args.batch_size)))
    rng = np.random.default_rng(seed + 1009)
    best_val = np.inf
    best_weights = None
    wait_lr = 0
    wait_stop = 0
    start = time.time()
    history = {
        "train_total_loss": [],
        "train_supervised_mse": [],
        "train_uncertainty_nll": [],
        "train_physics_loss": [],
        "train_boundary_loss": [],
        "train_mono_loss": [],
        "train_smooth_loss": [],
        "train_mean_std": [],
        "val_total_loss": [],
        "val_supervised_mse": [],
        "val_uncertainty_nll": [],
        "val_physics_loss": [],
        "learning_rate": [],
    }

    for epoch in range(args.epochs):
        order = rng.permutation(len(x_train_scaled))
        sums = {key: 0.0 for key in history if key.startswith("train_")}
        for batch_idx in range(n_batches):
            idx = order[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            if args.train_noise_prob > 0:
                x_batch = transform_batch(x_train_raw[idx], scaler_x, rng, args.train_noise_prob)
            else:
                x_batch = x_train_scaled[idx]
            y_batch = y_train[idx]
            with tf.GradientTape() as tape:
                outputs = model(x_batch, training=True)
                if config.use_uncertainty:
                    y_mu, y_log_var = outputs
                    nll, clipped_log_var = uncertainty_nll(y_batch, y_mu, y_log_var)
                    mse = supervised_mse(y_batch, y_mu)
                    data_loss = mse + args.uq_weight * nll
                    mean_std = tf.reduce_mean(tf.sqrt(tf.exp(clipped_log_var) + VAR_EPS))
                else:
                    y_mu = outputs
                    nll = tf.constant(0.0, dtype=tf.float32)
                    mse = supervised_mse(y_batch, y_mu)
                    data_loss = mse
                    mean_std = tf.constant(0.0, dtype=tf.float32)

                prior, boundary, mono, smooth = physics_loss_components(y_mu)
                physics = prior if config.use_physics else tf.constant(0.0, dtype=tf.float32)
                total = data_loss + args.physics_weight * physics

            grads = tape.gradient(total, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            sums["train_total_loss"] += float(total)
            sums["train_supervised_mse"] += float(mse)
            sums["train_uncertainty_nll"] += float(nll)
            sums["train_physics_loss"] += float(physics)
            sums["train_boundary_loss"] += float(boundary)
            sums["train_mono_loss"] += float(mono)
            sums["train_smooth_loss"] += float(smooth)
            sums["train_mean_std"] += float(mean_std)

        val_total_sum = 0.0
        val_mse_sum = 0.0
        val_nll_sum = 0.0
        val_physics_sum = 0.0
        val_count = 0
        for start_idx in range(0, len(x_val_scaled), args.batch_size):
            end_idx = min(start_idx + args.batch_size, len(x_val_scaled))
            x_val_batch = x_val_scaled[start_idx:end_idx]
            y_val_batch = y_val[start_idx:end_idx]
            val_outputs = model(x_val_batch, training=False)
            if config.use_uncertainty:
                val_mu, val_log_var = val_outputs
                val_nll, _ = uncertainty_nll(y_val_batch, val_mu, val_log_var)
                val_mse = supervised_mse(y_val_batch, val_mu)
                val_data = val_mse + args.uq_weight * val_nll
            else:
                val_mu = val_outputs
                val_nll = tf.constant(0.0, dtype=tf.float32)
                val_mse = supervised_mse(y_val_batch, val_mu)
                val_data = val_mse
            val_prior, _, _, _ = physics_loss_components(val_mu)
            val_physics = val_prior if config.use_physics else tf.constant(0.0, dtype=tf.float32)
            val_total_batch = val_data + args.physics_weight * val_physics
            batch_n = end_idx - start_idx
            val_total_sum += float(val_total_batch) * batch_n
            val_mse_sum += float(val_mse) * batch_n
            val_nll_sum += float(val_nll) * batch_n
            val_physics_sum += float(val_physics) * batch_n
            val_count += batch_n
        val_total = val_total_sum / max(1, val_count)
        val_mse_value = val_mse_sum / max(1, val_count)
        val_nll_value = val_nll_sum / max(1, val_count)
        val_physics_value = val_physics_sum / max(1, val_count)

        for key, value in sums.items():
            history[key].append(value / n_batches)
        history["val_total_loss"].append(val_total)
        history["val_supervised_mse"].append(val_mse_value)
        history["val_uncertainty_nll"].append(val_nll_value)
        history["val_physics_loss"].append(val_physics_value)
        history["learning_rate"].append(float(optimizer.learning_rate.numpy()))

        if val_total < best_val - args.min_delta:
            best_val = val_total
            best_weights = model.get_weights()
            wait_lr = 0
            wait_stop = 0
        else:
            wait_lr += 1
            wait_stop += 1
            if wait_lr >= args.lr_patience:
                new_lr = max(args.min_lr, float(optimizer.learning_rate.numpy()) * args.lr_factor)
                optimizer.learning_rate.assign(new_lr)
                wait_lr = 0
            if args.early_stop_patience > 0 and wait_stop >= args.early_stop_patience:
                print(f"{config.key} early stopped at epoch {epoch + 1}")
                break

        if (epoch + 1) % args.print_every == 0 or epoch == 0:
            print(
                f"{config.key} epoch {epoch + 1}/{args.epochs}: "
                f"val_total={val_total:.4e}, val_mse={val_mse_value:.4e}, "
                f"lr={float(optimizer.learning_rate.numpy()):.2e}"
            )

    if best_weights is not None:
        model.set_weights(best_weights)
    history["meta"] = {
        "model": asdict(config),
        "epochs_requested": args.epochs,
        "epochs_ran": len(history["val_total_loss"]),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": seed,
        "training_seconds": time.time() - start,
        "best_val_total": best_val,
    }
    return model, history


def predict_model(model, config, x_scaled):
    outputs = model.predict(x_scaled, verbose=0, batch_size=1024)
    if config.use_uncertainty:
        mu, log_var = outputs
    else:
        mu = outputs
        log_var = np.zeros_like(mu, dtype=np.float32)
    return mu.astype(np.float32), log_var.astype(np.float32)


def calculate_metrics(y_true, y_pred, log_var=None):
    abs_error = np.abs(y_pred - y_true)
    mse = float(np.mean(np.square(y_pred - y_true)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_error))
    mape = float(np.mean(abs_error / (np.abs(y_true) + 1e-6)) * 100.0)
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    boundary_violation = (y_pred < V_MIN) | (y_pred > V_MAX)
    mono_violation = np.diff(y_pred, axis=1) < -1e-6
    sample_violation = np.any(boundary_violation, axis=1) | np.any(mono_violation, axis=1)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "r2": float(1.0 - ss_res / (ss_tot + 1e-12)),
        "violation_rate_percent": float(np.mean(sample_violation) * 100.0),
        "layer_mae": np.mean(abs_error, axis=0).tolist(),
    }
    if log_var is not None:
        clipped = np.clip(log_var, LOG_VAR_MIN, LOG_VAR_MAX)
        pred_std = np.sqrt(np.exp(clipped) + VAR_EPS)
        flat_std = pred_std.reshape(-1)
        flat_error = abs_error.reshape(-1)
        corr = 0.0 if np.std(flat_std) < 1e-12 or np.std(flat_error) < 1e-12 else float(np.corrcoef(flat_std, flat_error)[0, 1])
        lower = y_pred - 1.96 * pred_std
        upper = y_pred + 1.96 * pred_std
        bins = []
        for bin_id, idx in enumerate(np.array_split(np.argsort(flat_std), 5), start=1):
            bins.append(
                {
                    "bin": bin_id,
                    "mean_std": float(np.mean(flat_std[idx])),
                    "mean_abs_error": float(np.mean(flat_error[idx])),
                    "count": int(len(idx)),
                }
            )
        metrics.update(
            {
                "corr_std_abs_error": corr,
                "picp_95": float(np.mean((y_true >= lower) & (y_true <= upper))),
                "mpiw_95": float(np.mean(upper - lower)),
                "mean_pred_std": float(np.mean(pred_std)),
                "uncertainty_bins": bins,
            }
        )
    return metrics


def fit_logvar_shift(y_true, pred_mu, log_var):
    clipped = np.clip(log_var, LOG_VAR_MIN, LOG_VAR_MAX)
    var = np.exp(clipped) + VAR_EPS
    sq = np.square(y_true - pred_mu)
    scale2 = float(np.mean(sq) / (np.mean(var) + VAR_EPS))
    scale2 = min(max(scale2, 1.0e-4), 1.0e4)
    return float(np.log(scale2))


def make_eval_sets(x_raw, scaler_x, seed: int):
    eval_sets = {"clean": scaler_x.transform(x_raw).reshape(-1, 256, 1).astype(np.float32)}
    for snr in [30, 20, 10]:
        rng = np.random.default_rng(seed + snr * 17)
        noisy = add_gaussian_noise_by_snr(x_raw, snr, rng)
        eval_sets[f"{snr}dB"] = scaler_x.transform(noisy).reshape(-1, 256, 1).astype(np.float32)
    return eval_sets


def save_dataset_tables_and_plots(x_train, y_train, x_val, y_val, x_test, y_test, dirs):
    freq_axis = np.arange(1, x_train.shape[1] + 1)
    layer_axis = np.arange(1, y_train.shape[1] + 1)
    pd.DataFrame(
        {
            "item": [
                "radial_range_m",
                "n_layers",
                "frequency_samples",
                "train_samples",
                "validation_samples",
                "test_samples",
                "total_samples",
                "v_min_used",
                "v_max_used",
                "v_ref_min",
                "v_ref_max",
                "max_total_drop",
                "noise_levels",
            ],
            "value": [
                "0-1",
                N_LAYERS,
                x_train.shape[1],
                len(x_train),
                len(x_val),
                len(x_test),
                len(x_train) + len(x_val) + len(x_test),
                V_MIN,
                V_MAX,
                V_REF_MIN,
                V_REF_MAX,
                MAX_TOTAL_DROP,
                "Clean, 30 dB, 20 dB, 10 dB",
            ],
        }
    ).to_csv(dirs["tables"] / "table_1_dataset_settings.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    plt.plot(freq_axis, mean, linewidth=2, label="Mean")
    plt.fill_between(freq_axis, mean - std, mean + std, alpha=0.25, label="±1 std")
    plt.xlabel("Frequency Sample Index")
    plt.ylabel("Dispersion Response")
    plt.title("Input Dispersion Response Distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "fig_1_input_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.boxplot([y_train[:, i] for i in range(y_train.shape[1])], tick_labels=[str(i) for i in layer_axis], showfliers=False)
    plt.xlabel("Layer Index")
    plt.ylabel("Shear Velocity (m/s)")
    plt.title("Output Velocity Model Distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "fig_2_output_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_history(histories, fig_dir: Path):
    plt.figure(figsize=(8, 5))
    for key, history in histories.items():
        plt.plot(history["val_supervised_mse"], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.yscale("log")
    plt.title("Validation Loss Across Ablation Models")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_3_training_validation_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_standard_figures(ablation_df, noise_df, y_test, predictions_by_model, log_var_full, dirs):
    base.plot_ablation(ablation_df, str(dirs["figures"]))
    base.plot_noise(noise_df, str(dirs["figures"]))
    # Patch base constants for plotting uncertainty with the wider variance range.
    base.LOG_VAR_MIN = LOG_VAR_MIN
    base.LOG_VAR_MAX = LOG_VAR_MAX
    base.VAR_EPS = VAR_EPS
    base.calculate_metrics = calculate_metrics
    base.plot_uncertainty((y_test, predictions_by_model["M5_Full"], log_var_full), str(dirs["figures"]))
    base.plot_typical_samples(y_test, predictions_by_model, str(dirs["figures"]))


def write_final_summary(run_dir: Path, args, ablation_df, uncertainty_raw_df, uncertainty_cal_df, scale_df=None):
    lines = [
        "# Strict Standard Rerun Summary",
        "",
        "## Run Settings",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Learning rate: `{args.learning_rate}`",
        f"- Train noise probability: `{args.train_noise_prob}`",
        f"- UQ weight: `{args.uq_weight}`",
        f"- Physics weight: `{args.physics_weight}`",
        f"- Seeds: `{args.seed}`",
        f"- Dataset: `{DATA_FILE.name}`",
        "",
        "## Ablation Results",
        "",
        ablation_df.to_markdown(index=False),
        "",
        "## Raw Uncertainty Results",
        "",
        uncertainty_raw_df.to_markdown(index=False),
        "",
        "## Validation-Calibrated Uncertainty Results",
        "",
        uncertainty_cal_df.to_markdown(index=False),
    ]
    if scale_df is not None:
        lines += ["", "## Data-Scale Results", "", scale_df.to_markdown(index=False)]
    (run_dir / "FINAL_STRICT_RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_complete(args):
    global V_REF_MIN, V_REF_MAX, MAX_TOTAL_DROP
    V_REF_MIN = args.v_ref_min
    V_REF_MAX = args.v_ref_max
    MAX_TOTAL_DROP = args.max_total_drop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.output_name or f"strict_standard_rerun_{timestamp}"
    run_dir = SCRIPT_DIR / run_name
    dirs = ensure_dirs(run_dir)
    set_seed(args.seed)

    x_raw, y_raw = load_data(DATA_FILE)
    data = prepare_split(x_raw, y_raw, args.seed)
    (x_train, x_train_scaled, y_train), (x_val, x_val_scaled, y_val), (x_test, x_test_scaled, y_test), scaler_x, split = data
    configure_initializers(y_train)
    np.savez(dirs["results"] / "split_indices.npz", **split)
    with (dirs["results"] / "scaler_x.pkl").open("wb") as f:
        pickle.dump(scaler_x, f)
    save_json(
        dirs["results"] / "run_config.json",
        {
            **vars(args),
            "data_file": str(DATA_FILE),
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "test_samples": int(len(y_test)),
            "v_min": V_MIN,
            "v_max": V_MAX,
            "v_ref_min": V_REF_MIN,
            "v_ref_max": V_REF_MAX,
            "max_total_drop": MAX_TOTAL_DROP,
            "log_var_min": LOG_VAR_MIN,
            "log_var_max": LOG_VAR_MAX,
            "direct_bias": DIRECT_BIAS.tolist(),
            "structured_bias": STRUCTURED_BIAS.tolist(),
            "log_var_bias": LOG_VAR_BIAS,
        },
    )
    save_dataset_tables_and_plots(x_train, y_train, x_val, y_val, x_test, y_test, dirs)

    val_eval_sets = make_eval_sets(x_val, scaler_x, args.seed + 3000)
    test_eval_sets = make_eval_sets(x_test, scaler_x, args.seed + 9000)
    histories = {}
    metrics_by_model_noise = {}
    predictions_by_model = {}
    full_log_vars = {}
    full_calibrated_log_vars = {}

    requested_models = {item.strip() for item in args.model_keys.split(",") if item.strip()}
    model_configs = [cfg for cfg in base.MODEL_CONFIGS if cfg.key in requested_models]
    if not model_configs:
        raise ValueError(f"No valid model keys selected: {args.model_keys}")

    for config in model_configs:
        print("\n" + "=" * 80)
        print(f"Training {config.key}: {config.label}")
        print("=" * 80)
        model, history = train_one_model(config, data, args, args.seed)
        histories[config.key] = history
        save_json(dirs["results"] / f"history_{config.key}.json", history)
        model.save_weights(str(dirs["models"] / f"{config.key}.weights.h5"))

        metrics_by_model_noise[config.key] = {}
        val_shifts = {}
        if config.use_uncertainty:
            for noise_key, x_val_eval in val_eval_sets.items():
                val_mu, val_log = predict_model(model, config, x_val_eval)
                val_shifts[noise_key] = fit_logvar_shift(y_val, val_mu, val_log)
            save_json(dirs["results"] / f"calibration_shifts_{config.key}.json", val_shifts)

        for noise_order, (noise_key, x_eval) in enumerate(test_eval_sets.items()):
            pred_mu, pred_log = predict_model(model, config, x_eval)
            raw_metrics = calculate_metrics(y_test, pred_mu, pred_log if config.use_uncertainty else None)
            raw_metrics["noise_order"] = noise_order
            metrics_by_model_noise[config.key][noise_key] = {"raw": raw_metrics}
            if config.use_uncertainty:
                shift = val_shifts.get(noise_key, val_shifts.get("clean", 0.0))
                cal_log = pred_log + shift
                cal_metrics = calculate_metrics(y_test, pred_mu, cal_log)
                cal_metrics["noise_order"] = noise_order
                cal_metrics["logvar_shift"] = shift
                metrics_by_model_noise[config.key][noise_key]["calibrated"] = cal_metrics
            np.savez_compressed(
                dirs["predictions"] / f"{config.key}_{noise_key}.npz",
                y_true=y_test,
                pred_mu=pred_mu,
                pred_log_var=pred_log,
            )
            if noise_key == "clean":
                predictions_by_model[config.key] = pred_mu
                if config.use_uncertainty:
                    full_log_vars[config.key] = pred_log
                    full_calibrated_log_vars[config.key] = pred_log + val_shifts.get("clean", 0.0)

    save_json(dirs["results"] / "complete_metrics_nested.json", metrics_by_model_noise)

    ablation_rows = []
    for config in model_configs:
        clean = metrics_by_model_noise[config.key]["clean"]["raw"]
        noise10 = metrics_by_model_noise[config.key]["10dB"]["raw"]
        degradation = (noise10["mae"] - clean["mae"]) / (clean["mae"] + 1e-12) * 100.0
        ablation_rows.append(
            {
                "Model": config.label,
                "SP": "Y" if config.use_structured_prior else "N",
                "PC": "Y" if config.use_physics else "N",
                "UQ": "Y" if config.use_uncertainty else "N",
                "MAE": clean["mae"],
                "RMSE": clean["rmse"],
                "MAPE (%)": clean["mape_percent"],
                "R2": clean["r2"],
                "Violation Rate (%)": clean["violation_rate_percent"],
                "MAE@10dB": noise10["mae"],
                "Degradation (%)": degradation,
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(dirs["tables"] / "table_2_ablation_metrics.csv", index=False, encoding="utf-8-sig")

    noise_rows = []
    for config in model_configs:
        for noise_key, item in metrics_by_model_noise[config.key].items():
            metrics = item["raw"]
            noise_rows.append(
                {
                    "Model": config.label,
                    "Noise Label": noise_key,
                    "Noise Order": metrics["noise_order"],
                    "MAE": metrics["mae"],
                    "RMSE": metrics["rmse"],
                    "MAPE (%)": metrics["mape_percent"],
                    "Violation Rate (%)": metrics["violation_rate_percent"],
                }
            )
    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(dirs["tables"] / "table_3_noise_metrics.csv", index=False, encoding="utf-8-sig")

    uncertainty_raw_rows = []
    uncertainty_cal_rows = []
    if "M5_Full" in metrics_by_model_noise:
        for noise_key, item in metrics_by_model_noise["M5_Full"].items():
            raw = item["raw"]
            uncertainty_raw_rows.append(
                {
                    "Dataset": noise_key,
                    "Corr(std, abs error)": raw["corr_std_abs_error"],
                    "PICP@95%": raw["picp_95"],
                    "MPIW@95%": raw["mpiw_95"],
                    "Mean Pred Std": raw["mean_pred_std"],
                }
            )
            cal = item["calibrated"]
            uncertainty_cal_rows.append(
                {
                    "Dataset": noise_key,
                    "Corr(std, abs error)": cal["corr_std_abs_error"],
                    "PICP@95%": cal["picp_95"],
                    "MPIW@95%": cal["mpiw_95"],
                    "Mean Pred Std": cal["mean_pred_std"],
                    "LogVar Shift": cal["logvar_shift"],
                }
            )
    uncertainty_raw_df = pd.DataFrame(uncertainty_raw_rows)
    uncertainty_cal_df = pd.DataFrame(uncertainty_cal_rows)
    uncertainty_raw_df.to_csv(dirs["tables"] / "table_4_uncertainty_metrics_raw.csv", index=False, encoding="utf-8-sig")
    uncertainty_cal_df.to_csv(dirs["tables"] / "table_4b_uncertainty_metrics_calibrated.csv", index=False, encoding="utf-8-sig")
    if "M5_Full" in metrics_by_model_noise:
        pd.DataFrame(metrics_by_model_noise["M5_Full"]["clean"]["calibrated"]["uncertainty_bins"]).to_csv(
            dirs["tables"] / "table_5_uncertainty_bins_clean_calibrated.csv", index=False, encoding="utf-8-sig"
        )

    layer_rows = []
    for config in model_configs:
        clean = metrics_by_model_noise[config.key]["clean"]["raw"]
        for layer_idx, layer_mae in enumerate(clean["layer_mae"], start=1):
            layer_rows.append({"Model": config.label, "Layer": layer_idx, "Layer MAE": layer_mae})
    pd.DataFrame(layer_rows).to_csv(dirs["tables"] / "table_6_layer_mae.csv", index=False, encoding="utf-8-sig")

    plot_training_history(histories, dirs["figures"])
    if "M5_Full" in predictions_by_model:
        plot_standard_figures(
            ablation_df,
            noise_df,
            y_test,
            predictions_by_model,
            full_calibrated_log_vars["M5_Full"],
            dirs,
        )
    else:
        base.plot_ablation(ablation_df, str(dirs["figures"]))
        base.plot_noise(noise_df, str(dirs["figures"]))
    scale_df = None
    if args.run_data_scale:
        scale_df = run_data_scale(args, data, dirs)
    write_final_summary(run_dir, args, ablation_df, uncertainty_raw_df, uncertainty_cal_df, scale_df)
    print("\nStrict standard rerun finished.")
    print(f"Run directory: {run_dir}")
    return run_dir


def run_data_scale(args, full_data, dirs):
    (x_train_raw, x_train_scaled, y_train), val_data, test_data, scaler_x, split = full_data
    full_config = [cfg for cfg in base.MODEL_CONFIGS if cfg.key == "M5_Full"][0]
    rng = np.random.default_rng(args.seed + 777)
    rows = []
    available = len(x_train_scaled)
    for requested in [int(x) for x in args.sample_sizes.split(",") if x.strip()]:
        sample_size = min(requested, available)
        idx = rng.choice(available, size=sample_size, replace=False)
        subset_data = ((x_train_raw[idx], x_train_scaled[idx], y_train[idx]), val_data, test_data, scaler_x, split)
        old_epochs = args.epochs
        args.epochs = args.scale_epochs
        print(f"\nTraining data-scale M5 with {sample_size} samples for {args.scale_epochs} epochs")
        start = time.time()
        model, _ = train_one_model(full_config, subset_data, args, args.seed + sample_size)
        train_seconds = time.time() - start
        args.epochs = old_epochs
        _, x_test_scaled, y_test = test_data
        pred_mu, pred_log = predict_model(model, full_config, x_test_scaled)
        metrics = calculate_metrics(y_test, pred_mu, pred_log)
        rows.append(
            {
                "Requested Sample Size": requested,
                "Actual Sample Size": sample_size,
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "MAPE (%)": metrics["mape_percent"],
                "Violation Rate (%)": metrics["violation_rate_percent"],
                "Training Time (s)": train_seconds,
            }
        )
        model.save_weights(str(dirs["models"] / f"M5_Full_datascale_{sample_size}.weights.h5"))
    df = pd.DataFrame(rows)
    df.to_csv(dirs["tables"] / "table_7_data_scale_metrics.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(8, 5))
    plt.plot(df["Actual Sample Size"], df["MAE"], marker="o", linewidth=2, label="MAE")
    plt.plot(df["Actual Sample Size"], df["RMSE"], marker="s", linewidth=2, label="RMSE")
    plt.xlabel("Training Sample Size")
    plt.ylabel("Error")
    plt.title("Data Scale vs Prediction Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "fig_11_data_scale_error.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(df["Actual Sample Size"], df["Training Time (s)"], marker="o", linewidth=2)
    plt.xlabel("Training Sample Size")
    plt.ylabel("Training Time (s)")
    plt.title("Data Scale vs Training Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "fig_12_data_scale_time.png", dpi=300, bbox_inches="tight")
    plt.close()
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Run strict standard rerun.")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--model-keys", default="M1_Baseline,M2_SP,M3_PC,M4_SP_PC,M5_Full")
    parser.add_argument("--v-ref-min", type=float, default=V_REF_MIN)
    parser.add_argument("--v-ref-max", type=float, default=V_REF_MAX)
    parser.add_argument("--max-total-drop", type=float, default=MAX_TOTAL_DROP)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=18)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--uq-weight", type=float, default=0.05)
    parser.add_argument("--train-noise-prob", type=float, default=0.35)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--run-data-scale", action="store_true")
    parser.add_argument("--sample-sizes", default="5000,10000,25000,41347")
    parser.add_argument("--scale-epochs", type=int, default=70)
    return parser.parse_args()


if __name__ == "__main__":
    run_complete(parse_args())
