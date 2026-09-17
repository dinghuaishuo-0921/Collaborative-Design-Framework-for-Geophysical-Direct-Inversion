from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import run_strict_standard_rerun as strict  # noqa: E402


N_LAYERS = 10
MLP_DEPTH_UNITS = 256


@dataclass(frozen=True)
class ArchitectureSpec:
    key: str
    label: str
    use_structured_prior: bool = False
    use_physics: bool = False
    use_uncertainty: bool = False


ARCHITECTURE_SPECS = [
    ArchitectureSpec("MLP", "MLP"),
    ArchitectureSpec("MLP_SP", "MLP-SP", True, False, False),
    ArchitectureSpec("MLP60K_SP", "MLP-60K-SP", True, False, False),
    ArchitectureSpec("CNN1D", "1D-CNN"),
    ArchitectureSpec("CNN1D_SP", "1D-CNN-SP", True, False, False),
    ArchitectureSpec("CNN1D100K_SP", "1D-CNN-100K-SP", True, False, False),
    ArchitectureSpec("ResCNN1D", "ResCNN"),
    ArchitectureSpec("BiLSTM", "BiLSTM"),
    ArchitectureSpec("BiLSTM_SP", "BiLSTM-SP", True, False, False),
    ArchitectureSpec("CNN_BiLSTM", "CNN-BiLSTM"),
    ArchitectureSpec("CNN_BiLSTM_Attention", "CNN-BiLSTM-Att"),
    ArchitectureSpec("CNN_BiLSTM_Attention_SP", "CNN-BiLSTM-Att-SP", True, False, False),
    ArchitectureSpec("TCN", "TCN"),
    ArchitectureSpec("Transformer", "Transformer"),
    ArchitectureSpec("Transformer_SOFT", "Transformer-Soft", False, True, False),
    ArchitectureSpec("Transformer_SP", "Transformer-SP", True, False, False),
    ArchitectureSpec("Transformer_SP_PC", "Transformer-SP-PC", True, True, False),
    ArchitectureSpec("Proposed_SP_PC_UQ", "Proposed", True, True, True),
    ArchitectureSpec("Transformer_UQ", "Transformer-UQ", False, False, True),
    ArchitectureSpec("Transformer_SP_UQ", "Transformer-SP-UQ", True, False, True),
    ArchitectureSpec("Transformer_SP_PC_UQ", "Transformer-SP-PC-UQ", True, True, True),
    ArchitectureSpec("ConvTransformer", "ConvTransformer"),
    ArchitectureSpec("ConvTransformer_SP_UQ", "ConvTransformer-SP-UQ", True, False, True),
    ArchitectureSpec("Fusion_MLP_Transformer", "MLP-Transformer-Fusion"),
    ArchitectureSpec("Fusion_MLP_Transformer_SP_UQ", "MLP-Transformer-Fusion-SP-UQ", True, False, True),
]


ARCHITECTURE_ALIASES = {
    "MLP_SP": "MLP",
    "CNN1D_SP": "CNN1D",
    "BiLSTM_SP": "BiLSTM",
    "CNN_BiLSTM_Attention_SP": "CNN_BiLSTM_Attention",
}


SHORT_NAME_MAP = {
    "MLP": "mlp",
    "MLP_SP": "mlpsp",
    "MLP60K_SP": "mlp60ksp",
    "CNN1D": "cnn1d",
    "CNN1D_SP": "cnn1dsp",
    "CNN1D100K_SP": "cnn1d100ksp",
    "ResCNN1D": "rescnn",
    "BiLSTM": "bilstm",
    "BiLSTM_SP": "bilstmsp",
    "CNN_BiLSTM": "cnnlstm",
    "CNN_BiLSTM_Attention": "cbatt",
    "CNN_BiLSTM_Attention_SP": "cbattsp",
    "TCN": "tcn",
    "Transformer": "tr",
    "Transformer_SOFT": "trsoft",
    "Transformer_SP": "trsp",
    "Transformer_SP_PC": "trsppc",
    "Proposed_SP_PC_UQ": "prop",
    "Transformer_UQ": "truq",
    "Transformer_SP_UQ": "trspuq",
    "Transformer_SP_PC_UQ": "trsppcuq",
    "ConvTransformer": "convtr",
    "ConvTransformer_SP_UQ": "convtruq",
    "Fusion_MLP_Transformer": "fusion",
    "Fusion_MLP_Transformer_SP_UQ": "fusionuq",
    "internal_test": "itest",
    "strict_external_zero_overlap_96009_unique": "ext96009",
    "strict_similar_domain_89713_unique": "sim89713",
    "strict_train_wide_140760_unique": "wide140760",
    "clean": "cln",
    "30dB": "n30",
    "20dB": "n20",
    "10dB": "n10",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {}
    for name in ["models", "predictions", "histories", "tables", "figures", "reports", "splits"]:
        path = run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def configure_strict_globals(cfg: dict) -> None:
    global MLP_DEPTH_UNITS
    MLP_DEPTH_UNITS = int(cfg.get("mlp_hidden_units", 256))
    strict.V_MIN = float(cfg.get("v_min", 1200.0))
    strict.V_MAX = float(cfg.get("v_max", 3500.0))
    strict.V_REF_MIN = float(cfg.get("structured_v_ref_min", strict.V_MIN))
    strict.V_REF_MAX = float(cfg.get("structured_v_ref_max", strict.V_MAX))
    strict.MAX_TOTAL_DROP = float(cfg.get("structured_max_total_drop", strict.V_REF_MAX - strict.V_MIN))
    strict.LOG_VAR_MIN = float(cfg.get("log_var_min", -4.0))
    strict.LOG_VAR_MAX = float(cfg.get("log_var_max", 18.0))


def make_training_args(cfg: dict, epochs: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=int(epochs or cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        min_lr=float(cfg["min_learning_rate"]),
        lr_factor=float(cfg["lr_factor"]),
        lr_patience=int(cfg["lr_patience"]),
        early_stop_patience=int(cfg.get("early_stop_patience", 0)),
        early_stop_min_delta=float(cfg.get("early_stop_min_delta", 1.0e-6)),
        physics_weight=float(cfg.get("physics_weight", 1.0)),
        uq_weight=float(cfg.get("uq_weight", 0.05)),
        train_noise_prob=float(cfg.get("train_noise_probability", 0.0)),
        train_noise_levels_db=[int(v) for v in cfg.get("train_noise_levels_db", [30, 20, 10])],
        print_every=int(cfg.get("print_every", 10)),
    )


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_raw_data(data_file: Path):
    mat = sio.loadmat(str(data_file))
    return mat["velocity2"].astype("float32"), mat["param2"].astype("float32")


def selected_architectures(cfg: dict) -> list[ArchitectureSpec]:
    selected = set(cfg["architectures"])
    specs = [spec for spec in ARCHITECTURE_SPECS if spec.key in selected]
    dynamic = []
    for key in sorted(selected):
        match = re.fullmatch(r"MLP_DEPTH_([1-9][0-9]*)", str(key))
        if match:
            depth = int(match.group(1))
            dynamic.append(ArchitectureSpec(key, f"MLP-{depth} hidden layer" + ("s" if depth != 1 else "")))
    specs.extend(dynamic)
    known = {spec.key for spec in ARCHITECTURE_SPECS} | {spec.key for spec in dynamic}
    missing = selected.difference(known)
    if missing:
        raise ValueError(f"Unknown architecture keys: {sorted(missing)}")
    return specs


def compact_name(name: str, max_len: int = 20) -> str:
    mapped = SHORT_NAME_MAP.get(str(name))
    if mapped:
        return mapped
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(name)).lower()
    if not cleaned:
        return "x"
    return cleaned[:max_len]


def transform_batch(x_raw, scaler_x, rng, noise_prob: float, noise_levels: list[int]):
    if noise_prob > 0 and noise_levels and rng.random() < noise_prob:
        snr = int(rng.choice(noise_levels))
        x_raw = strict.add_gaussian_noise_by_snr(x_raw, snr, rng)
    return scaler_x.transform(x_raw).reshape(-1, 256, 1).astype(np.float32)


def make_eval_sets(x_raw, scaler_x, cfg: dict, seed_offset: int) -> dict[str, np.ndarray]:
    sets = {"clean": scaler_x.transform(x_raw).reshape(-1, 256, 1).astype(np.float32)}
    for level in cfg["eval_noise_levels_db"]:
        if str(level).lower() == "clean":
            continue
        snr = int(level)
        rng = np.random.default_rng(int(cfg["split_seed"]) + seed_offset + snr * 1009)
        noisy = strict.add_gaussian_noise_by_snr(x_raw, snr, rng)
        sets[f"{snr}dB"] = scaler_x.transform(noisy).reshape(-1, 256, 1).astype(np.float32)
    return sets


def attention_pool(sequence, prefix: str):
    score = layers.Dense(64, activation="tanh", name=f"{prefix}_att_tanh")(sequence)
    score = layers.Dense(1, name=f"{prefix}_att_score")(score)
    weights = layers.Softmax(axis=1, name=f"{prefix}_att_weights")(score)
    return layers.Lambda(
        lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=1),
        name=f"{prefix}_att_context",
    )([sequence, weights])


class PositionEmbedding(layers.Layer):
    def __init__(self, max_length: int, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = int(max_length)
        self.channels = int(channels)
        self.embedding = layers.Embedding(input_dim=self.max_length, output_dim=self.channels)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        pos = self.embedding(positions)
        return inputs + tf.expand_dims(pos, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({"max_length": self.max_length, "channels": self.channels})
        return config


def conv_bn_relu(x, filters: int, kernel_size: int, name: str, dilation_rate: int = 1):
    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def residual_block(x, filters: int, kernel_size: int, name: str, dilation_rate: int = 1):
    shortcut = x
    x = conv_bn_relu(x, filters, kernel_size, f"{name}_a", dilation_rate=dilation_rate)
    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
        name=f"{name}_b_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_b_bn")(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same", use_bias=False, name=f"{name}_skip_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(x)


def transformer_block(x, channels: int, heads: int, ff_dim: int, name: str):
    attn = layers.MultiHeadAttention(num_heads=heads, key_dim=channels // heads, dropout=0.1, name=f"{name}_mha")(x, x)
    x = layers.Add(name=f"{name}_att_add")([x, attn])
    x = layers.LayerNormalization(name=f"{name}_att_ln")(x)
    ff = layers.Dense(ff_dim, activation="relu", name=f"{name}_ff_1")(x)
    ff = layers.Dropout(0.1, name=f"{name}_ff_drop")(ff)
    ff = layers.Dense(channels, name=f"{name}_ff_2")(ff)
    x = layers.Add(name=f"{name}_ff_add")([x, ff])
    return layers.LayerNormalization(name=f"{name}_ff_ln")(x)


def transformer_encoder(inputs, prefix: str, channels: int = 64, blocks: int = 3):
    # Downsample before self-attention to keep the attention matrix tractable
    # on 6 GB-class GPUs while preserving the full frequency trend.
    x = layers.Conv1D(channels, 5, strides=4, padding="same", name=f"{prefix}_projection")(inputs)
    x = PositionEmbedding(64, channels, name=f"{prefix}_pos_embedding")(x)
    for idx in range(1, blocks + 1):
        x = transformer_block(x, channels=channels, heads=4, ff_dim=channels * 2, name=f"{prefix}_block_{idx}")
    x = layers.GlobalAveragePooling1D(name=f"{prefix}_gap")(x)
    x = layers.Dense(128, activation="relu", name=f"{prefix}_dense_128")(x)
    return layers.Dense(64, activation="relu", name="shared_feature")(x)


def conv_transformer_encoder(inputs):
    convs = [
        layers.Conv1D(32, k, activation="relu", padding="same", name=f"ct_conv_k{k}")(inputs)
        for k in [3, 7, 11]
    ]
    x = layers.Concatenate(name="ct_multiscale_concat")(convs)
    x = layers.BatchNormalization(name="ct_multiscale_bn")(x)
    x = residual_block(x, 96, 5, "ct_res_1")
    x = layers.Conv1D(96, 5, strides=4, padding="same", activation="relu", name="ct_downsample")(x)
    x = PositionEmbedding(64, 96, name="ct_pos_embedding")(x)
    for idx in range(1, 3):
        x = transformer_block(x, channels=96, heads=4, ff_dim=192, name=f"ct_block_{idx}")
    gap = layers.GlobalAveragePooling1D(name="ct_gap")(x)
    att = attention_pool(x, "ct")
    x = layers.Concatenate(name="ct_pool_concat")([gap, att])
    x = layers.Dense(160, activation="relu", name="ct_dense_160")(x)
    x = layers.Dropout(0.15, name="ct_dropout_160")(x)
    return layers.Dense(64, activation="relu", name="shared_feature")(x)


def fusion_mlp_transformer_encoder(inputs):
    mlp = layers.Flatten(name="fusion_mlp_flatten")(inputs)
    mlp = layers.Dense(256, activation="relu", name="fusion_mlp_dense_256")(mlp)
    mlp = layers.Dropout(0.2, name="fusion_mlp_dropout_256")(mlp)
    mlp = layers.Dense(128, activation="relu", name="fusion_mlp_dense_128")(mlp)

    tr = layers.Conv1D(64, 5, strides=4, padding="same", name="fusion_tr_projection")(inputs)
    tr = PositionEmbedding(64, 64, name="fusion_tr_pos_embedding")(tr)
    for idx in range(1, 4):
        tr = transformer_block(tr, channels=64, heads=4, ff_dim=128, name=f"fusion_tr_block_{idx}")
    tr = layers.GlobalAveragePooling1D(name="fusion_tr_gap")(tr)
    tr = layers.Dense(128, activation="relu", name="fusion_tr_dense_128")(tr)

    x = layers.Concatenate(name="fusion_concat")([mlp, tr])
    x = layers.Dense(192, activation="relu", name="fusion_dense_192")(x)
    x = layers.Dropout(0.15, name="fusion_dropout_192")(x)
    x = layers.Dense(96, activation="relu", name="fusion_dense_96")(x)
    return layers.Dense(64, activation="relu", name="shared_feature")(x)


def build_encoder(arch_key: str, inputs):
    dynamic_match = re.fullmatch(r"MLP_DEPTH_([1-9][0-9]*)", str(arch_key))
    if dynamic_match:
        depth = int(dynamic_match.group(1))
        x = layers.Flatten(name=f"mlp_depth_{depth}_flatten")(inputs)
        for idx in range(1, depth + 1):
            x = layers.Dense(MLP_DEPTH_UNITS, activation="relu", name=f"mlp_depth_{depth}_dense_{idx}")(x)
        return layers.Dense(64, activation="relu", name=f"mlp_depth_{depth}_shared_feature")(x)

    arch_key = ARCHITECTURE_ALIASES.get(arch_key, arch_key)

    if arch_key == "MLP60K_SP":
        # 256 -> 160 -> 80 -> 64 plus the structured head: 59,899 parameters.
        x = layers.Flatten(name="mlp60k_flatten")(inputs)
        x = layers.Dense(160, activation="relu", name="mlp60k_dense_160")(x)
        x = layers.Dropout(0.2, name="mlp60k_dropout_160")(x)
        x = layers.Dense(80, activation="relu", name="mlp60k_dense_80")(x)
        return layers.Dense(64, activation="relu", name="shared_feature")(x)

    if arch_key == "MLP":
        x = layers.Flatten(name="mlp_flatten")(inputs)
        x = layers.Dense(256, activation="relu", name="mlp_dense_256")(x)
        x = layers.Dropout(0.2, name="mlp_dropout_256")(x)
        x = layers.Dense(128, activation="relu", name="mlp_dense_128")(x)
        return layers.Dense(64, activation="relu", name="shared_feature")(x)

    if arch_key == "CNN1D":
        x = conv_bn_relu(inputs, 32, 7, "cnn1")
        x = layers.MaxPooling1D(2, name="cnn_pool_1")(x)
        x = conv_bn_relu(x, 64, 5, "cnn2")
        x = layers.MaxPooling1D(2, name="cnn_pool_2")(x)
        x = conv_bn_relu(x, 128, 3, "cnn3")
        x = layers.GlobalAveragePooling1D(name="cnn_gap")(x)
        x = layers.Dense(128, activation="relu", name="cnn_dense_128")(x)
        return layers.Dense(64, activation="relu", name="shared_feature")(x)

    if arch_key == "CNN1D100K_SP":
        # 32 -> 64 -> 192 convolution channels and a 198-unit dense layer.
        # With the shared 64-unit feature layer and structured head this is
        # exactly 100,017 trainable parameters.
        x = conv_bn_relu(inputs, 32, 7, "cnn100k_1")
        x = layers.MaxPooling1D(2, name="cnn100k_pool_1")(x)
        x = conv_bn_relu(x, 64, 5, "cnn100k_2")
        x = layers.MaxPooling1D(2, name="cnn100k_pool_2")(x)
        x = conv_bn_relu(x, 192, 3, "cnn100k_3")
        x = layers.GlobalAveragePooling1D(name="cnn100k_gap")(x)
        x = layers.Dense(198, activation="relu", name="cnn100k_dense_198")(x)
        return layers.Dense(64, activation="relu", name="shared_feature")(x)

    if arch_key == "ResCNN1D":
        x = conv_bn_relu(inputs, 64, 7, "res_stem")
        x = residual_block(x, 64, 5, "res_block_1")
        x = layers.MaxPooling1D(2, name="res_pool_1")(x)
        x = residual_block(x, 96, 5, "res_block_2")
        x = layers.MaxPooling1D(2, name="res_pool_2")(x)
        x = residual_block(x, 128, 3, "res_block_3")
        x = layers.GlobalAveragePooling1D(name="res_gap")(x)
        x = layers.Dense(128, activation="relu", name="res_dense_128")(x)
        return layers.Dense(64, activation="relu", name="shared_feature")(x)

    if arch_key == "BiLSTM":
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_1")(inputs)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False), name="bilstm_2")(x)
        x = layers.Dense(128, activation="relu", name="bilstm_dense_128")(x)
        return layers.Dense(64, activation="relu", name="shared_feature")(x)

    if arch_key == "CNN_BiLSTM":
        x = conv_bn_relu(inputs, 48, 7, "cb_conv_1")
        x = conv_bn_relu(x, 64, 5, "cb_conv_2")
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="cb_bilstm_1")(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False), name="cb_bilstm_2")(x)
        x = layers.Dense(128, activation="relu", name="cb_dense_128")(x)
        return layers.Dense(64, activation="relu", name="shared_feature")(x)

    if arch_key in {"CNN_BiLSTM_Attention", "Proposed_SP_PC_UQ"}:
        convs = [
            layers.Conv1D(32, k, activation="relu", padding="same", name=f"conv_k{k}")(inputs)
            for k in [3, 5, 7, 9]
        ]
        x = layers.Concatenate(name="multi_scale_concat")(convs)
        x = layers.BatchNormalization(name="multi_scale_bn")(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="att_bilstm_1")(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="att_bilstm_2")(x)
        x = attention_pool(x, "freq")
        x = layers.Dense(128, activation="relu", name="att_dense_128")(x)
        x = layers.Dropout(0.15, name="att_dropout_128")(x)
        return layers.Dense(64, activation="relu", name="shared_feature")(x)

    if arch_key == "TCN":
        x = conv_bn_relu(inputs, 64, 3, "tcn_stem")
        for idx, dilation in enumerate([1, 2, 4, 8, 16], start=1):
            x = residual_block(x, 64, 3, f"tcn_block_{idx}", dilation_rate=dilation)
        x = layers.GlobalAveragePooling1D(name="tcn_gap")(x)
        x = layers.Dense(128, activation="relu", name="tcn_dense_128")(x)
        return layers.Dense(64, activation="relu", name="shared_feature")(x)

    if arch_key in {
        "Transformer",
        "Transformer_SOFT",
        "Transformer_SP",
        "Transformer_SP_PC",
        "Transformer_UQ",
        "Transformer_SP_UQ",
        "Transformer_SP_PC_UQ",
    }:
        return transformer_encoder(inputs, "tr", channels=64, blocks=3)

    if arch_key in {"ConvTransformer", "ConvTransformer_SP_UQ"}:
        return conv_transformer_encoder(inputs)

    if arch_key in {"Fusion_MLP_Transformer", "Fusion_MLP_Transformer_SP_UQ"}:
        return fusion_mlp_transformer_encoder(inputs)

    raise ValueError(f"Unsupported architecture: {arch_key}")


def architecture_batch_size(spec: ArchitectureSpec, default_batch_size: int) -> int:
    arch_key = ARCHITECTURE_ALIASES.get(spec.key, spec.key)
    attention_heavy = {
        "Transformer",
        "Transformer_SOFT",
        "Transformer_SP",
        "Transformer_SP_PC",
        "Transformer_UQ",
        "Transformer_SP_UQ",
        "Transformer_SP_PC_UQ",
        "ConvTransformer",
        "ConvTransformer_SP_UQ",
        "Fusion_MLP_Transformer",
        "Fusion_MLP_Transformer_SP_UQ",
    }
    if arch_key in attention_heavy:
        return min(int(default_batch_size), 256)
    return int(default_batch_size)


def build_benchmark_model(spec: ArchitectureSpec, input_shape=(256, 1)):
    inputs = layers.Input(shape=input_shape, name="dispersion_input")
    shared = build_encoder(spec.key, inputs)

    if spec.use_structured_prior:
        raw = layers.Dense(
            11,
            name="structured_raw",
            bias_initializer=tf.keras.initializers.Constant(strict.STRUCTURED_BIAS),
        )(shared)
        mu = layers.Lambda(strict.structured_velocity_mapping, name="mu_output")(raw)
    else:
        mu = layers.Dense(
            N_LAYERS,
            name="mu_output",
            bias_initializer=tf.keras.initializers.Constant(strict.DIRECT_BIAS),
        )(shared)

    if spec.use_uncertainty:
        log_var_hidden = layers.Dense(64, activation="relu", name="log_var_hidden")(shared)
        log_var = layers.Dense(
            N_LAYERS,
            name="log_var_output",
            bias_initializer=tf.keras.initializers.Constant(strict.LOG_VAR_BIAS),
        )(log_var_hidden)
        return Model(inputs=inputs, outputs=[mu, log_var], name=spec.key)

    return Model(inputs=inputs, outputs=mu, name=spec.key)


def make_compiled_train_step(model, optimizer, spec: ArchitectureSpec, args: SimpleNamespace):
    """Create an isolated compiled step for one model and one optimizer.

    TensorFlow 2.10 does not allow a global ``tf.function`` to lazily create
    Adam slot variables for a second optimizer. Binding the function to the
    current model/optimizer and initializing the slots first keeps separate
    benchmark combinations independent.
    """
    optimizer._create_all_weights(model.trainable_variables)

    @tf.function(reduce_retracing=True)
    def step(xb, yb):
        with tf.GradientTape() as tape:
            outputs = model(xb, training=True)
            if spec.use_uncertainty:
                mu, log_var = outputs
            else:
                mu, log_var = outputs, None
            mse = tf.reduce_mean(tf.square(yb - mu))
            physics_loss = tf.constant(0.0, dtype=tf.float32)
            uq_nll = tf.constant(0.0, dtype=tf.float32)
            loss = mse
            if spec.use_physics:
                physics_loss, _, _, _ = strict.physics_loss_components(mu)
                loss = loss + args.physics_weight * physics_loss
            if spec.use_uncertainty:
                uq_nll, _ = strict.uncertainty_nll(yb, mu, log_var)
                loss = loss + args.uq_weight * uq_nll
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mse, physics_loss, uq_nll

    return step


def train_one_architecture(spec: ArchitectureSpec, split_data, args: SimpleNamespace, seed: int):
    (x_train_raw, _, y_train), (_, x_val_scaled, y_val), _, scaler_x, _ = split_data
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_benchmark_model(spec)
    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    train_step = make_compiled_train_step(model, optimizer, spec, args)
    batch_size = architecture_batch_size(spec, args.batch_size)
    n_batches = max(1, int(np.ceil(len(y_train) / batch_size)))
    rng = np.random.default_rng(seed + 1009)
    best_val = np.inf
    best_weights = None
    wait_lr = 0
    wait_stop = 0
    start = time.perf_counter()
    history = {
        "train_total_loss": [],
        "train_supervised_mse": [],
        "train_physics_loss": [],
        "train_uq_nll": [],
        "val_supervised_mse": [],
        "learning_rate": [],
        "epochs_ran": 0,
        "early_stopped": False,
    }

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y_train))
        total_loss_sum = 0.0
        mse_sum = 0.0
        physics_sum = 0.0
        uq_sum = 0.0
        seen = 0
        for batch_no in range(n_batches):
            idx = order[batch_no * batch_size : (batch_no + 1) * batch_size]
            if len(idx) == 0:
                continue
            xb = transform_batch(
                x_train_raw[idx],
                scaler_x,
                rng,
                args.train_noise_prob,
                args.train_noise_levels_db,
            )
            yb = y_train[idx].astype(np.float32)
            loss, mse, physics_loss, uq_nll = train_step(xb, yb)
            current_batch_size = len(idx)
            total_loss_sum += float(loss.numpy()) * current_batch_size
            mse_sum += float(mse.numpy()) * current_batch_size
            physics_sum += float(physics_loss.numpy()) * current_batch_size
            uq_sum += float(uq_nll.numpy()) * current_batch_size
            seen += current_batch_size

        val_outputs = model.predict(x_val_scaled, batch_size=batch_size, verbose=0)
        val_mu = val_outputs[0] if spec.use_uncertainty else val_outputs
        val_mse = float(np.mean(np.square(y_val - val_mu)))
        history["train_total_loss"].append(total_loss_sum / max(seen, 1))
        history["train_supervised_mse"].append(mse_sum / max(seen, 1))
        history["train_physics_loss"].append(physics_sum / max(seen, 1))
        history["train_uq_nll"].append(uq_sum / max(seen, 1))
        history["val_supervised_mse"].append(val_mse)
        history["learning_rate"].append(float(tf.keras.backend.get_value(optimizer.learning_rate)))

        if val_mse < best_val - args.early_stop_min_delta:
            best_val = val_mse
            best_weights = model.get_weights()
            wait_lr = 0
            wait_stop = 0
        else:
            wait_lr += 1
            wait_stop += 1
            if wait_lr >= args.lr_patience:
                old_lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
                new_lr = max(args.min_lr, old_lr * args.lr_factor)
                tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
                wait_lr = 0

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"    epoch {epoch:03d}/{args.epochs} "
                f"loss={history['train_total_loss'][-1]:.4f} "
                f"val_mse={val_mse:.4f} "
                f"lr={history['learning_rate'][-1]:.2e}"
            )

        history["epochs_ran"] = epoch
        if args.early_stop_patience > 0 and wait_stop >= args.early_stop_patience:
            history["early_stopped"] = True
            print(
                f"    early stopping at epoch {epoch:03d}/{args.epochs} "
                f"(best_val_mse={best_val:.4f}, patience={args.early_stop_patience})"
            )
            break

    if best_weights is not None:
        model.set_weights(best_weights)
    train_seconds = time.perf_counter() - start
    history["best_val_supervised_mse"] = best_val
    history["train_seconds"] = train_seconds
    return model, history


def predict_model(model: Model, spec: ArchitectureSpec, x_eval, batch_size: int):
    outputs = model.predict(x_eval, batch_size=batch_size, verbose=0)
    if spec.use_uncertainty:
        return outputs[0].astype(np.float32), outputs[1].astype(np.float32)
    return outputs.astype(np.float32), None


def corrcoef_safe(a, b) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rank_corr(a, b) -> float:
    ra = pd.Series(np.asarray(a).reshape(-1)).rank(method="average").to_numpy()
    rb = pd.Series(np.asarray(b).reshape(-1)).rank(method="average").to_numpy()
    return corrcoef_safe(ra, rb)


def point_metrics(y_true, y_pred) -> dict:
    abs_error = np.abs(y_pred - y_true)
    mse = float(np.mean(np.square(y_pred - y_true)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_error))
    mape = float(np.mean(abs_error / (np.abs(y_true) + 1e-6)) * 100.0)
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    boundary_violation = (y_pred < strict.V_MIN) | (y_pred > strict.V_MAX)
    mono_violation = np.diff(y_pred, axis=1) < -1.0e-6
    sample_violation = np.any(boundary_violation, axis=1) | np.any(mono_violation, axis=1)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "r2": float(1.0 - ss_res / (ss_tot + 1.0e-12)),
        "violation_rate_percent": float(np.mean(sample_violation) * 100.0),
        "layer_mae": np.mean(abs_error, axis=0).tolist(),
    }


def fit_logvar_shift(y_true, y_pred, log_var) -> float:
    clipped = np.clip(log_var, strict.LOG_VAR_MIN, strict.LOG_VAR_MAX)
    var = np.exp(clipped) + strict.VAR_EPS
    scale2 = float(np.mean(np.square(y_true - y_pred)) / (np.mean(var) + strict.VAR_EPS))
    scale2 = min(max(scale2, 1.0e-4), 1.0e4)
    return float(np.log(scale2))


def uncertainty_metrics(y_true, y_pred, log_var, coverages: list[float]) -> dict:
    clipped = np.clip(log_var, strict.LOG_VAR_MIN, strict.LOG_VAR_MAX)
    var = np.exp(clipped) + strict.VAR_EPS
    std = np.sqrt(var)
    err = y_pred - y_true
    abs_err = np.abs(err)
    nll = 0.5 * np.mean(np.square(err) / var + clipped + math.log(2.0 * math.pi))
    coverage_rows = []
    ece_terms = []
    for nominal in coverages:
        z = NormalDist().inv_cdf((1.0 + float(nominal)) / 2.0)
        lower = y_pred - z * std
        upper = y_pred + z * std
        empirical = float(np.mean((y_true >= lower) & (y_true <= upper)))
        width = float(np.mean(upper - lower))
        coverage_rows.append({"nominal": float(nominal), "empirical": empirical, "width": width})
        ece_terms.append(abs(empirical - float(nominal)))
    return {
        "nll": float(nll),
        "corr_std_abs_error": corrcoef_safe(std, abs_err),
        "spearman_std_abs_error": rank_corr(std, abs_err),
        "mean_pred_std": float(np.mean(std)),
        "picp_95": next(item["empirical"] for item in coverage_rows if abs(item["nominal"] - 0.95) < 1e-9),
        "mpiw_95": next(item["width"] for item in coverage_rows if abs(item["nominal"] - 0.95) < 1e-9),
        "ece": float(np.mean(ece_terms)),
        "coverage_curve": coverage_rows,
    }


def uncertainty_bins(y_true, y_pred, log_var, n_bins: int = 10) -> list[dict]:
    clipped = np.clip(log_var, strict.LOG_VAR_MIN, strict.LOG_VAR_MAX)
    std = np.sqrt(np.exp(clipped) + strict.VAR_EPS).reshape(-1)
    abs_error = np.abs(y_pred - y_true).reshape(-1)
    quantiles = np.quantile(std, np.linspace(0.0, 1.0, n_bins + 1))
    rows = []
    for i in range(n_bins):
        low = quantiles[i]
        high = quantiles[i + 1]
        if i == n_bins - 1:
            mask = (std >= low) & (std <= high)
        else:
            mask = (std >= low) & (std < high)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": i + 1,
                "std_min": float(low),
                "std_max": float(high),
                "mean_pred_std": float(np.mean(std[mask])),
                "mean_abs_error": float(np.mean(abs_error[mask])),
                "sample_count": int(np.sum(mask)),
            }
        )
    return rows


def measure_inference(model: Model, x_eval, batch_size: int, repeats: int = 5) -> dict:
    n = min(len(x_eval), 4096)
    sample = x_eval[:n]
    model.predict(sample, batch_size=batch_size, verbose=0)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(sample, batch_size=batch_size, verbose=0)
        times.append(time.perf_counter() - start)
    mean_seconds = float(np.mean(times))
    return {
        "timed_samples": int(n),
        "inference_seconds": mean_seconds,
        "ms_per_sample": mean_seconds / max(n, 1) * 1000.0,
        "samples_per_second": n / max(mean_seconds, 1.0e-12),
    }


def aggregate_mean_std(df: pd.DataFrame, group_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
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


def style_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def savefig(fig_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    plt.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def architecture_order(df: pd.DataFrame) -> list[str]:
    order = [spec.key for spec in ARCHITECTURE_SPECS]
    available = set(df["architecture"].unique())
    return [key for key in order if key in available]


def labels_for(order: list[str]) -> list[str]:
    mapping = {spec.key: spec.label for spec in ARCHITECTURE_SPECS}
    return [mapping.get(key, key) for key in order]


def plot_internal_clean_bar(agg: pd.DataFrame, fig_dir: Path) -> None:
    clean = agg[(agg["dataset"].eq("internal_test")) & (agg["noise"].eq("clean"))]
    if clean.empty:
        return
    order = architecture_order(clean)
    clean = clean.set_index("architecture").loc[order]
    plt.figure(figsize=(8.8, 4.4))
    x = np.arange(len(order))
    plt.bar(x, clean["mae_mean"], yerr=clean["mae_std"], capsize=3, color="#4C78A8", alpha=0.9)
    plt.xticks(x, labels_for(order), rotation=25, ha="right")
    plt.ylabel("MAE (m/s)")
    plt.title("Internal Clean Test Accuracy Across Network Backbones")
    savefig(fig_dir, "fig_b01_internal_clean_mae")


def plot_noise_curves(agg: pd.DataFrame, fig_dir: Path) -> None:
    data = agg[agg["dataset"].eq("internal_test")]
    if data.empty:
        return
    noise_order = ["clean", "30dB", "20dB", "10dB"]
    x = np.arange(len(noise_order))
    plt.figure(figsize=(8.8, 5.0))
    for arch in architecture_order(data):
        g = data[data["architecture"].eq(arch)].set_index("noise")
        if not set(noise_order).issubset(set(g.index)):
            continue
        g = g.loc[noise_order]
        plt.errorbar(x, g["mae_mean"], yerr=g["mae_std"], marker="o", linewidth=1.8, capsize=2.5, label=labels_for([arch])[0])
    plt.xticks(x, noise_order)
    plt.xlabel("Evaluation Noise Level")
    plt.ylabel("MAE (m/s)")
    plt.title("Noise Robustness Across Network Backbones")
    plt.legend(ncol=2)
    savefig(fig_dir, "fig_b02_internal_noise_robustness")


def plot_external_heatmap(agg: pd.DataFrame, fig_dir: Path) -> None:
    ext = agg[(~agg["dataset"].eq("internal_test")) & (agg["noise"].eq("clean"))]
    if ext.empty:
        return
    pivot = ext.pivot(index="architecture", columns="dataset", values="mae_mean")
    order = [arch for arch in architecture_order(ext) if arch in pivot.index]
    pivot = pivot.loc[order]
    plt.figure(figsize=(8.4, 4.8))
    im = plt.imshow(pivot.values, aspect="auto", cmap="magma_r")
    plt.colorbar(im, label="MAE (m/s)")
    plt.yticks(np.arange(len(order)), labels_for(order))
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    plt.title("External Synthetic-Set Generalization (Clean)")
    savefig(fig_dir, "fig_b03_external_clean_heatmap")


def plot_efficiency_tradeoff(per_seed_df: pd.DataFrame, fig_dir: Path) -> None:
    clean = per_seed_df[(per_seed_df["dataset"].eq("internal_test")) & (per_seed_df["noise"].eq("clean"))]
    if clean.empty:
        return
    summary = clean.groupby(["architecture", "architecture_label"]).agg(
        mae=("mae", "mean"),
        params=("parameter_count", "mean"),
        ms=("ms_per_sample", "mean"),
    ).reset_index()
    plt.figure(figsize=(7.2, 5.2))
    sizes = np.clip(summary["ms"] / summary["ms"].max() * 700.0, 70.0, 700.0)
    plt.scatter(summary["params"] / 1.0e6, summary["mae"], s=sizes, alpha=0.75, color="#F58518", edgecolor="k", linewidth=0.5)
    for _, row in summary.iterrows():
        plt.annotate(row["architecture_label"], (row["params"] / 1.0e6, row["mae"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    plt.xscale("log")
    plt.xlabel("Trainable Parameters (million, log scale)")
    plt.ylabel("Clean MAE (m/s)")
    plt.title("Accuracy-Complexity-Inference Tradeoff")
    savefig(fig_dir, "fig_b04_efficiency_tradeoff")


def plot_layer_heatmap(layer_df: pd.DataFrame, fig_dir: Path) -> None:
    clean = layer_df[(layer_df["dataset"].eq("internal_test")) & (layer_df["noise"].eq("clean"))]
    if clean.empty:
        return
    pivot = clean.groupby(["architecture", "layer"])["layer_mae"].mean().unstack("layer")
    order = [arch for arch in architecture_order(clean) if arch in pivot.index]
    pivot = pivot.loc[order]
    plt.figure(figsize=(8.4, 4.6))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Layer MAE (m/s)")
    plt.yticks(np.arange(len(order)), labels_for(order))
    plt.xticks(np.arange(N_LAYERS), np.arange(1, N_LAYERS + 1))
    plt.xlabel("Radial Layer")
    plt.title("Layer-Wise Clean MAE Across Network Backbones")
    savefig(fig_dir, "fig_b05_layer_mae_heatmap")


def plot_reliability(coverage_df: pd.DataFrame, fig_dir: Path) -> None:
    required = {"architecture", "dataset", "calibration", "noise", "nominal", "empirical"}
    if coverage_df.empty or not required.issubset(set(coverage_df.columns)):
        return
    data = coverage_df[
        coverage_df["architecture"].eq("Proposed_SP_PC_UQ")
        & coverage_df["dataset"].eq("internal_test")
        & coverage_df["calibration"].eq("calibrated")
    ]
    if data.empty:
        return
    plt.figure(figsize=(5.8, 5.2))
    for noise, group in data.groupby("noise"):
        mean = group.groupby("nominal")["empirical"].mean().reset_index()
        plt.plot(mean["nominal"], mean["empirical"], marker="o", linewidth=2, label=noise)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
    plt.xlabel("Nominal Coverage")
    plt.ylabel("Empirical Coverage")
    plt.title("Reliability of Proposed Uncertainty Output")
    plt.xlim(0.45, 1.0)
    plt.ylim(0.45, 1.02)
    plt.legend()
    savefig(fig_dir, "fig_b06_proposed_reliability")


def plot_uncertainty_bins(bin_df: pd.DataFrame, fig_dir: Path) -> None:
    required = {"architecture", "dataset", "noise", "calibration", "bin", "mean_pred_std", "mean_abs_error"}
    if bin_df.empty or not required.issubset(set(bin_df.columns)):
        return
    data = bin_df[
        bin_df["architecture"].eq("Proposed_SP_PC_UQ")
        & bin_df["dataset"].eq("internal_test")
        & bin_df["noise"].eq("clean")
        & bin_df["calibration"].eq("calibrated")
    ]
    if data.empty:
        return
    mean = data.groupby("bin")[["mean_pred_std", "mean_abs_error"]].mean().reset_index()
    plt.figure(figsize=(6.6, 4.4))
    plt.plot(mean["bin"], mean["mean_abs_error"], marker="o", linewidth=2, label="Mean absolute error")
    plt.plot(mean["bin"], mean["mean_pred_std"], marker="s", linewidth=2, label="Mean predicted std")
    plt.xlabel("Predicted-Uncertainty Quantile Bin")
    plt.ylabel("Velocity (m/s)")
    plt.title("Uncertainty Ranking Versus Prediction Error")
    plt.legend()
    savefig(fig_dir, "fig_b07_uncertainty_error_bins")


def load_external_eval_sets(cfg: dict, scaler_x, smoke: bool) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    result = {}
    max_samples = int(cfg.get("external_eval_max_samples", 20000))
    if smoke:
        max_samples = min(max_samples, 1000)
    for offset, rel in enumerate(cfg.get("external_data_files", []), start=1):
        path = (HERE / rel).resolve()
        x_raw, y_raw = load_raw_data(path)
        if max_samples > 0 and len(x_raw) > max_samples:
            rng = np.random.default_rng(int(cfg["split_seed"]) + 77000 + offset)
            idx = rng.choice(len(x_raw), size=max_samples, replace=False)
            x_raw = x_raw[idx]
            y_raw = y_raw[idx]
        dataset_name = path.stem
        result[dataset_name] = (y_raw, make_eval_sets(x_raw, scaler_x, cfg, seed_offset=60000 + offset * 1000))
    return result


def write_report(run_dir: Path, cfg: dict, point_agg: pd.DataFrame, uq_agg: pd.DataFrame, efficiency_df: pd.DataFrame) -> None:
    lines = ["# Architecture Benchmark Report", ""]
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Training data: `{cfg['train_data_file']}`")
    lines.append(f"- External data: `{cfg.get('external_data_files', [])}`")
    lines.append(f"- Architectures: `{cfg['architectures']}`")
    lines.append(f"- Seeds: `{cfg['train_seeds']}`")
    lines.append(f"- Epochs: `{cfg['epochs']}`")
    lines.append(f"- Batch size: `{cfg['batch_size']}`")
    lines.append("")
    lines.append("## Internal Clean Accuracy")
    internal_clean = point_agg[(point_agg["dataset"].eq("internal_test")) & (point_agg["noise"].eq("clean"))]
    if not internal_clean.empty:
        cols = ["architecture_label", "mae_mean", "mae_std", "rmse_mean", "r2_mean", "violation_rate_percent_mean"]
        lines.append(internal_clean[cols].sort_values("mae_mean").to_markdown(index=False))
    lines.append("")
    lines.append("## External Clean Accuracy")
    external_clean = point_agg[(~point_agg["dataset"].eq("internal_test")) & (point_agg["noise"].eq("clean"))]
    if not external_clean.empty:
        cols = ["dataset", "architecture_label", "mae_mean", "mae_std", "rmse_mean", "violation_rate_percent_mean"]
        lines.append(external_clean[cols].sort_values(["dataset", "mae_mean"]).to_markdown(index=False))
    if not uq_agg.empty:
        lines.append("")
        lines.append("## Proposed Uncertainty Metrics")
        uq_show = uq_agg[
            uq_agg["architecture"].eq("Proposed_SP_PC_UQ")
            & uq_agg["dataset"].eq("internal_test")
            & uq_agg["calibration"].eq("calibrated")
        ]
        if not uq_show.empty:
            cols = ["noise", "nll_mean", "corr_std_abs_error_mean", "spearman_std_abs_error_mean", "picp_95_mean", "mpiw_95_mean", "ece_mean"]
            lines.append(uq_show[cols].sort_values("noise").to_markdown(index=False))
    if not efficiency_df.empty:
        lines.append("")
        lines.append("## Efficiency")
        lines.append(efficiency_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Suggested Paper Use")
    lines.append("- Main table: internal clean/noisy MAE-RMSE-MAPE-R2 for all backbones.")
    lines.append("- Robustness figure: MAE versus clean/30/20/10 dB.")
    lines.append("- Generalization table/heatmap: external clean MAE on 89713, 96009 and 140760 datasets.")
    lines.append("- Efficiency figure: parameter count and inference time versus clean MAE.")
    lines.append("- UQ subsection: reliability curve and uncertainty-error quantile bins for the proposed model.")
    (run_dir / "reports" / "ARCHITECTURE_BENCHMARK_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict, run_dir: Path, smoke: bool = False, config_source: Path | None = None) -> None:
    style_plots()
    dirs = ensure_dirs(run_dir)
    configure_strict_globals(cfg)
    source = config_source or (HERE / "benchmark_config.json")
    shutil.copy2(source, dirs["reports"] / "benchmark_config_used.json")

    if smoke:
        cfg = dict(cfg)
        cfg["train_seeds"] = [int(cfg["train_seeds"][0])]
        if cfg.get("architectures"):
            cfg["architectures"] = cfg["architectures"][: min(3, len(cfg["architectures"]))]
        else:
            cfg["architectures"] = ["MLP", "Transformer", "Proposed_SP_PC_UQ"]
        cfg["epochs"] = 2
        cfg["external_eval_max_samples"] = 1000
        print(
            "Running architecture benchmark smoke test: "
            f"seeds=1, architectures={cfg['architectures']}, epochs=2"
        )

    data_file = (HERE / cfg["train_data_file"]).resolve()
    x_raw, y_raw = load_raw_data(data_file)
    split_data = strict.prepare_split(x_raw, y_raw, int(cfg["split_seed"]))
    (x_train_raw, _, y_train), (x_val_raw, x_val_scaled, y_val), (x_test_raw, _, y_test), scaler_x, split = split_data
    strict.configure_initializers(y_train)
    np.savez(dirs["splits"] / "split_indices.npz", **split)

    dataset_summary = {
        "train_data_file": str(data_file),
        "x_shape": list(x_raw.shape),
        "y_shape": list(y_raw.shape),
        "train": int(len(y_train)),
        "validation": int(len(y_val)),
        "test": int(len(y_test)),
        "y_min": float(np.min(y_raw)),
        "y_max": float(np.max(y_raw)),
        "monotonic_violation_percent": float(np.mean(np.any(np.diff(y_raw, axis=1) < -1.0e-6, axis=1)) * 100.0),
        "v_min_constraint": strict.V_MIN,
        "v_max_constraint": strict.V_MAX,
        "structured_v_ref_min": strict.V_REF_MIN,
        "structured_v_ref_max": strict.V_REF_MAX,
        "structured_max_total_drop": strict.MAX_TOTAL_DROP,
    }
    save_json(dirs["reports"] / "dataset_summary.json", dataset_summary)

    val_sets = make_eval_sets(x_val_raw, scaler_x, cfg, seed_offset=20000)
    internal_test_sets = make_eval_sets(x_test_raw, scaler_x, cfg, seed_offset=40000)
    external_sets = load_external_eval_sets(cfg, scaler_x, smoke=smoke)
    specs = selected_architectures(cfg)
    args = make_training_args(cfg)
    coverages = [float(v) for v in cfg.get("uq_nominal_coverages", [0.5, 0.68, 0.8, 0.9, 0.95])]

    point_rows = []
    uq_rows = []
    coverage_rows = []
    bin_rows = []
    layer_rows = []
    efficiency_rows = []

    for train_seed in cfg["train_seeds"]:
        seed_dir = dirs["models"] / f"seed_{train_seed}"
        seed_dir.mkdir(exist_ok=True)
        for spec in specs:
            spec_file_id = compact_name(spec.key)
            print("\n" + "=" * 88)
            print(f"[seed {train_seed}] Training {spec.key}: {spec.label}")
            print("=" * 88)
            model, history = train_one_architecture(spec, split_data, args, int(train_seed))
            save_json(dirs["histories"] / f"hist_s{train_seed}_{spec_file_id}.json", history)
            model.save_weights(str(seed_dir / f"{spec_file_id}.weights.h5"))

            eval_batch_size = architecture_batch_size(spec, args.batch_size)
            inference = measure_inference(model, internal_test_sets["clean"], eval_batch_size)
            efficiency_rows.append(
                {
                    "seed": int(train_seed),
                    "architecture": spec.key,
                    "architecture_label": spec.label,
                    "actual_batch_size": int(eval_batch_size),
                    "parameter_count": int(model.count_params()),
                    "train_seconds": float(history["train_seconds"]),
                    **inference,
                }
            )

            calibration_shifts = {}
            if spec.use_uncertainty:
                for noise, x_val_eval in val_sets.items():
                    val_mu, val_log = predict_model(model, spec, x_val_eval, eval_batch_size)
                    calibration_shifts[noise] = fit_logvar_shift(y_val, val_mu, val_log)
                save_json(dirs["reports"] / f"cal_s{train_seed}_{spec_file_id}.json", calibration_shifts)

            eval_groups = {"internal_test": (y_test, internal_test_sets)}
            eval_groups.update(external_sets)
            for dataset_name, (y_eval, eval_sets) in eval_groups.items():
                dataset_file_id = compact_name(dataset_name)
                for noise, x_eval in eval_sets.items():
                    noise_file_id = compact_name(noise)
                    pred_mu, pred_log = predict_model(model, spec, x_eval, eval_batch_size)
                    pred_path = dirs["predictions"] / f"s{train_seed}_{spec_file_id}_{dataset_file_id}_{noise_file_id}.npz"
                    np.savez_compressed(pred_path, y_true=y_eval, pred_mu=pred_mu, pred_log_var=pred_log)
                    pm = point_metrics(y_eval, pred_mu)
                    point_rows.append(
                        {
                            "seed": int(train_seed),
                            "architecture": spec.key,
                            "architecture_label": spec.label,
                            "dataset": dataset_name,
                            "noise": noise,
                            "actual_batch_size": int(eval_batch_size),
                            "parameter_count": int(model.count_params()),
                            "train_seconds": float(history["train_seconds"]),
                            "ms_per_sample": inference["ms_per_sample"],
                            **{k: v for k, v in pm.items() if k != "layer_mae"},
                        }
                    )
                    for layer_idx, layer_mae in enumerate(pm["layer_mae"], start=1):
                        layer_rows.append(
                            {
                                "seed": int(train_seed),
                                "architecture": spec.key,
                                "architecture_label": spec.label,
                                "dataset": dataset_name,
                                "noise": noise,
                                "layer": layer_idx,
                                "layer_mae": float(layer_mae),
                            }
                        )
                    if spec.use_uncertainty:
                        for calibration in ["raw", "calibrated"]:
                            log_for_eval = pred_log
                            shift = 0.0
                            if calibration == "calibrated":
                                shift = calibration_shifts.get(noise, calibration_shifts.get("clean", 0.0))
                                log_for_eval = pred_log + shift
                            um = uncertainty_metrics(y_eval, pred_mu, log_for_eval, coverages)
                            uq_rows.append(
                                {
                                    "seed": int(train_seed),
                                    "architecture": spec.key,
                                    "architecture_label": spec.label,
                                    "dataset": dataset_name,
                                    "noise": noise,
                                    "calibration": calibration,
                                    "logvar_shift": float(shift),
                                    **{k: v for k, v in um.items() if k != "coverage_curve"},
                                }
                            )
                            for item in um["coverage_curve"]:
                                coverage_rows.append(
                                    {
                                        "seed": int(train_seed),
                                        "architecture": spec.key,
                                        "architecture_label": spec.label,
                                        "dataset": dataset_name,
                                        "noise": noise,
                                        "calibration": calibration,
                                        **item,
                                    }
                                )
                            for item in uncertainty_bins(y_eval, pred_mu, log_for_eval):
                                bin_rows.append(
                                    {
                                        "seed": int(train_seed),
                                        "architecture": spec.key,
                                        "architecture_label": spec.label,
                                        "dataset": dataset_name,
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
    efficiency_df = pd.DataFrame(efficiency_rows)

    point_df.to_csv(dirs["tables"] / "seed_architecture_point_metrics.csv", index=False, encoding="utf-8-sig")
    uq_df.to_csv(dirs["tables"] / "seed_architecture_uncertainty_metrics.csv", index=False, encoding="utf-8-sig")
    coverage_df.to_csv(dirs["tables"] / "seed_architecture_coverage_curves.csv", index=False, encoding="utf-8-sig")
    bin_df.to_csv(dirs["tables"] / "seed_architecture_uncertainty_bins.csv", index=False, encoding="utf-8-sig")
    layer_df.to_csv(dirs["tables"] / "seed_architecture_layer_mae.csv", index=False, encoding="utf-8-sig")
    efficiency_df.to_csv(dirs["tables"] / "seed_architecture_efficiency.csv", index=False, encoding="utf-8-sig")

    point_agg = aggregate_mean_std(
        point_df,
        ["architecture", "architecture_label", "dataset", "noise"],
        ["actual_batch_size", "mae", "rmse", "mape_percent", "r2", "violation_rate_percent", "parameter_count", "train_seconds", "ms_per_sample"],
    )
    point_agg.to_csv(dirs["tables"] / "aggregate_architecture_point_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    uq_agg = pd.DataFrame()
    if not uq_df.empty:
        uq_agg = aggregate_mean_std(
            uq_df,
            ["architecture", "architecture_label", "dataset", "noise", "calibration"],
            ["nll", "corr_std_abs_error", "spearman_std_abs_error", "picp_95", "mpiw_95", "ece", "mean_pred_std"],
        )
        uq_agg.to_csv(dirs["tables"] / "aggregate_architecture_uncertainty_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    efficiency_agg = aggregate_mean_std(
        efficiency_df,
        ["architecture", "architecture_label"],
        ["actual_batch_size", "parameter_count", "train_seconds", "ms_per_sample", "samples_per_second"],
    )
    efficiency_agg.to_csv(dirs["tables"] / "aggregate_architecture_efficiency_mean_std.csv", index=False, encoding="utf-8-sig")

    plot_internal_clean_bar(point_agg, dirs["figures"])
    plot_noise_curves(point_agg, dirs["figures"])
    plot_external_heatmap(point_agg, dirs["figures"])
    plot_efficiency_tradeoff(point_df, dirs["figures"])
    plot_layer_heatmap(layer_df, dirs["figures"])
    plot_reliability(coverage_df, dirs["figures"])
    plot_uncertainty_bins(bin_df, dirs["figures"])
    write_report(run_dir, cfg, point_agg, uq_agg, efficiency_agg)
    print(f"\nArchitecture benchmark finished: {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run architecture benchmark for radial shear velocity inversion.")
    parser.add_argument("--config", default="benchmark_config.json")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    cfg_path = (HERE / parsed.config).resolve()
    cfg = load_config(cfg_path)
    suffix = "_smoke" if parsed.smoke else ""
    run_name = parsed.output_name or f"{cfg['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    run_dir = Path.cwd() / "runs" / run_name if not Path(run_name).is_absolute() else Path(run_name)
    run(cfg, run_dir, smoke=parsed.smoke, config_source=cfg_path)
