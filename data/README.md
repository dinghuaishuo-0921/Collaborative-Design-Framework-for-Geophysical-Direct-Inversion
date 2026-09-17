# Reported Aggregate Results

This folder contains compact result tables used by the manuscript. It does not
contain raw training pools, checkpoints, or figure-generation code.

| Manuscript experiment | Aggregate table(s) |
| --- | --- |
| Main Transformer-SP noise robustness | `main_model_noise_metrics_d3.csv`, `main_model_layer_mae_by_noise_d3.csv`, `prediction_profiles_all_noise_d3.csv` |
| Sampling organization (D1/D2/D3) | `sampling_strategy.csv` |
| Training-set scale (10k to 200k) | `transformer_scale_consistent_10k_200k.csv` |
| Data-quality contamination | `data_quality_pollution.csv` |
| Free, soft-constrained, and structured output | `structured_output.csv`, `structured_output_noise_metrics.csv` |
| Five-encoder comparison | `encoder_replaceability.csv`, `encoder_noise_metrics.csv` |
| MLP depth and scale | `aggregate_architecture_scale_point_metrics_mean_std.csv` |

MAE and RMSE fields are in m/s unless a column name says otherwise. Values
with `_mean` and `_std` are the mean and standard deviation across the reported
independent seeds. The full data file is released separately through Mendeley
Data; see `docs/data_format.md`.
