# exp078_compact_surface_longtail_gate セッションノート

## 現在の状態

- status: `local_oof_completed_no_submit_candidate`
- route: `ml_model`
- actual experiment name: `exp078_compact_surface_longtail_gate`
- parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- compact parent: `exp075_compact_tracker_pfbeam_feature_repro_guard`
- requested backlog name had an exp073 prefix, but the implemented experiment correctly increments to exp078 and removes that prefix from the name.

## 実装内容

- `docs/legacy/steering/20260618-exp078-compact-surface-longtail-gate/` を作成。
- `experiments/exp078_compact_surface_longtail_gate/` を exp077 の postprocess audit 基盤から作成。
- `config.yaml` を compact surface long-tail gate 用に更新。
- `run_compact_surface_longtail_gate()` を追加。
  - exp073 OOF predictions を base として読む。
  - exp075 OOF predictions を compact branch として読む。
  - id / well で align し、target と anchor columns の一致を確認する。
  - `tail_rank_ge1000`、`tail_or_len_long`、`tail_rank_ge1000_diff_p50`、`tail_rank_ge1000_diff_p75` を `w=0.05/0.10/0.20` で比較する。
  - discussion #698860 を反映し、RMSE だけでなく SSE / delta SSE を保存する。
  - discussion #700340 を反映し、well-level regression と worst-well regression を保存する。
  - row-level predictions は全 policy ではなく best policy のみ保存する。
- `run_compact_surface_longtail_inference()` を追加。
  - exp073 / exp075 の saved inference predictions を読む。
  - train audit で固定した policy を test rows に適用する。
  - `submission.csv`、prediction SHA、submission SHA を保存する。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp078_compact_surface_longtail_gate
uv run python scripts/new_experiment.py --name exp078_compact_surface_longtail_gate --source experiments/exp077_full_replay_postprocess_guard
```

## 次のアクション

## ローカル OOF audit 結果

- command: `uv run python - <<'PY' ... run_compact_surface_longtail_gate(...)`
- output: `experiments/exp078_compact_surface_longtail_gate/artifacts/`
- rows / wells: `3,783,989` / `773`
- exp073 prediction source SHA: `986e26c5c6617ade714623d44433e9beacdb2b1027d46c4a4e70825bc8ab87fc`
- exp075 prediction source SHA: `a41f27848c2937950492c6c391da3d12f6d537ad85afc0aaf2d68fa613f09c0a`
- selected policy: `baseline_exp073`
- selected RMSE: `9.526374749390682`
- best compact diagnostic by RMSE: `tail_or_len_long_w020`
- best compact diagnostic RMSE: `9.362945426943881`
- best compact diagnostic delta RMSE: `-0.16342932244680064`
- best compact diagnostic delta SSE: `-11681440.0`
- rejection reason: `max_well_rmse_regression=2.908365249633789` exceeds guard `0.25`
- decision: compact gate is not a submit candidate in this form.

## 検証

- `uv run python -m py_compile experiments/exp078_compact_surface_longtail_gate/exp063_full_replay_reproducibility_guard.py experiments/exp078_compact_surface_longtail_gate/public_notebook_replay_audit.py experiments/exp078_compact_surface_longtail_gate/settings.py`: PASS
- `uv run python -m json.tool experiments/exp078_compact_surface_longtail_gate/exp078_compact_surface_longtail_gate_train.ipynb`: PASS
- `uv run python -m json.tool experiments/exp078_compact_surface_longtail_gate/exp078_compact_surface_longtail_gate_inference.ipynb`: PASS
- `uv run ruff check experiments/exp078_compact_surface_longtail_gate/exp063_full_replay_reproducibility_guard.py experiments/exp078_compact_surface_longtail_gate/public_notebook_replay_audit.py experiments/exp078_compact_surface_longtail_gate/settings.py`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp078_compact_surface_longtail_gate`: PASS

## 次のアクション

1. compact gate を続けるなら、悪化 well を特定して `tail_or_len_long_w020` の除外条件を作る。
2. ただし現時点では `projection_only_on_exp073` や `modelpkg_tiny_gate_on_exp073` の方が優先。
