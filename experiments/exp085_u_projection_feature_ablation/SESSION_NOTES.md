# exp085_u_projection_feature_ablation セッションノート

## 現在の状態

- status: `kaggle_train_v1_timeout_log_eval`
- route: `ml_model`
- parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- blocked: none

## 実装内容

- `.steering/20260620-exp085-u-projection-feature-ablation/` を作成。
- `experiments/exp085_u_projection_feature_ablation/` を exp080 から作成。
- `settings.py` の experiment name を exp085 に更新。
- `config.yaml` を U-space projection feature ablation 用に更新。
- 補助実装を `u_projection_feature_ablation.py` に整理。
  - exp072 deterministic 196-feature train cache を読む。
  - raw train known-prefix rows から well ごとの `T0` / `Z0` を復元する。
  - `U = candidate_tvt + Z - (T0 + Z0)` の local U-space を使う。
  - PF ANCC、PF-Z、Beam mean/median、likelihood-PF mean から robust polynomial projection を well 内で fit する。
  - projection correction、residual、absolute residual、residual MAD、slope/curvature、candidate disagreement を生成する。
  - 初期 active variants は control、projection correction、U-space disagreement、両者の和。
  - target は `TVT - last_known_tvt` のままにし、target ablation は混ぜない。
  - LGB OOF 由来 U-space feature は nested fold が必要なため default 無効。誤って有効化された場合は `NotImplementedError` で停止する。
  - fold/pooled metrics、well metrics、distance/tail buckets、OOF predictions、feature schema、feature importance、model manifest を保存する。
- train notebook を exp085 用の読みやすい 4 セクション構成に更新。
- inference notebook は selected variant 未設定なら停止する guard notebook として更新。

## 実行コマンド

```bash
python3 scripts/new_steering.py --experiment exp085_u_projection_feature_ablation
python3 scripts/new_experiment.py --name exp085_u_projection_feature_ablation --source experiments/exp080_u_space_target_ablation
```

## 次のアクション

1. Kaggle train を実行して、variant 別 pooled RMSE、worst-well、distance/tail bucket、feature importance を確認する。
2. 改善候補が出た場合だけ inference port を実装し、raw-test regenerated features と projection feature parity を監査する。

## 検証

- `uv run python -m py_compile experiments/exp085_u_projection_feature_ablation/u_projection_feature_ablation.py experiments/exp085_u_projection_feature_ablation/public_notebook_replay_audit.py experiments/exp085_u_projection_feature_ablation/settings.py`: PASS
- `uv run python -m json.tool experiments/exp085_u_projection_feature_ablation/exp085_u_projection_feature_ablation_train.ipynb`: PASS
- `uv run python -m json.tool experiments/exp085_u_projection_feature_ablation/exp085_u_projection_feature_ablation_inference.ipynb`: PASS
- `uv run ruff check experiments/exp085_u_projection_feature_ablation/u_projection_feature_ablation.py experiments/exp085_u_projection_feature_ablation/public_notebook_replay_audit.py experiments/exp085_u_projection_feature_ablation/settings.py`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp085_u_projection_feature_ablation`: PASS
- synthetic frame による `build_u_projection_features()` smoke test: PASS、16 rows / 71 columns、feature groups は projection correction 20、projection shape 20、U-disagreement 24。
- `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp085_u_projection_feature_ablation --notebook train --kernel-id kentookumura/exp085-u-projection-feature-ablation-train --title "exp085 u projection feature ablation train" --run-on-push --strict`: PASS
- generated train package: `experiments/exp085_u_projection_feature_ablation/kaggle/train`
- generated kernel id: `kentookumura/exp085-u-projection-feature-ablation-train`
- generated metadata: GPU enabled, internet disabled, run_on_push true, competition source `rogii-wellbore-geology-prediction`, kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- generated bootstrap manifest includes `config.yaml` SHA `e027438e3d9e3d83a12454bfe129b39fff40ec6faa13b99d60390be61c97c003` and `u_projection_feature_ablation.py` SHA `1570207bb77b875f0a54511e900e2071fc087e9c0b664e3448a24351204278ef`。

## Kaggle train v1

- 2026-06-20: `kaggle kernels push -p experiments/exp085_u_projection_feature_ablation/kaggle/train`: `Kernel version 1 successfully pushed`
- kernel id: `kentookumura/exp085-u-projection-feature-ablation-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp085-u-projection-feature-ablation-train`
- pull existence check: PASS at `/tmp/kaggle-pull/exp085-u-projection-feature-ablation-train-v1`
- initial normal logs: empty
- 3 minute `logs -f --interval 15` polling: no output before timeout; treated as Kaggle API log lag or still-running state, not failure.
- output probe: `/tmp/kaggle-output/exp085_u_projection_feature_ablation/train_v1_probe` contained no files yet.
- status probe after output check: `KernelWorkerStatus.RUNNING`
- User requested to stop monitoring for now.

## Kaggle train v1 log-derived evaluation

- 2026-06-20: Kaggle train v1 は timeout。正式 metrics CSV / pooled OOF artifact は保存されていない。
- `kaggle kernels logs kentookumura/exp085-u-projection-feature-ablation-train` を取得し、全ログを `/tmp/exp085_train_v1_logs.json` に保存。
- logs から fold-model JSON 行を抽出し、以下を保存。
  - `experiments/exp085_u_projection_feature_ablation/artifacts/exp085_train_v1_log_fold_metrics.csv`
  - `experiments/exp085_u_projection_feature_ablation/artifacts/exp085_train_v1_log_variant_model_summary.csv`
- 実行は 11.937h まで進み、59/60 fold-model metrics を回収。最後は `u_projection_correction_plus_disagreement` / `lgb2` / fold3 まで完了し、fold4 が未到達。

### Log-derived fold metrics summary

注意: 以下は fold RMSE の単純平均であり、正式 pooled RMSE ではない。

| variant | model | completed folds | mean fold RMSE |
| --- | --- | ---: | ---: |
| `u_projection_correction_plus_disagreement` | `lgb1` | 5 | 9.291006 |
| `u_disagreement` | `lgb2` | 5 | 9.392842 |
| `u_disagreement` | `lgb1` | 5 | 9.417772 |
| `u_projection_correction` | `lgb2` | 5 | 9.450537 |
| `u_projection_correction_plus_disagreement` | `lgb0` | 5 | 9.499037 |
| `u_projection_correction` | `lgb1` | 5 | 9.503377 |
| `control_exp073_base196` | `lgb1` | 5 | 9.534549 |
| `control_exp073_base196` | `lgb2` | 5 | 9.541831 |
| `u_projection_correction` | `lgb0` | 5 | 9.557780 |
| `control_exp073_base196` | `lgb0` | 5 | 9.633324 |
| `u_disagreement` | `lgb0` | 5 | 9.726577 |
| `u_projection_correction_plus_disagreement` | `lgb2` | 4 | 9.037692 |

Interpretation:

- `u_projection_correction_plus_disagreement` が最有望。`lgb1` は control `lgb1` から -0.243542、`lgb0` は control `lgb0` から -0.134287 改善。
- `u_projection_correction_plus_disagreement` / `lgb2` は fold4 未完了だが、完了済み 4 folds は 9.037692。fold4 が他モデル同様に 10.3 前後なら平均は約 9.29 で、`lgb1` と同程度に有望。
- `u_disagreement` 単独は `lgb1/lgb2` では改善したが、`lgb0` では悪化。単独採用より correction との併用を優先する。
- 次に回す場合は全 variant 再実行ではなく、`u_projection_correction_plus_disagreement` だけに絞って full pooled OOF / bucket metrics / feature importance を完走させる。
