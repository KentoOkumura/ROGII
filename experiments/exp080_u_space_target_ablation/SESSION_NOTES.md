# exp080_u_space_target_ablation セッションノート

## 現在の状態

- status: `kaggle_train_v1_timeout_log_metrics_recovered`
- route: `ml_model`
- parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- blocked: none. Kaggle train v1 は timeout したが、logs から target/model/fold 別 RMSE を回収済み。

## 実装内容

- `.steering/20260618-exp080-u-space-target-ablation/` を作成。
- `experiments/exp080_u_space_target_ablation/` を exp073 から作成。
- `settings.py` の experiment name を exp080 に更新。
- `config.yaml` を U-space target ablation 用に更新。
- 補助実装を `u_space_target_ablation.py` に整理。
  - exp072 deterministic 196-feature train cache を読む。
  - raw train known-prefix rows から well ごとの `T0` / `Z0` を復元する。
  - `last_known_tvt` と復元 `T0` の差が 0.05ft を超える場合は fail する。
  - `dTVT`、`dTVT_plus_dZ`、`TVT_plus_Z_abs`、`TVT_plus_Z_minus_T0`、`TVT_plus_Z_minus_T0Z0` を同一 folds / 同一 features / 同一 LightGBM config で比較する。
  - 各 target の予測は inverse transform で TVT 空間に戻して RMSE を計算する。
  - pooled/fold metrics、well metrics、distance/tail buckets、target summary、OOF predictions、feature schema、model manifest を保存する。
- train notebook を exp080 用の読みやすい 4 セクション構成に更新。
- inference notebook は selected target 未設定なら停止する guard notebook として更新。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp080_u_space_target_ablation
uv run python scripts/new_experiment.py --name exp080_u_space_target_ablation --source experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay
```

## 次のアクション

1. target ablation は baseline `dTVT` 維持で閉じる。
2. 追加実行するなら、全 target を回すのではなく `dTVT` と必要最小限の比較 target だけに絞る。
3. U-space を続けるなら target 変更ではなく、projection / U-space disagreement feature の add-only 方向を優先する。

## Kaggle train v1

- 2026-06-18: `uv run python scripts/validate_experiment.py --experiment exp080_u_space_target_ablation`: PASS
- 2026-06-18: `uv run ruff check experiments/exp080_u_space_target_ablation/u_space_target_ablation.py experiments/exp080_u_space_target_ablation/public_notebook_replay_audit.py experiments/exp080_u_space_target_ablation/settings.py`: PASS
- 2026-06-18: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp080_u_space_target_ablation --notebook train --kernel-id kentookumura/exp080-u-space-target-ablation-train --title "exp080 u space target ablation train" --run-on-push --strict`: PASS
- 2026-06-18: `kaggle kernels push -p experiments/exp080_u_space_target_ablation/kaggle/train`: `Kernel version 1 successfully pushed`
- kernel id: `kentookumura/exp080-u-space-target-ablation-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp080-u-space-target-ablation-train`
- metadata: GPU enabled, internet disabled, run_on_push true, kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- pull existence check: PASS at `/tmp/kaggle-pull/exp080-u-space-target-ablation-train-v1`
- initial normal logs: empty
- 3 minute `logs -f --interval 15` polling: no output before timeout; treated as Kaggle API log lag or still-running state, not failure.
- post-poll output probe: `/tmp/kaggle-output/exp080_u_space_target_ablation/train_v1_probe` contained no files yet.
- 2026-06-19: timeout 後に `kaggle kernels logs kentookumura/exp080-u-space-target-ablation-train` を再取得し、`/tmp/exp080_train_v1_logs.json` に保存。
- 2026-06-19: logs から `{"target", "model", "fold", "rmse_tvt"}` JSON 行を抽出し、以下を保存。
  - `experiments/exp080_u_space_target_ablation/artifacts/exp080_train_v1_log_fold_metrics.csv`
  - `experiments/exp080_u_space_target_ablation/artifacts/exp080_train_v1_log_target_model_summary.csv`
- 2026-06-19: output 取得は一部モデルファイルを `/tmp/kaggle-output/exp080_u_space_target_ablation/train_v1` に取得したが、正式 metrics CSV / summary JSON は timeout のため保存されていない。途中で `User cancelled operation` として停止。

### Log-derived fold metrics

注意: 以下は fold RMSE の単純平均であり、正式 pooled RMSE ではない。ただし target 間の差が十分大きいため、target 優劣判断には使える。

| target | model | completed folds | mean fold RMSE |
| --- | --- | ---: | ---: |
| `dTVT` | `lgb1` | 5 | 9.534549 |
| `dTVT` | `lgb2` | 5 | 9.541831 |
| `dTVT` | `lgb0` | 5 | 9.633329 |
| `dTVT_plus_dZ` | `lgb0` | 5 | 12.058333 |
| `dTVT_plus_dZ` | `lgb2` | 5 | 13.433934 |
| `dTVT_plus_dZ` | `lgb1` | 5 | 13.792792 |
| `TVT_plus_Z_minus_T0` | `lgb0` | 5 | 18.893572 |
| `TVT_plus_Z_minus_T0` | `lgb1` | 5 | 27.568903 |
| `TVT_plus_Z_minus_T0` | `lgb2` | 1 | 31.333343 |
| `TVT_plus_Z_abs` | `lgb1` | 5 | 52.900531 |
| `TVT_plus_Z_abs` | `lgb2` | 5 | 53.171079 |
| `TVT_plus_Z_abs` | `lgb0` | 5 | 63.381782 |

Interpretation:

- `dTVT` が全モデルで最良。exp073 baseline target を維持するのが妥当。
- `dTVT_plus_dZ` は最も良い `lgb0` でも 12.06 で、`dTVT` の 9.53-9.63 から大きく悪化。
- 絶対 U-space 系 (`TVT_plus_Z_abs`, `TVT_plus_Z_minus_T0`) は大幅悪化。well 固有 offset / target scale を吸収できず破綻している可能性が高い。
- `TVT_plus_Z_minus_T0Z0` は timeout 前に未到達。既に近い U-space variants が大きく悪化しているため、追加で回す優先度は低い。

## 検証

- `uv run python -m py_compile experiments/exp080_u_space_target_ablation/u_space_target_ablation.py experiments/exp080_u_space_target_ablation/public_notebook_replay_audit.py experiments/exp080_u_space_target_ablation/settings.py`: PASS
- `uv run python -m json.tool experiments/exp080_u_space_target_ablation/exp080_u_space_target_ablation_train.ipynb`: PASS
- `uv run python -m json.tool experiments/exp080_u_space_target_ablation/exp080_u_space_target_ablation_inference.ipynb`: PASS
- `uv run ruff check experiments/exp080_u_space_target_ablation/u_space_target_ablation.py experiments/exp080_u_space_target_ablation/public_notebook_replay_audit.py experiments/exp080_u_space_target_ablation/settings.py`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp080_u_space_target_ablation`: PASS
- `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp080_u_space_target_ablation --notebook train --kernel-id kentookumura/exp080-u-space-target-ablation-train --title "exp080 u space target ablation train" --run-on-push --strict`: PASS
- generated train package: `experiments/exp080_u_space_target_ablation/kaggle/train`
- generated kernel id: `kentookumura/exp080-u-space-target-ablation-train`
- generated metadata: GPU enabled, internet disabled, run_on_push true, kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
