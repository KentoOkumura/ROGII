# exp086_oof_feature_importance_error_readout セッションノート

## 現在の状態

- status: `readout_completed`
- route: `ml_model`
- requested name: `exp073_oof_feature_importance_error_readout`
- actual experiment name: `exp086_oof_feature_importance_error_readout`
- parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- readout parent: `exp077_full_replay_postprocess_guard`
- Kaggle train: v1 `COMPLETE`
- Kaggle inference: not applicable
- submission: not applicable

## 実装内容

- `.steering/20260620-exp086-oof-feature-importance-error-readout/` を作成。
- `experiments/exp086_oof_feature_importance_error_readout/` を exp077 から作成。
- ユーザー指定の `exp073_...` は新規実験 ID としては使わず、最新 `exp085` の次として `exp086_oof_feature_importance_error_readout` にした。
- `oof_feature_importance_error_readout.py` を追加。
  - exp077 policy predictions から `baseline_exp073_lgb_mean` と `longtail_likpf_tiny_gate_w006` だけを chunk read する。
  - exp077 feature importance mean から上位特徴量を選ぶ。
  - exp072 full replay feature cache から `id` / `well` / 上位特徴量 / 診断用列だけを読む。
  - OOF rows を id join し、policy metrics、feature quantile metrics、feature summary、well summary、error lift plot、error correlation plot、summary JSON を保存する。
- train notebook を exp086 診断用に更新。
- inference notebook は diagnostic only とし、`submission.csv` を生成しない構成にした。
- `config.yaml` を exp086 用に更新し、ローカル `/tmp/kaggle-output` と Kaggle source の両方の入力候補を登録した。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp086_oof_feature_importance_error_readout
uv run python scripts/new_experiment.py --name exp086_oof_feature_importance_error_readout --source experiments/exp077_full_replay_postprocess_guard
```

## 次のアクション

## 検証

静的検証:

```bash
uv run python -m py_compile settings.py oof_feature_importance_error_readout.py
uv run python -m json.tool experiments/exp086_oof_feature_importance_error_readout/exp086_oof_feature_importance_error_readout_train.ipynb
uv run python -m json.tool experiments/exp086_oof_feature_importance_error_readout/exp086_oof_feature_importance_error_readout_inference.ipynb
uv run ruff check experiments/exp086_oof_feature_importance_error_readout/oof_feature_importance_error_readout.py experiments/exp086_oof_feature_importance_error_readout/settings.py
uv run python scripts/validate_experiment.py --experiment exp086_oof_feature_importance_error_readout
```

結果:

- py_compile: pass
- train notebook JSON: pass
- inference notebook JSON: pass
- ruff: pass
- validate_experiment strict: pass

Kaggle train package:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp086_oof_feature_importance_error_readout --notebook train --kernel-id kentookumura/exp086-oof-feature-importance-error-readout-train --title "exp086 oof feature importance error readout train" --run-on-push --strict
uv run python -m py_compile experiments/exp086_oof_feature_importance_error_readout/kaggle/train/settings.py experiments/exp086_oof_feature_importance_error_readout/kaggle/train/oof_feature_importance_error_readout.py
uv run python -m json.tool experiments/exp086_oof_feature_importance_error_readout/kaggle/train/exp086_oof_feature_importance_error_readout_train.ipynb
```

結果:

- prepare train package: pass
- package py_compile: pass
- package notebook JSON: pass

local smoke:

- `/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train_v1/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz` が 0 byte だったため、local full readout smoke は未実行。
- resolver は 0 byte ファイルを入力候補から除外するように修正済み。
- Kaggle 実行では `kentookumura/exp072-exp063-full-replay-feature-cache-train` source の feature cache を読む前提。

## 次のアクション

## Kaggle train v1

- kernel id: `kentookumura/exp086-oof-feature-importance-error-readout-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp086-oof-feature-importance-error-readout-train`
- metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`
- kernel sources:
  - `kentookumura/exp077-full-replay-postprocess-guard-train`
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - `kentookumura/exp073-full-replay-repro-guard-train`
- push: `Kernel version 1 successfully pushed`
- pull existence check: PASS at `/tmp/kaggle-pull/exp086-oof-feature-importance-error-readout-train-v1`
- initial normal logs: empty
- `timeout 180 kaggle kernels logs -f --interval 15 ...`: no output before timeout
- status during run: `KernelWorkerStatus.RUNNING`
- later `timeout 300 kaggle kernels logs -f --interval 20 ...`: completed logs returned
- final status: `KernelWorkerStatus.COMPLETE`
- output path: `/tmp/kaggle-output/exp086_oof_feature_importance_error_readout/train_v1`
- elapsed seconds in summary: `355.393`

結果:

- rows / wells: 3,783,989 / 773
- features read: 34
- missing top feature rows: 0
- input policy predictions SHA: `754e389abe8a4492bcd31426617465530c7010a9f90cfc4734c7ed13f1cc5468`
- input policy metrics SHA: `3c504885271ab91d203887350174bff1a268dbc4a168edf77f85f2a047988c0a`
- feature importance mean SHA: `3ed39f7d91c1869e138425eb77e922b4dbb422dc438cb2c25e535593d47c5e5e`
- feature cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`

Policy metrics:

- `baseline_exp073_lgb_mean`: RMSE `9.52637482601992`, MAE `6.159766047813564`
- `longtail_likpf_tiny_gate_w006`: RMSE `9.47051480056479`, MAE `6.110920032404956`
- delta vs baseline: `-0.05586002545513047`

主要所見:

- baseline error lift が大きい特徴量 bucket は `pf_vs_dense`、`tvt_densew_d`、`tvt_dense50_d`、`tvt_dense_d`、`pf_vs_z`、`dense_dist`、`dz`、`beam_std_d`、`slp_b_d_50`、`likpf_mean_d`。
- selected important features の absolute-error correlation は `beam_std_d`、`dense_dist`、`dense_nb_std`、`eval_len`、`tvt_densew_d` が上位。
- `pf_vs_dense` worst bucket は MAE lift `+2.632059`、baseline RMSE `12.783611`。
- `tvt_densew_d` worst bucket は MAE lift `+2.533809`、baseline RMSE `12.851624`。
- `dense_dist` は Spearman abs error correlation `0.119421`。
- `beam_std_d` は Spearman abs error correlation `0.175250`。

生成物 SHA:

- summary: `fdbd0ea8db5d4f7a5ce17298fce1665055af933f5e162a7a4ef12a894acefeac`
- policy metrics: `06c21b0876749d19d7feb417fa8d3fba62b7645bea584bfd94f80b0add671744`
- feature summary: `158554cb16d1061a97fb8620efe91e3bddf97d421b84349af2e3eb5907691bd3`
- feature quantile metrics: `c0d4bf4de2834b7cb10d7b0269fcb3816a68d4323b717b2e53c5bc0dddb05bc4`
- well summary: `40da07f10f3ab8de9fd56de3a62671c7157347f08440e74d4ca55e9f468e8beb`
- error lift plot: `c907783224849bcb69c33a15c8e4e892a1425227513bb4e7d79065722f7b40ec`
- error correlation plot: `eba6285829b0ffdb9f4087e3c49740dfaaf132981a76798fef85988a7596b397`

## 次のアクション

1. `pf_beam_disagreement_sample_weight` では `beam_std_d`、`dense_dist`、`dense_nb_std`、`eval_len`、`pf_vs_dense` を優先候補にする。
2. direct replacement ではなく confidence feature / sample weight / guard 条件に限定する。
3. worst-well 悪化があるため、次実験では well-level guard と distance bucket を必ず見る。
