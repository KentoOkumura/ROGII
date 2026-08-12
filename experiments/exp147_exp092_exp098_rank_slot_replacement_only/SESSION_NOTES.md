# exp147_exp092_exp098_rank_slot_replacement_only セッションノート

## 現在の状態

- status: `completed_train_side_rejected_no_submit`
- route: `ml_model`
- parent: `exp092_u_projection_correction_disagreement_fullrun`
- rank-slot source parent: `exp098_selector_rank_slot_features_on_exp073`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- CV: best `lgb2` RMSE 9.397013393、`lgb1` 9.423893838、`lgb_mean` 9.438575715
- LB: 未提出
- blocked: none

## 実装内容

- `docs/legacy/steering/20260627-exp147-exp092-exp098-rank-slot-replacement-only/` を作成。
- `experiments/exp147_exp092_exp098_rank_slot_replacement_only/` を exp139 から作成。
- 補助実装を `exp092_exp098_rank_slot_replacement_only.py` に変更。
- `feature_columns_for_variant()` に `drop_columns` 対応を追加。
- config に 22 dropped generated columns と 25 replacement rank-slot columns を明記。
- base 196 features は残し、exp092 generated columns の overlap 部分だけを落とす。
- train / inference notebook は exp147 の helper と variant 名を参照するように更新。

## 実行計画

- active variant: `u_projection_rank_slot_replacement_only` の 1 本
- LightGBM config 数: 3
- folds: 5
- 合計 booster 数: 15
- 親実験 / control 再学習: なし
- 比較基準: exp092 / exp098 / exp139 の保存済み metrics

## 実行コマンド

```bash
make new-steering EXP=exp147_exp092_exp098_rank_slot_replacement_only
make new-exp EXP=exp147_exp092_exp098_rank_slot_replacement_only SOURCE=experiments/exp139_exp092_exp098_small_rank_slot_merge
```

## 次のアクション

1. inference port / submit は行わない。
2. rank-slot replacement / pruning 系は追わない。
3. 残す場合は backlog `exp098_full_rank_slot_addonly_on_exp092` を実験化した `exp153_full_rank_slot_addonly_on_exp092` を低優先の対照実験として扱う。

## 検証

- `uv run python -m py_compile experiments/exp147_exp092_exp098_rank_slot_replacement_only/exp092_exp098_rank_slot_replacement_only.py experiments/exp147_exp092_exp098_rank_slot_replacement_only/public_notebook_replay_audit.py experiments/exp147_exp092_exp098_rank_slot_replacement_only/settings.py`: PASS
- `python3 -m json.tool experiments/exp147_exp092_exp098_rank_slot_replacement_only/exp147_exp092_exp098_rank_slot_replacement_only_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp147_exp092_exp098_rank_slot_replacement_only/exp147_exp092_exp098_rank_slot_replacement_only_inference.ipynb`: PASS
- `make validate-exp EXP=exp147_exp092_exp098_rank_slot_replacement_only`: PASS
- `uv run ruff check experiments/exp147_exp092_exp098_rank_slot_replacement_only/exp092_exp098_rank_slot_replacement_only.py experiments/exp147_exp092_exp098_rank_slot_replacement_only/public_notebook_replay_audit.py experiments/exp147_exp092_exp098_rank_slot_replacement_only/settings.py`: PASS
- `uv run ruff format --check experiments/exp147_exp092_exp098_rank_slot_replacement_only/exp092_exp098_rank_slot_replacement_only.py experiments/exp147_exp092_exp098_rank_slot_replacement_only/public_notebook_replay_audit.py experiments/exp147_exp092_exp098_rank_slot_replacement_only/settings.py`: PASS
- synthetic frame による `build_u_projection_features()` + `build_selector_rank_slot_features()` + `feature_columns_for_variant()` smoke test: PASS。projection features 69、rank features 70、synthetic selected features 53、dropped columns 22、replacement columns 25。
- `make prepare-kaggle-notebooks EXP=exp147_exp092_exp098_rank_slot_replacement_only EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp147-rank-slot-replacement-train --title 'exp147 rank slot replacement train' --run-on-push --strict"`: PASS
- generated train package: `experiments/exp147_exp092_exp098_rank_slot_replacement_only/kaggle/train`
- generated metadata: kernel id `kentookumura/exp147-rank-slot-replacement-train`, title `exp147 rank slot replacement train`, GPU enabled, internet disabled, run_on_push true, competition source `rogii-wellbore-geology-prediction`, kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- generated bootstrap manifest includes:
  - `config.yaml` SHA `3fc2f779d42b3616ae6cccb2c75446309d66b06cdda818867e909ca1fea9b2c1`
  - `exp092_exp098_rank_slot_replacement_only.py` SHA `c56602ae04d0e53afc4d4769a5cd180ca09124014d2d6f4283160e2960a198bf`
  - `public_notebook_replay_audit.py` SHA `c46da772d09595cb3ff6d1c7f04233f0522fd672a2d83f70808b8f7e0e117a60`
  - `settings.py` SHA `7b6bd332ac0e1e6e348f82520e68ad3478587200bde6acdbaeac3c1aadf42f83`

## Kaggle train v1

```bash
make push-kaggle-train EXP=exp147_exp092_exp098_rank_slot_replacement_only
kaggle kernels pull kentookumura/exp147-rank-slot-replacement-train -p /tmp/kaggle-pull/exp147-rank-slot-replacement-train-v1 -m
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp147-rank-slot-replacement-train
kaggle kernels logs kentookumura/exp147-rank-slot-replacement-train
kaggle kernels output kentookumura/exp147-rank-slot-replacement-train -p experiments/exp147_exp092_exp098_rank_slot_replacement_only/kaggle/output/train_v1
kaggle kernels status kentookumura/exp147-rank-slot-replacement-train
```

- kernel id: `kentookumura/exp147-rank-slot-replacement-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp147-rank-slot-replacement-train`
- version: 1
- push: `Kernel version 1 successfully pushed`
- pull existence check: PASS at `/tmp/kaggle-pull/exp147-rank-slot-replacement-train-v1`
- short log monitor: 300 sec timeout with empty CLI log output.
- normal logs after timeout: empty CLI log output.
- output download after timeout: no files downloaded yet.
- status check after user completion notice: `KernelWorkerStatus.COMPLETE`
- output download: metrics / by-well / bucket / feature summaries / feature importance / schema / summary / lgb model manifest / 15 model txt files downloaded.
- local prediction `.csv.gz`: not downloaded; prediction SHA は Kaggle metrics CSV と `metrics.json` に記録。
- current decision: train-side rejected; do not port inference or submit.

## Kaggle train v1 完了結果

- runtime: 15,532.769 sec
- rows / wells: 3,783,989 / 773
- feature count: 243
- model count: 15
- local model files downloaded: 15 / 15
- best model: `lgb2`
- best RMSE: 9.397013392978039
- `lgb0` RMSE: 9.66561251619198
- `lgb1` RMSE: 9.42389383771385
- `lgb2` RMSE: 9.397013392978039
- `lgb_mean` RMSE: 9.438575715119802
- delta best vs exp092 `lgb1`: +0.07453349747411156
- delta `lgb1` vs exp092 `lgb1`: +0.10141394220992302
- delta `lgb2` vs exp092 `lgb2`: +0.05882098833285965
- delta `lgb_mean` vs exp092 `lgb_mean`: +0.09551164912472914
- delta best vs exp139 `lgb1`: +0.0721057523184359
- decision: `completed_train_side_rejected_no_submit`

Distance bucket:

| model | `000_050` RMSE | `1000_plus` RMSE |
| --- | ---: | ---: |
| `lgb1` | 1.3012446165084839 | 10.34688949584961 |
| `lgb2` | 1.3518543243408203 | 10.317358016967773 |
| `lgb_mean` | 1.1437119245529175 | 10.365809440612793 |

Prediction SHA:

- `lgb0`: `85a67e32a4fc5eb2c13d0200bf1f58eec68affb8659e07436f8456a378184909`
- `lgb1`: `a7d910034f128de685b7bcabfb71e4d3ba85c67abaa5ed511acd1566a17059a2`
- `lgb2`: `80ded477646c24f9fd1899c3fee6e61c55675ef5ac07996c9d3a3f98e815d005`
- `lgb_mean`: `cce131d6d77f4e976d4c277a1c85d4fde8de61af235fb8a1e5fb2bbda0935674`

Artifact SHA:

- log: `c72df9e2db19e3297c4b36670911531024c345c82a04c8f7cc9420cbf80053ca`
- metrics: `679c02f7ed5ccf3bc7a8449740bab5be537c20e8d40b6e89cea5b54f0794e23b`
- by-well: `76a27cea28edccc5bc02c417db982fe88b2d09bf41e3a1cb9c11b1e6fad09cb4`
- bucket metrics: `fc9b16481c06e7ba3e07f308780cd2aa7e949fefa796503fbf04fa0b55f2fa30`
- feature importance mean: `e8e9cacc22975d1d990f91eae2476b9700b70547e6336e652d7a2c6c4ea8b850`
- feature schema: `678eccdd08c1dd3235018da27a0f75d7ef66c2258b88551f8bb944e45ebfbd53`
- summary: `cd14152fcb87a039d15d46bd497ae075323f6fc195159e6310cdf801e40b91d1`
- model manifest: `2302a59ec1d93725f1bbc2161b3087de3ca893b9e700fd6d78b8eb74b5042579`

解釈:

- rank-slot U-shape features は feature importance 上位に残る。
- ただし exp092 generated overlap columns を落とすと global OOF が明確に悪化する。
- exp139 add-only は微小悪化、exp147 replacement-only は大きく悪化したため、rank-slot 系は replacement / pruning では追わない。
