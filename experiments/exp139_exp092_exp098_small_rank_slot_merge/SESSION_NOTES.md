# exp139_exp092_exp098_small_rank_slot_merge セッションノート

## 現在の状態

- status: `completed_train_side_rejected_no_submit`
- route: `ml_model`
- parent: `exp092_u_projection_correction_disagreement_fullrun`
- rank-slot source parent: `exp098_selector_rank_slot_features_on_exp073`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- CV: best `lgb1` pooled RMSE 9.324907641
- LB: 未提出
- blocked: none

## 実装内容

- `.steering/20260627-exp139-exp092-exp098-small-rank-slot-merge/` を作成。
- `experiments/exp139_exp092_exp098_small_rank_slot_merge/` を exp092 から作成。
- 補助実装を `exp092_exp098_small_rank_slot_merge.py` に変更。
- exp098 の target-free rank-slot feature generator を移植。
- exp092 の U-projection correction / disagreement features と、small rank-slot add-only columns を同じ train/inference flow で生成するように更新。
- 初期 small columns は config の `model.feature_ablation.active_variants[0].extra_columns` に限定。
- train notebook は rank-slot config を渡し、projection summary と rank-slot summary を表示する。
- inference notebook は train-side review 後に使う前提で、同じ rank-slot config を渡す。

## 実行コマンド

```bash
make new-steering EXP=exp139_exp092_exp098_small_rank_slot_merge
make new-exp EXP=exp139_exp092_exp098_small_rank_slot_merge SOURCE=experiments/exp092_u_projection_correction_disagreement_fullrun
```

## 次のアクション

1. exp139 は rejected として閉じる。
2. inference port / submit は行わない。
3. 今後 rank-slot 系を続ける場合は、full union / pruning ではなく normalized shape または candidate quality diagnostic として別 backlog に分ける。

## 検証

- `uv run python -m py_compile experiments/exp139_exp092_exp098_small_rank_slot_merge/exp092_exp098_small_rank_slot_merge.py experiments/exp139_exp092_exp098_small_rank_slot_merge/public_notebook_replay_audit.py experiments/exp139_exp092_exp098_small_rank_slot_merge/settings.py`: PASS
- `python3 -m json.tool experiments/exp139_exp092_exp098_small_rank_slot_merge/exp139_exp092_exp098_small_rank_slot_merge_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp139_exp092_exp098_small_rank_slot_merge/exp139_exp092_exp098_small_rank_slot_merge_inference.ipynb`: PASS
- `make validate-exp EXP=exp139_exp092_exp098_small_rank_slot_merge`: PASS
- `uv run ruff check experiments/exp139_exp092_exp098_small_rank_slot_merge/exp092_exp098_small_rank_slot_merge.py experiments/exp139_exp092_exp098_small_rank_slot_merge/public_notebook_replay_audit.py experiments/exp139_exp092_exp098_small_rank_slot_merge/settings.py`: PASS
- `uv run ruff format --check experiments/exp139_exp092_exp098_small_rank_slot_merge/exp092_exp098_small_rank_slot_merge.py experiments/exp139_exp092_exp098_small_rank_slot_merge/public_notebook_replay_audit.py experiments/exp139_exp092_exp098_small_rank_slot_merge/settings.py`: PASS
- synthetic frame による `build_u_projection_features()` + `build_selector_rank_slot_features()` + `feature_columns_for_variant()` smoke test: PASS。projection features 69、rank features 70、selected smoke features 61、rank summary rows 15。
- `make prepare-kaggle-notebooks EXP=exp139_exp092_exp098_small_rank_slot_merge EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp139-small-rank-slot-train --title 'exp139 small rank slot train' --run-on-push --strict"`: PASS
- generated train package: `experiments/exp139_exp092_exp098_small_rank_slot_merge/kaggle/train`
- generated metadata: kernel id `kentookumura/exp139-small-rank-slot-train`, title `exp139 small rank slot train`, GPU enabled, internet disabled, run_on_push true, competition source `rogii-wellbore-geology-prediction`, kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- generated bootstrap manifest includes:
  - `config.yaml` SHA `d85e870db425a02e781382c3a9b0d3eaa6996c035fe5a36694d69a8f1fe753dd`
  - `exp092_exp098_small_rank_slot_merge.py` SHA `cf49fdf6a64414629d8794c3514d88f7296e389e94a4e4332a985e5051209e55`
  - `public_notebook_replay_audit.py` SHA `c46da772d09595cb3ff6d1c7f04233f0522fd672a2d83f70808b8f7e0e117a60`
  - `settings.py` SHA `03acab031ab8f8ce4b1f91563951eb69f2e1f06d36128aa1bd104994a0ec1bcb`

## Kaggle train v1

```bash
make push-kaggle-train EXP=exp139_exp092_exp098_small_rank_slot_merge
kaggle kernels pull kentookumura/exp139-small-rank-slot-train -p /tmp/kaggle-pull/exp139-small-rank-slot-train-v1 -m
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp139-small-rank-slot-train
```

- kernel id: `kentookumura/exp139-small-rank-slot-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp139-small-rank-slot-train`
- version: 1
- push: `Kernel version 1 successfully pushed`
- pull existence check: PASS at `/tmp/kaggle-pull/exp139-small-rank-slot-train-v1`
- short log monitor: no CLI log output before user requested to stop monitoring.
- monitoring status: stopped locally by user request; Kaggle run completion will be checked after user reports completion.

## Kaggle train v1 completion

```bash
kaggle kernels logs kentookumura/exp139-small-rank-slot-train
kaggle kernels output kentookumura/exp139-small-rank-slot-train -p experiments/exp139_exp092_exp098_small_rank_slot_merge/kaggle/output/train_v1
```

- completed: yes
- runtime: 15,866.843 sec
- rows / wells: 3,783,989 / 773
- features: 255
- model count: 15
- output dir: `experiments/exp139_exp092_exp098_small_rank_slot_merge/kaggle/output/train_v1`
- output download note: full output download was interrupted while downloading model text files. Essential log summary, bucket metrics, by-well metrics, feature importance mean, feature schema, and model manifest were retrieved.
- saved full log JSON: `kaggle/output/train_v1/exp139-small-rank-slot-train.log.json`
- log SHA: `377c43609000ee4d57c75ed798c3d6d3714aff4908b2050d890298244b65f504`

### Pooled OOF metrics

| model | pooled RMSE | delta vs exp092 same model | prediction SHA |
| --- | ---: | ---: | --- |
| `lgb0` | 9.613226293 | +0.080099855 | `6027d47b76a13a89a964c4af6311dfd8733ec23fa73f2ed1fc4f76b4c0b89563` |
| `lgb1` | 9.324907641 | +0.002427745 | `b309bc3c24b003a95d7b9afc31c16ce7977151bebf998994fbac2b5d3a6861b3` |
| `lgb2` | 9.337578311 | -0.000614093 | `19291ff4438c2835cea030a74c015981a2c4a5f1d3b396f5d47b5e8a01278646` |
| `lgb_mean` | 9.370584225 | +0.027520159 | `09ced87e6de69c101d0a31b4e06012c730c7303ea7b1311b099feaea719a0e28` |

### Readout

- Best model is `lgb1`, RMSE 9.324907641.
- exp092 best `lgb1` 9.322479896 から +0.002427745 悪化。
- exp098 `lgb1` 9.358151052 よりは -0.033243412 改善。
- `lgb2` は exp092 `lgb2` から -0.000614093 微小改善したが、exp092 best より悪い。
- `lgb1` distance bucket: `000_050` RMSE 1.410827、`1000_plus` RMSE 10.234380。
- `lgb1` worst wells: `86454a6f` 57.459114、`1b1eba53` 42.140781、`fb03ae90` 41.229336。
- Rank-slot shape features are important: `rank1_u_curvature`, `rank2_u_curvature`, `rank2_u_slope`, `rank1_u_slope` are top-10 mean importance features.

### Decision

Rejected. Inference port / submit は行わない。exp098 rank-slot signal は exp073 には効いたが、exp092 上では U-projection correction / disagreement と重複または微小ノイズになったと判断する。
