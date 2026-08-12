# exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction セッションノート

## 目的

KAGGLE_DIRECTION backlog `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction` を実装する。Connor Tynan の公開 notebook `ROGII K16 spline + kernel kNN + adaptive kappa` から外部 weight 依存の v7/v8 を除き、deterministic v6 fallback を source-port して exp206 の線形 `dTVT ~= a*dZ+b` 失敗と切り分ける。

## 現在の状態

- Route: `pf_beam`
- 状態: 実装済み / Kaggle train 未実行
- CV: 未計測
- LB: 未提出
- GPU cost: なし
- Active variants: 1 (`v6_k16_geometry_gr_u_projection`)
- LightGBM config count: 0
- Fold count: 5 group-safe folds
- Booster count: 0
- Parent/control retraining: なし
- External weights: v7 neural committee / v8 GBM meta-layer とも無効

## 実装内容

- `docs/legacy/steering/20260709-exp226-connortynan-k16-spline-kernel-knn-adaptive-kappa-reproduction/` を作成。
- `experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/` を template から作成。
- `connortynan_k16_reproduction.py`
  - K=16 segment spline coefficient fit。
  - raw / smoothed donor field generation。
  - XY local-linear kernel kNN。
  - donor-distance regime 別 adaptive kappa fit。
  - near-strike gate と ANCC local theta substitute column。
  - typewell GR correction と U-projection。
  - train group-safe CV と full-train inference / submission writer。
- train notebook source:
  - `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train.py`
- inference notebook source:
  - `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_inference.py`

## リークガード

- validation target well は donor field から除外。
- validation target well は kappa fit から除外。
- validation target well の ANCC surface sample も field / local theta fit から除外。
- unknown suffix true TVT は metrics と artifact summary のみに使用。
- target known prefix `TVT_input`、GR、geometry、typewell は test と同等に利用可能な入力として扱う。
- v7/v8 external weights は探索、ロード、代替学習しない。
- blind LB weight search はしない。

## push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- active rule variants: 1
- LightGBM configs: 0
- folds: 5
- total boosters: 0
- parent/control retraining: なし
- expected train loop: group-safe kappa refit 5 回 + validation prediction all train wells

## コマンドログ

```bash
make new-steering EXP=exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction
make new-exp EXP=exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction
```

- result: steering / experiment scaffold 作成。

## 次の確認

```bash
.venv/bin/python -m py_compile \
  experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/connortynan_k16_reproduction.py \
  experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train.py \
  experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_inference.py \
  experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/settings.py
.venv/bin/ruff check experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 \
  experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 \
  experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_inference.py
.venv/bin/python scripts/validate_experiment.py --experiment exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction
```

- result: PASS
- `py_compile`: PASS
- `ruff check --select F821,F401`: PASS
- `jupytext --to ipynb --set-kernel python3`: train / inference とも PASS
- `jupytext --to ipynb --test`: train / inference とも PASS
- `validate_experiment.py`: PASS

## ローカル関数 smoke

Notebook 実行ではなく、helper の最小関数経路だけ確認した。

```bash
.venv/bin/python -c "import sys; from pathlib import Path; sys.path.insert(0, 'experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction'); from settings import ExperimentPaths, load_config; from connortynan_k16_reproduction import params_from_config, load_train_wells, build_fields, fit_kappa, predict_well, rmse; cfg=load_config(); p=params_from_config(cfg); paths=ExperimentPaths(); wells=load_train_wells(paths.train_data_dir, p, max_wells=8); src=wells[:-1]; val=wells[-1]; fields=build_fields(src, p); k=fit_kappa(src, fields, p); pred=predict_well(val, fields, k, p); print('smoke_wells', len(wells), 'kappa_dim', len(k), 'pred_len', len(pred.pred), 'rmse', round(rmse(val.tvt[val.s+1:], pred.pred), 6))"
```

- result: `smoke_wells 8 kappa_dim 12 pred_len 5108 rmse 8.419873`
- note: これは Kaggle notebook 実行の代替評価ではなく、移植コードの関数 smoke のみ。

## Kaggle train v1

初回 push は kernel slug が長すぎる可能性がある `kentookumura/exp226-connortynan-k16-spline-kernel-knn-adaptive-kappa-reproduction-train` で Kaggle API 400 になった。Kaggle slug 制約に合わせ、意味を残した短縮 id `kentookumura/exp226-k16-kappa-repro-train` に変更して再 push した。

```bash
make prepare-kaggle-notebooks EXP=exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp226-k16-kappa-repro-train --title 'exp226 k16 kappa repro train' --run-on-push --strict"
make push-kaggle-train EXP=exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction
kaggle kernels status kentookumura/exp226-k16-kappa-repro-train
kaggle kernels logs kentookumura/exp226-k16-kappa-repro-train
```

- kernel: `kentookumura/exp226-k16-kappa-repro-train`
- version: 1
- status: COMPLETE
- runtime log time: 約 346 sec
- train wells: 773
- OOF rows: 3,783,989
- CV folds: 5
- RMSE: 9.427109596582213
- MAE: 6.148527797393756
- bias: -0.29961900506691624
- within10: 0.8077095361535142
- within25: 0.9767446469849674
- OOF decompressed SHA256: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- leakage guard: target well excluded from donor fields and kappa fit
- errors: none

## Kaggle inference v1

```bash
make prepare-kaggle-notebooks EXP=exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction \
  EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp226-k16-kappa-repro-infer --title 'exp226 k16 kappa repro inference' --run-on-push --strict"
make push-kaggle-infer EXP=exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction
kaggle kernels status kentookumura/exp226-k16-kappa-repro-inference
kaggle kernels logs kentookumura/exp226-k16-kappa-repro-inference
kaggle kernels output kentookumura/exp226-k16-kappa-repro-inference \
  -p /tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/inference_v1
make submit-check EXP=exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction \
  SUBMISSION=/tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/inference_v1/submission.csv
```

- kernel: `kentookumura/exp226-k16-kappa-repro-inference`
- version: 1
- status: COMPLETE
- note: metadata id は `...-infer` で push したが、title-derived slug により Kaggle URL / status slug は `...-inference` になった。
- train wells: 773
- test wells: 3
- submission rows: 14,151
- submission SHA256: `b71e15f7dc7e66f7be70db4a81d9ec72e1001ff2ba13907c3aba24938e906047`
- TVT min/max: 11590.507200740965 / 12237.326047949082
- TVT mean/std: 11905.948938813102 / 277.98497549360616
- submit-check: PASS
- local output: `/tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/inference_v1`
- code submit: 未実施

## Code submission

ユーザーから scoring 完了の連絡を受け、Kaggle submission history を確認した。

```bash
kaggle competitions submissions rogii-wellbore-geology-prediction | head -10
make record-submission EXP=exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction \
  SUBMISSION=/tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/inference_v1/submission.csv \
  EXTRA_ARGS="--cv 9.427109596582213 --public-lb 9.837 --notes 'ref=54491603; kernel=kentookumura/exp226-k16-kappa-repro-inference v1; selected=v6_k16_geometry_gr_u_projection; submit-check PASS; worse than exp218 ML anchor 7.843 and ensemble anchor 7.601, not adopted'"
```

- submission ref: `54491603`
- status: COMPLETE
- Public LB: 9.837
- Private LB: -
- `SUBMISSIONS.md`: v060
- judgment: exp206 よりは大幅改善したが、exp218 ML anchor 7.843 / exp148 CPU runtime 7.921 / exp082 ensemble 7.601 に届かないため不採用。

## 2026-07-27 offset root-cause audit

ユーザー依頼により、保存済みgroup-safe OOFを対象に、exp226で見える低周波
vertical offsetの根本原因をread-onlyで監査した。新規学習、exp226再生成、
Kaggle push、inference、submissionは0。

```bash
.venv/bin/python studies/exp226_offset_root_cause_audit.py
```

- 入力: `3,783,989 rows / 773 wells / 5 folds`
- OOF decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- global bias `-0.299619 ft`を除いてもRMSEは
  `9.427110 -> 9.422347`だけで、説明MSEは`0.1010%`。
- K16 segment mean offsetをoracle診断上だけ除くとRMSEは
  `1.130603`、説明MSEは`98.5617%`。
- K16 segment mean errorと前segment end errorのPearsonは`0.982951`。
  境界error jump中央値は`0.008190 ft`で、segment境界の不連続は原因ではない。
- suffix 0--50 / 2000+ RMSEは`1.741257 / 11.151214`。
  最後の既知TVT anchorの一律誤りではなく、suffix内の増分誤差が距離とともに
  累積する。
- persistent offsetは645 episodes / 449 wells。全行の`18.9943%`だが
  SSEの`82.0073%`を占める。onset一行jump中央値は`0.021148 ft`。
- geometry / pre-U / final RMSEは
  `10.077950 / 9.500816 / 9.427110`。GRとU projectionはpooled・5/5 foldsで
  改善するため単独の根本原因ではないが、一部episodeのthreshold crossingを
  起こすproximal triggerにはなる。
- donor distance maxのbottom / top quartileでwell RMSE中央値は
  `4.099483 / 7.774613`、episode well率は`43.52% / 72.68%`。
- 公開deterministic v6 coreとportの
  `segment_geometry / fit_coeffs / local_linear / kernel_mean /
  build_columns / affine_cal / project_u / gr_correction`
  を固定synthetic入力で比較し、9 checksすべて最大絶対差`0.0`。

結論は、`最後の既知TVTを一度だけanchor + donor由来の相対増分を累積 +
unknown suffix内にabsolute re-anchorなし`という機構である。target local
structureとdonor fieldの小さなsigned rate mismatchが積分され、後続K16
segmentへほぼ定数offsetとして継承される。遠いdonor、長いsuffix / TVT range、
遠距離kappa binの弱支持が増幅条件。GR/Uは不十分な補正・一部triggerであり、
global bias、row order、特定fold、K16境界jump、K=16単独、v6 port bugは否定した。

詳細:
`docs/surveys/exp226_offset_root_cause_audit_20260727.md`

生成物:
`studies/exp226_offset_root_cause_audit_20260727/`
