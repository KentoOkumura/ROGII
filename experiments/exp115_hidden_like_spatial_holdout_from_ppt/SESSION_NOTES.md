# exp115_hidden_like_spatial_holdout_from_ppt セッションノート

## 目的

公式 PPT slide10 の Verification map から赤い Verification well の空間分布を抽出し、train wells から hidden-like な固定 holdout を作る。成果物は後続で `exp092` / `exp073` / `exp098` を再採点するための split/readout 入力であり、提出候補ではない。

## 現在の状態

- Route: ml_model
- 状態: Kaggle train v1 完了、output 検証済み
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
make new-steering EXP=exp115_hidden_like_spatial_holdout_from_ppt
make new-exp EXP=exp115_hidden_like_spatial_holdout_from_ppt
.venv/bin/python -m py_compile experiments/exp115_hidden_like_spatial_holdout_from_ppt/hidden_like_spatial_holdout_from_ppt.py
make validate-exp EXP=exp115_hidden_like_spatial_holdout_from_ppt
.venv/bin/python experiments/exp115_hidden_like_spatial_holdout_from_ppt/hidden_like_spatial_holdout_from_ppt.py --allow-local
.venv/bin/ruff check experiments/exp115_hidden_like_spatial_holdout_from_ppt
.venv/bin/ruff format experiments/exp115_hidden_like_spatial_holdout_from_ppt/hidden_like_spatial_holdout_from_ppt.py
make prepare-kaggle-notebooks EXP=exp115_hidden_like_spatial_holdout_from_ppt EXTRA_ARGS="--notebook train --strict"
.venv/bin/python experiments/exp115_hidden_like_spatial_holdout_from_ppt/hidden_like_spatial_holdout_from_ppt.py --allow-local
make prepare-kaggle-notebooks EXP=exp115_hidden_like_spatial_holdout_from_ppt EXTRA_ARGS="--notebook train --run-on-push --strict --title 'exp115 hidden like spatial holdout from ppt train'"
kaggle kernels push -p experiments/exp115_hidden_like_spatial_holdout_from_ppt/kaggle/train
kaggle kernels status kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train
kaggle kernels output kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train -p /tmp/kaggle-output/exp115_hidden_like_spatial_holdout_from_ppt/train_v1
```

## 変更点

- `.steering/20260623-exp115-hidden-like-spatial-holdout-from-ppt/` を作成し、要件、設計、タスクを記録。
- `config.yaml` を hidden-like fixed holdout audit 用に更新。
- `hidden_like_spatial_holdout_from_ppt.py` を追加。
  - PPTX を zip として読み、slide10 の embedded PNG を標準ライブラリだけで decode。
  - 赤 component を抽出し、plot bbox 内の正規化座標に変換。
  - train wells の centroid / azimuth / eval length / prefix length / GR coverage / exact typewell group を再計算。
  - `verification_like_spatial` と `verification_like_typewell_purged` の 2 split を保存。
- train notebook を audit notebook に更新。
- inference notebook は no-submission と明示。
- Kaggle train package を `--run-on-push --strict` で生成し、kernel v1 を実行。
- Kaggle output を取得し、`experiments/exp115_hidden_like_spatial_holdout_from_ppt/kaggle/output/train_v1/` に保存。

## 再現性メモ

- seed policy: 乱数なし。well_id、PPT red distance、component order の deterministic sort。
- stochastic components: なし。
- CPU/GPU runtime: CPU only。GPU 不要。
- Kaggle kernel id / version: `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train` v1、status `COMPLETE`。
- input / feature schema SHA: PPTX SHA256 `c083c59df01f0fdf1fea860bc977fcdcf278eb12a33282a0200044c8abf950fc`、slide image SHA256 `e8ed99d562ae0a630a1780ac4bdb73cf2c5fd3f40d733723ddaece80f5e17901`。
- feature content SHA: 予測特徴量 cache は作らない。生成物 CSV は local smoke output として `artifacts/`、Kaggle output として `kaggle/output/train_v1/artifacts/` に保存。
- model manifest / model SHA: モデルなし。
- prediction SHA: 予測なし。
- submission SHA: 提出なし。
- rerun check: local smoke 2 回と Kaggle train v1 で red component count 45、spatial valid wells 200、typewell-purged valid wells 200、purged excluded 16 が一致。

## Local smoke 結果

- PPT extraction status: `ok`
- red component count: 45
- train wells: 773
- `verification_like_spatial`: valid wells 200、distinct typewell groups 196、median PPT red distance 0.018609910、max 0.080668542
- `verification_like_typewell_purged`: valid wells 200、distinct typewell groups 200、purged train excluded wells 16

## Kaggle train v1 結果

- Kernel: `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`
- Status: `COMPLETE`
- Output: `experiments/exp115_hidden_like_spatial_holdout_from_ppt/kaggle/output/train_v1/`
- PPT extraction status: `ok`
- PPTX SHA256: `c083c59df01f0fdf1fea860bc977fcdcf278eb12a33282a0200044c8abf950fc`
- Slide image SHA256: `e8ed99d562ae0a630a1780ac4bdb73cf2c5fd3f40d733723ddaece80f5e17901`
- red component count: 45
- red pixel count: 6217
- train wells: 773
- `verification_like_spatial`: valid wells 200、distinct typewell groups 196、median PPT red distance 0.018609910、max 0.080668542
- `verification_like_typewell_purged`: valid wells 200、distinct typewell groups 200、purged train excluded wells 16
- output row check: `holdout_wells.csv` 400 rows + header、`fold_assignments.csv` 773 rows + header、`ppt_red_points.csv` 45 rows + header

## 次のアクション

1. 保存済み Kaggle output の `fold_assignments.csv` に `exp092` / `exp073` / `exp098` の OOF prediction を merge し、hidden-like holdout score readout を作る。
2. `verification_like_spatial` と `verification_like_typewell_purged` の両方で overall RMSE、bucket、worst-well delta を読む。
