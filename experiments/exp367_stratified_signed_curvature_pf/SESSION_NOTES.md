# exp367_stratified_signed_curvature_pf セッションノート

## 目的

signed-curvatureを層別維持するlikelihood-PFの設計を確定する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_stage0_failed_close_without_rescue`
- CV / LB: なし
- Stage 0: Kaggle private CPU version 1で完了。
- technical gate: PASS
- scientific gate: FAIL
- Stage 1 PF / inference / submission: 不適格・未実装・未実行。
- 分岐: no-rescue契約どおり完了・閉鎖。

## コマンドログ

### 2026-07-23 実行済み

```bash
make new-steering EXP=exp367_stratified_signed_curvature_pf
make new-exp EXP=exp367_stratified_signed_curvature_pf
```

### 2026-07-25 実行済み

```bash
.venv/bin/python -m py_compile \
  experiments/exp367_stratified_signed_curvature_pf/exp367_stratified_signed_curvature_pf_compact_selfcontained_train.py \
  experiments/exp367_stratified_signed_curvature_pf/tests/test_exp367_stratified_signed_curvature_pf.py
.venv/bin/ruff check \
  experiments/exp367_stratified_signed_curvature_pf/exp367_stratified_signed_curvature_pf_compact_selfcontained_train.py \
  experiments/exp367_stratified_signed_curvature_pf/tests/test_exp367_stratified_signed_curvature_pf.py
.venv/bin/pytest -q experiments/exp367_stratified_signed_curvature_pf/tests/test_exp367_stratified_signed_curvature_pf.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp367_stratified_signed_curvature_pf/exp367_stratified_signed_curvature_pf_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp367_stratified_signed_curvature_pf/exp367_stratified_signed_curvature_pf_compact_selfcontained_train.py
.venv/bin/python scripts/validate_experiment.py \
  --experiment exp367_stratified_signed_curvature_pf
```

- `py_compile`: PASS
- `ruff`: PASS
- 専用test: `9 passed`
- Jupytext変換 / round-trip: PASS
- `validate-exp`: PASS
- Kaggle / local notebook実行: なし
- 全体test: `967 passed, 6 skipped, 5 failed`。失敗はexp296 / exp348 / exp358 /
  exp391の既存config状態と既存test期待値の不一致で、exp367由来ではない。

### 2026-07-25 Stage 0実行承認

- ユーザー指示: `実行してください。`
- 実行先: Kaggle private CPU
- scientific variant: 0（固定3 pathの診断のみ）
- fixed signed paths: 3 (`-1/0/+1`)
- reporting folds: 5
- PF seed-well runs: 0
- LightGBM config / trained fold / booster: 0 / 0 / 0
- 親exp072 PF control replay: 0
- Stage 1 treatment: 0（未実装・未承認）
- inference / submission: なし

### 2026-07-25 Kaggle package preflight

```bash
make prepare-kaggle-notebooks \
  EXP=exp367_stratified_signed_curvature_pf \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp367-stratified-signed-curvature-pf-train \
  --title 'exp367 stratified signed curvature pf train' \
  --run-on-push --strict"
```

- kernel id: `kentookumura/exp367-stratified-signed-curvature-pf-train`
- title: `exp367 stratified signed curvature pf train`
- private / CPU / internet off / run-on-push
- competition source: `rogii-wellbore-geology-prediction`
- kernel source: `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`
- package 19 cells = bootstrap 1 + canonical 18、canonical source完全一致
- package Notebook SHA256:
  `92fc49c40f78ecb1953ff911ef75a8cdc40d1fd5e514108f640c353ccacff0b8`
- metadata SHA256:
  `0aebfc9f62dad15446c5b4a41e4271e99d147371c1d6a2b5c659e7caf46bd678`
- config SHA256:
  `cdbaa8299ea6204a17391b1e70e429009b44c3e13023c26564ad61a9c922fc8d`
- `run_stage_0=true`、`kaggle_push_approved=true`をpackage前に再確認した。

### 2026-07-25 Kaggle Stage 0 version 1

```bash
kaggle kernels push \
  -p experiments/exp367_stratified_signed_curvature_pf/kaggle/train
kaggle kernels pull \
  kentookumura/exp367-stratified-signed-curvature-pf-train \
  -p /tmp/exp367-kaggle-pull-v1 -m
```

- push: success
- Kaggle version: 1
- id_no: `128528103`
- pulled id / title: canonical一致
- private / CPU (`machine_shape=None`) / internet off
- 監視開始UTC: `2026-07-25 00:38:25 UTC`

### 2026-07-25 Kaggle Stage 0完了

- status: `COMPLETE`
- runtime: `267.914282461 sec`
- input / scored wells: `773 / 772`
- complete 512-row blocks: `13,631`
- rows with block overlap: `6,979,072`
- technical gate: 全PASS
- overall top1: `0.469591`（`>=0.40` PASS）
- zero-first top1 / gain: `0.008510 / +0.461081`
- MRR: `0.687550`
- zero-first MRR / gain: `0.410779 / +0.276771`（`>=0.01` PASS）
- circular top1: `0.464016`
- real-minus-circular top1: `+0.005576`（`>=0.03` FAIL）
- selected / zero path RMSE: `90.506527 / 91.336129 ft`
- selected path RMSE gain: `+0.829601 ft`
- 1000+ RMSE gain: `+0.911161 ft`（PASS）
- hidden-like spatial / typewell-purged RMSE gain:
  `+0.306996 / +0.447147 ft`（PASS / PASS）
- fold gate:
  - fold 0: FAIL
  - fold 1: FAIL
  - fold 2: FAIL
  - fold 3: PASS
  - fold 4: PASS
  - passing folds: `2/5`（`>=4/5` FAIL）
- decision: `STAGE0_FAIL_CLOSE_WITHOUT_RESCUE`
- Stage 1 eligible: false
- Kaggle log records: 91
- Kaggle log SHA256:
  `a3272196ebc84a409a88cb04d38f6a3a448a0a421c86be32158c16f875cfcf53`
- AGENTS.md方針に従い、gate、fold、variant、生成物SHAをlogsから確認できたため
  Kaggle output archiveはダウンロードしていない。

## 変更点

- curvature drift/transition、初期層、resampling最低quotaを固定した。
- Stage 0の3軌道GR識別gateを固定した。
- Stage 0をcompact self-contained Jupytext source 1,630行 / 18 notebook cellsで実装した。
- 512行完全block、stride 256、1-block circular GR control、同点順`0/-1/+1`を固定した。
- candidate path / GR scoreをgzip decompressed SHAでfreezeし、horizontal truthと
  hidden-like roleをSHA readback後だけlate joinする。
- scope方向はselected pathのpooled block RMSE gain vs zero path、fold方向はtop1 /
  MRR / real-vs-circularの3条件がすべて正、と固定した。
- 親exp072には`*compact_selfcontained*_train.py`が存在しないため直接の章・行数比較は
  不可。exp367正規train notebook自体にImports、runtime/config/SHA、scientific contract、
  truth-free input/path、freeze、late truth、metrics/gates、orchestrationを展開した。
- 既存inference placeholderのsample submissionコピーを削除し、Stage 1未実装中は生成物を
  作らず停止するfail-closed self-contained inference notebookへ置換した。
- Stage 1は1 variant、500 particles、128 seeds、773 wells、98,944 seed-well runs。
- LightGBM config / trained fold / booster / parent PF replayはすべて0。

## 再現性メモ

- seed: `SHA256(experiment|well|family|seed_index)`。
- Stage 0は固定3軌道でRNGなし。
- 将来のStage 1 stochastic: initialization、c transition、propagation、
  stratified systematic resampling、jitter。
- global RNGとthread schedule依存は禁止。将来もCPU single worker。
- train/testはrawから別生成し、row predictionはdecompressed content SHAを記録する。
- kernel package SHAはpackage preflightに記録済み。
- candidate path / score / manifest / gate / summary content SHAは`config.yaml`と
  `metrics.json`に記録済み。
- prediction / submission SHA: Stage 1不適格のため未生成。

## 次のアクション

exp367は完了・閉鎖。Stage 1 PF、inference、submission、同一Stage 0結果に対する
quota / curvature / transition / gateのparameter rescueは行わない。
