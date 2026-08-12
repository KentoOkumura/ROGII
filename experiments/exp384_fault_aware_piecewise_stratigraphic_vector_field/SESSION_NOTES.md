# exp384_fault_aware_piecewise_stratigraphic_vector_field セッションノート

## 目的

exp383のsmooth vector fieldに、fault-aware piecewise domainを追加した価値を測る。

## 現在の状態

- Route: `pf_beam`
- 状態: `closed_by_exp383_stage0_resource_fail`
- 実装: 完了
- CV・LB: なし
- Notebook: compact self-contained trainを正規trainへ採用、inferenceはfail-closed
- Kaggle package/push/run: exp383 Stage 0 resource FAILにより未開始のまま停止
- inference/submission: なし

## コマンドログ

2026-07-24:

```bash
make new-steering EXP=exp384_fault_aware_piecewise_stratigraphic_vector_field
make new-exp EXP=exp384_fault_aware_piecewise_stratigraphic_vector_field
.venv/bin/python -m py_compile experiments/exp384_fault_aware_piecewise_stratigraphic_vector_field/exp384_fault_aware_piecewise_stratigraphic_vector_field_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp384_fault_aware_piecewise_stratigraphic_vector_field/exp384_fault_aware_piecewise_stratigraphic_vector_field_compact_selfcontained_train.py experiments/exp384_fault_aware_piecewise_stratigraphic_vector_field/tests/test_exp384_fault_aware_piecewise_stratigraphic_vector_field.py
.venv/bin/pytest -q experiments/exp384_fault_aware_piecewise_stratigraphic_vector_field/tests/test_exp384_fault_aware_piecewise_stratigraphic_vector_field.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp384_fault_aware_piecewise_stratigraphic_vector_field/exp384_fault_aware_piecewise_stratigraphic_vector_field_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp384_fault_aware_piecewise_stratigraphic_vector_field/exp384_fault_aware_piecewise_stratigraphic_vector_field_compact_selfcontained_train.py
make validate-exp EXP=exp384_fault_aware_piecewise_stratigraphic_vector_field
make test
kaggle kernels pull kentookumura/exp383-all-tvt-stratigraphic-vector-drift-field-train -p /tmp/kaggle-pull/exp383-all-tvt-stratigraphic-vector-drift-field-train -m
kaggle kernels list --mine --search exp383 --sort-by dateRun --format json
kaggle kernels list --mine --search exp384 --sort-by dateRun --format json
```

## 変更点

- exp383を固定し、fault graph、component field、soft domain posteriorだけを追加する設計を確定。
- graph閾値、component条件、base floor、Stage 0/1 gateを事前固定。
- 2026-07-24のユーザー指示をコード実装と正規Notebook採用の承認として記録。
- 256 ft node graph、12 unique-well neighbor、formation AND structural cut、
  stable connected component、small/no-fault component fallbackを実装。
- exp383と同じ6-surface relative absolute/vector fit、最大8 component、
  signature/XY/surface uncertainty/prefix likelihood posteriorを実装。
- base floor 0.25、mixture uncertainty、prefix bias、exp226 shrink、
  hard-prefix banded WLS、no-component exact exp383 fallbackを実装。
- target-free logical SHA freeze後だけtruthをjoinするStage 0/1 orchestrationを実装。
- 2026-07-24のユーザー指示「実行してください」を、exp384のKaggle package /
  push / CPU実行承認として記録。

## 予定実行量

- 1 candidate / 5 reporting folds
- fitted model / HMM / PF / Beam / booster: `0 / 0 / 0 / 0 / 0`
- exp383 control再実行: 0
- parent control再実行: 0
- Kaggle CPU package / push / 実行: 過去に条件付き承認済みだったが、
  exp383 Stage 0 resource FAILにより現在は無効
- 実行開始: 0（exp383必須生成物がないため、package前に停止）

## 2026-07-24 実行前提監査

- OAuth credentialとlegacy environment credentialの存在を確認。
- ローカルにexp383 manifest、donor nodes、query fields、OOF成果物なし。
- exp383は`design_frozen_not_implemented`で、Stage 0/1は未実行。
- exact exp383 train kernel pullは`403 Forbidden`。
- Kaggleの自分のkernelを`exp383`、`exp384`で検索し、どちらも`Not found`。
- `expected_manifest_logical_sha256`は`null`のまま。
- この状態ではexp384はparent manifest gateで必ずfail-closedになるため、
  無効なpackage / push / runは開始しなかった。

## 2026-07-25 親実験確定

- exp383 version 1はtruth join前にcode errorで停止した。
- code fix後も5-fold donor-surface stage投影`30.52 h`が固定gate`8.5 h`をFAIL。
- exp383は`stage0_resource_fail_closed`、PASS artifact/manifest SHAなし。
- exp384の実行authorizationを無効化し、未実行のまま閉じた。
- ユーザーがexp383と後続実験の閉鎖を明示確認したため、再開候補としても扱わない。

## 再現性メモ

- RNGなし、stable node/edge/component/query順
- exp383 input SHAをhard pin
- graph/component/posterior/path/prediction logical SHAを実行時に記録
- deterministic anchorはrerun一致まで主張しない
- `expected_manifest_logical_sha256`は意図的に`null`で、exp383 PASS前はfail-closed

## 検証

- 専用contract test: `14 passed`
- scaffold/notebook test: `11 passed`
- Ruff: PASS
- py_compile: train/inference PASS
- Jupytext train/inference round-trip: PASS
- `make validate-exp ...`: strict PASS
- 全repository test: exp384を含む`853 passed / 6 skipped`。
  unrelatedなexp296既存config期待不一致2件
  (`status` prefix、`execution.run_variant`)だけFAILし、exp384 testは全PASS。
- 親比較: exp383はtemplate scaffoldのみでcompact sourceが存在しない。
  exp384 trainは9章・2,221行で、helper importだけの薄いNotebookではない。

## 次のアクション

1. exp383がStage 0 resource FAILで閉じたため、exp384も未実行で閉じる。
2. PASS時だけexp384 configへparent manifest SHAをhard pinする。
3. 承認済みの1 candidate / 5 folds / model・HMM・PF・Beam・booster各0 /
   parent control再実行0で、16-well resource auditから実行する。
