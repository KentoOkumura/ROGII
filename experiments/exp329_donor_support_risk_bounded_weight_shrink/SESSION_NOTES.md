# exp329 セッションノート

## 目的

fold-safe donor supportがexp226寄与の有害区間を識別できるかを先に診断し、通過時だけexp263固定式のexp226重みをboundedに縮める。

## 現在の状態

- 2026-07-21: steering、scaffold、config、判定式を確定。
- 2026-07-21: ユーザー承認によりStage 0を実装。
- 2026-07-21: ユーザーの「実行してください」をKaggle CPU Stage 0 push / 実行の明示承認として記録。Stage 1、inference、submissionは対象外。
- 2026-07-21: 53文字の初期slug `exp329-donor-support-risk-bounded-weight-shrink-train` はKaggle SaveKernel 400で作成されず。Kaggle上限に合わせ、canonical slugを `exp329-donor-support-risk-shrink-train` に短縮して再prepare。
- 2026-07-21: canonical Kaggle CPU kernel `kentookumura/exp329-donor-support-risk-shrink-train` version 1をpush。pullしたmetadataでprivate、GPU/TPU/internet無効、exp115/exp226/exp263 kernel sourceを確認。
- 2026-07-21: version 1は28.6秒でERROR。exp263 component identity guardが文字列/object列にもNumPy 2の`array_equal(equal_nan=True)`を適用し、`isnan` TypeError。入力不整合や科学判定前の失敗ではない。pandas Seriesのdtype-aware exact equalityへ修正し、文字列・aligned NaN・mismatchの回帰テストを追加。
- 2026-07-21: 修正後はfocused test 10/10、strict validation、Jupytext round-trip、構文、ruff F821がPASS。同一canonical IDへversion 2をpush。
- 2026-07-21: version 2はKaggle `COMPLETE`。3,783,989 rows / 773 wells / 12,368 segments、Stage 0 runtime 209.829秒。technical hard checksとcoverage checksは全PASSしたが、pooled AUC 0.562091、control差0.005310、top-risk benefit -0.674259 ft、1000+・hidden-like 2面方向FAILで総合`FAIL`。Stage 1 unavailable、rescue gridなしでbranch close。
- `exp329_donor_support_risk_bounded_weight_shrink_compact_selfcontained_train.py`からcompact/正規train Notebookを生成。
- exp226 source foldとexp263 readout foldを別列で保持し、K16/k50 donor ledger、6 support primitive、outer-train ECDF、SHA256 circular control、target-free freeze、late-truth AUC/benefit/tail判定を実装。
- Stage 1のbounded shrinkは未実装のまま閉鎖。Stage 0全gate PASS条件を満たさなかった。
- Route: `pf_beam`。
- CV/LB: 予測candidateを生成しないStage 0 readoutのみ。LB/submissionなし。

## 固定事項

- riskは6個のtarget-free donor-support featureだけ。outer-train percentileを使用する。
- signed neighbor error/bias、K-scale instability、GR likelihood、HMM posteriorを使わない。
- Stage 0はrisk AUCとwithin-well circular control。FAIL時は救済gridなし。
- Stage 1は最大shrink 25%、最大移動5 ft、250 ft未満vetoの1式だけ。
- exp263、exp226、likPF、exact-HMMの保存予測は再生成しない。

## 実行量

- Stage 0: 1 scientific risk、1 control、773 support well-runs、model/booster/decoder 0。
- Stage 1最大: 1 fixed candidate、1 matched control、model/booster/decoder 0。
- 親/control再学習0、inference/submission 0。

実装時確認: active Stage 0 scientific risk 1、diagnostic control 1、support well-runs 773、LightGBM config 0、fold学習0、booster 0、HMM/PF/Beam 0。既存parent/controlの再学習と予測再生成は含まない。

push前再確認（2026-07-21）: 実行variantはStage 0 scientific risk 1 + diagnostic control 1、LightGBM config 0、学習fold 0、合計booster 0、HMM/PF/Beam decoder 0。CPUのみ、親/control再学習0、Stage 1/inference/submission 0。

## 再現性

RNGは使わない。fold/well/segment/donor順を固定し、input、risk feature、CDF、risk score、activation、real/control predictionのschema/content SHAを記録する。circular controlだけwell idのSHA256でoffsetを固定する。

実装ではdonor ledger、raw/smoothed donor field SHA、6 primitive、fold別ECDF reference、segment risk/control risk、saved parent/destination row contractをtruth attach前に保存する。gzipはdecompressed SHAを主証拠にする。

## 実装検証

- `.venv/bin/python -m py_compile ...compact_selfcontained_train.py tests/test_exp329_...py`: PASS。
- `.venv/bin/ruff check ... --select F821`: PASS。
- `.venv/bin/pytest -q tests/test_exp329_donor_support_risk_bounded_weight_shrink.py`: 10 passed。
- `.venv/bin/pytest -q tests/test_kaggle_notebooks.py tests/test_scaffold.py`: 11 passed。
- `.venv/bin/pytest -q`: 465 passed / 1 skipped / 2 failed。失敗2件はいずれも既存exp296の完了後config（`completed_train_side_guard_failed_closed`、`run_variant: false`）と、旧testが期待する実行前状態（`kaggle_cpu_*`、`run_variant: true`）の不一致で、exp329差分からは独立。exp329専用9件と共通Notebook/scaffold 11件は全PASS。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...compact_selfcontained_train.py`: PASS。
- `make validate-exp EXP=exp329_donor_support_risk_bounded_weight_shrink`: strict PASS（`task` CLIは環境に無いためMakefile経由）。
- 保存済みinput read-only preflightはexp263 manifest SHAとexp226 decompressed SHAを確認。ローカル取得物にはexp263 full partitionのfold 1--4が無く、3,783,989行のfull loadは未完了。Kaggle kernel source上のfull cacheで確認する。
- 親compact比較: exp302 trainは9章/3,252行、exp329 trainは10章/1,961行。exp329はcandidate-bank再生成を持たないStage 0診断のため短いが、runtime、入力、exp226 geometry、donor support、risk/control、freeze、late truth、decision、生成物、executionの全役割をNotebook内に展開しており、同一exp helper importだけの薄い構成ではない。

## Kaggle Stage 0結果

- canonical kernel: `kentookumura/exp329-donor-support-risk-shrink-train` / id_no `128104811` / successful version 2。
- 発火: 762,529 rows（20.151459%）、433 wells、5 folds。coverage checks全PASS。
- AUC: real 0.562090518、control 0.556780893、差0.005309625、AUC>0.5は5/5 folds。
- benefit: top risk -0.674259008 ft、bottom risk -1.343711194 ft、top-bottom +0.669452186 ft。
- scientific FAIL: pooled AUC、control separation、top-risk benefit、1000+、hidden-like spatial、hidden-like typewell-purged。
- target-free contract SHA256: `03049211fdf9c394ff7c34426e0cbb0ab424da3ae440ab92136c106b805f3000`。
- stage0 decision raw SHA256: `9c25c68c12527002fd1171dd5dea39448f8eb1928d495a88b48513c65cf0f8a2`。

## 次

threshold、alpha、clip、destination、featureの救済gridを行わずexp329を閉じる。Stage 0 PASSを必須依存とするexp330も閉じ、新しい同系救済backlogは追加しない。
