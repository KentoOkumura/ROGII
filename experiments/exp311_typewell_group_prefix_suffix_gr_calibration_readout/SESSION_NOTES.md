# exp311 セッションノート

## 現在の状態

- 2026-07-21: steering、scaffold、configの設計を確定。
- 2026-07-21: ユーザーの実装指示を受け、compact self-contained train/inferenceとテストを実装。Kaggle実行なし。
- 2026-07-21: ユーザーの実行指示を、compact trainのcanonical採用とKaggle private CPU runの承認として受領。
- Route: `pf_beam`
- 実行契約: 1 diagnostic variant / 5 folds / 0 model / 0 booster / 0 decoder。
- train-side readout: Kaggle private CPU version 1 completed / gate failed。推論・submission: disabled and not run。

## Kaggle実行契約

- active diagnostic variant: 1。
- model / LightGBM config / booster / decoder: `0 / 0 / 0 / 0`。
- folds: 5。親control・baseline再実行: なし。
- runtime: Kaggle CPU、internet disabled、single process。
- canonical kernel: `kentookumura/exp311-typewell-gr-calibration-readout-train`。
- title: `exp311 typewell gr calibration readout train`。
- inference / submission: 実行しない。

## Kaggle実行ログ

- 2026-07-21 initial push:
  - command: `make push-kaggle-train EXP=exp311_typewell_group_prefix_suffix_gr_calibration_readout`
  - kernel: `kentookumura/exp311-typewell-group-prefix-suffix-gr-calibration-readout-train`
  - result: `SaveKernel` 400、詳細messageなし。Kaggle側の存在とslug長/正規化を確認してから再pushする。
  - 同一IDの`kernels pull -m`は403で、作成済みkernelを確認できなかった。
  - id/titleのslugは一致していたが64文字でKaggleの一般的なslug上限50文字を超えるため、意味を保った44文字の`exp311-typewell-gr-calibration-readout-train`へ短縮する。別実験は作らない。
- 2026-07-21 version 1 push:
  - kernel: `kentookumura/exp311-typewell-gr-calibration-readout-train`
  - result: push成功、private CPU run開始。
  - URL: `https://www.kaggle.com/code/kentookumura/exp311-typewell-gr-calibration-readout-train`
  - id_no: `128085784`。status: `COMPLETE`。
  - runtime: summary計測`246.630865 sec`。private CPU / internet off。
  - contract: 1 diagnostic / 5 folds / 0 model / 0 booster / 0 decoder。parent/control再実行なし。
  - promotion: FAIL。8 checks中6 PASS、`group_loo_fit_rmse_r2`と`worst_well_delta`がFAIL。
  - primary `native_overlap_1` / same-group held-out-well: coverage 760/773、identity→transfer gain `0.376220` GR API、5/5 folds改善、real-minus-shuffled `0.240055`。
  - noise R² `0.202320`、fit-RMSE R² `-0.003255`、worst-well delta `+12.914716` GR API。
  - leakage guard: outer-valid truth rows before freeze `0`、fold freeze SHA 5件をsummaryへ記録。
  - 9 manifestのraw SHAとpair table展開後SHAをローカル再計算し全件一致。pair decompressed SHA `14f506da...e335`、summary raw SHA `821499b8...a29`。
  - outputはSHA確認に必要なため`/tmp/exp311-kaggle-output.AcBbRe`へ取得したが、大容量artifactはrepositoryへ保存しない。

## 設計契約

- outer-train TVT truthのみでnative-overlap群のaffine/noise priorを作る。
- outer-valid truthはpair/table/featureを凍結した後だけscoreへ結合する。
- well等重み、identity shrinkage、group shuffle、GR circular shiftを固定する。
- input/typewell/fold/pair/group/readoutのschema/content SHAを記録する。
- Huber IRLSはdelta 1.345 / 最大50反復、identity shrinkageは`alpha=n/(n+200)`、group aggregateはwell等重みmedian。
- suffix scoreはTVT候補やdecoderを作らないためhorizontal GR API unit。旧`*_ft` gate名を`*_gr_*`へ修正し数値閾値は維持。

## 仮説と変更点

同群peerのbias/noise/reliability統計がheld-out suffixへ転送できるという仮説を、設計-only scaffoldから実行可能な0-booster readoutへ変更した。親exp211の直接補正やPF/Beamは再実行せず、GR再構成readoutだけを追加した。

## 実装

- `exp311_typewell_group_prefix_suffix_gr_calibration_readout_compact_selfcontained_train.py/.ipynb`
  - raw horizontalは最初に`GR`/`TVT_input`だけを読み、outer-valid `TVT`を露出しない。
  - outer-train truthだけでreal/circular-shift statsを作り、group-shuffleを含む全priorをSHA freezeする。
  - freeze SHAを必須引数としてouter-valid suffix truthをlate joinする。
  - `native_overlap_1` primary、exact hash sensitivity、same-group / group-LOO / exp115 spatial+typewell-purgedを評価する。
  - 9 CSV/CSV.GZ + summary JSON、metrics.jsonを出力する。
- `*_compact_selfcontained_inference.py/.ipynb`
  - inference/submission configの全flagをassertしてfail-closedにする。
- compact trainを正規`*_train.ipynb`へ採用した。正規inferenceはplaceholderのままで、compact inferenceもfail-closed。

## 検証

- `.venv/bin/pytest -q experiments/exp311_typewell_group_prefix_suffix_gr_calibration_readout/tests/test_exp311_typewell_group_prefix_suffix_gr_calibration_readout.py`: 7 passed。
- Jupytext変換と`--test`: train/inferenceともPASS。
- `py_compile`、ruff/F821、`make validate-exp`、`make validate-template`: PASS（最終検証時点）。
- `task` executableは環境になかったため、同等のMakefile targetへ切り替えた。
- 親exp211にcompact self-contained版は存在しない。通常train source 206行に対しexp311 compact trainは1,388行で、readout helperと上位orchestrationをnotebookセルに展開した。
- `__file__`参照なし。

## 次

固定gate FAILのため同一OOFで閾値・shrinkage・group定義を救済調整せずbranchを閉じる。exp311全gate PASSを先行条件としたexp312〜320は停止する。独立した新しい事前根拠がない限りType Well群transfer branchは再開せず、既存P0のexp321 Stage A/Bとexp304 PASS後のexp305を優先する。
