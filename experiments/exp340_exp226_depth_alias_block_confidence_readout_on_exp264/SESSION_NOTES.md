# exp340 セッションノート

## 2026-07-22 設計確定

- 目的: robust emissionで識別力を弱める前に、depth aliasの検知可能性だけを測る。
- 実行規模: scientific readout 7 family、circular block-order control 1、booster 0、HMM run 0。
- leakage guard: Q1/Q4と特徴SHAをtruth join前に固定する。
- long-tail guard: pooled差、4/5 fold、stress bucket、AUCを併用する。
- 禁止: 閾値探索、候補修正、selector再学習、errorを使ったfeature選択。

## 2026-07-23 Stage 0実装

- ユーザー依頼を0-booster Stage 0の実装承認として扱った。Kaggle package/push/run、
  add-only feature化、推論、提出の承認は含めない。
- compact self-contained train/inference sourceと正規Notebookを実装した。同一exp helper
  import、`__file__`、model、HMM、候補/予測変更はない。
- 実行予定量は1 Stage 0 / 7 fixed family / circular control 1 / LightGBM config 0 /
  trained fold 0 / booster 0 / HMM well-run 0 / 保存済みcontrol再学習0。
- exp280 score gzip decompressed SHA `c6e9e39a...d99c3`、score content SHA
  `4a546cfe...aa46`、scientific contract SHA `60d32ba9...7978`を固定した。
- exp264 Stage D v3 OOF SHA `b11c5005...9ae2`、exp226 OOF decompressed SHA
  `709eb726...c609`、exp115 hidden-like assignment SHA `5f9ac9fa...6597`を固定した。
- exp280/exp226診断foldとexp264 Stage D outer foldは別契約であることを保存実schemaから
  確認した。fold guardはexp280/exp226 foldを使い、exp264 outer foldはprovenanceのみ。
- exp264保存truthはfloat32なのでexp226 truthとのidentity toleranceを`1e-3 ft`に固定し、
  評価値はexp264保存`actual_tvt`を正とする。

### 固定feature・評価

- 7 familyはmargin、unit-temperature softmax entropy、likelihood-weighted shift
  population std、normalized zero rank、absolute top1 shift、prior-block top1 jump、
  直近3 block nonzero sign pairwise disagreement share。
- marginだけ符号反転し、全familyを「高いほどalias risk」へ揃える。
- fold内target-free scoreだけでQ25/Q75をfreezeし、feature/schema/quantile/content SHAを
  保存後にだけexp264/exp226 truth/errorを読む。
- `abs_error>=10 ft` AUCは、block riskを各rowへ反復したものと厳密に等価なtie-aware
  row-weighted AUC。sequence 2 familyのcontrolはwell内top1 shift列のSHA256 circular rotation。
- 1000+はblock最小`md_since>=1000 ft`、hidden-like 2面はexp115 roleで固定。

### 検証

```bash
.venv/bin/pytest -q experiments/exp340_exp226_depth_alias_block_confidence_readout_on_exp264/tests/test_exp340_exp226_depth_alias_block_confidence_readout_on_exp264.py
# 10 passed

.venv/bin/python -m py_compile \
  experiments/exp340_exp226_depth_alias_block_confidence_readout_on_exp264/*compact_selfcontained*.py

.venv/bin/ruff check \
  experiments/exp340_exp226_depth_alias_block_confidence_readout_on_exp264/*compact_selfcontained*.py \
  experiments/exp340_exp226_depth_alias_block_confidence_readout_on_exp264/tests/test_exp340_exp226_depth_alias_block_confidence_readout_on_exp264.py --select F821,E9
# All checks passed

JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp340_exp226_depth_alias_block_confidence_readout_on_exp264/*compact_selfcontained*.py

make validate-exp EXP=exp340_exp226_depth_alias_block_confidence_readout_on_exp264
# strict validation passed
```

- 最初のread-only全行ID preflightはexp226/exp264を同時にpandasへ全量展開したため、
  ローカルRAM上限でexit 137になった。親ファイル変更・生成物書き出しはない。実装を
  250,000-row chunk / Parquet batchごとのblock集約へ変更し、全量ID indexを廃止した。
- 修正後read-only parent preflightは101,231 score rows / 7,787 blocks / 773 wells /
  5 folds / 7 family / 35 fold-family quantile rows / finite 100%を確認した。late source側も
  exp226/exp264各7,787 blocks、3,783,989 rows、block coverage 7,787/7,787、row count・
  first/last row identity一致、block truth mean最大差`0.000800781 ft`、exp264保存RMSE
  `8.460811237612477`完全一致を確認した。family科学scoreやgateは計算していない。
- 構成比較はscore source exp280 trainが1,165行・9章、exp340 compact trainが
  1,592行・11章。exp340は親score再生成をせず、代わりにsource SHA loader、7 family、
  freeze、late OOF join、AUC/scopes/gate、生成物保存をNotebook上で追える。

## 2026-07-23 Stage 0 Kaggle実行承認

- 2026-07-23 21:09:03 JST、ユーザーがStage 0の実行を明示承認した。
- push前の実行量を再確認した。実行対象は1 Stage 0 / 7 fixed family /
  circular block-order control 1 / LightGBM config 0 / trained fold 0 /
  booster 0 / HMM well-run 0 / 親・control再学習0。
- Kaggle runtimeはCPU、internet off、run-on-pushを使用する。正規kernel IDは
  `kentookumura/exp340-exp226-alias-readout-on-exp264-train`、
  titleは`exp340 exp226 alias readout on exp264 train`。
- 最初にexperiment名をそのままslug化した66文字のIDを指定するとKaggle SaveKernelが
  HTTP 400で保存前に拒否した。kernelが未作成であることを確認し、50文字上限内で
  exp340・exp226・alias readout・exp264を残す43文字の正規名へ短縮した。
- 短縮後の正規IDへkernel version 1をpushし、run-on-pushで実行を開始した。
  URL: https://www.kaggle.com/code/kentookumura/exp340-exp226-alias-readout-on-exp264-train
- 推論、提出、readout通過後の補正実験は承認範囲外のため無効のままとする。

## 2026-07-23 Stage 0完了

- Kaggle private CPU version 1、id_no `128356047`を`26.400168 sec`で完了した。
  3,783,989 rows / 7,787 blocks / 773 wells、7 family finite coverage 100%、
  親SHA、block identity、truth tolerance、Q1/Q4非重複を満たしtechnical gateはPASS。
- scientific gateは7/7 family FAIL。pooled Q4-Q1 mean block RMSE差はzero rank
  `+0.905341 ft`、absolute top1 shift `+1.359545 ft`、prior-block jump
  `+2.253795 ft`、3-block sign inconsistency `+1.513854 ft`で、いずれも5/5 folds正方向。
- 一方、全familyが必須bad10 AUC `>=0.60`を満たさなかった。最良のprior-block jumpも
  AUC `0.574392`で、realがcircularを上回るfoldも3/5に留まった。3-block sign
  inconsistencyはcircular 5/5だがAUC `0.548155`。他familyのAUCは`0.486936--0.544737`。
- decisionは`close_depth_alias_confidence_branch_without_rescue`。予測・候補・selector
  変更、family/threshold探索、補正、推論、提出、再実行は行わない。
- 成果物12件を`kaggle/output/train_v1/artifacts/`へ取得した。feature content SHAは
  `70748900...e437`、quantile content SHAは`e207e559...ee98`、post-freeze readout
  decompressed SHAは`a31f4091...f9bc`。
- 期待成果物12/12、SHA manifest 11/11、family gate 7/7をローカルで再検証し、
  専用pytest 10件、Ruff、strict experiment validation、template validationも再PASS。
- 失敗原因は数値・coverageではなく、target-free shift形状とbad10 failureの識別力が
  事前閾値に届かないこと。sequence familyもcircular controlに対するfold安定性が弱い。
  同familyの救済backlogは追加せず、独立既存候補exp342、exp343をP3のまま残す。

## 未実施

予測補正、selector学習、HMM、推論、提出は行っていない。scientific FAILのため、
readout通過後を前提にした後続実験も作成しない。
