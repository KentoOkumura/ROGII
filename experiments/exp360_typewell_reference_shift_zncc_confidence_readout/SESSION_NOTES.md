# exp360 セッションノート

## 2026-07-23 設計確定

### 目的

- 全wellで通常matching `δ=0` と固定shifted Type Well reference viewsを併置する。
- raw-finite ZNCCのzero-vs-nonzero score surfaceが、exp264のbad blockをexp280 raw
  Gaussian scoreより安定して予告するかを0-booster OOF readoutで判定する。
- exp340はclosedのままとし、既存familyの閾値・blend・selectorを再探索しない。

### 承認範囲

- 追加のユーザー依頼により、Stage 0コード、正規Notebook採用、契約テストまで実装した。
- Kaggle package/push/run、output取得、推論、提出は未承認・未実施。
- 正規train/inference Notebookはcompact self-contained sourceから再生成した。

### 単一変更と固定条件

- 単一変更: exp280 absolute-residual raw Gaussian scoreをraw-finite ZNCCへ置換。
- 固定: exp226 path/fold、512-row block、13 shifts、exp264 prediction/error readout、
  pooled/fold/1000+/hidden-like scopes。
- shift定義: horizontal row/MDを動かさず `GR_typewell(TVT_geop + δ)` を評価。
- 全773 OOF wellsを対象にし、元から相関が悪いwellだけへ絞らない。
- 4 sentinel wellsはnon-gating descriptive outputだけ。

### 実行予算

- real ZNCC score variant 1。
- stable SHA256 shift-label permutation control 1。
- 保存済みexp280 raw Gaussian baseline 1。
- core feature family 6、primary 1。
- LightGBM config 0、trained fold 0、booster 0。
- PF/Beam/HMM 0、親control再学習0。
- CPU-only、GPU/internet off。

### 再現性メモ

- seed policy: real pathはRNGなし。controlだけstable SHA256 per well/block。
- parallel policy: well/block/shiftのstable sortを固定し、global RNGや完了順に依存しない。
- leakage barrier: input→score/mask→control/features→fold quantile→SHA manifestをfreezeし、
  `truth_access_count=0`確認後だけtruth/errorをlate joinする。
- input SHA: exp340/exp280/exp264/exp226/exp115の既知SHAを`config.yaml`に固定。
- future feature SHA: score、valid mask、control、feature schema/content、quantile、
  post-freeze readoutをKaggle実行時に記録する実装を入れた。
- model/prediction/submission SHA: 生成しないため非該当。
- deterministic anchor: いいえ。既存prediction anchorを変更しないdiagnostic。
- kernel id/version: 未作成、未実行。

### preregistered decision

- primaryは `best_nonzero_minus_zero_zncc` だけ。
- coverage、RMSE quartile lift、4/5 fold、bad10 AUC、1000+/hidden-like、
  raw Gaussian増分、permutation増分のAND gate。
- primaryが1条件でもFAILならsupporting familyやsentinelで救済せずbranch close。
- PASSしてもexp360内でpredictionを変更せず、別add-only ML feature実験を提案するだけ。

## 2026-07-23 Stage 0 実装

- safe horizontal loaderは`MD`, `GR`, `TVT_input`だけをmaterializeし、exp226 safe
  OOFも`well_id`, `row_idx`, `suffix_offset`, `fold`, `tvt_geop`だけを読む。
- Type Well GRはTVT昇順、missing GRのforward/backward fill、重複TVTの平均後に、
  endpoint hold付き線形補間で`GR_typewell(TVT_geop + δ)`を作る。
- horizontal GRは補間せずraw finite pairだけを使い、pair数32、両std `>1e-6`の
  block/shiftだけをvalidとする。invalid scoreは`-1.0`。
- exact tieは`0,-2,+2,-5,+5,-10,+10,-20,+20,-40,+40,-80,+80`の順に固定した。
- real ZNCC、保存済みexp280 raw Gaussian、valid score内stable SHA256 shift-label
  permutationの3面で同じ6 familyを作り、real core-supported blockに揃えて比較する。
- score、valid mask、3面feature、fold quantile、schema、input/contract、content SHAを
  freezeして`truth_access_count=0`を確認した後だけexp264/exp226 truthを読む。
- primaryだけを対象にtechnical/scientific AND gateを機械判定し、FAIL時は
  `close_zncc_confidence_branch_without_rescue`、PASS時も別add-only実験の提案だけとする。

### 実行予定量

- real ZNCC variant 1、stable permutation control 1、保存済みraw baseline 1。
- core family 6、reporting fold 5。
- LightGBM config 0、trained fold 0、booster 0、PF/Beam/HMM 0。
- 親/control再学習0。Kaggle CPU実行は未承認。

### Notebook構成

- 親compact train exp340は1,592行・11章、exp360 compact trainは2,318行・11章。
  exp360は親のlate OOF join/AUC/scope骨格に加え、raw safe loader、ZNCC生成、
  valid mask、matched/permutation control、primary gateをNotebook上へ展開した。
- 同一exp helper importと`__file__`はtrain/inference sourceに含めない。

## 2026-07-23 Kaggle Stage 0 実行承認

- ユーザーの「実行してください」を、exp360のprivate Kaggle CPU Stage 0
  package/push/run承認として2026-07-23 23:16:42 JSTに記録した。
- 実行対象はreal ZNCC variant 1、stable permutation control 1、保存済みraw
  baseline 1、core family 6、reporting fold 5。
- LightGBM/model config 0、trained fold 0、booster 0、PF/Beam/HMM 0。
- 親実験・controlの再学習は0。GPU/internetは無効で、inference/submissionも
  引き続き無効・未承認。
- canonical kernelは
  `kentookumura/exp360-typewell-shift-zncc-readout-train`
  (`exp360 typewell shift ZNCC readout train`) とする。

### Kaggle version 1

- `id_no=128366385`、private CPU、GPU/TPU/internet off、5 kernel sourcesで
  canonical kernel version 1をpushした。
- bootstrap 18 filesとexecution contract表示は成功したが、約31秒で
  `FileNotFoundError: raw train root not found`となった。
- competition dataは
  `/kaggle/input/competitions/rogii-wellbore-geology-prediction/train`
  にmountされる一方、config候補が旧direct pathまでだったことが原因。
- score生成、truth join、gate判定には未到達で、科学条件やgridは変更していない。
- 既存の成功実験と同じcanonical competition pathを候補へ1件追加し、同じkernel
  IDのversion 2で再実行する。

### Kaggle version 2

- 同じcanonical kernelをversion 2としてpushし、status `COMPLETE`を確認した。
- id_no `128366385`、private CPU、GPU/TPU/internet off、runtime
  `125.393474 sec`。
- real ZNCC 1、stable permutation 1、保存済みraw Gaussian 1、core family 6、
  reporting fold 5、model/config/trained fold/booster/PF/Beam/HMM各0、
  親control再学習0のまま完走した。
- 3,783,989 rows / 773 wells / expected 7,787 blocksを処理し、core-supportedは
  7,700 blocks、coverage `0.9888275331`。`896d15b9`だけsupported blockがなく、
  supported wellsは772/773だった。
- freeze前truth accessは0、expected blocks/folds/wells、coverage、Q1/Q4非重複はPASS。
  全well supportだけがtechnical FAIL。

#### primary `best_nonzero_minus_zero_zncc`

- pooled Q4−Q1 mean / median block RMSE:
  `+0.107478737 / +0.085354479 ft`。meanは`+0.50 ft` gateをFAIL。
- fold mean lift:
  `+0.076389 / -0.272526 / +0.163209 / +0.329903 / +0.247506 ft`。
  正方向4/5 foldsはPASS。
- pooled row-weighted bad10 AUC `0.5051641873`で`0.60` gateをFAIL。
  AUC `>0.50`は4/5 folds。
- 1000+ / hidden-like spatial / hidden-like typewell-purgedのQ4−Q1 meanは
  `-0.169027081 / +0.276701116 / +0.214993422 ft`。1000+がFAIL。
- exp280 raw Gaussian pooled AUCは`0.5499488829`。ZNCC gainは
  `-0.044784696`、fold勝利1/5で両gate FAIL。
- stable permutation pooled AUCは`0.4885198628`。ZNCC gainは
  `+0.016644325 < +0.02`でFAIL、fold勝利4/5はPASS。
- technical / scientificともFAILし、decisionは
  `close_zncc_confidence_branch_without_rescue`。

#### 生成物と再現性

- score / valid mask / feature / quantile content SHA:
  `1c16ae1c269936cec724cc4faf164581a3177584f0ff3ffa7a9db7f73634bab7` /
  `bf975b3b3a2e1f73bbc0aa04fb7e48251385b8e349e372d7563d1680ed313377` /
  `642b93dc46d6efd8977a00c643487e6a3083699459a4258b4eb559af645106c2` /
  `8733750c2a66cda6c8d52f059c8f32188ef85292b687842aecaa7ad3c88190df`。
- post-freeze readout decompressed SHA:
  `7d41349afdb4f954b7a73f672364c2b0b7416ae996a34210dbccb152396a5f58`。
- version 2 outputを
  `kaggle/output/train_v2/`へ取得し、SHA manifest 13件をローカル再計算した。
  mismatchは0件で、gate JSONとsummary内gateも一致した。
- model / prediction / submission SHAは生成物がないため非該当。

## コマンドログ

```bash
make new-steering EXP=exp360_typewell_reference_shift_zncc_confidence_readout
make new-exp EXP=exp360_typewell_reference_shift_zncc_confidence_readout

.venv/bin/pytest -q \
  experiments/exp360_typewell_reference_shift_zncc_confidence_readout/tests/test_exp360_typewell_reference_shift_zncc_confidence_readout.py
# 10 passed

.venv/bin/ruff check \
  experiments/exp360_typewell_reference_shift_zncc_confidence_readout/*compact_selfcontained*.py \
  experiments/exp360_typewell_reference_shift_zncc_confidence_readout/tests/test_exp360_typewell_reference_shift_zncc_confidence_readout.py \
  --select F821,E9
# All checks passed

JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp360_typewell_reference_shift_zncc_confidence_readout/*compact_selfcontained*.py

make prepare-kaggle-notebooks \
  EXP=exp360_typewell_reference_shift_zncc_confidence_readout \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp360-typewell-shift-zncc-readout-train \
  --title 'exp360 typewell shift ZNCC readout train' --run-on-push --strict"

make push-kaggle-train \
  EXP=exp360_typewell_reference_shift_zncc_confidence_readout
# version 1: input root解決前ERROR
# version 2: COMPLETE

kaggle kernels output \
  kentookumura/exp360-typewell-shift-zncc-readout-train \
  -p experiments/exp360_typewell_reference_shift_zncc_confidence_readout/kaggle/output/train_v2

make test
# 728 passed, 5 skipped, 2 failed
# failures: experiments/exp296_exp223_self_gr_known_tvt_support_gate/tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py の既存config/status契約2件
# exp360専用10件とnotebook共通4件は全PASS
```

- `make new-steering`と`make new-exp`で設計用scaffoldを作成した。
- Kaggle Stage 0はversion 2で完了した。推論、提出は実行していない。

## 次のアクション

1. primary-only fail-closed規則どおりZNCC confidence branchを閉じる。
2. threshold、family、shift grid、pair/std、sentinel、supporting familyで救済しない。
3. prediction変更、add-only feature化、再実行、inference、submissionへ進まない。
4. exp360をbacklogから削除し、この結果だけを根拠とする同family救済候補を追加しない。
