# exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264 セッションノート

## 目的

exp368の連続weak riskが、exp264 selectorにとってactionableな候補回復情報かを
別実験で確認する。Stage 0のimplementation-onlyを完了し、受領した別承認に
基づいてKaggle private CPUで実行する。

## 現在の状態

- Route: `ml_model`
- 状態: Stage 0完了・scientific gate FAIL・branch閉鎖
- CV / LB: なし
- steering:
  `.steering/20260726-exp401-exp368-weak-risk-candidate-advantage-readout-on-exp264/`
- canonical train Notebook: compact self-contained候補を採用
- inference / `settings.py`: 未編集placeholder
- compact self-contained Jupytext / Notebook候補: train / fail-closed inference実装済み
- helper: なし。train候補はself-contained
- 専用test: 9件PASS
- Kaggle package: strict準備済み
- Kaggle push: version 1成功
- Kaggle run: version 4 COMPLETE

## 2026-07-26 設計確定

### 根拠

- exp368 marginalized-PF branchはknown-prefix NLL gain
  `0.037356% < 1%`、weak mass `0.009689 < 0.02`でFAILを維持する。
- 一方、saved suffix bad10 AUCは`0.636675`、circular差`+0.058264`、
  AUC>0.5が5/5 folds、hidden-likeは`0.641795 / 0.636115`だった。
- したがってPF likelihood変更ではなく、selector用のrisk contextを独立仮説にした。

### 固定入力

- exp368 block ledger content SHA:
  `7327ce8e6383d76f99c51cec6982c1db181e6f05257df28e7268d7a0549ba30a`
- exp368 weak posterior content SHA:
  `4ffa4fc761fc4db6b1c7de42c132b8102e33f9910bf5dc56752b20e95c2520ae`
- exp264 corrected Stage C v6 candidate score logical SHA:
  `a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc`
- exp226 fold decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- exp115 hidden-like assignment SHA:
  `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`
- rows / wells / blocks / folds:
  `3,783,989 / 773 / 15,174 / 5`
- candidate-long rows:
  `45,407,868 = 3,783,989 × 12`

### Stage 0実行量

- diagnostic variant: 1
- reporting folds: 5
- model config / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / parent control replay / prediction / submission: `0 / 0 / 0 / 0`

featureとtruthの順序、label、2 legal domains、technical/scientific gateは
steering `design.md`と`config.yaml`に固定した。一つでもFAILなら、反転、
threshold、block、domain、subset、metric、gateの調整なしで閉じる。

### 条件付きStage 1実行量

Stage 0全gate PASSと別承認が必要。

- scientific variant: 1
- LightGBM config: 1
- objectives: 2
- outer folds: 5
- inner folds: 4
- CPU selector boosters: `1 × 2 × 5 × 4 = 40`
- parent/control再学習: 0
- PF/HMM/Beam replay: 0
- downstream TVT / GPU booster / inference / submission: 0

Stage 1前にraw train replay parityとraw current-test 14,151 rows / 3 wellsの
feature coverageを確認する。

## コマンドログ

scaffold作成時:

```bash
make new-steering EXP=exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264
make new-exp EXP=exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264 SOURCE=templates/experiment
```

## 2026-07-26 implementation-only

- ユーザーの「exp401を実装してください」を、既存designのStage 0
  implementation-only承認として記録した。
- 正規Notebookは上書きせず、次を別名候補として追加した。
  - 11章 / 2,054行のcompact self-contained train Jupytext source / Notebook
  - submission非生成のfail-closed inference Jupytext source / Notebook
  - 専用contract test 9件
- 親exp264の正規train sourceは7章 / 465行。exp401候補は、
  runtime/config、input preflight、target-free block集約、strict-nested
  selector surface、late truth、readout、gate、生成物/SHAのrole slotを
  すべてNotebookセルへ展開しており、同一exp helperをimportする薄い構成ではない。
- exp368 block ledger / weak posteriorはsafe projectionだけを読み、
  512-row / stride 256 / tail keepの全overlap blockをrow算術平均する。
- exp226からfold-only projectionを先に読み、exp264 corrected Stage C v6を
  Parquet row group単位で45,407,868行 / 12候補宣言順 / outer-valid /
  nested model count 4までfail closedで検査する。候補TVTは一時float32
  memmapとし、wide candidate DataFrameを常駐させない。
- primary 11候補とsecondary 7候補でanchor以外の`pred_abs_error`最小を
  宣言順tie-breakで固定し、margin decileとweak quartileをother 4 foldsだけで
  target-freeに固定する。
- row risk gzipのdecompressed content SHA、schema SHA、selector surface SHA、
  scientific contract SHAを揃えた後にだけ`TruthAccessLedger`が`tvt_true`を
  読める。early truth/error projectionは例外で停止する。
- truth join後はnominated recovery10、oracle headroom、realized advantage、
  real / circular AUC、fold / hidden-like、margin-conditional AUC、
  Q4-Q1を計算し、事前固定gateを全ANDで判定する。AUC反転や救済分岐はない。
- 実装コードにはLightGBM fit、PF/Beam replay、TVT prediction、
  submission生成を含めていない。Stage 1 / inferenceはfail closedのまま。

実行した検証:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264/\
exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264/\
exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264/\
exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264/\
exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264_compact_selfcontained_inference.py
.venv/bin/python -m py_compile \
  experiments/exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264/\
*compact_selfcontained*.py
.venv/bin/ruff check \
  experiments/exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264/\
*compact_selfcontained*.py \
  experiments/exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264/tests/test_exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264.py \
  --select F821,F401,F841,E9
.venv/bin/pytest -q \
  experiments/exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264/tests/test_exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264.py
make validate-exp EXP=exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264
make validate-template
.venv/bin/pytest -q
```

結果:

- Jupytext conversion / round-trip: PASS
- py_compile: PASS
- Ruff `F821/F401/F841/E9`: PASS
- 専用test: `9 passed`
- strict experiment validation: PASS
- template validation: PASS
- `__file__`: train / inference候補とも0
- 正規Notebook上書き: 0
- Kaggle package / push / run: 0
- model / booster / PF / prediction / submission: 0
- repo全体pytest: `1111 passed, 7 skipped, 2 failed`。failure 2件はいずれも
  未変更のexp296 contract testで、2026-07-20 00:07のtestが
  2026-07-20 09:36に完了状態へ更新されたconfig
  (`status=completed_train_side_guard_failed_closed`,
  `run_variant=false`)より古い実行前期待
  (`status.startswith("kaggle_cpu_")`、approval guard順序)を保持しているもの。
  exp401専用testおよび他1111件はPASSしており、exp296は本依頼のscope外として
  変更していない。

## 再現性メモ

- Stage 0 seed policy: RNGなし、CPU single worker
- Stage 1 seed policy: exp264 seed 42 / deterministic LightGBMを継承
- stochastic components: 現設計ではなし
- PF/Beam/HMM: 保存値load-only、replay 0
- gzip evidence: decompressed logical-content SHA
- Parquet evidence: schema SHA + logical-content SHA
- model / prediction / submission SHA: 現時点では非該当
- Kaggle kernel id / version:
  `kentookumura/exp401-weak-risk-readout-on-exp264-train` / 4 /
  id_no `128626512`
- 実装候補は`Path.cwd()`起点で、Notebook上に`__file__`を残していない。
- gzip row featureはmtime 0で書き、decompressed CSV content SHAを主証拠にする。
- exp264候補TVT memmapはreadout後に削除する一時領域で、prediction生成物ではない。

## 2026-07-26 Stage 0実行承認

- ユーザーの「実行してください」により、正規train Notebook採用、
  Kaggle private CPU package / push / Stage 0 runを承認済みとして記録した。
- 実行前に再提示した固定量:
  - diagnostic variant: 1
  - reporting folds: 5
  - candidate-long rows: 45,407,868
  - model config / LightGBM config / trained fold / booster: 0 / 0 / 0 / 0
  - PF / parent-control retraining / prediction / submission: 0 / 0 / 0 / 0
- runtime: CPU、internet off、GPU off、private、run-on-push true。
- Kaggle inputs:
  - `kentookumura/exp368-marginalized-reliability-pf-train`
  - `kentookumura/exp264-exp263-confidence-dual-selector-train`
  - `kentookumura/exp226-k16-kappa-repro-train`
  - `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`
- Stage 1の40 CPU selector boosters、inference、submissionは承認対象外であり、
  自動実行しない。
- strict package生成をPASSし、metadataを次で固定した。
  - kernel:
    `kentookumura/exp401-weak-risk-readout-on-exp264-train`
  - private: true
  - internet / GPU: false / false
  - run-on-push: true
  - kernel sources: 上記4件
- 最初にexp名全体を含む67文字のslug/titleでpushを試したが、Kaggle APIが
  `400 Bad Request`で拒否した。コードは実行開始していない。Kaggleのmetadata
  長制約に合わせ、意味を保つ40文字の上記slug/titleへ短縮して再packageする。
- 短縮slugでversion 1のpushに成功した。
  - id_no: `128626512`
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp401-weak-risk-readout-on-exp264-train`
  - Kaggle pullでprivate、internet/GPU off、入力4件を再確認した。
- version 1は約16秒でtechnical preflight ERROR。
  `TruthAccessLedger`の部分文字列guardが、configでsafe projectionとして固定した
  exp264 outer-valid予測列`pred_abs_error`を実測error列と誤判定した。
  block集約、truth読込、readout、gateへは到達していない。設計変更ではなく
  exact-name `pred_abs_error`だけをtarget-free predicted-score allowlistへ追加し、
  `abs_error`、`tvt_true`、`bad10`等の拒否を維持してversion 2へ進む。
- allowlist regression test、専用test 9件、Jupytext、py_compile、Ruff、
  strict experiment validationをPASSし、同じ
  1 diagnostic / 5 folds / 0 model / 0 booster / 0 PF / 0 predictionで
  version 2をpushした。
- version 2は約36秒、selector row group 0でtechnical ERROR。exp264の
  `outer_fold == downstream_outer_fold`はstrict-nested score生成foldであり、
  exp226のreporting foldとは独立なのに、実装が両者の一致を要求していた。
  design時に記録済みの既知631/773-well不一致を誤ってjoin errorにしたもの。
  `outer_fold == downstream_outer_fold`、5-fold coverage、row/candidate identityは
  維持し、exp226 reporting foldはcross-fit metricだけに使う。両fold ledgerの
  mismatch rows/wellsをtarget-free selector auditとsurface SHAへ追加して
  version 3へ進む。
- 独立fold ledger regression test、専用test 9件、Jupytext、py_compile、
  Ruff、strict experiment validationをPASSし、同じ
  1 diagnostic / 5 folds / 0 model / 0 booster / 0 PF / 0 predictionで
  version 3をpushした。
- version 3は45,407,868 candidate-long rowsのscan、late truth、readout、
  gate計算、成果物保存まで約125秒で完了したが、最後のgate表示だけが
  `numpy.bool_`を標準JSON encoderへ渡してERROR。科学計算やartifact保存後の
  presentation-only failureである。既存のartifact JSON保存に使っている
  `_json_default`を最終表示にも指定し、serialization regression test後に
  version 4を同一条件で実行する。
- serialization regression test、専用test 9件、Jupytext、py_compile、
  Ruff、strict experiment validationをPASSし、同一科学契約のversion 4をpush。

## 2026-07-26 Stage 0完了

- Kaggle private CPU version 4はCOMPLETE。
- Stage 0 gate出力時刻: `129.300203457 sec`
- kernel最終ログ時刻: `139.263892941 sec`
- 実行量:
  - diagnostic variant / reporting folds: 1 / 5
  - candidate-long / base rows: 45,407,868 / 3,783,989
  - model config / LightGBM config / trained fold / booster: 0 / 0 / 0 / 0
  - PF / parent-control retraining / prediction / submission: 0 / 0 / 0 / 0
- technical gate: 15/15 PASS。入力SHA、3,783,989 rows、773 wells、
  15,174 blocks、12 candidates、5 folds、finite/range/coverage、
  truth-before-freeze 0、feature/readout SHA、zero-compute contractを全確認。
- exp264 generation foldとexp226 reporting foldの不一致は
  3,074,825 rows / 631 wellsで、独立ledgerとしてSHAへ記録した。
- primary `primitive_pair_bank` overall:
  - cohort: 859,755 rows / 595 wells
  - nominated recovery10 AUC: `0.520213769`
  - circular AUC / real-minus-circular: `0.523466649 / -0.003252880`
  - margin-conditional AUC: `0.458846380`
  - hidden-like spatial / typewell-purged AUC:
    `0.527468467 / 0.513625628`
  - Q4-Q1 realized advantage: `+3.879371674 ft`
- secondary `primitive_fixed_bank` overall:
  - nominated recovery10 AUC: `0.504233454`
  - circular差: `-0.003987552`
  - margin-conditional AUC: `0.461268505`
  - Q4-Q1 realized advantage: `+3.676855784 ft`
- scientific gate: 4/12 PASS、総合FAIL。pooled AUC、circular差、
  margin-conditional AUC、hidden-like 2面が固定閾値を満たさない。
- decision: `stage_0_failed_close_without_rescue`。
- 完走後は誤再実行防止のため`execution.run_stage_0=false`、
  `runtime.kaggle.train_run_on_push=false`へ戻した。承認履歴は保持する。
- outputを`kaggle/output/train_v4`へ取得し、実ファイルで照合:
  - frozen feature raw SHA:
    `199dbf853bfd5cee8c5eff7ad42962be1bb7bf3c7fd7fa44fabf6ea79c349973`
  - frozen feature decompressed content SHA:
    `b71b7d5728cf2d61fa709a46ecd87ad69ded269a235a8c16cc3b5ad359955ff7`
  - selector surface SHA:
    `986a02c8c2daa20579faefe4107521b4739cc3fb3ac297ca82450ea612a2db83`
  - scope metrics SHA:
    `a8d7713ec0132856fcc8e8018c296d63c28e957df595d7e7adf471ae23ec0be3`
  - nomination distribution SHA:
    `8faa7a4788f2c20fab59ed68f590b1318b6012f88bc6e9e60179183d67455d79`
  - Stage 0 summary SHA:
    `2b46291b6258450f047a4aff89d01781360acf72d040aab66b3ab7690b145909`

## 次のアクション

固定all-AND gate FAILに従い、threshold、score反転、bucket、domain、
candidate subset、gateの救済なしでbranchを閉じる。Stage 1の40 CPU
selector boosters、downstream TVT、inference、submissionは実装・実行しない。
