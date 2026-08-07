# exp358 セッションノート

## 目的

旧exp308のmissing-distance仮説をfailed exp307 observationから分離し、
exp209直結の0-HMM technical auditとして固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU Stage 1 scientific gate FAIL、rescueなしでclosed
- CV/LB: `12.012569787442315` / なし

## 2026-07-25 Stage 1実装・実行承認

- ユーザー指示「Stage 1に進んでください」を、Stage 1の実装、正規train
  Notebook採用、同一canonical Kaggle kernelへのpackage/push/run、完了監視の
  承認として記録した。inferenceとsubmissionは承認範囲外である。
- Stage 0はKaggle version 1で23/23 technical checksがPASS済みであり、
  Stage 1はその技術契約を前提とする。
- 実行前に固定した量:
  - scientific variant: 1
  - reporting folds: 5
  - exact-HMM well-runs: 773
  - model / LightGBM config / trained fold / booster: 0 / 0 / 0 / 0
  - PF / Beam / parent-control rerun: 0 / 0 / 0
  - GPU / TPU / internet: off
- saved exp209 HMM、saved exp072 LikPF、exp226 fold/truth、exp115 hidden-like
  assignmentはread-only inputとして使い、親controlは再実行しない。
- exp389の検証済みcompact exact-HMM構成を実装骨格に使い、Huber emissionだけを
  exp358の`confidence_weight * (-0.5 * min(z^2, 600))`へ置換した。
  exp308は仮説の履歴参照のみで、failed exp307 observationをimportしない。
- exp209のknown-prefix zero-fill population sigma、absolute TVT grid、rate grid、
  transition、prior、posterior mean出力を固定し、変更はraw-missing行の
  Gaussian log-emission multiplier 1回だけである。
- raw GR mask、missing-distance surface、candidate predictionをunknown-suffix
  truthより先にcontent SHA付きでfreezeする。truth、fold、saved control、
  hidden-like assignmentはfreeze後にlate attachする。
- Stage 1 sourceは11 markdown / 10 codeの21 cells、2,108行。
  比較元のexp389は1,884行、履歴参照exp308は1,987行であり、差分は
  missing-distance observation audit、weighted-Gaussian契約、全promotion
  readoutの展開である。同一exp helper importと`__file__`は使用していない。
- compact候補と正規train Notebookは21 cellsのsource content SHA
  `05677072435578e372c9e7819e006253adb3bfb388d8f422f7549cea3900b00c`
  で一致した。正規inference Notebookはfail-closedのままである。
- 事前登録済みAND gateはexp209 raw-HMM比0.05 ft以上、4/5 folds改善、
  `md_since_1000_plus`、2つのhidden-like surface、by-well p95、worst well
  0.25 ft以内、fixed LikPF 50/50非劣化である。失敗時はhalf-life/floor grid、
  hard mask、sigma/transition/prior/blend rescueを行わずcloseする。
- 実装検証:
  - `py_compile`: train / inference PASS
  - Ruff full check / format: PASS
  - dedicated pytest: `14 passed`
  - dedicated + common Kaggle Notebook pytest: `18 passed`
  - Jupytext変換 / round-trip: PASS
  - `make validate-exp EXP=exp358_exp209_missing_distance_emission_downweight`: strict PASS
  - `make validate-template`: PASS
- Kaggle push前の時点ではStage 1 Notebookをローカル実行しておらず、
  Stage 1のscientific CV結果もまだ存在しない。
- `make prepare-kaggle-notebooks`を同一canonical id/title、
  `--run-on-push --strict`で実行した。metadataはprivate、CPU、
  GPU/TPU/internet off、competition source 1、read-only kernel source 3、
  dataset/model source 0である。
- package Notebookはbootstrap 1 cellと正規train 21 cellsの計22 cells。
  bootstrap後21 cellsは正規trainと完全一致し、support 25 filesのbytes/SHAと
  package loose filesのbyte一致を確認した。
- package Notebook cell-source SHA:
  `560239fd268ec25095416a647e9aff08567bdbd00c323dc0a3346008625a5322`
- bootstrap config SHA:
  `645fcd33f72197908e02087d814cac11e81f1b98ff4aa4eaa2c4811eec02402f`
- bootstrap Stage 1 source SHA:
  `ac0d6ea584ffa77f25d0f5aac94952078eb0bb4160a9089270f5f886d936fb26`
- 同一canonical kernelへversion 2をpushし、Kaggle実行開始を確認した。
  pull後metadataはid_no `128528105`、private、CPU、GPU/TPU/internet off、
  read-only kernel source 3でpackage契約と一致した。
- packageとpull後Notebookは22 cellsのsource content SHA
  `fbe348f196f8a5ddcd74938a82480ca969da3e32d4020053ef666958f2c18356`
  で一致した。Stage 1の1回実行承認はversion 2 pushで消費済みである。

## 2026-07-25 Kaggle Stage 1結果

- canonical private CPU version 2、id_no `128528105`を完了した。
  runtimeは`17475.55788087845 sec`。1 variant / 5 reporting folds /
  773 exact-HMM well-runs、model / LightGBM config / trained fold / booster /
  PF / Beam / parent-control rerun各0で、事前固定量と一致した。
- direct candidate / exp209 control RMSEは
  `12.012569787442315 / 11.938287234887435`、improvementは
  `-0.07428255255488025 ft`。必要`+0.05 ft`に届かず、0/5 folds改善だった。
- fold improvementは順に
  `-0.271713 / -0.031463 / -0.029050 / -0.007788 / -0.054276 ft`で、
  全foldが悪化した。
- raw observed / missing improvementは
  `-0.048384 / -0.129738 ft`。gap 1--3 / 4--15 / 16+は
  `-0.124171 / -0.142545 / -0.111850 ft`で、全gap bucketが悪化した。
- missing-fraction low / mid / highは
  `-0.012642 / +0.038165 / -0.162302 ft`。midだけ小さく改善した。
- `md_since_1000_plus`、hidden-like spatial、hidden-like typewell-purgedは
  `-0.082776 / -0.224970 / -0.229587 ft`で、required scope全てFAILした。
- by-well improved / regressedは`358 / 415`、median deltaは
  `+0.000786 ft`、p95は`+0.469370 ft`。worst well `f5859199`は
  `+6.630365 ft`で、上限`+0.25 ft`を大きく超えた。
- fixed LikPF 50:50 candidate / controlは
  `10.3066733932215 / 10.269692505026358`、delta `+0.036981 ft`でFAILした。
- Stage 1のformal technical gateは`missing_weight_formula_exact=false`だけで
  FAIL表示となった。切り分けのため50,553,974 byteのfrozen raw-GR emission
  contractだけを取得し、1,200,837 missing rows中753 rowsがgzip CSV再読込後に
  bit-exact不一致、最大絶対差`5.551115123125783e-17`であることを確認した。
  `rtol=0, atol=1e-16`では全件一致し、生成・HMM適用時の式逸脱ではなく
  post-CSV float parseへの過剰なexact guardである。事後にgateは変更しない。
- rows/wells/input SHA/control parity、finite、posterior normalization
  `3.997e-15`、observed exact 1、weight range/unique、clip 600、
  emission application count 1、truth read before freeze 0は全てPASSした。
- reproducibility:
  - scientific contract SHA:
    `90e02546e56e9b0b3c1d58f944fa6f3fe82fc63c6467bfde8e4b94882399ab65`
  - input/control manifest SHA:
    `c07b2ee089457d07257661d452c8511452f9dd03ca41827fa3e3eaa480af00f0`
  - prediction decompressed SHA:
    `5d5c1cb9a0682d5f352e56dd19fffd44574816ce26b0a0e85cfc49d16cc14742`
  - raw-GR emission contract decompressed / raw gzip SHA:
    `36499c6d7d81eb90e98f9181f427db5023eda3857dc3877503e64ac1bdfb7e14` /
    `eebf8350458c0ee46e0924ff5827fd9db124dbdb7e95282e12b619bfdc425d85`
  - observation audit decompressed SHA:
    `bf88a2e069e0b8eb906b6a90fdf11e4e4374cf5153a0f3e7dc40b9ea4949e4d4`
- decision:
  `missing_distance_exp209_failed_close_without_rescue`
- scientific gateはtechnical表示に関係なく明確にFAILした。事前契約どおり
  half-life/floor grid、hard mask、sigma/transition/prior/clip/temperature変更、
  blend/same-OOF rescue、再実行、inference、submissionは行わない。

## 2026-07-25 Stage 0実装

- ユーザー指示「exp358を実装してください」をStage 0実装承認として記録した。
- compact self-contained train候補をJupytext percent形式で実装し、compact
  `.ipynb`へ変換した。既存の正規train/inference Notebookは上書きしていない。
- fail-closed inference候補を実装し、Stage 1、raw-test生成、prediction、
  submissionを明示停止した。
- whole-well raw GRからraw-finite/missing mask、nearest-finite row distance、
  `max(0.25, 2^(-distance/8))`、missing run、gap bucketを決定し、
  unknown suffix 3,783,989 rowsへsliceする契約を実装した。
- exp209と同じ両方向linear interpolationを使い、well内all-missing時だけ
  Type Well GR平均をfallbackにする。observed row weightはexact 1、
  all-missing weightはexact 0.25である。
- raw mask、distance、weight、interpolated GR、per-well summaryをcontent SHA付きで
  freezeするまでunknown-suffix truthを読まない。Stage 0ではfold、hidden-like、
  exp209 saved predictionも不要なためloadしない。
- Stage 0 gateは773 wells / 3,783,989 rows、row identity、mask partition、finite、
  observed exact 1、missing formula/range、above-floor値、複数weight値、
  all-missing fallback、truth-read 0、HMM/model/booster/control-rerun 0のAND gate。
- technical PASSでもStage 1は未実装・未承認のままで、自動進行しない。

## Notebook構成比較

- 科学的親exp209にはcompact self-contained版が存在しないため、親compactとの
  直接比較は非該当。
- exp358 train候補は8章、17 cells（markdown 9 / code 8）、967行で、
  runtime/config/SHA、raw input preflight、distance/interpolation、
  per-well surface、freeze、gate、orchestrationをNotebook上に展開した。
- inference候補は3章、7 cells（markdown 4 / code 3）、117行で明示停止する。
- 同一exp helper importと`__file__`は使用していない。

## 静的検証

- `py_compile`: train / inference / dedicated testともPASS。
- Ruff `F821,F401,F841`: PASS。
- Ruff full check / format check: PASS。
- dedicated pytest: `9 passed`。
- dedicated + common Kaggle Notebook pytest: `13 passed`。
- Jupytext変換 / round-trip: PASS。
- `make validate-exp EXP=exp358_exp209_missing_distance_emission_downweight`: strict PASS。
- `make validate-template`: PASS。
- repository全体は`969 collected / 961 passed / 6 skipped / 2 failed`。失敗2件は
  既存exp296の完了後configに対してtestが旧`kaggle_cpu_*` statusと
  `run_variant=true`を期待する不一致で、exp358専用9件は全PASSしている。
- この実装検証時点ではローカルNotebook実行、Kaggle package/push/runは行っていない。

## 2026-07-25 Kaggle Stage 0実行承認

- ユーザー指示「実行してください」を、compact self-contained Stage 0候補の
  正規train Notebook採用、Kaggle CPU package/push/run、完了監視の承認として
  記録した。
- canonical kernel id:
  `kentookumura/exp358-missing-distance-emission-downweight-train`
- canonical title:
  `exp358 missing-distance emission downweight train`
- push前の実行量:
  - technical audit: 1
  - reporting folds: 0
  - HMM well-runs: 0
  - model / LightGBM config / trained fold / booster: 0 / 0 / 0 / 0
  - PF / Beam / parent-control rerun: 0 / 0 / 0
  - GPU / TPU / internet: off
- Stage 1の1 variant / 5 reporting folds / 773 HMM well-runs、inference、
  submissionは承認範囲外のまま。
- compact候補と正規train Notebookは17 cellsのsource content SHA
  `e649c14a5a02a561d4ba8430751876f3307cd8e696e6712f366634849b8c41f4`
  で一致した。正規inference Notebookはplaceholderのまま。
- `make prepare-kaggle-notebooks`をcanonical id/title、`--run-on-push --strict`
  で実行した。metadataはprivate、CPU、GPU/TPU/internet off、
  competition source 1、kernel/dataset/model source 0。
- loose / package / bootstrapのconfigとtrain sourceはbyte一致した。
  bootstrap SHAはconfig
  `b571e9840a4d04ea999a4b19a903826a61b033f0351fcd808a7d47a906a312f9`、
  train source
  `00837174709743ace8e40b9e8e9aaa55c341306ca88de27b6030bd069722559f`。
- 初回pushはKaggle `SaveKernel`の400で停止した。response bodyを資格情報や
  request bodyを出さない診断で確認し、原因が
  `The title cannot exceed 50 characters.` であることを特定した。
- 科学条件と実行量は変えず、canonical id/titleだけを上記49文字の値へ短縮した。

## 2026-07-25 Kaggle Stage 0結果

- canonical private CPU version 1、id_no `128528105`を完了した。
- remote metadataはprivate、CPU、GPU/TPU/internet off、ROGII competition source 1、
  dataset/kernel/model source 0で、package契約と一致した。
- packageとKaggle pull後Notebookは18 cellsのsource content SHA
  `9930e5ff8025b7ec37d0356de6311a7438ef95eb914a93060931dcf657da7562`
  で一致した。
- Stage 0 runtimeは`82.49706196784973 sec`。3,783,989 rows / 773 wells、
  raw observed 2,583,152 rows、raw missing 1,200,837 rowsを監査した。
- missing fractionは`0.3173468527524789`。missing weightは
  min `0.25`、max `0.9170040432046712`、unique `16`、
  above-floor 1,199,379 rows、all-missing well 0だった。
- expected rows/wells、raw identity、mask partition、finite、exp209 interpolation、
  observed exact 1、missing formula/range/non-degeneracy、freeze-before-truth、
  truth read 0、HMM/model/booster/control-rerun 0を含む23/23 checksがPASSした。
- execution countsはtechnical audit 1、reporting fold 0、HMM 0、
  model config 0、trained fold 0、booster 0、parent-control rerun 0。
- scientific contract logical SHA:
  `4eee8ae34beec8bb849e8a2becebe4958675b5a8c6a33aec64e4efb6ba760ddd`
- confidence weight logical SHA:
  `5de63cc6ddb5daae920fe88e7f7fecc28a02247fab8ae2f8ab6b1e90e489eb02`
- weight surface logical / decompressed / raw gzip SHA:
  `4683a75d19a0a73d0a819339a879ebdeaf089e21493e87dfed5d86f3191252f7` /
  `89fe2fcc6465533097e7f4a54c278ac44010bc0886fdefdceac5e51202bc8a54` /
  `98e76fd60cc3fc861a382270b1a016167263a58962817edc98948ac0d4d1f52b`
- Kaggle output一覧で期待Stage 0生成物8件を確認した。logsでgate、counts、
  SHAを確認できたため、44,121,300 byteのweight surfaceを含むoutput archiveは
  ダウンロードしていない。
- decision:
  `stage_0_technical_pass_awaiting_separate_stage_1_approval`
- Stage 0 PASSは技術的適格性だけで、CVやexp209比の科学的改善は未評価。
  Stage 1、inference、submissionは実装・実行していない。

## 2026-07-23 設計

- ユーザー依頼によりexp358を採番し、steeringとscaffoldを作成した。
- parentはtrusted exact-HMMのexp209に固定した。
- Stage 0はtechnical audit 1 / HMM・model・trained fold・booster各0。
- Stage 1予約は1 variant / 5 reporting folds / 773 HMM runs / parent-control再実行0。
- 実装、Notebook採用、Kaggle package/push/run、inference、submissionは行っていない。

## 再現性メモ

- RNGなし。raw well、row、nearest-distance、reduction順を固定する。
- Stage 0ではraw identityをhard guardし、Stage 1時だけexp209 controlの
  decompressed SHAとfold/hidden-like assignmentを追加でhard guardする。
- missing mask、distance、weight、interpolated GR、Stage 0 summaryのcontent SHAを記録する。
- Stage 1時だけdecoder contractとprediction SHAを記録する。
- deterministic anchorとは扱わない。

## 次のアクション

branchをclosedとして維持する。inference、submission、rescue、再実行へ進まない。
