# exp277 PF ANCC small-seed mean add-only selector audit

> **CORRECTED MEAN4 SELECTOR PASS:** 旧version 1出力はquarantineしたまま、修正版exp264の
> raw-test-only 88特徴からmean4 nested selector 40 CPU boostersをversion 2で完了・監査済み。

## 状態

- route: `ensemble`
- status: `corrected_mean4_nested_completed_score_guard_passed`
- 親: `exp271_pf_ancc_small_seed_mean_candidate_audit`
- selector親: `exp264_exp263_candidate_confidence_dual_selector`
- CV / LB: selector score guard PASS / TVT downstream未実行・未提出

## 仮説

exp271 version 2の固定PF ANCC mean4/mean8 pathでexp263 core-12内の`pf_ancc`を差し替え、exp264型nested
selector compactからclean 273 downstream TVT modelへadd-onlyする。比較は`mean4_only`、
`mean8_only`、`mean4_mean8_disagreement`の3つ。修正版exp264 Stage D v3 clean-273
matched-control OOFはSHA固定baselineとして読むだけで、controlは再学習しない。

single variantは12候補のまま、both variantだけmean4/mean8の2 slotとなるため13候補になる。
既存`pf_ancc`とmean pathを同時に残す追加監査ではない。

## 検証方針

actual train/current-test headerで`MD/X/Y/Z/GR`のavailabilityを確認し、formation raw/deltaを
schema guardで拒否する。その後outer 5 × inner 4のfold-safe selector compactをvariantごとに生成し、
clean 273 + compact 74、固定3 configs × 5 foldsのadd-only downstreamを保存済みcontrol OOFと比較する。
overall、3-of-5 folds、1000+、hidden-like 2面、worst-wellを全guardとする。

## 所見

### version 1 の技術監査

`kentookumura/exp277-pf-ancc-mean4-selector` version 1はCPU 40 boostersを約6,984秒で完走した。
40 model、25 compact partitions、18,919,945 compact rowsのmanifest・model・partition SHAは一致し、
候補列も旧`pf_ancc`を含まず`pf_ancc_seed_mean_4`へ正しく差し替わっている。

一方、出力`feature_schema.json`にはhidden testに存在しないformation 6列
（ANCC / ASTNU / ASTNL / EGFDU / EGFDL / BUDA）のraw値とlast-known差分が計12特徴含まれる。
このためnotebook内score/leakage guardのPASSは性能根拠にせず、outputはquarantineする。
PF再生成、hard top1、candidate平均、inference、submissionは実装scope外のまま維持する。

### 修正版親へのport

- exp264 Stage A v4: 88特徴、logical SHA `aaef4ffd...ddd3a4`、formation raw/delta 0。
- raw context: `MD/X/Y/Z/GR`、train 773/773・current-test 3/3 filesでavailability PASS。
- exp264 Stage C v6: 40/40 model SHA、model manifest `3f28b04a...2422d2`、compact manifest
  `f4855726...aecf1c`。
- downstream: source 380から非fold-safe 107列を落としたclean 273、allowlist SHA
  `d01a73cc...677bf`。修正版Stage D v3 control RMSE `10.476169179`、OOF SHA
  `7367983f...6dafee`。
- exp277 notebookはraw header audit、formation schema guard、273 + 74 = 347列guardを持つ。
- pushed packageは`nested_selector_mean4_only` / `run_approved=true` / CPU / internet off。
- local configは重複push防止のためpush後に`run_approved=false`へ戻した。

### corrected mean4 version 2

- Kaggle: version 2 / `COMPLETE`、stage完了5,707.598秒、CPU 40 model、生成物86件。
- candidate: 12候補のslot 4を旧`pf_ancc`から`pf_ancc_seed_mean_4`へ差し替え。
- schema: 88特徴。corrected exp264との差はcandidate ID one-hot 1列の置換だけ。
- raw-test guard: `MD/X/Y/Z/GR`がtrain 773/773・current-test 3/3で全存在、formation
  raw/delta hit 0。
- integrity: 40 model実体のSHA 40/40一致。model manifest `6cf60fa8...563d9`、compact manifest
  `50cf8e0d...1298e`。25 partitions / 18,919,945 compact rowsのmanifest契約もPASS。
- selector guard: expected-error MAE `3.793764`対prior `5.708749`、within10 logloss
  `0.360024`対`0.509003`、Brier `0.112272`対`0.164560`。すべてpooledかつ5/5 folds改善。
- corrected exp264 original `pf_ancc` selector比は、expected-error MAE `-0.005055`（4/5 folds改善）、
  logloss `+0.000612`、Brier `+0.000441`（各2/5 folds改善）でmixed。mean4の一様優位は未確認。
- hard top1、downstream、PF再生成、inference、submissionは実行していない。

## 次

corrected mean4 selector stageは有効な結果としてPASSしたが、TVT downstream価値は未評価。
`downstream_mean4_only`を続ける場合は、version 2 compactを固定入力とし、1 variant × 3 configs ×
5 folds = 15 GPU boosters、control再学習0を別途承認してから実行する。mean8/both、推論、提出も未承認。
