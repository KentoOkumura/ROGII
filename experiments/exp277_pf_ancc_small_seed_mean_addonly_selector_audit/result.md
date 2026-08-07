# exp277 PF ANCC small-seed mean add-only selector audit 結果

> **VERSION 1 INVALID / CORRECTED MEAN4 VERSION 2 VALID:** version 1出力はquarantineを維持する。
> 修正版exp264の88列schemaによるcorrected mean4 selectorは40 CPU boostersで完了し、guard PASS。

## 状態

`mean4_only` version 1は技術的に完走したが、旧親schemaが無効なため出力を使用しない。
2026-07-19に修正版exp264 Stage A v4 / Stage C v6 / Stage D v3へ親契約を差し替えた。
corrected version 2はKaggle `COMPLETE`で、raw-test-only schema、nested leakage、selector scoreの
全guardを通過した。有効なselector結果として採用する。TVT downstream CV / LBは未算出。

## corrected version 2 結果

- kernel: `kentookumura/exp277-pf-ancc-mean4-selector` version 2 / id_no `127737879`
- runtime: stage完了まで5,707.598秒（CPU）
- cost: 1 variant、2 objectives、outer 5 × inner 4 = 40 models、control再学習0
- candidate: 12候補。旧`pf_ancc`なし、slot 4は`pf_ancc_seed_mean_4`
- schema: 88特徴、logical SHA `0e92f643...8e36`。corrected exp264との差はcandidate identity
  `id__candidate__pf_ancc` → `id__candidate__pf_ancc_seed_mean_4`の1列だけ
- availability: `MD/X/Y/Z/GR`がtrain 773/773・current-test 3/3で全存在
- forbidden formation raw/delta: 0件
- leakage: outer-valid除外、inner train/valid well分離、outer-train inner OOF、outer-valid 4-inner
  ensembleを確認しPASS
- integrity: 40 model実体のSHA 40/40一致、best iteration 83〜224。model manifest
  `6cf60fa8...563d9`、compact manifest `50cf8e0d...1298e`
- compact: 25 partitions / 18,919,945 rows。各downstream outer foldは3,783,989 rows、
  train 4 partitions × 1 modelとvalid 1 partition × 4 modelsの契約に一致
- outer-valid candidate-long: 45,407,868 rows
- hard top1: disabled

selector scoreはexpected-error MAE `3.793764`対outer-train prior `5.708749`、within10 logloss
`0.360024`対`0.509003`、Brier `0.112272`対`0.164560`で、3指標ともpooledかつ5/5 folds改善し、
事前定義したscore guardをPASSした。

corrected exp264 Stage C v6のoriginal `pf_ancc` selectorとの直接比較では、mean4置換の
expected-error MAEは`-0.005055`で4/5 folds改善した。一方、within10 loglossは`+0.000612`、
Brierは`+0.000441`で、改善foldはいずれも2/5だった。したがって「mean4 selectorはpriorから
有効に学習できる」は支持するが、「元のpf_anccより全selector objectiveで優れる」は支持しない。

## 修正版親契約

- selector raw context: `MD/X/Y/Z/GR`のみ。actual train/current-test headerをfit前に照合。
- selector schema guard: formation 6列のraw/delta 12特徴を明示拒否。
- selector parent: corrected 88特徴、Stage C v6 model manifest `3f28b04a...2422d2`。
- downstream base: clean 273 allowlist `d01a73cc...677bf`。旧380列surfaceは不使用。
- fixed control: corrected Stage D v3 `matched_control` RMSE `10.476169179`、OOF SHA
  `7367983f...6dafee`。control再学習は0。
- final downstream feature count: 273 + variant compact 74 = 347。

## version 1 技術監査

- kernel: `kentookumura/exp277-pf-ancc-mean4-selector` version 1 / id_no `127737879`
- runtime: stage完了まで約6,984.277秒（CPU）
- cost: 1 variant、2 objectives、outer 5 × inner 4 = 40 boosters、control再学習0
- artifact: 40 model / 25 compact partitions / 18,919,945 compact rows / 45,407,868 outer-valid candidate-long rows
- integrity: 40 model SHAと25 partition SHAが全一致。compact schema 74特徴、hard readoutなし、submissionなし
- candidate contract: 12候補。旧`pf_ancc`なし、slot 4は`pf_ancc_seed_mean_4`

notebook内ではexpected-error MAE `3.768452`対prior `5.708749`、within10 logloss
`0.356230`対prior `0.509003`、Brier `0.110910`対prior `0.164560`で、各指標5/5 folds改善と
reportされた。ただし以下のavailability leakageにより、これらの値とguard PASSを性能根拠にしない。

## 無効化根拠

出力`feature_schema.json`の選択100特徴に、hidden testで利用できないtraining-only formation 6列の
raw値とlast-known差分が12特徴含まれていた。

- `ctx__raw__{ancc,astnu,astnl,egfdu,egfdl,buda}`
- `ctx__raw_delta_last__{ancc,astnu,astnl,egfdu,egfdl,buda}`

outer-valid selectorがhidden testでは作れない特徴を見ているため、内部のwell分離guardがPASSしても
feature availability leakageは解消されない。compact、selector score、importanceをすべてquarantineし、
downstream入力へ使わない。

## 仮説

exp263/exp264 nested selectorの既存`pf_ancc`をexp271の固定mean4/mean8 pathへ差し替え、
seed/particle disagreementをboth variantだけへ加えれば、exp264 Stage D fixed controlに対してoverall、3-of-5 folds、1000+、hidden-like
2面、worst-wellを守りながらdownstream TVT RMSEを改善できる。mean8依存gainがなければ、
raw-test計算契約を4 seedへ縮約できる。

## 実装範囲

- `mean4_only`、`mean8_only`、`mean4_mean8_disagreement`の3 variant。
- mean4/mean8 singleは`pf_ancc`を置換して12候補、bothは2 meanへ置換して13候補。
- nested selectorはvariantごと40 CPU boosters。
- downstream add-onlyはvariantごと15 GPU boosters、control再学習0。
- exp271 gzip、exp263 cache、exp264 fixed control OOF、exp218 surfaceをSHA guardする。
- PF再生成、hard top1、candidate平均、inference、submissionはなし。

## 親と変更点

exp271の固定PF path、exp263のcore-12 candidate bank、exp264のnested selector、exp218の
downstream surfaceを親とする。selector側の変更は既存`pf_ancc` slotのmean4/mean8への置換と
target-free disagreement blockだけで、fold、selector objective、downstream model config、control OOFは固定した。

## 未実行・禁止

- `mean8_only` / `mean4_mean8_disagreement` nested selector train
- downstream train
- aggregate compare
- inference / submission

mean4 selector compactは有効なdownstream入力候補になったが、TVT RMSEをまだ測っていないため、
add-only仮説全体の支持・棄却や4 seed縮約は判断しない。

## 次

corrected `nested_selector_mean4_only`は完了し、次の実行stageは`downstream_mean4_only`。
実行する場合はversion 2 compactを固定し、1 variant × 3 configs × 5 folds = 15 GPU boosters、
control再学習0を別途承認する。mean8/both、aggregate、inference、submissionは自動では進めない。
