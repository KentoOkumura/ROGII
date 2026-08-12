# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `pf_ancc_small_seed_mean_addonly_selector_audit` を
`exp277_pf_ancc_small_seed_mean_addonly_selector_audit` として実装する。

## 制約

- Route は `ensemble`。exp263 core12内の既存`pf_ancc`をexp271のPF ANCC mean candidateで
  差し替え、exp263/exp264 selector compactを介して
  exp218系 downstream TVT modelへadd-onlyする。
- exp271 version 2の保存済みcandidate gzipだけを読み、train-side PFを再生成しない。
- exp263 core 12、outer 5 folds、inner 4 folds、selector 2 objectives、exp218 3 configsを固定する。
- selector raw contextは修正版exp264 Stage A v4の`MD/X/Y/Z/GR`だけとし、actual train/current-test
  header availabilityをfit前に確認する。formation 6列のraw/delta 12特徴は禁止する。
- 比較variantは`mean4_only`、`mean8_only`、`mean4_mean8_disagreement`の3つに固定する。
- single variantは`pf_ancc`をmean4またはmean8で置換して12候補を維持し、both variantは
  `pf_ancc`をmean4+mean8で置換して13候補とする。
- 修正版exp264 Stage D v3のclean 273 `matched_control` OOFをSHA固定baselineとして読み、
  controlを再学習しない。旧380列control OOFは使用しない。
- target / error / oracle / hidden-like roleをfeature、gate、candidate選択へ使わない。
- hard top1、oracle routing、candidate平均、raw-test PF再生成、inference、submissionを実装scopeに入れない。
- Kaggle pushはstage別costの明示承認後に行う。実装時点では`run_approved=false`とする。

## 受け入れ基準

- exp271 candidate gzipのraw/decompressed SHA、schema、3,783,989 rows / 773 wellsをfail-closed検証する。
- exp263 60 partitionとmanifest SHA、exp264 fixed-control OOF SHA、exp218 surface契約をfail-closed検証する。
- 修正版親の88 selector schema、clean 273 allowlist、Stage C v6/Stage D v3 manifest SHAを固定する。
- nested selectorはouter-trainをinner OOF、outer-validをinner model ensembleで作り、fold leakageを防ぐ。
- 各selector variantは40 CPU boosters、各downstream variantは15 GPU boosters、control再学習0本である。
- downstream結果はoverall、fold、1000+、hidden-like 2面、worst-wellをfixed controlとの差で出力する。
- mean4 / mean8 / both+disagreementを同じfold・config・base surfaceで比較できる0-booster aggregate stageを持つ。
- Jupytext notebook、構文、F821、unit test、strict experiment validationが通る。
- feature content SHA、model manifest SHA、prediction SHA、Kaggle kernel versionを記録できる。
- gzip生成物はdecompressed content SHAを主証拠にする。
- 旧mean4 version 1はquarantineしたまま、新しいcorrected runの親または比較値として使用しない。
