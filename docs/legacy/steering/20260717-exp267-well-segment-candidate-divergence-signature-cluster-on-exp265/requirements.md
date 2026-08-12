# 要件

## 依頼

`KAGGLE_DIRECTION.md` の高優先バックログ
`well_segment_candidate_divergence_signature_cluster_on_exp265` を
`exp267_well_segment_candidate_divergence_signature_cluster_on_exp265` として実装する。

exp265で512-row blockの長さとraw MD spanに支配されたK=3 regimeを救済するのではなく、
exp263の6 primitive candidateがwell内の序盤・中盤・終盤でどの程度広がるかを
target-freeな18次元well署名へ固定する。outer-fold target-free clusteringと、cluster確定後の
保存済みexp264 Stage B score監査までを0 boosterで実装する。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親はexp265、candidate sourceはexp263、post-assignment scoreは保存済みexp264 Stage Bとする。
- evaluation progressの3区間、18特徴、K=3、clip、seed、guardをgridしない。
- cluster fitにはtrue TVT、TVT_input、target/error/oracle、exp264 scoreを入れない。
- raw MD span、block row count、absolute candidate TVTを特徴にしない。
- Stage Aは0 variant / 0 LightGBM config / 0 fold training / 0 booster、親/control再学習0。
- conditional Stage Bの10 CPU boosters、downstream GPU、inference、submissionは無効のままにする。
- Kaggle pushは別途ユーザー承認を得る。

## 受け入れ基準

- `.steering`、`config.yaml`、candidate contract、Jupytext train/inference、共有監査module、
  targeted unit test、`SESSION_NOTES.md`、`result.md`、`metrics.json`が揃う。
- wellごとにevaluation progressを`[0,1/3)`, `[1/3,2/3)`, `[2/3,1]`へ固定し、
  6指標×3区間=18特徴をrow-weightedで生成する。
- outer 5 foldsごとにouter-train median、RobustScaler、clip `[-10,10]`、KMeans K=3をfitし、
  outer-validへlow/middle/highのsoft membershipをOOF付与する。
- 773 wells / 5 folds / 18特徴 / forbidden hit 0、3区間coverage/fallback、occupancy、
  別seed stability、divergence profile、fold semantic整合をfail-closedで監査する。
- cluster assignment凍結後にのみexp264 scoreをjoinし、candidate winner、calibration差、
  worst clusterのsingle-well依存を保存する。
- Stage A全guard通過時だけ別承認の10 CPU booster add-onlyへ進める判定を出す。
- Jupytext round-trip、py_compile、Ruff F821、targeted tests、strict experiment validationを通す。
- deterministic anchorとは扱わない。Stage Aではmodel/prediction/submissionを作らず、input、
  feature schema/content、preprocessor/centroid/assignment SHAとKaggle kernel versionだけを記録対象にする。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
