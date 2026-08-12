# 要件

## 依頼

閉鎖済み`exp314_label_derived_typewell_gr_quality_addonly`のsoft ML feature着眼点を、
`exp311` / `exp313` のpromotionに依存しない0-booster preflightとして新番号へ切り出す。
Type Well群のsupport/noise/reliability統計が、outer-valid wellの固定ML control誤差と
fold横断で関連するかを、LightGBM学習前に判定する。

初回依頼では設計確定までとする。2026-07-23のexp352実行後、平均signalが改善した場合は
次へ進むというユーザー指示を受け、直接補正gateは維持したままStage 0のfeature generator、
compact train、contract tests、正規train Notebook、Kaggle CPU実行までを追加承認範囲とする。
Stage 1の15 boosters、raw-test再生成、inference、submissionは引き続き実装・実行しない。

## 制約

- Route: `ml_model`。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 科学的親は`exp148_learned_likelihood_fulltrain_addonly_on_exp092`、設計参照は旧exp314とする。
- exp148のwell GroupKFold、保存済みOOF、base feature、3 LightGBM configを固定する。
- outer-train wellsだけから6列の群統計を作り、outer-validはType Well contentだけでjoinする。
- 6列は`log_support_wells`、`log_support_rows`、`residual_sigma`、`fit_rmse`、
  `bias_abs_gr50`、`prior_available`に固定し、結果後の列選択をしない。
- fallbackはouter-train global priorと`prior_available=0`に固定する。
- Stage 0は1 preflight / 5 folds / model・booster各0。Stage 1はStage 0全PASSと別承認時だけ
  1 variant × 3 configs × 5 folds = 15 boosters、control再学習0とする。
- calibrated GR、TVT correction、suffix target statisticの直接付与、well ID featureは禁止する。

## 受け入れ基準

- Stage 0 coverage `>=0.90`、fallback `<=0.10`、全feature finiteを要求する。
- outer-valid well単位で`group_residual_sigma`と保存済みexp148 RMSEのSpearman `>=0.15`、
  正方向`>=4/5 folds`を要求する。
- residual-sigma上位quartileと下位quartileのexp148 RMSE差 `>=0.25 ft`を要求する。
- real groupとfold内group-label shuffleのSpearman差 `>=0.05`を要求する。
- Stage 0 PASS時だけStage 1実装を別承認し、exp148 `lgb_mean`比 `>=0.03 ft`改善、
  `>=4/5 folds`、全距離帯・hidden-like非悪化、worst `<=+0.25 ft`を要求する。
- deterministic preflightとして扱う場合は、fold/group/prior/feature/readoutのcontent SHAを記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
