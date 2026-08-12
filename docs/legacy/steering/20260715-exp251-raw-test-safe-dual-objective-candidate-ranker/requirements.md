# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `raw_test_safe_dual_objective_candidate_ranker` を実験化する。
exp248 `original_only` の within10 classifier、expected-error regressor、固定Viterbiを維持し、raw testで完全再生成できる特徴だけに限定して再監査する。

## 制約

- 対象実験は `exp251_raw_test_safe_dual_objective_candidate_ranker`。
- Routeは `ensemble`。固定候補bankにPF/Beam、dense、HMM、geometryが本質的に含まれる。
- exp248の固定11候補、outer well 5-fold、2目的、row sampling cap、LightGBM parameter、固定Viterbi 1規則を変更しない。
- 最初は297 long featureすべてについて、train/raw-test生成可否、provenance、fallback率、分布差、schema/content SHAを監査する。feature audit中はboosterを学習しない。
- raw testに物理的に存在しない列とtrain-only exp226 auxiliary列をmedian/0で救済しない。
- `copcf_*`は一律にOOF-onlyとしてrejectしない。trainではfold外wellだけを参照する既存OOF値、raw testではraw-test wellをsource poolから除外したfull-train参照値を生成し、同じ165列のschemaを再現する。
- visible raw testのwell IDがtrain cacheにも存在するため、typewell/spatial priorのlabel sourceとreference curveからraw-test well IDを必ず除外する。test側の`TVT`列やtrain側の同一well targetを読まない。
- augmentation、candidate追加、Viterbi grid、true target/error/oracle gate、inference、submissionを行わない。
- exp248 original-onlyを保存済みcontrolとし、親/control/PF/HMM/Beam/dense/geometry/exp218を再学習しない。
- Kaggle CPU、GPU false、internet false。ローカルnotebook実行は行わない。
- `docs/06_reproducibility.md` に従い、gzipはdecompressed content SHAを主証拠にする。

## 受け入れ基準

- `execution.stage=feature_audit_only` が既定で、0 variant、0 config、0 fold、0 boosterで停止する。
- exp248の297 featureを過不足なく監査し、各列のprovenance、train/raw-test missing率、missing率差、quantile、standardized mean difference、PSI、採否理由を保存する。
- selected featureはraw test上で生成済み、許可provenance、missing率5%以下、train/raw-test missing率差5%以下をすべて満たす。
- `copcf_*` 165列について、raw-test生成、finite coverage、train OOF/full-train分布差、source-well exclusionを監査する。guardを通る列はselected schemaへ入れてよい。
- `exp226_gr_delta`、`exp226_geop_tvt`、`exp226_geop_minus_pred*`、その他raw-test未生成列はselected schemaへ入らない。
- feature auditはhard-coded prefix denylistで採否を決めず、実際に生成したraw-test列とprovenance contractで判定する。
- selected train/raw-test feature sample、schema、contractのSHAを保存する。gzip sampleはdecompressed content SHAを保存する。
- optional train stageはsame-run audit passを必須とし、1 variant × 2 objectives × 5 folds = 10 CPU boostersだけを学習する。
- train stageではoverall、distance 1000+、exp115 hidden-like 2群、worst-well、candidate calibration、fixed Viterbiを評価し、model/prediction/schema SHAを保存する。
- notebook、補助module、config、SESSION_NOTES、README/result/metrics、実験横断記録が静的検証を通る。
