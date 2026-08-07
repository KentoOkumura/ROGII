# 要件

## 依頼

`exp287_fold_safe_formation_tail_attribution_readout`候補を正式な実験として採番し、設計、readoutコード、Notebook、synthetic testsを実装する。その後のユーザーの`実行してください`という明示依頼により、canonical Notebookへ採用し、Kaggle CPUでStage A/Bを実行して固定判定まで記録する。inferenceとsubmissionは含めない。

## 背景

- corrected exp264はCV `8.460811237612477`、Public LB `7.562`。
- exp287はfold-safe formation 74列をadd-onlyし、CV `8.136708220359452`、exp264比`-0.3241030172530248 ft`、5/5 folds改善、Public LB `7.530`を得た。
- exp287はexp264比worst-well `+8.228409822385604 ft`、`+1/+3/+5 ft`悪化well数`140/40/19`でtrain-side guardをFAILした。
- exp334はwell均等lossでCVをexp287比`-0.04321069622868201 ft`改善したが、by-well p95 `+0.429584617 ft`、exp264比worst `+7.156485377 ft`、`+3/+5 ft`悪化well数`40/19`でsevere tailを回復できなかった。
- したがって、row数比例lossだけでなくformation reference品質にtail原因があるかを0-boosterで原因分解する。

## 仮説

exp287のtail悪化がformation reference品質に由来するなら、事前登録した6つのtarget-free formation risk familyの少なくとも1つで、高risk Q4ほど低risk Q1よりexp287のexp264比well RMSE deltaが安定して悪化する。

## 実験識別

- 実験名: `exp336_exp287_formation_tail_attribution_readout`
- Route: `ml_model`の診断readout
- 親: `exp287_fold_safe_formation_74_addonly_on_exp264`
- 比較: corrected `exp264_exp263_candidate_confidence_dual_selector`
- trigger evidence: `exp334_equal_well_loss_weighting_on_exp287`
- phase: Late。追加GPU train前の低コスト反証を優先する。

## 科学要件

- exp287/exp264の予測、fold、モデル、特徴量を変更しない。
- exp287 outer-valid formation cacheから推論可能なtarget-free属性だけを作る。
- 属性値、aggregation、risk方向、四分位境界、coverage、判定基準をtruth/error join前に固定する。
- 主endpointはwell別`RMSE(exp287)-RMSE(exp264)`とし、wellを等重みで扱う。
- primary risk familyは次の6件に固定する。
  1. plane reference距離
  2. dense reference距離
  3. dense近傍不確実性
  4. plane-dense予測不一致
  5. formation間spread
  6. known-prefix formation calibration error
- plane/dense reference availability、formation finite/missing、known-prefix/trajectory、generic signal disagreementはcontext readoutとし、単独でprimary PASSにしない。
- exp334 OOFはformation追加のclean comparisonではないため、主endpointへ混ぜない。

## Leakage制約

- Stage Aでは`TVT`、target、actual、prediction、error、abs/squared error、worst-well ID、by-well outcomeをloadしない。
- raw horizontal contextは`MD/X/Y/Z/TVT_input`だけを許可する。
- Stage A成果物のcontent SHAを固定後だけStage BのOOF joinを許可する。
- Stage Bは属性、方向、境界、coverage、gateを変更できない。
- errorを見てfamily、threshold、bin、transform、weightを選択しない。

## 判定要件

各primary familyの高risk Q4と低risk Q1を比較し、次のANDを満たしたfamilyだけをPASSとする。

- global Q4−Q1 mean well-delta `>= +0.25 ft`
- global Q4−Q1 median well-delta `> 0`
- 5 folds中4 folds以上でQ4−Q1 `> 0`
- hidden-like spatial / typewell-purgedの両方でQ4−Q1 `> 0`
- global Q1/Q4各100 wells以上、各fold各10 wells以上、各hidden-like各15 wells以上
- 四分位edgeがstrictに増加し、属性・境界・方向がerror非依存

少なくとも1 familyが全条件を通れば別の単一変更介入実験を設計可能とする。全family不通過ならformation attribution枝を閉じる。PASSはexp287/exp334の昇格、補正実装、推論、提出を意味しない。

## 実行・コスト制約

- risk family: 6
- model variant / LightGBM config / trained fold / booster / control再学習: `0 / 0 / 0 / 0 / 0`
- Kaggle CPU、GPU/internet off、single worker、BLAS thread 1で実行する。
- implementationとcompact self-contained Notebook候補は承認済み。
- canonical Notebook採用とKaggle CPU Stage A/B runは承認済み。inferenceとsubmissionは禁止する。

## 禁止事項

- worst-well ID rule、oracle risk、true error由来gate
- 同一OOF上のthreshold/feature/grid最適化
- formation列削除、clip/shrink/weight/gate救済
- corrected OOFや新predictionの生成
- model/control再学習、inference、submission、guard緩和

## 受け入れ基準

- steeringのrequirements/design/tasklistがTODOなしで完了状態と一致している。
- experimentのconfig/README/SESSION_NOTES/result/metricsがKaggle run完了・branch closeで一致している。
- `experiment.route=ml_model`、model/config/fold/booster 0、`implementation_approved=true`、完了後run flag falseが明記されている。
- 入力成果物の正規SHA、Stage A/B境界、6 family、四分位、coverage、判定gate、禁止事項がconfigとdesignで一致している。
- `docs/06_reproducibility.md`に従い、RNGなし、canonical order、Stage A freeze SHA、Kaggle bootstrap確認方針が記録されている。
- `KAGGLE_DIRECTION.md`へexp336の完了結果を記録し、未着手候補から削除している。
- compact self-contained train候補にStage A/B境界と全11生成物契約が実装されている。
- compact self-contained inference候補がprediction/submissionをfail-closedで拒否する。
- synthetic testsで禁止列、family集約、freeze改ざん、四分位eligibility、coverage/fold/hidden gate、OOF整合、SHA同一copy resolverを検証する。
- Jupytext往復、py_compile、ruff、専用pytest、strict experiment validation、experiment docs reviewがPASSする。
- canonical採用とKaggle実行は明示承認後に行い、fixed gate、artifact SHA、kernel version/runtimeを記録する。inference/submissionを生成しない。
