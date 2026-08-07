# exp336_exp287_formation_tail_attribution_readout

## 状態

- ルート: `ml_model`の診断readout
- 状態: Kaggle CPU version 2完了、6/6 family FAIL、formation attribution枝close
- CV / Public LB / Private LB: なし / なし / なし
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- GPU: 不要
- 作成日: 2026-07-22
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`
- 比較対象: corrected `exp264_exp263_candidate_confidence_dual_selector`
- 再開根拠: `exp334_equal_well_loss_weighting_on_exp287`のtail guard FAIL

## 仮説

exp287のwell-level tail悪化がfold-safe formation referenceの信頼性不足に由来するなら、誤差を見る前に固定したformation距離・不確実性・plane/dense不一致・formation spread・known-prefix calibrationのいずれかで、高risk wellほどexp287のexp264比RMSE deltaが一貫して悪化する。

exp334はwell均等lossでglobalとtailの一部を改善したが、by-well p95、worst-well、`+3/+5 ft`悪化well数を回復できなかった。このため、row数比例lossだけではなくformation品質を原因候補として診断する。

## 変更点

予測やモデルは変更しない。保存済みexp287 outer-valid formation cacheから、次の6 risk familyをtarget-freeにwell集約する。

1. plane reference距離
2. dense reference距離
3. dense近傍不確実性
4. plane–dense予測不一致
5. formation間spread
6. known-prefix formation calibration error

Stage Aで属性値と四分位境界を凍結し、content SHAを保存した後だけ、Stage Bでexp287/exp264 OOFの真値・予測を読み込む。exp334 OOFは主比較へ混ぜない。

## 検証方針

- split: exp287と同じ5 outer folds、group=`well`
- score rows: `TVT_input.isna()`の3,783,989行、773 wells
- Stage A: exp287の5 outer-valid formation cacheからtarget-free well属性と四分位を凍結
- Stage B: SHA固定したexp287/corrected-exp264 OOFをjoinし、well別RMSE deltaを評価
- hidden-like: spatial / typewell-purgedの固定200-well面
- leakage check: Stage A禁止列audit、freeze manifest SHA、Stage BのID/well/fold/actual照合

## 固定判定

各familyについて高risk Q4と低risk Q1のwell-level `RMSE(exp287)-RMSE(exp264)`を比較する。family PASSは次のAND条件とする。

- globalのQ4−Q1 mean deltaが`+0.25 ft`以上
- globalのQ4−Q1 median deltaが正
- 5 folds中4 folds以上でQ4−Q1が正
- hidden-like spatialとtypewell-purgedの両方でQ4−Q1が正
- global/fold/hidden-likeで固定したminimum coverageを満たす
- 属性、方向、境界がerror非依存

6 familyのいずれも全条件を通らなければformation attribution枝を閉じる。PASSしてもexp287をtrain-side昇格せず、別の単一変更実験を設計できる資格だけを与える。

## Leakageと禁止事項

- Stage Aでは`TVT`、target、actual、prediction、error、worst-well IDを読み込まない。
- raw contextは`MD/X/Y/Z/TVT_input`だけを許可する。
- 同一OOFでのthreshold、feature、weight、clip、shrink、gate探索は禁止する。
- formation列削除、救済train、corrected OOF生成、inference、submissionは行わない。

## 実行入口

Jupytext percent形式のcompact self-contained train候補へStage A/B readoutを実装し、inferenceはprediction/submissionを必ず拒否するfail-closed構成にした。ユーザーの実行承認後、両候補をcanonical Notebookへ採用した。Kaggle CPU train packageを実行入口とし、inference packageは作成・実行しない。

## 成果物

- target-free well属性とfreeze manifest
- exp287/exp264 well別OOF delta
- family四分位、fold方向、hidden-like方向の各metrics
- technical/context readout
- attribution decisionと再現性manifest

## 結果

固定判定は`NO_STABLE_FORMATION_ATTRIBUTION_CLOSE`、通過familyは`0/6`だった。全familyでstrict edge、error非依存、global/fold/hidden-like coverageは通ったが、効果量と方向再現性のANDを満たさなかった。

最も近かった`dense_reference_distance`はglobal Q4−Q1 mean`+0.350471 ft`、median正、5/5 folds正だった一方、hidden-like spatial / typewell-purgedが`-0.019368 / -0.037255 ft`と逆方向だった。formation consensus spreadとknown-prefix calibration errorはhidden-like 2面を通ったがglobal効果量不足だった。

Kaggle private CPU kernel `kentookumura/exp336-exp287-formtail-attribution-readout-train` version 2（id_no `128221753`）をreadout本体`92.458 sec`で完了した。Stage A freeze manifest SHAは`e65a9924c11f77008d1574070f71b6cf2d099993e8510eeaf7cc285c5d54979f`。11成果物の存在と、reproducibility manifestが記録する10子artifactのSHA一致を確認した。

## 所見

exp287はCVとPublic LBの方向がexp264比で整合する一方、少数wellの重い回帰を残した。exp334でloss寄与をwell均等化してもsevere tailが残ったため、追加GPU trainより先にformation品質の安定した事前riskがあるかを0-boosterで反証する価値がある。ただし本readoutは関連を診断するだけで、因果や補正方法を確定しない。

## 次

事前契約どおり同じOOFでのfamily/threshold救済を行わず、この枝を閉じる。inference、prediction correction、submissionは行わない。
