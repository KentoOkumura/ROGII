# 要件

## 依頼

「次に試す価値が高い対応」として、donor-support risk readoutからbounded weight shrinkへ進む実験を設計確定する。実装は行わない。

## 2026-07-21 追加依頼

ユーザーの「exp329を実装してください」をStage 0実装の明示承認として扱う。Stage 1、Kaggle push、実行、inference、submissionは従来どおり別承認まで行わない。

## 制約

- Routeは`pf_beam`、親は保存済みexp263固定式、anchorは保存済みexp226とする。
- donor supportは危険度だけに使い、近傍wellの誤差方向や補正量を転送しない。
- exp226 source foldのvalidation wellsをdonorから完全除外する。
- Stage 0を先に固定し、全gate PASSと別承認なしにStage 1へ進まない。
- Stage 1は最大25% shrink、5 ft cap、250 ft未満vetoの1候補だけとする。
- threshold、alpha、cap、destination、feature、blockのgrid救済を禁止する。
- 親再学習、primitive予測再生成、HMM/PF/Beam再実行、inference、submissionを行わない。

## 受け入れ基準

- 6 featureの定義、outer-train percentile、composite risk、controlが一意に定義される。
- Stage 0/1の式、成功条件、停止条件、実行量、承認境界が一意に記録される。
- exp303/322およびHMM varianceを変更するexp324と重複しない。
- Stage 0 implementationだけを有効化し、Kaggle実行、Stage 1、inference、submissionは無効のままにする。
