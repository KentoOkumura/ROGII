# student_t_gaussian_disagreement_continuous_risk_feature_on_exp264

- 候補名: `student_t_gaussian_disagreement_continuous_risk_feature_on_exp264`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `student_t_gaussian_disagreement_continuous_risk_feature_on_exp264`: Gaussian--Student-t予測差、posterior std、log-likelihoodだけを連続値のadd-only downstream risk featureとして監査する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度と理由: 低・P4・0-booster先行・別の必要性/承認時のみ・未採番
- `KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](../../KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・0-booster先行・別の必要性/承認時のみ・未採番 | `student_t_gaussian_disagreement_continuous_risk_feature_on_exp264`: Gaussian--Student-t予測差、posterior std、log-likelihoodだけを連続値のadd-only downstream risk featureとして監査する | exp374は単独平均を改善したが、exp388 fixed13 hard selectorは親比`+0.083572 ft`、2/5 folds、p95`+0.910123 ft`でFAIL。exp493 fixed12全面置換もpooled`-0.036294555 ft`、全固定scope改善、Student-t family top1`36.281580%`まで回復した一方、3/5 folds、p95`+0.540095855 ft`、worst`+10.472288433 ft`でFAILした。candidate数を戻してもhard replacement/additionのwell-tail不安定性は解消せず、この経路は閉じた。独立したfeature-only再開理由とユーザー承認がある場合だけ着手する | まず0 model / 0 boosterで保存済みGaussian/Student-t OOFからfeature availability、finite、fold別分布、既存exp264 featureとの相関、fixed12 error/riskへのfold transferだけを事前固定してreadoutする。4/5 folds、hidden-like 2面、by-well tail方向へtransferする単一feature familyを凍結できた場合だけ、さらに別承認でadd-only downstream ML実験を設計する | 13本目以降へのStudent-t / Huber TVT追加、hard replacement再開、selector weight/threshold/domain/gate救済、true error/oracle/worst-well IDの入力化、同一OOFでfeature/transformを選ぶこと、親再学習、current-test生成、inference、submitは禁止。自動優先しない |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `student_t_gaussian_disagreement_continuous_risk_feature_on_exp264`: Gaussian--Student-t予測差、posterior std、log-likelihoodだけを連続値のadd-only downstream risk featureとして監査する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp374は単独平均を改善したが、exp388 fixed13 hard selectorは親比`+0.083572 ft`、2/5 folds、p95`+0.910123 ft`でFAIL。exp493 fixed12全面置換もpooled`-0.036294555 ft`、全固定scope改善、Student-t family top1`36.281580%`まで回復した一方、3/5 folds、p95`+0.540095855 ft`、worst`+10.472288433 ft`でFAILした。candidate数を戻してもhard replacement/additionのwell-tail不安定性は解消せず、この経路は閉じた。独立したfeature-only再開理由とユーザー承認がある場合だけ着手する
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

まず0 model / 0 boosterで保存済みGaussian/Student-t OOFからfeature availability、finite、fold別分布、既存exp264 featureとの相関、fixed12 error/riskへのfold transferだけを事前固定してreadoutする。4/5 folds、hidden-like 2面、by-well tail方向へtransferする単一feature familyを凍結できた場合だけ、さらに別承認でadd-only downstream ML実験を設計する

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

13本目以降へのStudent-t / Huber TVT追加、hard replacement再開、selector weight/threshold/domain/gate救済、true error/oracle/worst-well IDの入力化、同一OOFでfeature/transformを選ぶこと、親再学習、current-test生成、inference、submitは禁止。自動優先しない

## リスク

移行前の「注意点」を原文のまま上に保持しています。leakage、hidden test、runtime、memory、再現性の分類は未整理です。

## 未決事項

- 観測事実と仮定の分離
- 根拠ファイルと保存済み生成物のパスおよびSHA
- input、target / objective、output、loss、decode、処理単位
- 親実験から変更するものと固定するもの
- 成功条件と停止条件
- 実装区分

## 判断履歴

- 2026-08-12: 移行前の内容を変更せず個別ファイルへ移した。未整理項目を推測で補わないため、状態を `検討メモ・設計不可` とした。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: いいえ
- 変更するものを一意に説明できる: いいえ
- 最小検証と停止条件を一意に説明できる: いいえ
- 実行しないことを一意に説明できる: 一部のみ
- 未決事項が明示されている: はい
