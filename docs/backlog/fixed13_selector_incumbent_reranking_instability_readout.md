# fixed13_selector_incumbent_reranking_instability_readout

- 候補名: `fixed13_selector_incumbent_reranking_instability_readout`
- 状態: `検討メモ・設計不可`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `fixed13_selector_incumbent_reranking_instability_readout`: 13候補selector再学習が追加候補を選ばない行でも既存12候補の順位を変え、候補横断で同じwell tailを壊すかだけを診断する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度と理由: 低・P4・0-booster・fixed13原因確認時のみ・未採番
- `KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](../../KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・0-booster・fixed13原因確認時のみ・未採番 | `fixed13_selector_incumbent_reranking_instability_readout`: 13候補selector再学習が追加候補を選ばない行でも既存12候補の順位を変え、候補横断で同じwell tailを壊すかだけを診断する | 固定exp264 / exp371 / exp373 / exp375 / exp388 / exp392 / exp496 / exp501 outer-valid candidate-scoreとhard choice SHAを入力にする。exp373/375は異なる追加候補でも同じ`b19b0395`を約`+29 ft`悪化させ、exp392ではworst `8902c3f6`がHuber利用0%でも`+7.875188 ft`悪化した。exp496はpooled`-0.191174 ft`、全7 scope改善でもp95`+1.109360 ft`、worst`+9.361781 ft`。exp501はpooled`-0.387642 ft`、5/5 folds、全7 scope改善でもp95`+2.904594 ft`、worst`+18.394664 ft`、追加候補非top1行のincumbent change率`35.007153%`だった。direct usageだけでは説明しにくい根拠がabsolute-geometryとmean-reverting HMMへ拡張した | 0 model / 0 booster。truth join前に「追加候補top1行」「追加候補非top1かつincumbent choice変更行」「incumbent choice不変行」、score margin/entropy/rank shiftを固定し、truth join後に7 fixed13 runで同じtail failureがfold・hidden-like・well bootstrapへtransferするかだけをreadoutする。再現しなければfixed13原因追跡を終了する | threshold/weight/domain/gate/feature/modelの救済、worst-well ID rule、true error/oracleによる分類、同一OOF最適化、selector再学習、current-test生成、downstream TVT、inference、submitは禁止。着手は独立した必要性とユーザー確認がある場合だけ |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `fixed13_selector_incumbent_reranking_instability_readout`: 13候補selector再学習が追加候補を選ばない行でも既存12候補の順位を変え、候補横断で同じwell tailを壊すかだけを診断する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: 固定exp264 / exp371 / exp373 / exp375 / exp388 / exp392 / exp496 / exp501 outer-valid candidate-scoreとhard choice SHAを入力にする。exp373/375は異なる追加候補でも同じ`b19b0395`を約`+29 ft`悪化させ、exp392ではworst `8902c3f6`がHuber利用0%でも`+7.875188 ft`悪化した。exp496はpooled`-0.191174 ft`、全7 scope改善でもp95`+1.109360 ft`、worst`+9.361781 ft`。exp501はpooled`-0.387642 ft`、5/5 folds、全7 scope改善でもp95`+2.904594 ft`、worst`+18.394664 ft`、追加候補非top1行のincumbent change率`35.007153%`だった。direct usageだけでは説明しにくい根拠がabsolute-geometryとmean-reverting HMMへ拡張した
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

0 model / 0 booster。truth join前に「追加候補top1行」「追加候補非top1かつincumbent choice変更行」「incumbent choice不変行」、score margin/entropy/rank shiftを固定し、truth join後に7 fixed13 runで同じtail failureがfold・hidden-like・well bootstrapへtransferするかだけをreadoutする。再現しなければfixed13原因追跡を終了する

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

threshold/weight/domain/gate/feature/modelの救済、worst-well ID rule、true error/oracleによる分類、同一OOF最適化、selector再学習、current-test生成、downstream TVT、inference、submitは禁止。着手は独立した必要性とユーザー確認がある場合だけ

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
