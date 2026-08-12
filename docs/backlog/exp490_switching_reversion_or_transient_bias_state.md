# exp490_switching_reversion_or_transient_bias_state

- 候補名: `exp490_switching_reversion_or_transient_bias_state`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `exp490_switching_reversion_or_transient_bias_state`: no-reversion / mean-reversionをK16 segmentでsoft switchingするか、residual offsetをpersistent datumとmean-reverting transient driftへ分解する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度と理由: 低・P4・高リスクalgorithm research・P2 evidence依存
- `KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](../../KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・高リスクalgorithm research・P2 evidence依存 | `exp490_switching_reversion_or_transient_bias_state`: no-reversion / mean-reversionをK16 segmentでsoft switchingするか、residual offsetをpersistent datumとmean-reverting transient driftへ分解する | switching state-space / IMM / Dynamic Model Averagingが構造参照。exp242 fixed regime、exp440 ambiguity gate、exp500 fixed PF mean-reversionはnegative context。`exp490_k16_segment_model_evidence_transfer_readout`がfold-stableにPASSすることを必須にする | 先にsmall technical/fixed32でmodel evidence、state normalization、mode occupancy、truth-late、runtimeを確認。exact state explosionならRao-Blackwellized/parallel expert approximationを一方式だけ事前固定する | evidence FAIL時は実装しない。mode transition/forgetting/temperature/state/noise/half-lifeのgrid、hard switch、truth gate、same-OOF救済、baseline GPU再学習、直接submitは禁止 |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `exp490_switching_reversion_or_transient_bias_state`: no-reversion / mean-reversionをK16 segmentでsoft switchingするか、residual offsetをpersistent datumとmean-reverting transient driftへ分解する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: switching state-space / IMM / Dynamic Model Averagingが構造参照。exp242 fixed regime、exp440 ambiguity gate、exp500 fixed PF mean-reversionはnegative context。`exp490_k16_segment_model_evidence_transfer_readout`がfold-stableにPASSすることを必須にする
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

先にsmall technical/fixed32でmodel evidence、state normalization、mode occupancy、truth-late、runtimeを確認。exact state explosionならRao-Blackwellized/parallel expert approximationを一方式だけ事前固定する

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

evidence FAIL時は実装しない。mode transition/forgetting/temperature/state/noise/half-lifeのgrid、hard switch、truth gate、same-OOF救済、baseline GPU再学習、直接submitは禁止

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
