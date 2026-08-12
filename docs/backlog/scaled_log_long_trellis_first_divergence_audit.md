# scaled_log_long_trellis_first_divergence_audit

- 候補名: `scaled_log_long_trellis_first_divergence_audit`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `scaled_log_long_trellis_first_divergence_audit`: exp458 v2と保存exp444のfixed4を使い、small denseでは見えなかった長系列の誤差増幅が最初に現れるrow/state/operatorをtarget-freeに特定する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度と理由: 低・P4・study-only
- `KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](../../KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・study-only | `scaled_log_long_trellis_first_divergence_audit`: exp458 v2と保存exp444のfixed4を使い、small denseでは見えなかった長系列の誤差増幅が最初に現れるrow/state/operatorをtarget-freeに特定する | exp458はruntime/RSSをPASSしたがmean/std/acceleration posterior parityをFAIL。新規HMM実行やcandidate生成は行わず、保存済みprediction/posterior/diagnostic/runtime manifestと既存sourceだけを使う | per-row scale、forward/backward message、acceleration/rate/position operator境界の差を再生可能なsynthetic long-trellisへ縮約し、固定1e-5/1e-7閾値を超える最初の地点を特定する。結果は原因説明だけで、exp458の再評価や昇格に使わない | exp458の再run、閾値緩和、precision/state/parameter/worker/thread/cache変更、別engine探索、Stage 0B/1、inference、submissionへの救済は禁止。既存P1/P2候補を追い越さない |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `scaled_log_long_trellis_first_divergence_audit`: exp458 v2と保存exp444のfixed4を使い、small denseでは見えなかった長系列の誤差増幅が最初に現れるrow/state/operatorをtarget-freeに特定する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp458はruntime/RSSをPASSしたがmean/std/acceleration posterior parityをFAIL。新規HMM実行やcandidate生成は行わず、保存済みprediction/posterior/diagnostic/runtime manifestと既存sourceだけを使う
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

per-row scale、forward/backward message、acceleration/rate/position operator境界の差を再生可能なsynthetic long-trellisへ縮約し、固定1e-5/1e-7閾値を超える最初の地点を特定する。結果は原因説明だけで、exp458の再評価や昇格に使わない

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

exp458の再run、閾値緩和、precision/state/parameter/worker/thread/cache変更、別engine探索、Stage 0B/1、inference、submissionへの救済は禁止。既存P1/P2候補を追い越さない

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
