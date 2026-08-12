# exp497_public_core_tail_correlation_failure_attribution_readout

- 候補名: `exp497_public_core_tail_correlation_failure_attribution_readout`
- 状態: `検討メモ・設計不可`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `exp497_public_core_tail_correlation_failure_attribution_readout`: exp497の小幅pooled改善とfold 0/4・hidden-like・well-tail悪化が同じpublic-core誤差相関regimeへ集中したかを説明する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度と理由: 低・P4・saved-artifact-only・0-model・tail/correlation原因分解
- `KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](../../KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・saved-artifact-only・0-model・tail/correlation原因分解 | `exp497_public_core_tail_correlation_failure_attribution_readout`: exp497の小幅pooled改善とfold 0/4・hidden-like・well-tail悪化が同じpublic-core誤差相関regimeへ集中したかを説明する | exp497 Stage E v1はtechnical契約、200 model count、exp413再学習0、5 positive weightsをPASSしたが、pooled gain`0.010314644 ft`、3/5 folds、hidden-like delta`+0.105138/+0.097410 ft`、p95/worst`+0.700720/+7.541588 ft`でterminal close。component OOF、strict public-core、exp413、fold/scope/by-well、meta5 weightとSHAを固定入力にする | 新規model / booster / prediction / PF/HMM/Beam 0。truth join前にpublic-coreとexp413のdisagreement、error covariance、public-core deviation、suffix horizon、visible-prefix supportの少数bucketを固定し、fold 0/4・hidden-like・tail harmが同一regimeへ集中するかだけをtruth-lateで記述する | exp497 gateを再評価しない。weight/bound/scope/threshold、well/row selector、family subset、rerun、same-OOF rescue、inference、submissionは禁止。原因説明の必要性と別承認があるまでP4とし、現行P0/P1を追い越さない |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `exp497_public_core_tail_correlation_failure_attribution_readout`: exp497の小幅pooled改善とfold 0/4・hidden-like・well-tail悪化が同じpublic-core誤差相関regimeへ集中したかを説明する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp497 Stage E v1はtechnical契約、200 model count、exp413再学習0、5 positive weightsをPASSしたが、pooled gain`0.010314644 ft`、3/5 folds、hidden-like delta`+0.105138/+0.097410 ft`、p95/worst`+0.700720/+7.541588 ft`でterminal close。component OOF、strict public-core、exp413、fold/scope/by-well、meta5 weightとSHAを固定入力にする
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

新規model / booster / prediction / PF/HMM/Beam 0。truth join前にpublic-coreとexp413のdisagreement、error covariance、public-core deviation、suffix horizon、visible-prefix supportの少数bucketを固定し、fold 0/4・hidden-like・tail harmが同一regimeへ集中するかだけをtruth-lateで記述する

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

exp497 gateを再評価しない。weight/bound/scope/threshold、well/row selector、family subset、rerun、same-OOF rescue、inference、submissionは禁止。原因説明の必要性と別承認があるまでP4とし、現行P0/P1を追い越さない

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
