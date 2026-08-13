# exp502_selector_transfer_scope_failure_attribution_readout

- 候補名: `exp502_selector_transfer_scope_failure_attribution_readout`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `backlog/KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `exp502_selector_transfer_scope_failure_attribution_readout`: exp501 compact77がselector-levelでは強く改善した一方、exp413 downstreamではfold 3 / 4とhidden-like 2面を悪化させた原因を保存exp502 OOFだけで分解する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度: P4
- 優先度の理由: saved-artifact-only・0-model・transfer原因分解
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・saved-artifact-only・0-model・transfer原因分解 | `exp502_selector_transfer_scope_failure_attribution_readout`: exp501 compact77がselector-levelでは強く改善した一方、exp413 downstreamではfold 3 / 4とhidden-like 2面を悪化させた原因を保存exp502 OOFだけで分解する | exp502 version 1はtechnical全PASS、pooled gain`0.002658891 ft`、fold 3 / 4 delta`+0.116026853 / +0.234685837 ft`、hidden-like delta`+0.139586563 / +0.140943998 ft`でterminal close。final373 feature/model/OOF SHAは固定済み | 新規model / booster / prediction / selector / HMM/PF/Beam 0。exp502とsaved exp413 OOF、fold/scope/by-well、selector score、final373 manifestをSHA固定し、selector correction magnitude、candidate disagreement、suffix horizon、visible-prefix supportの事前固定bucketでfold 3 / 4とhidden-like harm集中だけを記述する | exp501/502 FAILを再分類しない。feature subset、selector差替え、weight/blend/threshold/gate、well/row routing、same-OOF rescue、exp502 rerun、inference、submissionは禁止。独立した原因説明の必要性と別承認があるまでP4とし、現行P1/P2/P3を追い越さない |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `exp502_selector_transfer_scope_failure_attribution_readout`: exp501 compact77がselector-levelでは強く改善した一方、exp413 downstreamではfold 3 / 4とhidden-like 2面を悪化させた原因を保存exp502 OOFだけで分解する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp502 version 1はtechnical全PASS、pooled gain`0.002658891 ft`、fold 3 / 4 delta`+0.116026853 / +0.234685837 ft`、hidden-like delta`+0.139586563 / +0.140943998 ft`でterminal close。final373 feature/model/OOF SHAは固定済み
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

新規model / booster / prediction / selector / HMM/PF/Beam 0。exp502とsaved exp413 OOF、fold/scope/by-well、selector score、final373 manifestをSHA固定し、selector correction magnitude、candidate disagreement、suffix horizon、visible-prefix supportの事前固定bucketでfold 3 / 4とhidden-like harm集中だけを記述する

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

exp501/502 FAILを再分類しない。feature subset、selector差替え、weight/blend/threshold/gate、well/row routing、same-OOF rescue、exp502 rerun、inference、submissionは禁止。独立した原因説明の必要性と別承認があるまでP4とし、現行P1/P2/P3を追い越さない

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
