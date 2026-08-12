# gr_path_negative_control_power_audit

- 候補名: `gr_path_negative_control_power_audit`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `gr_path_negative_control_power_audit`: fixed-path rankingで使うcircular null自体の検出力をtruth-freeに監査する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度と理由: 低・P4・CPU・0-HMM・design-only
- `KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](../../KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・CPU・0-HMM・design-only | `gr_path_negative_control_power_audit`: fixed-path rankingで使うcircular null自体の検出力をtruth-freeに監査する | exp364はreal-minus-circular top1 `0.003081`、fold`3/5`、exp367はcircular差`0.005576`、fold`2/5`でFAIL。exp366もtrigger AUC差`0.000003`、fold`0/5`、GR-selected branch gain`-1.005307 ft`でFAILした。各branchを再開しない | 将来別のpath-ranking preflightを設計する前だけ、候補pathとGR scoreをtruth前に固定し、block長と互いに非整合な十分離れたcircular nullを少数事前固定してnull間分散・real差・fold再現性を測る。nullが一貫してrealを破壊できる場合だけ、新しい独立仮説のcontrolとして採用する | exp364/367/366のFAILを再分類しない。curvature/reset state、path、block、shift、emissionの救済grid、truth/errorを見たnull選択、exact HMM/PF、予測、inference、submissionは禁止。実装・実行は別承認 |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `gr_path_negative_control_power_audit`: fixed-path rankingで使うcircular null自体の検出力をtruth-freeに監査する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp364はreal-minus-circular top1 `0.003081`、fold`3/5`、exp367はcircular差`0.005576`、fold`2/5`でFAIL。exp366もtrigger AUC差`0.000003`、fold`0/5`、GR-selected branch gain`-1.005307 ft`でFAILした。各branchを再開しない
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

将来別のpath-ranking preflightを設計する前だけ、候補pathとGR scoreをtruth前に固定し、block長と互いに非整合な十分離れたcircular nullを少数事前固定してnull間分散・real差・fold再現性を測る。nullが一貫してrealを破壊できる場合だけ、新しい独立仮説のcontrolとして採用する

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

exp364/367/366のFAILを再分類しない。curvature/reset state、path、block、shift、emissionの救済grid、truth/errorを見たnull選択、exact HMM/PF、予測、inference、submissionは禁止。実装・実行は別承認

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
