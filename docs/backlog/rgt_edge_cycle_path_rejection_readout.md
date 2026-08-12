# rgt_edge_cycle_path_rejection_readout

- 候補名: `rgt_edge_cycle_path_rejection_readout`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `rgt_edge_cycle_path_rejection_readout`: exp386の固定graph/path contractでcycle residualとroute全棄却の段階だけを分解する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度と理由: 低・P4・exp406 FAILで先行条件成立・0-prediction・別承認待ち
- `KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](../../KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・exp406 FAILで先行条件成立・0-prediction・別承認待ち | `rgt_edge_cycle_path_rejection_readout`: exp386の固定graph/path contractでcycle residualとroute全棄却の段階だけを分解する | exp386 v1はRGT coverage `0.989847`とleakage/resourceをPASSしたが、cycle residual p95 `2.363303`、graph query / bank / finite-path coverage 0でStage 0 FAIL。独立GR-firstのexp406もgraph query `0.451157`、finite row coverage `0.755026`、full投影`65,543.109 sec`でtechnical FAILしたため、GR graph救済ではなくFormation側の既存route棄却段階を分解する先行条件だけが成立した。既存P1/P2/P3候補を追い越さずP4を維持する | 着手する場合だけ、exp386の同じ5 graph solves / fixed16でreject countをtarget-free保存する。prediction、truth join、scenario採用は0。新しいsteeringで「どの保存生成物だけで足りるか」を先に確認し、不要なgraph再計算を避ける | exp386/exp406 FAILを再分類しない。exp406のGR edge/solverとの混在、neighbor/stretch/Huber/scenario-count/diversity/monotonicity閾値grid、target GR/Formation/truth、best rejection rescue、full run、exp387再開は禁止 |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `rgt_edge_cycle_path_rejection_readout`: exp386の固定graph/path contractでcycle residualとroute全棄却の段階だけを分解する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp386 v1はRGT coverage `0.989847`とleakage/resourceをPASSしたが、cycle residual p95 `2.363303`、graph query / bank / finite-path coverage 0でStage 0 FAIL。独立GR-firstのexp406もgraph query `0.451157`、finite row coverage `0.755026`、full投影`65,543.109 sec`でtechnical FAILしたため、GR graph救済ではなくFormation側の既存route棄却段階を分解する先行条件だけが成立した。既存P1/P2/P3候補を追い越さずP4を維持する
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

着手する場合だけ、exp386の同じ5 graph solves / fixed16でreject countをtarget-free保存する。prediction、truth join、scenario採用は0。新しいsteeringで「どの保存生成物だけで足りるか」を先に確認し、不要なgraph再計算を避ける

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

exp386/exp406 FAILを再分類しない。exp406のGR edge/solverとの混在、neighbor/stretch/Huber/scenario-count/diversity/monotonicity閾値grid、target GR/Formation/truth、best rejection rescue、full run、exp387再開は禁止

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
