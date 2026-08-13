# five_surface_global_stratigraphic_potential_contract

- 候補名: `five_surface_global_stratigraphic_potential_contract`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `backlog/KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `five_surface_global_stratigraphic_potential_contract`: exp436でsource supportが十分だったANCC / ASTNU / ASTNL / EGFDU / EGFDLの5面だけを、target/truthを見る前に固定したglobal potential contractとして再設計できるか確認する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度: P3
- 優先度の理由: target-free fixed-five design・別実験/別承認待ち
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低-中・P3・target-free fixed-five design・別実験/別承認待ち | `five_surface_global_stratigraphic_potential_contract`: exp436でsource supportが十分だったANCC / ASTNU / ASTNL / EGFDU / EGFDLの5面だけを、target/truthを見る前に固定したglobal potential contractとして再設計できるか確認する | exp436 v2のcontact censusでは5面が全fold 555–618 wells、BUDAだけ4–6 wellsで、6面contractはStage 0 FAIL。exp436のFAIL、exp381 absolute contact-TVT FAIL、exp273 prefix-plane negative、exp383 resource FAILを再分類しない。BUDA除外はtarget-free source censusだけを根拠に固定し、別実験の`requirements.md`への契約記録・別実行承認を必須とする | まず0 predictionのdesign auditで5面固定、全5面support、fold-safe graph、query coverage gate、Stage 1 rolling-origin、truth-late順序を一意に固定する。実行する場合も新しいStage 0で5面solveとtarget-free query coverageだけを先行し、PASS後の別承認までtruth/CVへ進まない | exp436内のformation除外、BUDA contact定義/threshold救済、4面subsetやrow-wise面切替、weight/grid、fallback/blend/selector、target formation/GR、same-OOF rescueは禁止。既存P1/P2候補を追い越さない |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `five_surface_global_stratigraphic_potential_contract`: exp436でsource supportが十分だったANCC / ASTNU / ASTNL / EGFDU / EGFDLの5面だけを、target/truthを見る前に固定したglobal potential contractとして再設計できるか確認する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp436 v2のcontact censusでは5面が全fold 555–618 wells、BUDAだけ4–6 wellsで、6面contractはStage 0 FAIL。exp436のFAIL、exp381 absolute contact-TVT FAIL、exp273 prefix-plane negative、exp383 resource FAILを再分類しない。BUDA除外はtarget-free source censusだけを根拠に固定し、別実験の`requirements.md`への契約記録・別実行承認を必須とする
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

まず0 predictionのdesign auditで5面固定、全5面support、fold-safe graph、query coverage gate、Stage 1 rolling-origin、truth-late順序を一意に固定する。実行する場合も新しいStage 0で5面solveとtarget-free query coverageだけを先行し、PASS後の別承認までtruth/CVへ進まない

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

exp436内のformation除外、BUDA contact定義/threshold救済、4面subsetやrow-wise面切替、weight/grid、fallback/blend/selector、target formation/GR、same-OOF rescueは禁止。既存P1/P2候補を追い越さない

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
