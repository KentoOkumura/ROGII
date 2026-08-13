# segment_offset_selector_tail_failure_attribution_readout

- 候補名: `segment_offset_selector_tail_failure_attribution_readout`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `backlog/KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `segment_offset_selector_tail_failure_attribution_readout`: exp333のdirectおよびfixed13 selectorのnear/worst悪化がpredicted segment offsetの大きさ・符号、segment位置、well寄与偏りにfold横断で集中するかだけを診断する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度: P4
- 優先度の理由: 0-booster・exp371原因確認時のみ・未採番
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・0-booster・exp371原因確認時のみ・未採番 | `segment_offset_selector_tail_failure_attribution_readout`: exp333のdirectおよびfixed13 selectorのnear/worst悪化がpredicted segment offsetの大きさ・符号、segment位置、well寄与偏りにfold横断で集中するかだけを診断する | 固定exp333 OOF/segment prediction/model manifestとexp371 by-well/usage SHAを入力にし、予測・fold・featureを変更しない。exp333 directはexp228比near`+0.057439 ft`、worst`+8.099023 ft`。exp371 Stage Cはparent fixed12比pooled`-0.232535 ft`でもp95`+0.861529 ft`、worst`+10.757997 ft`、続くStage D add-onlyもpooled`-0.090815 ft`、3/5 folds、全scope改善に対してp95`+1.179312 ft`、worst`+4.637599 ft`だった。Stage C worst wellのexp333使用率は42.96%だが全wellのusage-delta相関は`-0.070004`で、usage単独gateは支持されない | 0 model / 0 booster。truth join前にoffset magnitude/sign/segment/distance/well-lengthとselector score/marginのbucketをfreezeし、directとfixed13の双方で悪化方向が4/5 folds、hidden-like、well bootstrapにtransferする単一familyがあるかだけをreadoutする。なければexp333 selector familyの原因追跡も終了する | 診断OOF上でthreshold、clip、shrink、taper、gate、feature、weightを選ばない。fixed13 / Stage D再学習、current-test prediction、推論、提出へ進まない。着手は独立した必要性とユーザー確認がある場合だけ |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `segment_offset_selector_tail_failure_attribution_readout`: exp333のdirectおよびfixed13 selectorのnear/worst悪化がpredicted segment offsetの大きさ・符号、segment位置、well寄与偏りにfold横断で集中するかだけを診断する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: 固定exp333 OOF/segment prediction/model manifestとexp371 by-well/usage SHAを入力にし、予測・fold・featureを変更しない。exp333 directはexp228比near`+0.057439 ft`、worst`+8.099023 ft`。exp371 Stage Cはparent fixed12比pooled`-0.232535 ft`でもp95`+0.861529 ft`、worst`+10.757997 ft`、続くStage D add-onlyもpooled`-0.090815 ft`、3/5 folds、全scope改善に対してp95`+1.179312 ft`、worst`+4.637599 ft`だった。Stage C worst wellのexp333使用率は42.96%だが全wellのusage-delta相関は`-0.070004`で、usage単独gateは支持されない
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

0 model / 0 booster。truth join前にoffset magnitude/sign/segment/distance/well-lengthとselector score/marginのbucketをfreezeし、directとfixed13の双方で悪化方向が4/5 folds、hidden-like、well bootstrapにtransferする単一familyがあるかだけをreadoutする。なければexp333 selector familyの原因追跡も終了する

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

診断OOF上でthreshold、clip、shrink、taper、gate、feature、weightを選ばない。fixed13 / Stage D再学習、current-test prediction、推論、提出へ進まない。着手は独立した必要性とユーザー確認がある場合だけ

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
