# h512_block_rank_tail_failure_attribution_readout

- 候補名: `h512_block_rank_tail_failure_attribution_readout`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `h512_block_rank_tail_failure_attribution_readout`: exp504のpooled gainとfold / hidden-like / by-well tail FAILが、どのtarget-free block regimeとhard-choice挙動に集中したかだけを保存生成物で分解する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度: P4
- 優先度の理由: saved-artifact-only・0-model・block-rank tail原因分解
- `KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](../../KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・saved-artifact-only・0-model・block-rank tail原因分解 | `h512_block_rank_tail_failure_attribution_readout`: exp504のpooled gainとfold / hidden-like / by-well tail FAILが、どのtarget-free block regimeとhard-choice挙動に集中したかだけを保存生成物で分解する | exp504 v1はtechnical全PASS、pooled`-0.124055 ft`だがnonworse`3/5`、hidden-like`+0.285759/+0.269833 ft`、p95/worst`+2.963656/+16.799044 ft`でterminal close。pair accuracy`0.741908`に対しtop-1`0.112624`で、pair判別から安全なwinner選択への転移失敗を説明する余地だけが残る | 新規model / booster / prediction / selector 0。保存block selection、pair probability、rank、by-well、feature importance、OOF SHAを固定し、truth join前にanchor fallback、Borda margin/entropy、candidate family、suffix horizon、GR missingness、candidate disagreementの少数bucketを固定する。fold 3/4・hidden-like・well-tailで同じharm concentrationが再現するかだけをreadoutし、次の独立仮説の根拠に限定する | exp504 FAILを再分類しない。H128/H256、loss/weight/model/threshold/guard、candidate、smooth/blend/gate、worst-well ID rule、同一OOF最適化、selector再学習、current-test、inference、submissionは禁止。別steering・別承認があるまでP4とし、現行P1/P2/P3を追い越さない |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `h512_block_rank_tail_failure_attribution_readout`: exp504のpooled gainとfold / hidden-like / by-well tail FAILが、どのtarget-free block regimeとhard-choice挙動に集中したかだけを保存生成物で分解する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp504 v1はtechnical全PASS、pooled`-0.124055 ft`だがnonworse`3/5`、hidden-like`+0.285759/+0.269833 ft`、p95/worst`+2.963656/+16.799044 ft`でterminal close。pair accuracy`0.741908`に対しtop-1`0.112624`で、pair判別から安全なwinner選択への転移失敗を説明する余地だけが残る
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

新規model / booster / prediction / selector 0。保存block selection、pair probability、rank、by-well、feature importance、OOF SHAを固定し、truth join前にanchor fallback、Borda margin/entropy、candidate family、suffix horizon、GR missingness、candidate disagreementの少数bucketを固定する。fold 3/4・hidden-like・well-tailで同じharm concentrationが再現するかだけをreadoutし、次の独立仮説の根拠に限定する

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

exp504 FAILを再分類しない。H128/H256、loss/weight/model/threshold/guard、candidate、smooth/blend/gate、worst-well ID rule、同一OOF最適化、selector再学習、current-test、inference、submissionは禁止。別steering・別承認があるまでP4とし、現行P1/P2/P3を追い越さない

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
