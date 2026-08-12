# global_sigma_pf_failure_regime_attribution_readout

- 候補名: `global_sigma_pf_failure_regime_attribution_readout`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `global_sigma_pf_failure_regime_attribution_readout`: 一律GR sigma緩和がhigh-missing / high-base-scale側で悪化した原因を、保存済みwell auditだけでmissingness・base scale・ESS/resampling・seed likelihood spreadへ分解する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度と理由: 低・P4・CPU・0-PF・output-only原因分解
- `KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](../../KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・CPU・0-PF・output-only原因分解 | `global_sigma_pf_failure_regime_attribution_readout`: 一律GR sigma緩和がhigh-missing / high-base-scale側で悪化した原因を、保存済みwell auditだけでmissingness・base scale・ESS/resampling・seed likelihood spreadへ分解する | exp400/404のx1.3 scientific FAILに加え、exp410 target-late counterfactualでもx1.3はepisode SSE`1.140067倍`、x3は11/16 episodesを改善しながらpooled`1.023661倍`・全suffix`1.309279倍`、GRほぼ無効は`8.835072倍`だった。GR効果は強く不均一だが通常は修正力で、global relaxation familyを再開する根拠はない | 新規PF・prediction・model 0。exp400/404 fixed well audit / by-well metricsとexp410保存paired readoutだけを使い、事前固定した少数regime指標が4/5 foldsで同方向かをreadoutする。FAILを再分類せず、目的は再利用可能な失敗原因の記録だけ。再現しなければ原因追跡も終了する | exp400/404/398/410のnegative判断を再分類しない。adaptive multiplier、well/row gate、threshold/grid、temperature/scale選択、parent再実行、PF/HMM/Beam/model、inference、submissionは禁止。既存P1/P2/P3候補を追い越さない |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `global_sigma_pf_failure_regime_attribution_readout`: 一律GR sigma緩和がhigh-missing / high-base-scale側で悪化した原因を、保存済みwell auditだけでmissingness・base scale・ESS/resampling・seed likelihood spreadへ分解する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp400/404のx1.3 scientific FAILに加え、exp410 target-late counterfactualでもx1.3はepisode SSE`1.140067倍`、x3は11/16 episodesを改善しながらpooled`1.023661倍`・全suffix`1.309279倍`、GRほぼ無効は`8.835072倍`だった。GR効果は強く不均一だが通常は修正力で、global relaxation familyを再開する根拠はない
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

新規PF・prediction・model 0。exp400/404 fixed well audit / by-well metricsとexp410保存paired readoutだけを使い、事前固定した少数regime指標が4/5 foldsで同方向かをreadoutする。FAILを再分類せず、目的は再利用可能な失敗原因の記録だけ。再現しなければ原因追跡も終了する

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

exp400/404/398/410のnegative判断を再分類しない。adaptive multiplier、well/row gate、threshold/grid、temperature/scale選択、parent再実行、PF/HMM/Beam/model、inference、submissionは禁止。既存P1/P2/P3候補を追い越さない

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
