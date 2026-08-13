# exp490_position_rate_mean_reversion_factorial

- 候補名: `exp490_position_rate_mean_reversion_factorial`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `backlog/KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `exp490_position_rate_mean_reversion_factorial`: exp490で同時変更したoffset centerとrate centerをposition-only / rate-onlyへ分離し、pooled gainとtail harmの因果成分を識別する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度: P3
- 優先度の理由: CPU fixed32・2 scientific variants別承認
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容へ大会後の運用条件だけを反映したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低-中・P3・CPU fixed32・2 scientific variants別承認 | `exp490_position_rate_mean_reversion_factorial`: exp490で同時変更したoffset centerとrate centerをposition-only / rate-onlyへ分離し、pooled gainとtail harmの因果成分を識別する | exp424/441/411/412はrate dynamics単独介入のnegative contextだが、exp490のK16 geometry-centered coordinateでoffset/rate寄与は未識別。複数variant数とCPUコストを実行前に明示承認し、保存exp357をcontrolにする | fixed32で2 candidates×32=64 HMM well-runs、親rerun0。persistent episode SSE、recovery、under-response、matched-control pooled/p95、fold方向、runtimeを各variant独立判定しwinnerを選ばない。full773は独立PASS candidateへの別承認のみ | half-life/sigma/momentum/grid/emission/gate、position+rate再現、blend/selector、same-fixed32 winner救済は禁止。大会後の原因分析に限定し、inference/submissionは行わない。P2 readout/add-onlyより後に扱う |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `exp490_position_rate_mean_reversion_factorial`: exp490で同時変更したoffset centerとrate centerをposition-only / rate-onlyへ分離し、pooled gainとtail harmの因果成分を識別する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp424/441/411/412はrate dynamics単独介入のnegative contextだが、exp490のK16 geometry-centered coordinateでoffset/rate寄与は未識別。複数variant数とCPUコストを実行前に明示承認し、保存exp357をcontrolにする
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

fixed32で2 candidates×32=64 HMM well-runs、親rerun0。persistent episode SSE、recovery、under-response、matched-control pooled/p95、fold方向、runtimeを各variant独立判定しwinnerを選ばない。full773は独立PASS candidateへの別承認のみ

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

half-life/sigma/momentum/grid/emission/gate、position+rate再現、blend/selector、same-fixed32 winner救済は禁止。大会後の原因分析に限定し、inference/submissionは行わない。P2 readout/add-onlyより後に扱う

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
