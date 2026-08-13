# outer_train_selector_well_risk_discriminator

- 候補名: `outer_train_selector_well_risk_discriminator`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `backlog/KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: exp255 assertiveで悪化するwellをouter-trainだけから識別できたか、保存済みOOF上で大会後の原因分析として確認する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度: P3
- 優先度の理由: 大会後のOOF原因分析・再開条件付き
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](KAGGLE_DIRECTION.md#未着手バックログ)

## 現行の実行範囲

- 保存済みのexp255 OOF、exp238 nested score、outer-fold情報だけを使い、well単位の悪化識別可能性を大会後の原因分析として確認する。
- raw-test featureやpredictionの生成、推論候補化、Kaggle Notebook実行、submissionは、分析結果にかかわらず行わない。
- 下記の「移行前の記録」は履歴であり、現行の実行条件ではない。

## 移行前の記録（履歴）

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 中・再開条件付き | `outer_train_selector_well_risk_discriminator`: exp255 assertiveのglobal gainを維持しつつ、補正で悪化するwellをouter-trainだけから識別する | exp255固定assertive correctionとexp238 nested scoreを入力にする。selector gain/margin分布、candidate family share、方向consistency、prefix/GR qualityなどraw-test生成可能なwell aggregateだけを使い、outer-valid wellのtarget/errorはrisk modelのfit・threshold選択から除外する | outer foldごとにouter-train wellsだけで`delta_rmse > 0.25` riskを学習しouter-validへ適用する。exp238 fallbackを常に保持し、overall / near / 1000+ / hidden-like非悪化、3/5 folds改善、worst-well +0.25 ft以下を全通過した場合だけ推論候補化する | exp255同一OOFでthreshold/gridを選ばない。candidate値、alpha、clipを同時調整しない。hard top1、guard緩和、outer-valid target/error feature、通過前のraw-test inference / submitは禁止。exp255では106 wellsが+0.25 ft超悪化したため優先度は高ではなく再開条件付きとする |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `outer_train_selector_well_risk_discriminator`: exp255 assertiveのglobal gainを維持しつつ、補正で悪化するwellをouter-trainだけから識別する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp255固定assertive correctionとexp238 nested scoreを入力にする。selector gain/margin分布、candidate family share、方向consistency、prefix/GR qualityなどraw-test生成可能なwell aggregateだけを使い、outer-valid wellのtarget/errorはrisk modelのfit・threshold選択から除外する
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

保存済みOOFをouter foldごとに分け、outer-train wellsだけで`delta_rmse > 0.25` riskを学習してouter-validへ適用する。overall / near / 1000+ / hidden-like、fold、worst-wellを原因分析として記録し、通過しても推論候補化しない。

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

exp255同一OOFでthreshold/gridを選ばない。candidate値、alpha、clipを同時調整しない。hard top1、guard緩和、outer-valid target/error featureを使わない。分析条件を満たしてもraw-test inference、推論候補化、submissionへ進めない。

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
- 2026-08-12: 最終提出締切後の現行方針に合わせ、保存済みOOFによる原因分析だけに限定し、推論・提出への移行を禁止した。移行前の記録は履歴として残した。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: いいえ
- 変更するものを一意に説明できる: いいえ
- 最小検証と停止条件を一意に説明できる: いいえ
- 実行しないことを一意に説明できる: 一部のみ
- 未決事項が明示されている: はい
