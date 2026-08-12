# exp516_stage24_vs_final_v96_contract_attribution_readout

- 候補名: `exp516_stage24_vs_final_v96_contract_attribution_readout`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `exp516_stage24_vs_final_v96_contract_attribution_readout`: 作者報告stage 2-4 standalone `twGR-prior PF alone`と、exp516が再現したfinal-v96 `pfA × twGR`の契約差だけを特定する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度: P4
- 優先度の理由: source/saved-output only・0-PF・0-submit
- `KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](../../KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 低・P4・source/saved-output only・0-PF・0-submit | `exp516_stage24_vs_final_v96_contract_attribution_readout`: 作者報告stage 2-4 standalone `twGR-prior PF alone`と、exp516が再現したfinal-v96 `pfA × twGR`の契約差だけを特定する | exp516 ref `55326266`はPublic `10.056` / Private `8.552`、作者報告は`7.88 / 7.78`。writeupはstage 2-2をfixed-lag 192として説明する一方、final v96はwhole smootherとlearned emissionを含む。両者の同一性は未証明 | 保存discussion、公開Notebook/config、exp516 manifest/log/submissionだけを入力にし、anchor、emission、smoother、PF params、decode、RNG/runtime、postprocessの`input -> target -> output -> loss -> decode -> context unit`差分表を作る。新規prediction/model/PF/GPU/submissionは0 | exp516のLBを見たparameter/seed/lag/emission/postprocess調整、final v96をstage 2-4と同一と仮定すること、91候補systemへの一般化、再提出は禁止。独立した原因説明の必要性とユーザー承認があるまで着手しない |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `exp516_stage24_vs_final_v96_contract_attribution_readout`: 作者報告stage 2-4 standalone `twGR-prior PF alone`と、exp516が再現したfinal-v96 `pfA × twGR`の契約差だけを特定する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp516 ref `55326266`はPublic `10.056` / Private `8.552`、作者報告は`7.88 / 7.78`。writeupはstage 2-2をfixed-lag 192として説明する一方、final v96はwhole smootherとlearned emissionを含む。両者の同一性は未証明
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

保存discussion、公開Notebook/config、exp516 manifest/log/submissionだけを入力にし、anchor、emission、smoother、PF params、decode、RNG/runtime、postprocessの`input -> target -> output -> loss -> decode -> context unit`差分表を作る。新規prediction/model/PF/GPU/submissionは0

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

exp516のLBを見たparameter/seed/lag/emission/postprocess調整、final v96をstage 2-4と同一と仮定すること、91候補systemへの一般化、再提出は禁止。独立した原因説明の必要性とユーザー承認があるまで着手しない

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
