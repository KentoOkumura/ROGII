# raw_test_safe_ranker_worst_well_attribution_readout

- 候補名: `raw_test_safe_ranker_worst_well_attribution_readout`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `未整理`
- 作成日: 2026-08-12
- 最終更新日: 2026-08-12
- 依頼原文: 旧 `backlog/KAGGLE_DIRECTION.md` の未着手表にあった記録を、内容を補完せず個別ファイルへ移行する。
- 期待する成果: `raw_test_safe_ranker_worst_well_attribution_readout`: exp251 130列版と295列版で共通して悪化した`fb03ae90`と、exp259の最大差分回帰`aed44918`のcandidate-family偏りを保存済み予測で分解する
- 親実験 / 比較対象: 未整理。下記の「先行条件 / 依存」の原文を参照する。
- 優先度: P3
- 優先度の理由: 再訪前readout
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [未着手バックログ](KAGGLE_DIRECTION.md#未着手バックログ)

## 移行前の記録

次の5項目は、移行前の索引に記録されていた内容を変更せず転記したものです。

| 優先度 | アイデア | 先行条件 / 依存 | 検証方法 | 注意点 |
| --- | --- | --- | --- | --- |
| 中・再訪前readout | `raw_test_safe_ranker_worst_well_attribution_readout`: exp251 130列版と295列版で共通して悪化した`fb03ae90`と、exp259の最大差分回帰`aed44918`のcandidate-family偏りを保存済み予測で分解する | exp259はexp251 v4比でoverall/1000+を改善したがhidden-like 2面と最大well回帰がFAILし、`fb03ae90`も+0.017093悪化した。exp248 original-only / exp251 v4 / exp259 OOFを固定入力にできるが、branch採用ではなくlongtail-only再訪可否を判断する0-booster readoutに限定する | well×candidate familyの選択率・連続segment・誤差delta、regenerated `copcf_*` risk bucket、1000+比率をreadoutし、改善386 / 悪化384 wellsのtarget-free属性差が5 outer foldsとhidden-likeで再現するか確認する | 旧「削除167列」を原因群として扱わない。学習、target/error/oracle gate、worst-well ID rule、guard緩和、candidate追加、Viterbi grid、raw-test inference、submitは禁止。target-freeな事前ruleが得られなければ`longtail_only_exact_datum_augmentation`も閉じる |

## 観測事実と仮定の整理状態

移行前の記録では、観測事実、そこからの解釈、実装上の仮定が独立した項目に分かれていません。推測による再分類は行っていません。

## 仮説と期待される観測

- 移行前のアイデア: `raw_test_safe_ranker_worst_well_attribution_readout`: exp251 130列版と295列版で共通して悪化した`fb03ae90`と、exp259の最大差分回帰`aed44918`のcandidate-family偏りを保存済み予測で分解する
- 仮説が正しい場合に期待する観測: 未整理
- 仮説を棄却する観測: 未整理

## 入力・予測対象・出力・推論方法

未整理です。設計前に、移行前の「先行条件 / 依存」と「検証方法」を根拠ファイルと照合して確定します。

## 親実験からの差分

- 先行条件 / 依存: exp259はexp251 v4比でoverall/1000+を改善したがhidden-like 2面と最大well回帰がFAILし、`fb03ae90`も+0.017093悪化した。exp248 original-only / exp251 v4 / exp259 OOFを固定入力にできるが、branch採用ではなくlongtail-only再訪可否を判断する0-booster readoutに限定する
- 変更するもの: 未整理
- 固定するもの: 未整理
- 再利用するコード / config / 生成物: 未整理
- 新しく作るもの: 未整理

## 最小の反証可能な検証

well×candidate familyの選択率・連続segment・誤差delta、regenerated `copcf_*` risk bucket、1000+比率をreadoutし、改善386 / 悪化384 wellsのtarget-free属性差が5 outer foldsとhidden-likeで再現するか確認する

## 成功条件と停止条件

未整理です。移行前の検証方法に数値条件が含まれる場合も、根拠と評価対象を確認するまでは設計契約として扱いません。

## 実行しないこと

旧「削除167列」を原因群として扱わない。学習、target/error/oracle gate、worst-well ID rule、guard緩和、candidate追加、Viterbi grid、raw-test inference、submitは禁止。target-freeな事前ruleが得られなければ`longtail_only_exact_datum_augmentation`も閉じる

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
