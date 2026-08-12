# 設計

## 要件の参照と承認済み差分

- 要件と手法契約: [`requirements.md`](requirements.md)
- `requirements.md`の未決事項が`なし`であることの確認: TODO
- 要件から変更する実装とユーザー承認: TODO (`N/A` または差分、承認日時 / 依頼メッセージ。契約本文はここへ複製しない)

## アプローチ

TODO

## 手法契約の実装対応

- inputの実装箇所と変換: TODO
- target / objectiveの構築箇所: TODO
- outputの生成箇所と表現: TODO
- lossの実装箇所: TODO
- decode / postprocessの実装箇所: TODO
- context unitを保つ処理箇所: TODO
- 参照sourceとの一致を確認するテスト: TODO
- 承認済み差分を確認するテスト: TODO
- 実験名と実装機構の整合: TODO

## 探索幅とpivot判定

変更classは`docs/glossary.md`に定義したこのリポジトリ内の管理用ラベルを使う。実際に変更する処理を先に具体的に記録する。

- 変更class: TODO (`parameter`、`add-only`、`selector-only`、`postprocess`、`mechanism`、`representation`)
- 同じ親 / familyで連続した小改善実験数: TODO
- positiveなoracle headroom / coverage / 誤差非相関性: TODO
- 比較したtarget、output、decode、context unitを変える案: TODO
- 小改善の継続またはpivotを選ぶ根拠: TODO
- `kaggle-idea-forge` の実行要否と根拠: TODO

## 実装範囲

- 対象実験: TODO
- 変更するファイル / component: TODO
- `requirements.md`の固定事項を保つ確認方法: TODO

## 再現性設計

- seed policy: TODO (`docs/06_reproducibility.md` を参照)
- stochastic 処理の有無: TODO
- PF/Beam / likelihood-PF / seed bagging の有無: TODO
- 並列処理と乱数の関係: TODO
- CPU/GPU runtime と deterministic flags: TODO
- train cache / test feature regeneration の SHA 記録方針: TODO
- model manifest / prediction / submission SHA 記録方針: TODO
- Kaggle package bootstrap 確認方針: TODO

## リスク

- リークリスク: TODO
- CV/LB 不一致リスク: TODO
- ランタイム/メモリリスク: TODO
- 再現性リスク: TODO
- 手法忠実性リスク: TODO
- 過度な縮小 / proxy化リスク: TODO
