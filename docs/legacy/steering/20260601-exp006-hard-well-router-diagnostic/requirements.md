# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログで最優先の `exp006_hard_well_router_diagnostic` を実装する。

## 制約

- 予測 model 自体は変更せず、exp002 / exp003 / exp005 の OOF well-level 差分診断に限定する。
- router の candidate rule は hidden test で利用できる well 条件だけを入力にする。
- OOF target outcome はタグ付けと評価には使ってよいが、将来の router 入力には混ぜない。
- train-only formation columns は直接使わない。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。

## 受け入れ基準

- `experiments/exp006_hard_well_router_diagnostic/` に実験一式がある。
- train notebook が OOF `well_metrics.csv` から router diagnostic artifacts を出力する。
- 既存の保存済み exp003/exp005 artifacts から同じ診断を再生成できる CLI がある。
- `hard_no_gr_candidate` と `public_like_keep_all_gr` のタグが well-level CSV に保存される。
- inference-safe candidate rules の OOF CV と選択 well 数が CSV に保存される。
- `task validate-exp EXP=exp006_hard_well_router_diagnostic` が通る。
