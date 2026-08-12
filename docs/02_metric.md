# 評価指標

このファイルは、公式メトリックをローカル実装と実験判断に落とし込むための運用メモです。実験テンプレートが読む metric 名は `project.yml` の `defaults.metric` を正とし、公式資料からの抜粋と出典は `docs/official/evaluation.md` に置きます。

## 公式メトリック要約

- 名前: Root Mean Squared Error (RMSE)
- 最適化方向: minimize
- 式: `sqrt(mean((y_true - y_pred)^2))`
- 実装の出典: Kaggle `Evaluation` ページ。取得日: 2026-05-27。
- 注意: Kaggle API の competition metadata は `evaluationMetric: Mean Squared Error` と返すが、公式 Evaluation ページ本文は RMSE と明記しているため、このリポジトリでは RMSE を採用する。

## ローカル実装

- 実装先: 実験の train notebook または実験固有の補助モジュールに `rmse(y_true, y_pred)` を置く。複数実験で共有する必要が生じた処理だけを `src/` に移す。
- 入力: evaluation zone の `TVT` 真値と、同じ `id` 順に並んだ予測 `tvt`。
- 出力: scalar RMSE。単位は ft。
- 公式例との照合: `data/raw/sample_submission.csv` は `id,tvt` の 2 列で 14,151 行。公開例では全 `tvt=0.0` なので、提出形式確認には使えるが RMSE の期待値照合には使えない。
- 評価対象: train CV では `TVT_input.isna()` の行だけを score する。既知区間の `TVT_input` 行を混ぜると推論時より簡単になり、CV が過大評価される。

## エッジケース

- 欠損予測: `submission.csv` に NaN がある場合は無効。推論後に有限値チェックを必須にする。
- 重複 ID: `sample_submission.csv` の `id` 順と完全一致させる。重複や順序違いは提出検証で落とす。
- 不正な値域: 公式の明示的な値域制約は未確認。極端な外れ値は RMSE を大きく悪化させるため、train 分布に基づく clipping は候補。
- 同点: 公式 Rules では同点時は先に提出された submission が上位。

## 解釈

- 意味のあるスコア変動: まず 5-fold well holdout の fold 間標準偏差を記録し、それを下回る改善は保留扱いにする。
- 想定される public/private のノイズ: hidden test は約 200 well、public/private の分布差は未確認。well ごとの長さや GR 欠損率が効く可能性がある。
- 既知の不一致リスク:
  - row-level random split は同一 well の情報が漏れるため不可。
  - train-only formation columns を使うとローカルでは良くても Kaggle inference で破綻する。
  - 公開 `test/` の 3 well は train 由来の例であり、Public LB そのものではない。
