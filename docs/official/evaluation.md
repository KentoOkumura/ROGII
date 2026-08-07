# 評価

公式資料から確認した一次情報と出典を置く場所です。ローカル実装、エッジケース、スコア解釈は `docs/02_metric.md` に記録します。提出形式の運用仕様は `docs/01_competition.md` に集約します。

## 公式メトリック

- 名前: Root Mean Squared Error (RMSE)
- 良い値の向き: 小さいほど良い
- 式/出典: Kaggle `Evaluation` ページ。取得日: 2026-05-27。
- 式:

```text
RMSE = sqrt((1 / n) * sum((y_i - yhat_i)^2))
```

- 備考: Kaggle API の competition metadata では `evaluationMetric` が `Mean Squared Error` と返るが、公式 Evaluation ページ本文では root mean squared error と説明されている。

## 公式例との照合

- 公式サンプル/例: `id,tvt`
- 例:

```csv
id,tvt
000d7d20_1442,0.0
000d7d20_1443,0.0
000d7d20_1444,0.0
000d7d20_1445,0.0
```

- 照合結果: `data/raw/sample_submission.csv` を取得済み。公開例は 14,151 行、2 列、欠損なし。
- 参照した実装/ページ: Kaggle `Evaluation` ページ、`data/raw/sample_submission.csv`。

## 提出形式の公式抜粋

- ID 列: `id`
- Target 列: `tvt`
- 必要行数: test set の各 row に対して 1 行。公開 sample では 14,151 行。
- サンプルファイル: `data/raw/sample_submission.csv`
- ファイル名: Kaggle Notebook 提出では `submission.csv`
