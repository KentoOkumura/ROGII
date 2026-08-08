# ドキュメント

このディレクトリには、エージェントが実験方針を決める前に読むべき文書を置きます。

- `01_competition.md`: コンペ概要、目的、提出形式。
- `02_metric.md`: 評価指標の理解とローカル実装メモ。
- `03_validation.md`: CV 設計、リークチェック、CV/LB 乖離の記録。
- `04_data.md`: データ構造、EDA、リスク。
- `05_workflow.md`: 実験、記録、提出の手順。
- `06_reproducibility.md`: seed、PF/Beam、GPU/CPU、Kaggle bootstrap、SHA 記録の再現性ガード。
- `glossary.md`: コンペや実験管理で使う用語。
- `backlog/`: 未着手の実験候補の詳細。`KAGGLE_DIRECTION.md`を優先度と要約の索引とする。
- `official/`: 公式ルール、データ説明、メトリックメモ、Kaggle API で取得した公式ページ要約。
- `discussions/`: Kaggle ディスカッションのアーカイブと要約。
- `notebooks/`: `kaggle-notebook-fetch`で取得した公開Notebookとmetadata。
- `papers/`: 関連論文または論文要約。
- `surveys/`: 完了した実験調査、モデル説明、OOF／結果EDA、特徴量・failure mode、複数実験比較、論文・公開Notebook調査。`surveys/README.md`を実験番号・種類・トピック別の検索入口とする。
- `analysis/`: 旧形式の分析文書。新規の完了レポートは追加せず、既存文書は次に更新または再利用するときに`surveys/`へ移す。

## ROGII 初期設定

- 公式情報取得日: 2026-05-27
- 評価指標: RMSE、minimize
- CV 方針: `well_id` の GroupKFold、`TVT_input` NaN 行のみ評価
- 提出: Notebook-only、internet disabled、`submission.csv`
