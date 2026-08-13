# ドキュメント

このディレクトリには、公式資料、検証・再現性の説明、保存資料、完了した調査など、作業内容に応じて参照する文書を置きます。対象が明示されたbacklog候補の実装では、この一覧を先に横断せず、`AGENTS.md`の引き継ぎ手順に従います。

- `01_competition.md`: コンペ概要、目的、提出形式。
- `02_metric.md`: 評価指標の理解とローカル実装メモ。
- `03_validation.md`: CV 設計、リークチェック、CV/LB 乖離の記録。
- `04_data.md`: データ構造、EDA、リスク。
- `05_workflow.md`: 実験、記録、提出の手順。
- `06_reproducibility.md`: seed、PF/Beam、GPU/CPU、Kaggle bootstrap、SHA 記録の再現性ガード。
- `agent-playbooks.md`: 作業内容から利用するskillを選ぶための参照入口。
- `pf_beam_explainer.md`: このコンペで使うPF/Beam実装の説明。
- `glossary.md`: コンペや実験管理で使う用語。
- 未着手候補と戦略索引はリポジトリ直下の [`backlog/`](../backlog/) に置く。
- `official/`: 公式ルール、データ説明、メトリックメモ、Kaggle API で取得した公式ページ要約。
- `discussions/`: Kaggle ディスカッションのアーカイブと要約。
- `notebooks/`: `kaggle-notebook-fetch`で取得した公開Notebookとmetadata。
- `papers/`: 関連論文または論文要約。
- `surveys/`: 完了した実験調査、モデル説明、OOF／結果EDA、特徴量・failure mode、複数実験比較、論文・公開Notebook調査。`surveys/README.md`を上位仮説・実験番号・種類・トピック別の検索入口とする。
- `analysis/`: 旧形式の分析文書。新規の完了レポートは追加せず、既存文書は次に更新または再利用するときに`surveys/`へ移す。
- `legacy/`: 廃止した旧運用の読み取り専用履歴。通常作業の参照入口にはしない。
- `images/`: ドキュメントから参照する図や画像。

## コンペ固有の設定

機械可読な設定は[`project.yml`](../project.yml)を正とします。公式情報、評価指標、CV設計、データ仕様の説明は上記の`01_competition.md`から`04_data.md`を参照し、この索引には設定値を重複記録しません。
