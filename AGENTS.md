# Kaggle 実験テンプレート用エージェントガイド

このリポジトリは、エージェントと一緒に Kaggle 実験を進めるための作業場所です。

## Codex の参照入口

- Codex は、このリポジトリでは `AGENTS.md` の作業ルールを最優先します。
- このリポジトリ固有の Codex スキルは `.agents/skills/` で管理します。Kaggle 系スキルはここを正とします。
- `.codex/` ディレクトリには依存しないでください。一部の Codex ランタイムでは、空の管理用ディレクトリとして見えることがあります。
- リポジトリ全体に常時適用する規約はこのファイル、作業別の実行手順は `.agents/skills/`、長い背景資料や仕様は `docs/`、タスク自動化は `Taskfile.yml`、`Makefile`、`scripts/` に置きます。同じ規則を複数箇所に複製せず、skill や docs からこのファイルの規約を参照します。
- グローバルの `~/.codex/skills/` は、Codex の system skills などリポジトリ非依存のスキルに使います。Kaggle 系スキルを `~/.codex/skills/` に重複配置しないでください。

## ユーザーへの説明で使う用語

- ユーザーへの説明では、コンペ主催者、公式資料、評価指標、データ仕様、参加者の公開 Notebook・Discussion、参照論文・実装で実際に使われている用語を優先します。
- データサイエンスや対象分野で一般に使われている英語の専門用語・略語は、日本語の文中で使用できます。英語であること自体を禁止理由にしません。
- 英語の専門用語・略語が一般的かどうかは、その分野の論文、公式資料、参加者の解説、一般的な実装で同じ意味に使われているかを基準に判断します。
- 概念を整理するための独自名称、独自分類、比喩的な工程名を新しく作りません。
- エージェントが説明の整理や実装管理のために作った英語名称、英単語を組み合わせた独自名称、出典を確認できない分類名を、一般的な専門用語のように使用しません。
- リポジトリ内部の管理用語を、コンペや手法の一般的な用語であるかのように説明しません。使用が必要な場合は「このリポジトリ内の管理用語」と明示し、先に平易な日本語で意味を説明します。
- 参加者固有の名称を使う場合は、「3位解法の説明で使われている名称」のように出所を明示します。
- 確立した名称がない概念は、名称を付けずに具体的な処理内容を説明します。短縮名がどうしても必要な場合は「この回答内だけの便宜的な呼称」と明示します。
- 一般的な専門用語か判断できない場合は、新しい名称を付けず、実際の処理を具体的に説明します。
- 略語と専門用語は初出時に定義します。ユーザーが使用していない略語を説明なしに導入しません。
- 手法を説明するときは、独自の包括的な呼称ではなく、入力、予測対象、モデル出力、損失関数、推論方法、処理単位を具体的に記述します。
- 回答前に、主要な用語について「誰が使っている名称か」を確認します。公式資料、参加者、論文、既存コードのいずれにも根拠がなければ、平易な表現へ書き換えます。

## 運用ルール

- 設計、実装方針、実験分岐、提出判断などで複数の妥当な選択肢があり、結果や作業方針に影響する場合は、独断で決めずユーザーに確認します。既存ルールやコードから明確に判断できる低リスクな細部は、作業を止めずに進めます。
- 実験を完了とするか、採用または不採用とするかは、エージェントだけで確定せずユーザーに判断を仰ぎます。判断を求める際は、比較対象、CV・LB、実行証拠、未解決事項を整理し、推奨する判断とその理由を伝えます。ユーザーが判断するまでは、`result.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` などに完了・採用・不採用が確定したものとして記録しません。
- ユーザーが実験の完了を判断した時点で、その実験に関係する変更だけを確認して `git commit` し、現在の作業ブランチを `git push` します。作業ツリーにある無関係な変更は commit に含めません。commit または push に失敗した場合は、失敗理由と未完了の操作をユーザーに報告します。
- リポジトリ全体の保存場所は次のように分けます。
  - 実験固有のコード、設定、記録は `experiments/expXXX_name/`。
  - 公式資料の要約は `docs/official/`、ディスカッションの保存は `docs/discussions/`、論文単位のメモは `docs/papers/`。
  - 完了した調査レポートは `docs/surveys/`、検索入口は `docs/surveys/README.md`。
  - 再利用するコードは `src/`、その場限りの調査コードと生の表・図は `studies/`。
  - 生のコンペデータは `data/raw/`。
- `experiments/<exp>/result.md` をその実験の結果・採用判断・実行証拠の正とし、`docs/surveys/` を実験完了後の調査・分析・統合説明の正とします。`studies/` や `experiments/` を完了した調査レポートの検索入口にしません。
- 生のコンペデータ、モデル重み、大きな生成物を Git に保存しません。

## Skill 入口

- 実験作成、コピー、Kaggle train/inference 実行、notebook 実装ルール、実験記録、再現性確認、実験レビューは `kaggle-review-exp` を使います。
- Kaggle API、Kaggle CLI、kernel push/pull/logs/output、slug/title 問題、network escalation、data sync は `kaggle-platform` を使います。
- Kaggle GPU quota が限られる場合の Colab notebook 作成、Google Drive 入出力、Colab runtime / session 対応は `colab-notebook-runner` を使います。
- 提出前の `submission.csv`、Kaggle Notebook output、`kernel-metadata.json`、sample submission 互換性の検証は `kaggle-submit-check` を使います。
- `kaggle competitions submit` 後の scoring 監視と LB 記録は `kaggle-submit-monitor` を使います。
- コード、notebook、失敗実行、OOF、leakage、実行時間、メモリのレビューは `kaggle-review` を使います。
- OOF、feature importance、feature cache、by-well metrics を結合した誤差分析は `kaggle-oof-readout` を使います。
- 現在路線の停滞時や次実験の発想時に、既存案の微調整に偏らない問題表現、情報源、候補生成、融合、data generation、validation 案を独立生成する場合は `kaggle-idea-forge` を使います。この skill は候補を採用・実装せず、反証可能な実験案として出力します。
- 次実験、ロードマップ、CV/LB 整合性、失敗パターン、優先順位付けの整理は `kaggle-strategy` を使います。
- 論文、過去解法、公開 notebook、ディスカッション、外部実装の調査は `kaggle-survey-papers` を使います。
- Kaggle の上位公開 notebook をメタデータ付きでローカル保存する場合は `kaggle-notebook-fetch` を使います。
- Kaggle discussion / forum を検索可能な Markdown として保存する場合は `kaggle-discussion-archive` を使います。

## 常時品質基準

- Kaggle Notebook のフル実行と公式評価を正とします。local smoke に必要な入力、依存関係、生成物がローカルに揃っている場合は、`task train-local` / `task infer-local` / `task execute-notebook-local` による local smoke を許可します。local smoke の結果だけで公式スコアや Kaggle 実行完了を判断しません。
- ハイパーパラメータは `config.yaml` に置きます。notebook や補助モジュール内の暗黙の定数は避けてください。

## GitHub 自動化

- このテンプレートは CI で `.github/workflows/template-check.yml` を使います。
- Anthropic の GitHub アプリと必要なシークレットをこのリポジトリで使う予定がない限り、Claude Code の GitHub Actions は追加しないでください。
