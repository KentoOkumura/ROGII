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
  - 未着手の実験候補の詳細は `docs/backlog/<candidate>.md`。`KAGGLE_DIRECTION.md` の「アイデアバックログ」は優先度と要約を示す索引とします。
  - 公式資料の要約は `docs/official/`、ディスカッションの保存は `docs/discussions/`、論文単位のメモは `docs/papers/`。
  - 取得した公開 Notebook と metadata は `docs/notebooks/`。
  - 完了した調査レポートは `docs/surveys/`、検索入口は `docs/surveys/README.md`。
  - `docs/analysis/` は旧形式の保存場所です。新しい完了レポートは追加せず、既存文書は次に更新または再利用するときに内容を確認して `docs/surveys/` へ移します。一括移行や推測による補完は行いません。
  - 再利用するコードは `src/`、その場限りの調査コードと生の表・図は `studies/`。
  - 生のコンペデータは `data/raw/`。
- 実験記録は、`metrics.json` を機械処理する数値、`result.md` を解釈・実行証拠・ユーザーの採否判断、`SESSION_NOTES.md` を実行中の作業ログの正とします。実験の `README.md` は状態概要とこれらへのリンクに留め、CV/LB を複数ファイルへ手作業で重複記録しません。
- `docs/surveys/` を実験完了後の調査・分析・統合説明の正とします。`studies/` や `experiments/` を完了した調査レポートの検索入口にしません。
- 生のコンペデータ、モデル重み、大きな生成物を Git に保存しません。

## 実験の状態

- 実験の状態は当面 `metrics.json` の単一の `status` で管理します。
- `planned`、`running`、`debug_completed`、`scaffold_completed`、`failed` は実行状態です。
- `usable`、`completed`、`deprecated`、`discarded` はユーザーの判断を表します。エージェントはユーザーの明示判断前にこれらへ変更しません。
- `leak-risk` は検証リークの注意表示で、採用・不採用・完了の判断ではありません。
- 既存実験の状態は一括変更せず、次に状態を更新するときにこの定義へ合わせます。

## アイデアバックログの引き継ぎ

- `KAGGLE_DIRECTION.md` の「アイデアバックログ」節と `docs/backlog/` の作成、内容更新、状態・優先度変更、削除は `kaggle-strategy` だけが担当します。`kaggle-idea-forge`、`kaggle-review-exp`、`kaggle-oof-readout` など他のskillは候補と根拠を作り、バックログへ反映する場合は同じターンで `kaggle-strategy` を使って引き渡します。
- ユーザーが壁打ち結果を「バックログ化」「バックログへ追加」と依頼した場合は、同じセッションで次の両方を更新します。
  - `KAGGLE_DIRECTION.md` の未着手バックログへ、優先度、短い要約、依存関係、状態、詳細ファイルへの相対リンクを追加する。
  - `docs/backlog/_TEMPLATE.md` を使って `docs/backlog/<candidate>.md` を作り、壁打ち時の根拠と実験境界を保存する。
- `docs/backlog/<candidate>.md` を未着手候補の詳細の正とします。`KAGGLE_DIRECTION.md` に同じ長文を複製しません。1候補につき1ファイルとし、ファイル名は未着手表の候補名と一致させます。
- バックログ化だけを依頼された時点では、exp番号の採番、`.steering/`、実験フォルダ作成、コード実装、Kaggle実行を行いません。同時に実験化や実装も依頼された場合は、その依頼範囲に従います。
- 候補の状態は、このリポジトリ内の管理用語として次のいずれかを使います。これは実験結果の採用・不採用・完了を表しません。
  - `検討メモ・設計不可`: 結果や実装方針に影響する未決事項が残っている。次セッションは推測で補完せず、設計や実装の前にユーザーへ確認する。
  - `設計可能・実験化未承認`: 仮説、根拠、親実験、変更するもの、固定するもの、最小検証、成功条件、停止条件、禁止する代替実装、未決事項が記録され、未決事項が`なし`である。ただし実験化はまだ承認されていない。
- 詳細ファイルには、観測事実と仮定を分け、関連する `result.md`、metrics、保存済み生成物、一次資料へのパス、壁打ちで採らなかった案と理由も記録します。情報不足を推測で埋めず、未決事項として残します。
- 別セッションで候補を設計・実装するときは、バックログ表だけから内容を再構成しません。最初に対応する `docs/backlog/<candidate>.md` と根拠ファイルを読み、固定するもの、変更するもの、最小検証、成功条件、停止条件、実行しないこと、未決事項を短く提示します。詳細記録と異なる解釈または重要な未決事項があれば、コード作成前にユーザーへ確認します。
- ユーザーが候補の実験化を承認したら、exp番号を採番して `.steering/YYYYMMDD-expXXX-title/` を作り、詳細ファイルの内容を `requirements.md`、`design.md`、`tasklist.md` へ欠落なく移します。移行を確認後、`kaggle-strategy` が `docs/backlog/<candidate>.md` と未着手バックログの行を削除し、以後は `.steering/` と `experiments/<exp>/` を正とします。
- この規則の導入前から詳細ファイルなしで存在する候補は、一括して推測補完しません。次に更新、設計、実装するとき、コード作成前に詳細ファイルを作ってユーザー確認を得ます。

## Skill 入口

- ユーザーが実験化を承認したバックログ候補のsteeringへの移行、実験作成、コピー、Kaggle train/inference 実行、notebook 実装ルール、実験記録、再現性確認、実験レビューは `kaggle-review-exp` を使います。バックログ側の状態変更や削除は `kaggle-strategy` を併用します。
- Kaggle API、Kaggle CLI、kernel push/pull/logs/output、slug/title 問題、network escalation、data sync は `kaggle-platform` を使います。
- Kaggle GPU quota が限られる場合の Colab notebook 作成、Google Drive 入出力、Colab runtime / session 対応は `colab-notebook-runner` を使います。
- 提出前の `submission.csv`、Kaggle Notebook output、`kernel-metadata.json`、sample submission 互換性の検証は `kaggle-submit-check` を使います。
- `kaggle competitions submit` 後の scoring 監視と LB 記録は `kaggle-submit-monitor` を使います。
- コード、notebook、失敗実行、OOF、leakage、実行時間、メモリのレビューは `kaggle-review` を使います。
- OOF、feature importance、feature cache、by-well metrics を結合した誤差分析は `kaggle-oof-readout` を使います。分析から出た候補をバックログへ反映する場合は `kaggle-strategy` を併用します。
- 現在路線の停滞時や次実験の発想時に、既存案の微調整に偏らない問題表現、情報源、候補生成、融合、data generation、validation 案を独立生成する場合は `kaggle-idea-forge` を使います。このskillは反証可能な実験案を生成するだけで、バックログを作成・更新しません。選択した候補をバックログ化する場合は `kaggle-strategy` を使います。
- 壁打ち結果や他skillが生成した候補のバックログ作成・更新、次実験、ロードマップ、CV/LB 整合性、失敗パターン、優先順位付けの整理は `kaggle-strategy` を使います。
- 論文、過去解法、公開 notebook、ディスカッション、外部実装の調査は `kaggle-survey-papers` を使います。
- Kaggle の上位公開 notebook をメタデータ付きでローカル保存する場合は `kaggle-notebook-fetch` を使います。
- Kaggle discussion / forum を検索可能な Markdown として保存する場合は `kaggle-discussion-archive` を使います。

## 常時品質基準

- Kaggle Notebook の最初のフル実行と公式評価を Kaggle 上で行います。local smoke に必要な入力、依存関係、生成物がローカルに揃っている場合は、別途のユーザー承認なしに `task train-local` / `task infer-local` / `task execute-notebook-local` による smoke debug を許可します。local smoke の結果だけで公式スコアや Kaggle 実行完了を判断しません。
- ハイパーパラメータは `config.yaml` に置きます。notebook や補助モジュール内の暗黙の定数は避けてください。

## GitHub 自動化

- このテンプレートは CI で `.github/workflows/template-check.yml` を使います。
- Anthropic の GitHub アプリと必要なシークレットをこのリポジトリで使う予定がない限り、Claude Code の GitHub Actions は追加しないでください。
