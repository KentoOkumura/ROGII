# Kaggle 実験テンプレート

Kaggle コンペの調査、実験、検証、提出、記録を一貫して管理するためのテンプレートです。

作業単位は`experiments/expXXX_title/`にまとめ、実装前の狙いと設計は`.steering/YYYYMMDD-expXXX-title/`に残します。コンペ単位の設定は`project.yml`、現在の戦略は`KAGGLE_DIRECTION.md`、実験比較は`experiment_summary.md`、完了した調査・判断履歴の検索入口は`docs/surveys/README.md`に集約します。

この README は人間向けの入口です。エージェント向けの詳細ルールは `AGENTS.md`、作業別の参照入口は `docs/agent-playbooks.md`、実際の手順は各 `.agents/skills/*/SKILL.md` を参照してください。

## クイックスタート

必要なもの:

- Python 3.11 以上
- `uv`
- `task`。未導入の場合は `make` で代替できます
- Kaggle を操作する場合は `uv run kaggle auth login`、`~/.kaggle/access_token`、実行環境のsecret store、またはlegacy `~/.kaggle/kaggle.json`による認証。client別の対応方式は`kaggle-platform`の認証設定を参照します

初回セットアップ:

```bash
uv sync --locked --extra dev
task validate-template
```

ルートの`task test` / `make test`は共通テストと実験固有テストを収集するため、全テストを扱う前にNotebook依存も同期します。

```bash
uv sync --locked --extra dev --extra notebook
task test
```

通常の実験変更では全テストを実行せず、対象実験だけを非破壊で確認します。`task test` / `make test`は、共通基盤やテンプレートを広く変更した場合、または全件確認を明示した場合に使います。

```bash
task check-exp EXP=exp001_baseline
task test-exp EXP=exp001_baseline
```

リポジトリ共通コードや同梱skillを変更した場合は、実験固有テストを収集せず、対応する共通検証だけを実行します。`check-skills`は各`SKILL.md`のfrontmatter・名前・本文、存在する`agents/openai.yaml`のUI metadata、skill内のPythonコードを検査します。`agents/openai.yaml`自体は推奨ファイルなので、存在しないskillも許可します。

```bash
task check-skills
task test-common
```

Notebook や Streamlit アプリを使う場合:

```bash
uv sync --locked --extra dev --extra notebook
uv sync --locked --extra dev --extra app
uv sync --locked --extra dev --extra notebook --extra app
```

`kagglehub` を使う Kaggle Platform の操作が必要な場合:

```bash
uv sync --locked --extra dev --extra kaggle-platform
```

`task`コマンドが使えない環境では、失敗する`task`を先に試さず、同名の`make`ターゲットを使います。Makefileは原則として`.venv/bin/...`を直接呼び、`uv`を使うターゲットには書き込み可能な`UV_CACHE_DIR=/tmp/uv-cache`を渡します。先に`uv sync --locked`を実行してください。

```bash
make validate-template
make validate-exp EXP=exp001_baseline EXTRA_ARGS="--allow-todo"
```

## このテンプレートで管理するもの

| 領域 | 管理する内容 |
| --- | --- |
| コンペ設定 | `project.yml` に competition、data、validation、submission、Kaggle runtime を記録 |
| 検証中の上位仮説 | `KAGGLE_DIRECTION.md` で複数の未着手候補・実験と残っている問いを対応付ける |
| 未着手候補 | `KAGGLE_DIRECTION.md` を索引、`docs/backlog/<candidate>.md` を候補詳細の正として管理 |
| 実験計画 | `.steering/` に要件、設計、タスクリストを作成 |
| 実験コード | `experiments/<exp>/` に `config.yaml`、`settings.py`、train/inference notebook、記録ファイルを配置 |
| 共通コード | 複数実験で使う処理を `src/` に集約 |
| 公開Notebook資料 | 取得した公開Notebookとmetadataを `docs/notebooks/` に保存 |
| 調査レポート | 完了した実験調査、モデル説明、OOF／結果EDA、外部調査を `docs/surveys/` に集約し、生成索引で検索 |
| 調査コード | その場限りの分析コードと生の表・図を `studies/` に保存 |
| 提出管理 | `submission.csv` の形式検証、Kaggle Notebook 実行、提出履歴の記録 |
| 実験比較 | `metrics.json`、`experiment_summary.md`、`SUBMISSIONS.md` で結果を追跡 |

## 機能一覧

- 新規実験ディレクトリをテンプレートまたは既存実験から作成できます。
- 複数の未着手候補・実験を共通の上位仮説IDで追跡できます。
- 実験前に steering document を作成し、仮説、設計、成功条件を明文化できます。
- `project.yml` と各実験の `config.yaml` を検証し、TODO や設定漏れを検出できます。
- Kaggle 上の学習・推論、提出形式チェックを `task` で統一して実行できます。
- Kaggle 実行用の notebook ディレクトリと `kernel-metadata.json` を生成できます。
- 実験結果と提出履歴を Markdown と JSON に記録し、比較表を更新できます。
- 完了した調査レポートを実験番号・種類・トピック別に自動索引化できます。
- Streamlit アプリで実験結果や OOF 分析を確認できます。
- Codex 用の repo-local skills を `.agents/skills/` に置き、Kaggle 作業の手順を共有できます。

## エージェント用 skills

このテンプレートには、Kaggle 作業で使う repo-local skills を `.agents/skills/` に同梱しています。Codex に依頼するときは、必要に応じて skill 名をそのまま指定できます。

| Skill | 使う場面 |
| --- | --- |
| `kaggle-platform` | Kaggle API、認証、データ取得、`project.yml` 設定、コンペ資料の準備 |
| `colab-notebook-runner` | Kaggle GPU quota が限られる場合の Colab notebook、Google Drive、runtime/session 対応 |
| `kaggle-review-exp` | 承認済みbacklog候補のsteering移行、実験の作成・実行・記録・レビュー。backlog更新はStrategyへ引き渡す |
| `kaggle-review` | 学習/推論コード、notebook、OOF、失敗実行のレビュー |
| `kaggle-oof-readout` | OOF、feature importance、feature cache、by-well metricsを結合した誤差分析。候補保存はStrategyへ引き渡す |
| `kaggle-idea-forge` | 反証可能な候補の独立生成。候補をbacklogへ直接保存しない |
| `kaggle-strategy` | 戦略と優先順位を整理し、アイデアbacklogを作成・更新・削除する唯一のSkill |
| `kaggle-submit-check` | `submission.csv`、Kaggle Notebook、`kernel-metadata.json` の提出前検証 |
| `kaggle-submit-monitor` | `kaggle competitions submit` 後の scoring 監視と LB 記録 |
| `kaggle-notebook-fetch` | 上位公開 notebook をメタデータ付きでローカル保存 |
| `kaggle-survey-papers` | 関連論文、過去解法、公開 notebook、ディスカッションの調査 |
| `kaggle-discussion-archive` | Kaggle discussion の HTML や本文を Markdown として保存 |

コンペ固有の観点が増えたら、対応する `SKILL.md` に追記します。汎用テンプレートとして使う場合は、コンペ名やドメイン固有のチェック項目を置き換えてください。

## 標準ワークフロー

初回は`project.yml`のコンペ固有項目を埋めてtemplate validationを行い、Kaggle認証後にコンペデータを取得します。`dl-kaggle-comp`は取得したzipを`data.raw_dir`へ安全に展開し、設定した`submission.sample_file`が存在することまで確認します。その後にstrict config validationを行います。詳しい設定項目は`kaggle-platform`の「Repository Template Setup」を正とします。

```bash
task validate-template
task dl-kaggle-comp
task validate-config
```

`submission.sample_file`を別の方法で配置済みなら、データ取得は省略できます。初期設定の検証後、steering documentを作ってから実験ディレクトリを作成します。

```bash
task new-steering EXP=exp002_next_idea
task new-exp EXP=exp002_next_idea SOURCE=experiments/exp001_baseline
task validate-exp EXP=exp002_next_idea EXTRA_ARGS="--allow-todo"
```

以後の実装、Kaggle train/inference、記録、レビューは `kaggle-review-exp`、Kaggle CLI と kernel 操作は `kaggle-platform` を使います。提出物の実ファイルを検証する場合は `kaggle-submit-check`、submit 後の監視は `kaggle-submit-monitor` を使います。ライフサイクル全体は `docs/05_workflow.md`、作業別の入口は `docs/agent-playbooks.md` を参照してください。

## 記録と判断

保存場所、各記録ファイルの役割、実験status、完了・採用・不採用の判断規則は`AGENTS.md`を正とし、このREADMEでは別定義しません。人間向けの横断入口は`experiment_summary.md`と`SUBMISSIONS.md`です。

## データと提出

データ配置は `AGENTS.md`、コンペ設定は `project.yml` を正とします。Kaggle Notebook のフル実行と公式評価を基準にし、local smoke だけで公式スコアや Kaggle 実行完了を判断しません。

CV は live logs、notebook cell、Kaggle UI から記録できます。提出物、OOF、model manifest、feature importance、SHA など実ファイルの確認が必要な場合だけ Kaggle output を取得します。Notebook-only code submission は対象 kernel version の `submission.csv` を確認して一度だけ実行し、同じ提出に raw CLI と task の両方を使いません。詳しい操作と失敗時の切り分けは `kaggle-platform` を参照してください。

## リポジトリ構成

- `.agents/skills/`: このリポジトリ固有の Codex skills。Kaggle 系スキルはここで管理します
- `.github/workflows/`: リポジトリテンプレートのCI設定
- `.steering/`: 実装前の要件、設計、タスクリスト
- `app/`: 実験や OOF を確認する Streamlit アプリ
- `data/`: ローカルデータキャッシュ。Git には入れません
- `docs/`: 公式情報、候補詳細、保存資料、調査レポート。保存先の一覧は [docs/README.md](docs/README.md)
- `experiments/`: 実験ごとのコード、設定、出力、記録
- `experiments/<exp>/artifacts/`: その実験が生成した出力。必要な分類はこの下のサブディレクトリで表現
- `experiments/<exp>/assets/`: その実験で参照する小規模な固定データ
- `experiments/<exp>/tests/`: その実験だけに属するテスト
- `scripts/`: テンプレート作成、検証、提出準備、記録更新用スクリプト
- `src/`: 複数実験で再利用する共通コード
- `studies/`: その場限りの EDA・調査コードと生の表・図。完了した結論は置かない
- `templates/`: 新規実験、steering、survey用テンプレート
- `tests/`: 複数実験やリポジトリ全体に関わる共通テスト
- `tools/`: Git で追跡しない外部ツールの clone やローカル配置

主要ファイル:

- `AGENTS.md`: エージェント向けの最優先運用ルール
- `Taskfile.yml`: 推奨コマンド定義
- `Makefile`: `task` が使えない環境向けの代替コマンド
- `project.yml`: コンペ単位のメタデータ、データ、検証、提出、Kaggle runtime の正
- `SUBMISSIONS.md`: submission refを専用列に持つ提出履歴
- `KAGGLE_DIRECTION.md`: コンペ戦略、検証方針、現在の重点、検証中の上位仮説、アイデアバックログ
- `experiment_summary.md`: 実験比較と自動更新される要約
- `docs/surveys/README.md`: 完了した調査レポートの生成索引
- `docs/agent-playbooks.md`: 実験、提出、レビュー、分析の作業別参照索引

## エージェントへの新実験依頼

新しい実験を依頼するときは、実験名だけでなく、仮説、変更点、検証方法、成功条件をセットで伝えてください。

```markdown
exp002_agg_features を作ってください。

ベース実験:
exp001_baseline

仮説:
カテゴリ単位の集約特徴量で CV が改善するはずです。

変更点:
- exp001_baseline をコピーして開始
- カテゴリ単位の mean/count/std 特徴量を追加
- fold 外の target を使わない

CV戦略:
- Fold method:
- Group key:
- Stratify key:
- Random seed:
- Number of folds:
- Score target:

成功条件:
- task validate-exp EXP=exp002_agg_features が通る
- debug train/inference が通る
- metrics.json と Kaggle inference output の submission.csv が生成される

リスク:
- 集約特徴量の fit 範囲を誤るとリークする
- 特徴量追加で実行時間が増える
```

レビューを依頼するときは、対象実験、見てほしい観点、直近の実行コマンド、エラーやスコアを添えると確認が速くなります。

## 人間が確認すること

- 公式ルール、外部データ可否、提出回数、Notebook の internet/GPU 制約
- `project.yml` の competition、data、validation、submission、runtime 設定
- Kaggle API token や秘密情報が Git に入っていないこと
- CV と LB がずれた場合の原因調査
- 採用する実験、提出する実験、public/private LB の確認
- 大きなデータ、モデル重み、生成物が Git に含まれていないこと
