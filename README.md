# Kaggle 実験テンプレート

Kaggle コンペの調査、実験、検証、提出、記録を一貫して管理するためのテンプレートです。

作業単位は `experiments/expXXX_title/` にまとめ、実装前の狙いと設計は `.steering/YYYYMMDD-expXXX-title/` に残します。コンペ単位の設定は `project.yml`、実験横断の戦略や履歴は `KAGGLE_DIRECTION.md` と `experiment_summary.md`、完了した調査レポートの検索入口は `docs/surveys/README.md` に集約します。

この README は人間向けの入口です。エージェント向けの詳細ルールは `AGENTS.md`、詳しい作業手順は `docs/agent-playbooks.md` を参照してください。

## クイックスタート

必要なもの:

- Python 3.11 以上
- `uv`
- `task`。未導入の場合は `make` で代替できます
- Kaggle API を使う場合は Kaggle token と認証用の環境変数

初回セットアップ:

```bash
uv sync --extra dev
task validate-template
```

Notebook や Streamlit アプリを使う場合:

```bash
uv sync --extra dev --extra notebook
uv sync --extra dev --extra app
uv sync --extra dev --extra notebook --extra app
```

`task` が使えない環境では、同名の `make` ターゲットを使います。`make` は `.venv/bin/...` を直接呼ぶため、先に `uv sync` を実行してください。

```bash
make validate-template
make validate-exp EXP=exp001_baseline EXTRA_ARGS="--allow-todo"
```

## このテンプレートで管理するもの

| 領域 | 管理する内容 |
| --- | --- |
| コンペ設定 | `project.yml` に competition、data、validation、submission、Kaggle runtime を記録 |
| 実験計画 | `.steering/` に要件、設計、タスクリストを作成 |
| 実験コード | `experiments/<exp>/` に `config.yaml`、`settings.py`、train/inference notebook、記録ファイルを配置 |
| 共通コード | 複数実験で使う処理を `src/` に集約 |
| 調査レポート | 完了した実験調査、モデル説明、OOF／結果EDA、外部調査を `docs/surveys/` に集約し、生成索引で検索 |
| 調査コード | その場限りの分析コードと生の表・図を `studies/` に保存 |
| 提出管理 | `submission.csv` の形式検証、Kaggle Notebook 実行、提出履歴の記録 |
| 実験比較 | `metrics.json`、`experiment_summary.md`、`submissions/SUBMISSIONS.md` で結果を追跡 |

## 機能一覧

- 新規実験ディレクトリをテンプレートまたは既存実験から作成できます。
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
| `kaggle-review-exp` | 実験の作成、コピー、Kaggle 実行、`SESSION_NOTES` や `metrics.json` の記録確認 |
| `kaggle-review` | 学習/推論コード、notebook、OOF、失敗実行のレビュー |
| `kaggle-oof-readout` | OOF、feature importance、feature cache、by-well metrics を結合した誤差分析 |
| `kaggle-idea-forge` | 停滞時や次実験の発想時に、既存案の微調整に偏らない反証可能な候補を独立生成 |
| `kaggle-strategy` | 実験履歴、提出履歴、調査メモを読んだ次の実験方針の整理 |
| `kaggle-submit-check` | `submission.csv`、Kaggle Notebook、`kernel-metadata.json` の提出前検証 |
| `kaggle-submit-monitor` | `kaggle competitions submit` 後の scoring 監視と LB 記録 |
| `kaggle-notebook-fetch` | 上位公開 notebook をメタデータ付きでローカル保存 |
| `kaggle-survey-papers` | 関連論文、過去解法、公開 notebook、ディスカッションの調査 |
| `kaggle-discussion-archive` | Kaggle discussion の HTML や本文を Markdown として保存 |

コンペ固有の観点が増えたら、対応する `SKILL.md` に追記します。汎用テンプレートとして使う場合は、コンペ名やドメイン固有のチェック項目を置き換えてください。

## 標準ワークフロー

まず対象コンペに合わせて `project.yml` を埋めます。competition、data、defaults、submission、runtime の項目を設定し、公式データや sample submission の配置が決まったら厳格な検証を通します。

```bash
task validate-config
```

新しい実験は、計画を作ってから実験ディレクトリを作成します。

```bash
task new-steering EXP=exp002_next_idea
# .steering/YYYYMMDD-exp002-next-idea/{requirements.md,design.md,tasklist.md} を記入

task new-exp EXP=exp002_next_idea SOURCE=experiments/exp001_baseline
task validate-exp EXP=exp002_next_idea EXTRA_ARGS="--allow-todo"
```

`config.yaml` と実装をコンペに合わせて埋めた後、学習前に厳格な検証を実行します。

Kaggle Notebook の slug は 50 文字以内にし、`kernel-metadata.json` の `id` と `title` 由来 slug を一致させます。実験ディレクトリ名全体では上限を超える場合、実験番号、意味のある短縮名、`train` / `inference` の種別を残して短縮します。push 直前には Kaggle UI の Active Sessions で対象 CPU / GPU session が上限未満であることを確認します。

```bash
task validate-exp EXP=exp002_next_idea
task prepare-kaggle-notebooks EXP=exp002_next_idea EXTRA_ARGS="--notebook train --kernel-id username/exp002-next-idea-train --title 'exp002 next idea train' --run-on-push --strict"
task push-kaggle-train EXP=exp002_next_idea
task kaggle-logs KERNEL=username/exp002-next-idea-train
```

CV はまず live logs、notebook cell、Kaggle UI で確認します。OOF、model manifest、feature importance など実ファイルの確認が必要な場合だけ output を取得します。

```bash
task kaggle-output KERNEL=username/exp002-next-idea-train OUT=/tmp/kaggle-output/exp002_next_idea/train
```

提出候補になったら、inference も Kaggle 上で実行して output を取得し、sample submission と照合します。

```bash
task prepare-kaggle-notebooks EXP=exp002_next_idea EXTRA_ARGS="--notebook inference --kernel-id username/exp002-next-idea-inference --title 'exp002 next idea inference' --run-on-push --strict"
task push-kaggle-infer EXP=exp002_next_idea
task kaggle-logs KERNEL=username/exp002-next-idea-inference
task kaggle-output KERNEL=username/exp002-next-idea-inference OUT=/tmp/kaggle-output/exp002_next_idea/inference
task submit-check EXP=exp002_next_idea SUBMISSION=/tmp/kaggle-output/exp002_next_idea/inference/submission.csv
```

## 実験管理ルール

実験結果は、コマンド、設定、CV、成果物、次のアクションまで書いて初めて記録済みとします。

主な記録先:

- `experiments/<exp>/README.md`: status、CV/LB、利用可否、リスク、次に使う実験
- `experiments/<exp>/SESSION_NOTES.md`: 実行したコマンド、作業ログ、エラー、途中結果
- `experiments/<exp>/result.md`: 最終評価、解釈、採用/不採用理由
- `experiments/<exp>/metrics.json`: スクリプトから読めるスコア要約
- `experiment_summary.md`: 実験間の比較、lineage、主要な発見
- `submissions/SUBMISSIONS.md`: Kaggle に提出した履歴
- `docs/surveys/README.md`: 完了した調査レポートを実験番号・種類・トピックから探す入口

スコアを記録し、比較表を更新する例:

```bash
task record-exp EXP=exp002_next_idea STATUS=usable CV=0.123 PUBLIC_LB=0.120 NOTES="stable baseline"
task compare-exp
task update-summary
```

実験構成・モデル説明、OOF／結果EDA、特徴量・failure mode、複数実験比較など、通常の`result.md`を越える完了調査は`docs/surveys/`へ記録します。

```bash
task new-survey-report \
  SURVEY_TITLE="exp238 selectorのモデル構成とOOF分析" \
  SURVEY_SLUG="exp238-selector-model-oof" \
  EXTRA_ARGS="--type experiment_review --type model_explanation --type oof_analysis --experiment exp238 --topic selector --topic confidence"
task update-survey-index
task validate-surveys
```

Kaggle 提出を行った場合:

```bash
task record-submission EXP=exp002_next_idea EXTRA_ARGS="--cv 0.123 --public-lb 0.120 --notes baseline"
task update-summary
```

実験の status は `planned`、`running`、`usable`、`failed`、`deprecated`、`leak-risk` を基本にします。CV と LB が合わない場合は、追加のチューニングに進む前に検証設計、データ分割、前処理差分、提出形式を確認します。

`config.yaml` の `experiment.route` は、ML を主対象にする `ml_model`、PF/Beam を主対象にする `pf_beam`、両方が予測生成に本質的に寄与する `ensemble` のいずれかにします。

## 学習/推論コードの鉄則

- 乱数 seed、fold、metric、主要ハイパーパラメータは `config.yaml` に置きます。
- notebook や補助モジュール内に暗黙の定数を増やさず、実験の差分が config と記録から追えるようにします。
- CV の分割単位、stratify、group、score 対象行を明記します。
- target encoding、集約特徴量、外部データ結合は fold 外の情報を使っていないか確認します。
- 学習時と推論時の前処理、特徴量順、欠損値処理、dtype を一致させます。
- 学習コードは学習済みモデルと推論に必要な前処理状態を保存し、特徴量名と順序、variant / mode / fold、ファイル形式、相対パス、SHA を model manifest に記録します。推論コードは manifest から再学習なしで読み込みます。
- metric の最大化/最小化の向きと、Kaggle 側の評価定義を確認します。
- Kaggle Notebook の offline、GPU、実行時間、メモリ制約で動く構成にします。
- code competition では公開 `test/` と `sample_submission.csv` は smoke test 用サンプルであり、提出時に hidden test 用入力へ差し替えられる前提で実装します。公開 test 固有の ID、行数、ファイル名、SHA を推論条件に使いません。
- 大きなデータ、モデル重み、生成物、token は Git に含めません。

## データと提出

標準のローカルデータ置き場は `data/raw/` です。手動で取得した公式データは `data/raw/`、外部データは `data/external/`、加工済みデータは `data/processed/` に置きます。

`project.yml` に competition slug を設定した後、Kaggle CLI で公式データを取得できます。

```bash
task dl-kaggle-comp
```

提出ファイルは Kaggle inference notebook output として生成し、`project.yml` の `submission.sample_file` を基準に検証します。

```bash
task prepare-kaggle-notebooks EXP=exp002_next_idea EXTRA_ARGS="--notebook inference --kernel-id username/exp002-next-idea-inference --title 'exp002 next idea inference' --run-on-push --strict"
task push-kaggle-infer EXP=exp002_next_idea
task kaggle-output KERNEL=username/exp002-next-idea-inference OUT=/tmp/kaggle-output/exp002_next_idea/inference
task submit-check EXP=exp002_next_idea SUBMISSION=/tmp/kaggle-output/exp002_next_idea/inference/submission.csv
```

実験コードの正の編集対象は notebook です。

- `experiments/<exp>/<exp>_train.ipynb`
- `experiments/<exp>/<exp>_inference.ipynb`

Kaggle Notebook のフル実行と公式評価を正とします。local smoke に必要な入力、依存関係、生成物がローカルに揃っている場合は、`task train-local` / `task infer-local` / `task execute-notebook-local` を使用できます。local smoke の結果だけで公式スコアや Kaggle 実行完了を判断しません。

```bash
task execute-notebook-local EXP=exp002_next_idea NOTEBOOK=train EXTRA_ARGS="--allow-local --debug"
task execute-notebook-local EXP=exp002_next_idea NOTEBOOK=inference EXTRA_ARGS="--allow-local"
```

Kaggle に notebook として push する場合:

```bash
task prepare-kaggle-notebooks EXP=exp002_next_idea EXTRA_ARGS="--notebook train --kernel-id username/exp002-next-idea-train --title 'exp002 next idea train' --run-on-push --strict"
task push-kaggle-train EXP=exp002_next_idea
task kaggle-logs KERNEL=username/exp002-next-idea-train
task kaggle-output KERNEL=username/exp002-next-idea-train OUT=/tmp/kaggle-output/exp002_next_idea/train
```

上の output 取得は、OOF、model manifest、feature importance など実ファイルの確認が必要な場合だけ実行します。

`prepare-kaggle-notebooks` は `kernel-metadata.json` の `competition_sources` に `project.yml` の competition slug を入れます。
そのため、Kaggle の Input 追加 UI は通常不要です。
Kaggle runtime は CPU をデフォルトにし、GPU が必要な実験だけ明示的に有効化します。
Kaggle CLI は notebook 本体だけを送るため、生成 notebook には補助ファイルを復元する base64 zip bootstrap セルを入れます。
VS Code Compatible URL の取得は Kaggle/VS Code 側で行います。

inference notebook を作成・更新する場合:

```bash
task prepare-kaggle-notebooks EXP=exp002_next_idea EXTRA_ARGS="--notebook inference --kernel-id username/exp002-next-idea-inference --title 'exp002 next idea inference' --strict"
task push-kaggle-infer EXP=exp002_next_idea
task kaggle-logs KERNEL=username/exp002-next-idea-inference
```

Kaggle inference notebook の output を提出する場合:

```bash
task prepare-kaggle-notebooks EXP=exp002_next_idea EXTRA_ARGS="--notebook inference --kernel-id username/exp002-next-idea-inference --title 'exp002 next idea inference' --run-on-push --strict"
task push-kaggle-infer EXP=exp002_next_idea
task kaggle-logs KERNEL=username/exp002-next-idea-inference
task kaggle-output KERNEL=username/exp002-next-idea-inference OUT=/tmp/kaggle-output/exp002_next_idea/inference
task submit-check EXP=exp002_next_idea SUBMISSION=/tmp/kaggle-output/exp002_next_idea/inference/submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k username/exp002-next-idea-inference -v VERSION -f submission.csv -m "exp002_next_idea"
```

submit 後に `task record-submission` と `task update-summary` で履歴へ反映します。

## リポジトリ構成

- `.agents/skills/`: このリポジトリ固有の Codex skills。Kaggle 系スキルはここで管理します
- `.steering/`: 実装前の要件、設計、タスクリスト
- `app/`: 実験や OOF を確認する Streamlit アプリ
- `data/`: ローカルデータキャッシュ。Git には入れません
- `docs/`: 公式情報、メトリック、検証方針、完了した調査レポート
- `docs/surveys/`: 実験調査、モデル説明、OOF／結果EDA、特徴量・failure mode、複数実験比較、外部調査の正
- `experiments/`: 実験ごとのコード、設定、出力、記録
- `notebooks/`: notebook 作業用
- `scripts/`: テンプレート作成、検証、提出準備、記録更新用スクリプト
- `src/`: 複数実験で再利用する共通コード
- `studies/`: その場限りの EDA・調査コードと生の表・図。完了した結論は置かない
- `submissions/`: 提出履歴
- `templates/`: 新規実験、steering 用テンプレート
- `tools/`: 補助ツールのメモやスクリプト置き場

主要ファイル:

- `AGENTS.md`: エージェント向けの最優先運用ルール
- `Taskfile.yml`: 推奨コマンド定義
- `Makefile`: `task` が使えない環境向けの代替コマンド
- `project.yml`: コンペ単位のメタデータ、データ、検証、提出、Kaggle runtime の正
- `KAGGLE_DIRECTION.md`: コンペ戦略、検証方針、現在の重点、アイデアバックログ
- `experiment_summary.md`: 実験比較と自動更新される要約
- `docs/surveys/README.md`: 完了した調査レポートの生成索引
- `docs/agent-playbooks.md`: 実験、提出、レビュー、分析の詳しい手順

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
- metrics.json と submission.csv が生成される

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
