# ワークフロー

## 新規実験

```bash
task new-steering EXP=expXXX_title
task new-exp EXP=expXXX_title
task validate-exp EXP=expXXX_title EXTRA_ARGS="--allow-todo"
```

ROGII では実装前に、steering に次を明記します。

- holdout 単位は `well_id`。
- 評価対象は `TVT_input` が NaN の行。
- train-only formation columns を使う場合は、隠しテストで使える代替特徴に変換する根拠を書く。
- stochastic feature generation、PF/Beam、GPU 学習、保存済み model inference を含む場合は、`docs/06_reproducibility.md` の seed policy、CPU/GPU mode、SHA 記録、rerun 方針を設計に書く。

学習、推論、提出は原則として同じ `EXP=expXXX_title` で完結させます。train-side CV が良かった候補を推論化するだけ、または提出するだけの理由で `expYYY_inference_submit` のような別実験を作らないでください。過去に分離した実験は履歴として残し、今後の新規実験からこの運用に揃えます。

## データ取得

最小確認用の `sample_submission.csv` は `data/raw/` に取得済みです。全公式データが必要になったら、`project.yml` の competition slug を使って取得します。

```bash
task dl-kaggle-comp
```

全データは約 1.33 GB で、`data/raw/` は Git 管理外です。

## Kaggle 実行パイプライン

```bash
task validate-exp EXP=expXXX_title
task prepare-kaggle-notebooks EXP=expXXX_title EXTRA_ARGS="--strict"
```

実験コードの正の編集対象は notebook です。

- `experiments/<exp>/<exp>_train.ipynb`
- `experiments/<exp>/<exp>_inference.ipynb`

notebook は、人間が上から読んで実験の目的、入力、比較 variant、評価、保存 artifact を理解できる構成にします。補助 `.py` に処理を分ける場合でも、notebook を `main()` だけの薄い entrypoint にしません。

標準の train notebook 構成:

1. `Setup and configuration`: `config.yaml`、親実験、anchor score、variant、出力先を確認する。
2. 入力確認: train files、OOF artifact、feature source、fold split などを読む。
3. 学習/監査: fold-safe な学習、OOF audit、ablation、postprocess など、この実験の主処理を実行する。
4. 評価: CV、bucket 別 score、variant summary、重要な group summary を表示する。
5. `Metrics and artifacts`: `metrics.json` と `artifacts/*.csv/json/log` を保存する。

標準の inference notebook 構成:

1. `Setup and paths`: model config、input/output path、選択 variant を確認する。
2. model/artifact 読み込みまたは final model 学習。
3. test prediction と postprocess。
4. `submission.csv` 作成と形式確認用の要約表示。

notebook の最初のフル実行と公式評価は Kaggle 上で行います。local smoke に必要な入力、依存関係、生成物が揃っていれば、別途のユーザー承認なしに `--allow-local` を付けてsmoke debugを実行できます。local smokeの結果だけで公式スコアやKaggle実行完了を判断しません。

```bash
task execute-notebook-local EXP=expXXX_title NOTEBOOK=train EXTRA_ARGS="--allow-local --debug"
task execute-notebook-local EXP=expXXX_title NOTEBOOK=inference EXTRA_ARGS="--allow-local"
```

## 結果の記録

```bash
task record-exp EXP=expXXX_title STATUS=running CV=0.123 PUBLIC_LB=0.120 NOTES="recorded result; awaiting user decision"
task compare-exp
task update-summary
```

## 提出

このコンペは Notebook-only code competition です。Kaggle 上では internet disabled、CPU/GPU ともに 9 hours 以内、提出ファイル名は `submission.csv` です。リポジトリのデフォルト Kaggle runtime は CPU です。GPU が必要な実験だけ `project.yml` または生成済み metadata で明示的に有効化します。

### Kaggle notebook 作成から実行まで

新規実験では `task new-exp` により、編集対象の notebook が作られます。

```text
experiments/<exp>/<exp>_train.ipynb
experiments/<exp>/<exp>_inference.ipynb
```

Kaggle に push する前に、Kaggle 用ディレクトリを生成します。生成 notebook には、`settings.py`、`config.yaml`、補助 `.py`、`project.yml`、`src/` を復元する base64 zip bootstrap セルが先頭に入ります。これは Kaggle CLI が `code_file` の notebook 本体だけを API に送るためです。

```bash
task prepare-kaggle-notebooks EXP=expXXX_title EXTRA_ARGS="--strict"
```

生成後に `experiments/<exp>/kaggle/<kind>/config.yaml` や補助 `.py` だけを手で直しても、notebook 先頭の bootstrap ZIP は古いままになることがあります。派生 CPU/GPU package などを手で作った場合は、push 前に bootstrap ZIP 内の `config.yaml` も期待値になっていることを確認するか、正の編集対象を直して `prepare-kaggle-notebooks` を再実行します。

Kaggle push と同時に実行したい場合は、準備時に `--run-on-push` を付けます。

```bash
task prepare-kaggle-notebooks EXP=expXXX_title EXTRA_ARGS="--notebook train --kernel-id username/expXXX-train --title 'expXXX train' --run-on-push --strict"
task push-kaggle-train EXP=expXXX_title
task kaggle-status KERNEL=username/expXXX-train
task kaggle-logs KERNEL=username/expXXX-train
```

train-side CV の評価だけなら、この時点で output archive は取得せず、CLI 2.2.3 の live SSE logs / notebook cell 出力 / Kaggle UI 上の metrics を確認して記録します。

Kaggle CLI は `2.2.3` を正とし、notebook のログ取得は実行中・完了後とも `kaggle kernels logs -f owner/slug` に統一します。このコマンドは Kaggle UI と同系統の live SSE に接続し、stdout/stderr を逐次取得します。`--interval` は deprecated で 2.2.3 では無視されるため使いません。queue / provisioning 中、または notebook がまだ stdout/stderr を出していない間は表示が空でも正常です。rich display、HTML、widget など stdout/stderr 以外の出力は notebook cell / Kaggle UI で確認します。live SSE が空であることだけを根拠に失敗、slug 間違い、停止、再 push と判断しません。

inference notebook を Kaggle に作成・更新する場合:

```bash
task prepare-kaggle-notebooks EXP=expXXX_title EXTRA_ARGS="--notebook inference --kernel-id username/expXXX-inference --title 'expXXX inference' --strict"
task push-kaggle-infer EXP=expXXX_title
task kaggle-status KERNEL=username/expXXX-inference
task kaggle-logs KERNEL=username/expXXX-inference
```

この inference は train と同じ実験 ID の一部として扱います。推論 port、submit-check、code submit、LB 記録は同じ `experiments/<exp>/SESSION_NOTES.md`、`result.md`、`metrics.json` に追記し、`experiment_summary.md` でも学習・推論を別行に分けません。

`kernel-metadata.json` の `competition_sources` に `project.yml` の competition slug を入れるため、Kaggle の Input 追加 UI は通常不要です。
`kaggle kernels init` は metadata 雛形を作るだけなので、この repo では `prepare-kaggle-notebooks` を使います。
VS Code Compatible URL の取得は Kaggle/VS Code 側の操作として残ります。

既存 kernel id を使わず、`KAGGLE_USERNAME` から既定 ID を生成することもできます。ただし再現性のため、実行・提出に使う kernel は `--kernel-id` を明示します。

```bash
task prepare-kaggle-notebooks EXP=expXXX_title EXTRA_ARGS="--notebook inference --kernel-id username/expXXX-inference --title 'expXXX inference' --run-on-push --strict"
task push-kaggle-infer EXP=expXXX_title
task kaggle-status KERNEL=username/expXXX-inference
task kaggle-logs KERNEL=username/expXXX-inference
kaggle competitions submit rogii-wellbore-geology-prediction -k username/expXXX-inference -v VERSION -f submission.csv -m "expXXX"
task submit-code EXP=expXXX_title KERNEL=username/expXXX-inference KERNEL_VERSION=VERSION OUTPUT_FILE=submission.csv MESSAGE="expXXX"
task record-submission EXP=expXXX_title EXTRA_ARGS="--allow-missing-file --cv 0.123 --public-lb 0.120 --notes baseline"
task update-summary
```

Kaggle output をローカルで検証したい場合だけ、実行後に出力を取得して明示パスを渡します。

CV 評価だけであれば、Kaggle output archive を取得せず、`kaggle kernels logs -f owner/slug`、notebook cell 出力、Kaggle UI 上の metrics を根拠に記録します。CLI 2.2.3 の `-f` は実行中の stdout/stderr を live SSE で逐次取得し、完了済み session では保存済みログを返します。`submission.csv`、OOF、`metrics.json`、feature importance、model manifest、SHA、後続実験の入力、提出形式検証など実ファイル確認が必要な場合だけ output を取得します。

```bash
task kaggle-output KERNEL=username/expXXX-inference OUT=/tmp/kaggle-output/expXXX_title/inference
task submit-check EXP=expXXX_title SUBMISSION=/tmp/kaggle-output/expXXX_title/inference/submission.csv
```

## 再現性の記録

deterministic anchor として扱う実験では、単に seed を固定しただけでは足りません。`docs/06_reproducibility.md` に従って次を記録します。

- Kaggle kernel id / version / source / runtime CPU-GPU / internet disabled。
- train cache、test feature、schema、model manifest、prediction、submission の SHA。
- gzip 生成物は raw `.csv.gz` SHA と decompressed content SHA を分け、determinism の主証拠は decompressed content SHA にする。
- PF/Beam や likelihood-PF は per-well stable seed を使い、global RNG と threaded random consumption を避ける。
- GPU 学習は bitwise 再現と仮定せず、必要なら CPU deterministic control と submission diff を作る。
- code competition の hidden test で raw-test feature regeneration が走る場合は、inference を 2 回以上 rerun し、feature content SHA、prediction SHA、submission SHA が安定するか確認する。

## 必須ログ

- `experiments/<exp>/README.md`: 状態概要と正の記録へのリンク。
- `experiments/<exp>/SESSION_NOTES.md`: 実行コマンドと現在の作業ログの正。
- `experiments/<exp>/result.md`: 解釈、実行証拠、ユーザーの採否判断の正。
- `experiments/<exp>/metrics.json`: CV/LBなど機械処理する数値の正。
- `experiment_summary.md`: 実験間の比較。

statusは当面`metrics.json`の1フィールドで管理します。`planned`、`running`、`debug_completed`、`scaffold_completed`、`failed`は実行状態、`usable`、`completed`、`deprecated`、`discarded`はユーザー判断後だけ設定する状態です。`leak-risk`は検証リークの注意表示で、採否や完了を意味しません。

## 提出前チェック

- `project.yml` の competition、validation、submission、runtime 設定が ROGII 用に設定済み。
- `submission.csv` は `id,tvt` の 2 列で、sample submission と `id` 順が一致している。
- 予測に NaN、inf、余分な列がない。
- CV は well holdout で、`TVT_input` NaN 行だけを score している。
- Kaggle Notebook metadata で internet が disabled になっている。
