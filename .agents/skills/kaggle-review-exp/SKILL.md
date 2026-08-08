---
name: kaggle-review-exp
description: "`docs/backlog/` の候補を実験へ移行し、`experiments/expXXX_name` 配下の Kaggle 実験を作成、コピー、実装、実行、debug、記録、要約、レビューする。実験開始、backlogからsteering docsへの引き継ぎ、train/inference Notebook実装、Code competitionのhidden test対応、過去実験のコピー、Kaggle train/inference実行、`SESSION_NOTES`/result/metricsの記録、`expXXX`のレビュー、実験ドキュメント確認、実験結果の信頼性監査を求められたときに使う。"
---

# Kaggle 実験ワークフローとレビュー

実験ライフサイクルの作業と、実験記録のレビューに使う。コードだけのレビューには `kaggle-review`、提出ファイルの検証には `kaggle-submit-check` を使う。

## 手法忠実性ガード

ユーザーが特定手法、論文、公開notebook、discussionの実装を求めた場合は、実験作成やコード編集の前に次を行う。

1. 一次資料または参照実装を確認し、`input -> target/objective -> output -> loss -> decode -> context unit`を手法契約として `.steering/.../requirements.md` と `design.md` に記録する。
2. 実装範囲を次のいずれかに分類する。
   - `faithful`: 手法契約の本質的な要素をすべて実装する。
   - `staged-faithful`: target、output表現、loss、decode、context unitを保ったまま、データ量、fold数、epoch数、解像度などだけを縮小する。
   - `proxy`: target、output表現、loss、decode、whole-group / local contextのいずれかを変更または省略する。
3. `proxy` の場合は、省略する機構、proxyで検証できない主張、完全実装との追加コストをユーザーに示す。明示承認を得るまで、実験フォルダ作成、コード実装、Kaggle pushを行わない。
4. 実験名と記録には実装した機構だけを書く。入力だけに使った表現をoutput headやtraining objectiveの実装と呼ばない。
5. negative resultは `(signal, representation, role, fusion, validation regime, compute regime)` のどこを閉じるかを `result.md` に記録する。`proxy` や単一実装の失敗で method family 全体を閉じない。

同じ親実験または機構familyの `parameter tuning`、`add-only feature`、`selector-only`、後処理が2件連続する場合、またはpositiveなoracle headroom / coverage / 誤差非相関性に対しend-to-end改善が得られない場合は、3件目の小改善の前に `kaggle-idea-forge` でrepresentation auditを行う。少なくとも1件の target / output / decode / context unit を変える案と小改善の継続案をユーザーに提示し、選択を得てから実験化する。

## GPU 学習コストガード

Kaggle GPU を使う train push の前に、必ず次を確認する。

- 実行される active variant 数、model/config 数、fold 数、合計 booster 数を数える。
- 親実験や既存 baseline/control を再学習する variant が含まれるか確認する。
- 既に信頼できる親実験の OOF / metrics / by-well / prediction がある場合は、それを baseline として参照し、新規実験では原則として新しい variant だけを学習する。
- 既存 baseline/control を Kaggle GPU で再学習する場合は、既存結果では代替できない理由、追加 GPU コスト、合計 booster 数をユーザーに説明し、明示承認を得る。承認なしに push しない。
- `SESSION_NOTES.md` に実行予定の variant/config/fold/booster 数と、control 再学習の有無を記録する。

比較の厳密性より GPU コストが問題になる場面では、保存済み parent metrics を baseline とし、runtime / code 差分は result に注意書きする。control 再学習は「必要なら後で相談して追加実行」に回す。

## 実験ライフサイクル

1. 現在の流れを把握する。
   - 存在する場合は `AGENTS.md` を読む。
   - `KAGGLE_DIRECTION.md`、`experiment_summary.md`、`docs/surveys/README.md`から対象実験・トピックの完了調査を読む。
   - 最近の `experiments/*/SESSION_NOTES.md` を確認する。
   - 特定手法の実装依頼では、参照sourceと手法契約を特定する。
   - 同じ親実験または機構familyの直近の子実験を `parameter / add-only / selector-only / postprocess / mechanism / representation` に分類し、representation auditの発動条件を確認する。
   - `KAGGLE_DIRECTION.md` の未着手候補から実験化する場合は、対応する `docs/backlog/<candidate>.md` とそこから参照される根拠を読む。未着手表だけから設計を再構成しない。
   - 導入前の候補で詳細ファイルがない場合は、コード作成前に `kaggle-strategy` を使って詳細ファイルを作り、推測できない事項を未決としてユーザーへ確認する。このskillから `docs/backlog/` を直接更新しない。
2. backlog候補から実験化する場合は、採番やコード作成の前に、固定するもの、変更するもの、最小検証、成功条件、停止条件、実行しないこと、未決事項を短く提示する。重要な解釈差または未決事項があればユーザー確認まで停止する。`設計可能・実験化未承認` は実験作成の承認を意味しない。
3. ユーザーの実験化承認後、実験を作成または変更する前に steering docs を作る。

```bash
task new-steering EXP=expXXX_title
```

Task が使えない場合は Makefile の同等コマンドを使う。

```bash
make new-steering EXP=expXXX_title
```

4. リポジトリに `templates/steering/` があれば、それを元に `.steering/YYYYMMDD-expXXX-title/{requirements.md,design.md,tasklist.md}` を埋める。backlogから移行する場合は、詳細ファイルの根拠、仮説、親との差分、固定事項、最小検証、成功条件、停止条件、実行しないこと、未決事項、判断履歴を欠落なく移す。移行確認後、`kaggle-strategy` を使って元の `docs/backlog/<candidate>.md` と未着手バックログ行を削除する。このskillからバックログを直接変更しない。
5. 実験を作成、またはコピーする。

```bash
task new-exp EXP=expXXX_title
task new-exp EXP=expXXX_title SOURCE=experiments/expYYY_parent
```

Makefile の同等コマンド:

```bash
make new-exp EXP=expXXX_title
make new-exp EXP=expXXX_title SOURCE=experiments/expYYY_parent
```

   - 学習、推論、提出は原則として同じ `experiments/expXXX_title/` で管理する。train-side CV が良かった候補を inference port / submit するだけなら新しい exp を作らず、同じ実験の `<exp>_inference.ipynb`、`SESSION_NOTES.md`、`result.md`、`metrics.json`、`submissions/SUBMISSIONS.md` を更新する。
   - 新しい exp を作るのは、仮説、特徴量面、モデル構造、評価条件、route の主目的が変わる場合に限定する。過去に学習・推論を分けて作成済みの exp は履歴として維持する。

6. 明らかに再利用できるコードでない限り、実装は実験フォルダ内に置く。
   - 実験固有のロジックは `experiments/expXXX_title/` に置く。
   - 共通 utility は `src/` に置く。
   - その場限りの調査コードと生の表・図は `studies/` に置く。完了した調査結論は、対象が単一実験でも `docs/surveys/` にメタデータ付きレポートとして記録する。
   - hyperparameter、route、系譜は `config.yaml` に置く。
   - `experiment.route` は `ml_model`、`pf_beam`、`ensemble` のいずれかにする。route をまたぐ場合は主目的の route を選ぶ。ML と PF/Beam の両方が予測生成に本質的に寄与する blend / public notebook replay / meta feature 化は `ensemble` を使い、詳細を `lineage.diff_summary` と `SESSION_NOTES.md` に記録する。
   - `<exp>_train.ipynb` / `<exp>_inference.ipynb` は実験コードの正の編集対象なので、人間が読める notebook 構成にする。薄い `from module import main; main()` だけの notebook は避け、setup、入力確認、学習/監査/推論、評価、metrics/生成物保存を Markdown 見出し付きのセルで追えるようにする。
   - 新規 notebook 実装、または既存 notebook の大きな作り替えでは、まず Jupytext percent 形式の `.py` を作成し、`# %%` / `# %% [markdown]` でセルを構造化してから `.ipynb` に変換する。
   - Jupytext 起点の notebook は compact self-contained を基本形にする。依存 `.py` を丸ごと貼り付けず、実験遂行に必要な関数・定数だけを AST 追跡または手動確認で抽出して notebook 内に持ち込む。親実験に `*_compact_selfcontained_train.py` / `*_compact_selfcontained_inference.py` が存在する場合は、通常版 `*_train.py` / `*_inference.py` ではなく compact self-contained 版を最優先の構成参照元にする。
   - 同じ実験ディレクトリ内の helper `.py` import は、ユーザーが明示的に許可した場合や既存正規 notebook の保守を除き、新規 self-contained notebook では避ける。外部ライブラリ、標準ライブラリ、Kaggle Dataset の許可済み package はこの制約の対象外。
   - 既存の同名 `.ipynb` はユーザーの明示承認なしに上書きしない。試行時は `_compact_selfcontained_train.py` / `_compact_selfcontained_inference.py` のような別名で生成し、採用判断後に正規名へ反映する。
   - marimo は標準採用しない。notebook の正は通常の `.ipynb` とし、上位ロジックは `.ipynb` のセルに展開する。重い helper や再利用ロジックだけを補助 `.py` に残す。
   - 既存の `.py` 実装を読める notebook に寄せるだけなら、新しい実験番号は切らない。仮説、特徴量、モデル構造、評価条件、route の主目的、推論方針、提出候補が変わる場合だけ新しい exp を作る。
7. フル実行の前に validation と静的チェックを実行する。最初のフル実行と公式評価は Kaggle 上で行う。local smoke に必要な入力、依存関係、生成物がローカルに揃っている場合は、別途のユーザー承認なしにsmoke debugを行ってよい。

Jupytext 変換と検証:

セル目次は固定章名ではなく、実験内容に応じた「役割スロット」として設計する。新規 notebook や大きな作り替えでは、薄い orchestration notebook にせず、次の役割が notebook 上で追えることを完了条件にする。

- Imports
- Runtime and configuration helpers
- Input / cache / raw data checks
- Core feature / candidate / replay generation helpers
- Route-specific execution helpers
  - `ml_model`: model training and artifact helpers
  - `pf_beam`: particle filter / beam search / candidate generation helpers
  - `ensemble`: blend / stacking / replay helpers
  - audit / diagnostic: metric / readout / diagnostic helpers
- Setup and configuration
- Execution orchestration
- Metrics, diagnostics, summaries, and generated artifacts

親実験に compact self-contained notebook が存在する場合は、実装完了前に必ず章立てと記載量を比較し、結果を `SESSION_NOTES.md` に記録する。親 compact と比べて章立てが大きく欠ける、または正規 notebook が同一 exp の helper module を呼ぶだけの薄い構成なら、実装完了扱いにしない。

```bash
rg -n "^# %% \\[markdown\\]|^# #|^# ##|^# [0-9]\\." experiments/expYYY_parent/*compact_selfcontained*_train.py experiments/expXXX_title/expXXX_title_train.py
wc -l experiments/expYYY_parent/*compact_selfcontained*_train.py experiments/expXXX_title/expXXX_title_train.py
```

compact self-contained 化で `settings.py` や helper `.py` から runtime helper を持ち込む場合、notebook セル上では `__file__` が未定義になるため使わない。`PACKAGE_DIR = Path.cwd()` を基本にし、`Path(__file__).resolve()` / `Path(__file__).with_name(...)` / `Path(__file__).resolve().parents[...]` は notebook-safe な形へ置き換える。Kaggle push 前に次を確認し、`__file__` が残っていれば修正する。

```bash
rg -n "__file__|Path\\(__file__\\)" experiments/expXXX_title/expXXX_title*_train.py experiments/expXXX_title/expXXX_title*_inference.py
```

ML train の目次例:

```python
# %% [markdown]
# # expXXX_title train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Train feature assembly helpers
# 4. Model training and artifact helpers
# 5. Setup and configuration
# 6. Input and feature contract
# 7. Train variants
# 8. Metrics and generated artifacts

# %%
# imports

# %% [markdown]
# ## 2. Runtime and configuration helpers
```

PF/Beam 生成の目次例:

```python
# %% [markdown]
# # expXXX_title train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and replay cache helpers
# 4. PF / Beam candidate generation helpers
# 5. Candidate scoring and path selection helpers
# 6. Setup and configuration
# 7. Run PF / Beam generation
# 8. Metrics, diagnostics, and generated artifacts
```

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/expXXX_title/expXXX_title_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/expXXX_title/expXXX_title_compact_selfcontained_train.py
.venv/bin/python -m py_compile experiments/expXXX_title/expXXX_title_compact_selfcontained_train.py
.venv/bin/ruff check experiments/expXXX_title/expXXX_title_compact_selfcontained_train.py --select F821
```

train / inference の両方に同じ検証を行う。`py_compile` で生成された確認用 `.pyc` は不要なら削除してよい。

Kaggle train push 前の追加チェック:
- `config.yaml` の `model.feature_ablation.active_variants` と `model.training.active_modes` を読み、学習対象数を数える。
- LightGBM family のように helper 内で複数 config を展開する場合は、その個数も数える。
- 親実験 control の再学習が含まれるなら、ユーザーの明示承認がない限り config から外すか `enabled: false` にする。

Kaggle train / inference push 前の必須runtime resource / quota確認:
- `task push-kaggle-train`、`task push-kaggle-infer`、または`kaggle kernels push`の直前に、`kaggle-platform`の「Push 前の runtime resource / quota 確認」を実行する。
- GPU / TPUでは`kaggle quota --format json`の残時間とrefresh時刻を確認し、想定runtimeと照合する。残時間不足リスクを承知で実行するユーザーの明示承認がある場合は、その承認とリスクを`SESSION_NOTES.md`へ記録してpushできる。
- Kaggle CLI 2.2.3ではアカウント全体のActive Sessions数を取得できないため、Active Sessions確認をpush前gateにしない。

```bash
task validate-template
task validate-exp EXP=expXXX_title
task fmt
task test
task prepare-kaggle-notebooks EXP=expXXX_title EXTRA_ARGS="--notebook train --kernel-id username/expXXX-title-train --title 'expXXX title train' --run-on-push --strict"
task push-kaggle-train EXP=expXXX_title
```

prepare 時は `kaggle-platform` の slug/title 手順に従い、canonical slug が 50 文字以内で `id` と `title` 由来 slug が一致していることを確認する。実験ディレクトリ名全体では上限を超える場合、意味を保った短縮名を明示的に決める。push 後の存在確認、CLI 2.2.3 live SSE logs、必要時の output 取得、`status` 500、API lag、slug/title 問題も `kaggle-platform` に従う。train-side CV の完了判定と評価は、原則として live SSE logs / cell 出力、metrics 保存表示、Kaggle UI 上の実行結果を根拠にし、Kaggle output archive は既定では取得しない。

Kaggle output archive 取得の判断:
- CV 評価だけなら、`kaggle kernels logs -f owner/slug`、notebook cell 出力、Kaggle UI に表示された metrics で確認し、`SESSION_NOTES.md` / `result.md` / `metrics.json` に記録する。output archive を丸ごとダウンロードしない。
- output を取得するのは、`submission.csv`、OOF、`metrics.json`、feature importance、model manifest、SHA、後続実験の入力、提出形式検証など、ローカルで実ファイルを読む必要がある場合だけにする。
- 学習完了時は、推論に必要なモデル、前処理状態、特徴量名と順序、variant / mode / fold、ファイル形式、相対パス、SHA が保存され、model manifest から同じ実験の inference notebook が再学習なしで解決・読み込みできることを確認する。
- logs や notebook 表示に CV、fold 別 score、variant/config、保存先パスが不足している場合は、まず notebook 側の表示を改善する。すでに実行済みで不足分を補う必要がある場合だけ output 取得を検討する。

train CV が良かった候補を推論化または提出する場合も、同じ `EXP=expXXX_title` のまま inference notebook を作成・実行する。

### Code competition の推論実装

- 公開 `test/` と `sample_submission.csv` は実行確認用のサンプルとして扱う。Code submissionでNotebookが再実行されると、入力は採点用のhidden testとそのsample submissionに差し替えられる。公開testで作った `submission.csv` はsmoke testであり、本番提出の予測結果ではない。
- Kaggle実行環境の現在の入力からcompetition root、testファイル、well IDを動的に列挙する。公開例のwell ID、well数、ファイル名、行数、ID一覧、内容SHA、hidden testの分布を固定値としてハードコードしない。
- 実行環境の `sample_submission.csv` を提出schema、ID集合、行順、行数の正とする。予測をIDで1対1に整列し、欠損予測、重複ID、余分なID、行数不一致を検証してから `submission.csv` を作る。
- hidden testに存在する入力と保存済みmodel manifest / model生成物だけで、特徴量生成から予測までを完結させる。train-only列、ローカル専用cache、公開testの保存済み予測に依存しない。
- 公開test固有のID、SHA、行数、予測値に基づく分岐、ゲート、fallbackを本番推論に入れない。公開例との一致やSHA検査を診断用に残す場合は、hidden testで不一致になることを正常とし、推論を中断しない。
- 公開例の小さなtestではなく、hidden testの可変なwell数・行数を前提にメモリと実行時間を設計する。必要に応じてwell単位の逐次処理、chunking、上限付き並列度を使う。

```bash
task prepare-kaggle-notebooks EXP=expXXX_title EXTRA_ARGS="--notebook inference --kernel-id username/expXXX-title-inference --title 'expXXX title inference' --run-on-push --strict"
task push-kaggle-infer EXP=expXXX_title
```

Kaggle output をローカルに取得した場合だけ、`kaggle-submit-check` の手順で提出形式を確認する。

local smoke に必要な入力、依存関係、生成物がローカルに揃っている場合だけ、local smoke debug を実行する。結果だけで公式スコアやKaggle実行完了を判断しない。

```bash
task train-local EXP=expXXX_title EXTRA_ARGS="--allow-local --debug"
task infer-local EXP=expXXX_title EXTRA_ARGS="--allow-local --debug"
```

8. 信頼できる結果は毎回記録する。
   - `SESSION_NOTES.md`: コマンド、現在の状態、出力、失敗、次アクション。
   - `result.md`: 解釈、実行証拠、ユーザーの採否判断の正。日本語で記載する。
   - `metrics.json`: CV/LBなど機械処理する数値の正。
   - 実験の`README.md`: 状態概要と正の記録へのリンク。CV/LBを手作業で重複記録しない。
   - `experiment_summary.md`: 実験横断の要約。
   - `submissions/SUBMISSIONS.md`: 提出した場合の提出履歴。
   - `KAGGLE_DIRECTION.md`: 実験結果と現行判断を記録する。アイデアバックログ節の削除・追加・更新が必要な場合は、候補、根拠、非使用条件、移行状態を `kaggle-strategy` へ引き渡し、同じターンで反映する。このskillからアイデアバックログ節や `docs/backlog/` を直接変更しない。
   - train CV、inference output、submit-check、code submit、Public LB は同じ実験記録に追記し、推論化だけを別実験として `experiment_summary.md` に分けない。
   - 通常の実験結果を記録するだけなら`result.md`で完結させる。実験構成・モデル説明、OOF／結果EDA、特徴量・failure mode、複数実験比較として再利用できる完了調査になった場合は、`docs/surveys/README.md`で既存レポートを探し、既存レポートの更新または新規surveyレポート作成を行う。
   - statusは`metrics.json`の単一フィールドを維持する。`planned`、`running`、`debug_completed`、`scaffold_completed`、`failed`は実行状態とする。`usable`、`completed`、`deprecated`、`discarded`はユーザーが判断した後だけ設定する。`leak-risk`は注意表示で、採否や完了を意味しない。既存実験の`config.yaml`に残るstatusは互換読み取り専用とし、次の状態更新は`metrics.json`へ記録する。

```bash
task update-summary
```

ユーザーが実験の完了を判断した場合は、回答を終了する前に `AGENTS.md` の完了時のGit手順を再確認して従う。手順の詳細はこのskillへ複製しない。

完了調査レポートを新規作成する場合:

```bash
task new-survey-report SURVEY_TITLE="..." SURVEY_SLUG="..." EXTRA_ARGS="--type experiment_review --experiment expXXX --topic ..."
task update-survey-index
task validate-surveys
```

surveyレポート本文には、結論、証拠範囲、実験構成・モデル説明、分析結果、解釈、関連する`result.md` / `metrics.json` / `studies/`、次のアクションを記載する。調査の正は`docs/surveys/`、実験の公式結果の正は`experiments/<exp>/result.md`として混同しない。

品質基準:
- ユーザーが依頼した手法契約と実装の `input / target / output / loss / decode / context unit` が一致し、`faithful` / `staged-faithful` / `proxy` の分類に根拠があること。
- `proxy` で省略した機構と検証できない主張が記録され、実装前のユーザー承認があること。
- negative resultが閉じる範囲と、残ったpositive submetric / oracle headroom / coverage / 誤差非相関性が記録されていること。
- CV を信頼する前に、validation 方針が明確であること。
- CV 評価だけであれば、Kaggle output archive を取得せず、logs / notebook cell 出力 / Kaggle UI 上の metrics を根拠としていること。
- notebook のフル実行と公式評価は Kaggle で行っていること。ローカル notebook 実行は、必要な入力と生成物が揃った smoke debug に限定する。
- code competition の inference は、このskillの「Code competition の推論実装」を満たし、公開 test 固有値のハードコードがなく、実行時の sample submission と ID で 1 対 1 に整列できること。
- 学習時と推論時の前処理が一致していること。
- すべての結果に、コマンド、config、CV、生成物、解釈、次アクションがあること。
- 結果と次アクションが `ml_model` / `pf_beam` / `ensemble` のどの route の anchor を更新するのか明確であること。
- 実装済みの backlog 項目を残さないため、必要な削除を `kaggle-strategy` へ引き渡して反映済みであること。
- 新規 backlog 候補は、完了した実験の証拠、非使用条件、未決事項を `kaggle-strategy` へ引き渡し、`docs/backlog/<candidate>.md` と `KAGGLE_DIRECTION.md` の整合および既存候補との優先度見直しが完了していること。
- backlogから始めた実験は、steering docsに固定事項、変更事項、最小検証、成功条件、停止条件、実行しないこと、判断履歴が引き継がれ、重要な未決事項が`なし`であること。

## 実験レイアウト

各実験には次を含める。

- `config.yaml`: パラメータ、route、派生元、実行メタデータ。
- `settings.py`: Kaggle Notebook 実行を正とするパス解決と設定読み込み。
- `<exp>_train.ipynb`: 学習 notebook。実験コードの正の編集対象。
- `<exp>_inference.ipynb`: 推論 notebook。実験コードの正の編集対象。
- `SESSION_NOTES.md`: 現在の状態、実行したコマンド、結果、次のアクション。
- `result.md`: 解釈、実行証拠、ユーザーの採否判断の正。日本語で記載する。
- 実験配下の`README.md`: 状態概要と正の記録へのリンク。日本語で記載する。
- `metrics.json`: ツールから読めるCV/LBなど数値の正。
- `artifacts/`、`features/`、`variants/`: 実験で生成した出力。提出 CSV はローカル実験ディレクトリには常設せず、Kaggle Notebook output として扱う。本文では artifact ではなく「生成物」と書く。

## Notebook 実装ルール

Kaggle Notebook が実行の正なので、`<exp>_train.ipynb` / `<exp>_inference.ipynb` は人間が読んで実験内容を追える形にする。

- marimo は標準では使わない。AI が編集しやすいことより、Kaggle Notebook と repo template の既存フローにそのまま乗る通常 `.ipynb` を優先する。
- notebook は、目的、設定確認、データ/OOF 読み込み、fold-safe な学習/推論、評価、生成物/metrics 保存がセル単位で分かる構成にする。
- セル構造は `.py` 側の `# %% [markdown]` 見出しで再現できるようにする。`## Contents` を置き、Imports、config、input checks、feature engineering、model、training/inference orchestration、metrics/artifacts などを出発点にする。ただし固定テンプレートではないため、実験内容に合わせて章を増減、統合、分割してよい。
- 薄い `run_*()` / `main()` 呼び出しだけの notebook は避ける。既存 `.py` に上位 orchestration がまとまっている場合は、次を notebook cell に展開する:
  - 設定、親実験、route、variant / mode / audit / split の確認。
  - 入力データ、OOF、feature cache、model manifest、sample submission の存在確認と preview。
  - どの fold / audit / mode / variant を実行するかの選択と validation。
  - 学習、推論、後処理、評価、metrics / 生成物 / SHA / summary 保存の手順。
- 重い helper 関数や再利用ロジックは補助 `.py` に置いてよい。具体的には PF/Beam、Numba、LightGBM fold 学習本体、重い特徴量生成、path resolver、validation / SHA utility は `.py` に残してよい。ただし notebook 側には「どの入力を読み、どの variant を比較し、何を保存するか」を明示する。
- `config.yaml` の主要値、variant 名、親実験、基準スコア、出力した生成物は notebook 上で確認できるようにする。
- train notebook では、モデルが特徴量重要度を出せる場合、fold ごとの特徴量重要度を平均した表を作り、上位特徴量を `matplotlib` でプロットする。
- PF/Beam 生成、public replay、診断 audit など model training がない notebook では、`Model training and artifact helpers` という章名を無理に使わず、`PF / Beam candidate generation helpers`、`Candidate scoring and path selection helpers`、`Replay helpers`、`Diagnostic metric helpers` など route-specific な章名に置き換える。
- Kaggle push 用 notebook には bootstrap セルが自動追加されるが、正の編集対象 notebook 自体は読みやすい構成を維持する。

## 再現性と記録

- stochastic feature generation、PF/Beam、GPU 学習、Kaggle bootstrap、保存済み model inference、code-submit hidden test 再生成を含む場合は、設計時点で `docs/06_reproducibility.md` を読む。
- 再現性ガードを満たせない場合は、理由を `SESSION_NOTES.md` に記録する。
- スコアを記録する場合は seed を固定し、検証方法を明記する。
- deterministic anchor として扱う場合は、seed だけでなく feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version を記録する。
- gzip 生成物は raw `.csv.gz` SHA ではなく、decompressed content SHA を主証拠にする。
- CV と LB が合わない場合は、追加チューニングに進む前に原因を確認する。
- 実行コマンド、設定、CV、生成物、次のアクションを書くまで、結果は記録済みと見なさない。
- スコア、失敗理由、次の候補は route 別に読めるように記録する。特に `ml_model` の CV 基準、`pf_beam` の決定的ルール候補、`ensemble` の ML + PF/Beam blend / public notebook replay は同じ根拠として混ぜない。

## 実験記録レビュー

1. ユーザーの依頼から、`exp003`、`expA07`、フォルダパスなどの実験識別子を特定する。
2. `docs/surveys/README.md`の実験番号別索引から、対象実験の完了済みモデル説明・OOF分析・比較レポートを確認する。
3. 同梱 reviewer を実行する。

```bash
python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py EXP_ID --root .
```

4. スクリプトが関連ありと判断したファイルを読む。
5. 次をレビューする。
   - 仮説が明示的で検証可能か。
   - 特定手法の実装では、手法契約と実装の `input / target / output / loss / decode / context unit` が一致するか。
   - `proxy` を忠実実装と呼んでいないか、proxyの結果からmethod familyを閉じていないか。
   - 元実験と変更点が明確か。
   - validation split がコンペや test 構造に合っているか。
   - CV/LB/result の数値に根拠があるか。CV だけなら logs / notebook cell 出力 / Kaggle UI の表示でよく、実ファイル確認が必要な場合だけ output 取得パスを求める。
   - 生成物、checkpoint、submission が命名され、再現可能か。
   - negative result が記録されているか。
   - 次アクションが証拠から自然に導かれているか。

## 出力

レビューでは次の形式を使う。

```markdown
**Findings**
- [Severity] [file:line] 問題、影響、修正案。

**Trust Assessment**
- Trustworthy / Partially trustworthy / Not trustworthy と、その理由。

**Next Action**
- 次にやる具体的な 1 ステップ。
```

ドキュメントが不足している場合は、補完して想像しない。どの証拠が欠けているか、最低限どのメモを追加すべきかを具体的に書く。
