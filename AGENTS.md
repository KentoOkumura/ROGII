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

- リポジトリ内の自動化コマンドは、対応するターゲットと`task`コマンドの両方が利用できる場合は`task <target>`を使います。`task`コマンドがない場合は同名の`make <target>`を使います。対応ターゲットがないrepo-local Python scriptだけ、リポジトリルートから`uv run python <script> ...`で実行します。PATH上の裸の`python` / `python3`を試してからfallbackしません。TaskfileとMakefileは`UV_CACHE_DIR=/tmp/uv-cache`と`PYTHONDONTWRITEBYTECODE=1`を既定で設定します。Ruff cacheは`/tmp/ruff-cache`へ置き、pytest cache providerは無効化します。managed sandboxでTask/Makeを経由せず`uv`を直接実行するときは、repo-local script、Kaggle CLI、`uv sync`などの用途を問わず、既定のuv cacheが書き込み可能だと確認できない場合は最初から`PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv ...`を使います。Makefile内部の`.venv/bin/...`は許可された実装です。`uv`も対応ターゲットも利用できず`.venv/`が準備済みの場合は、repo-local scriptを`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python`で直接実行できます。Kaggle Notebook、Colab、外部containerなど、リポジトリ外の実行環境内で指定されたinterpreterはこの規則の対象外です。
- リポジトリ内でKaggle CLIを直接実行する場合は、lockfileで固定した版を使うため`uv run kaggle ...`を使います。Taskfileは`uv run kaggle`、Makefileは`.venv/bin/kaggle`を使います。CLI構文を説明するだけの記載は裸の`kaggle`表記でも構いません。
- 検証範囲は変更範囲に合わせます。通常の実験変更では対象実験の`task check-exp`と`task test-exp`を使います。同梱skillだけの変更では`task check-skills`、共通コードやルート設定の限定変更では影響するルートテスト、テンプレート変更ではそれらに加えて`task validate-template`を実行します。複数のルート共通領域へ広く影響するが実験固有テストには影響しない変更では`task test-common`を使います。全実験のテストを収集する`task test`は、変更が複数の実験固有テストにも影響する場合、またはユーザーが全件確認を明示した場合だけ実行します。
- 設計、実装方針、実験分岐、提出判断などで複数の妥当な選択肢があり、結果や作業方針に影響する場合は、独断で決めずユーザーに確認します。既存ルールやコードから明確に判断できる低リスクな細部は、作業を止めずに進めます。
- 実際の Kaggle submission は、ユーザーがその提出を明示的に依頼または承認した場合だけ実行します。実験化、Notebook の準備・push・実行、提出前検証の承認を、submission の承認として扱いません。
- 実験を完了とするか、採用または不採用とするかは、エージェントだけで確定せずユーザーに判断を仰ぎます。判断を求める際は、比較対象、CV・LB、実行証拠、未解決事項を整理し、推奨する判断とその理由を伝えます。ユーザーが判断するまでは、`result.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` などに完了・採用・不採用が確定したものとして記録しません。
- ユーザーが実験の完了を判断した時点で、その実験に関係する変更だけを確認して `git commit` し、現在の作業ブランチを `git push` します。作業ツリーにある無関係な変更は commit に含めません。commit または push に失敗した場合は、失敗理由と未完了の操作をユーザーに報告します。
- リポジトリ全体の保存場所は次のように分けます。
  - 実験固有のコード、設定、記録は `experiments/expXXX_name/`。
  - 実験結果や OOF を対話的に確認するアプリケーションは `app/`。
  - 複数の未着手候補や実験を束ねる検証中の上位仮説は、`KAGGLE_DIRECTION.md` の「検証中の仮説」を索引の正とします。
  - 未着手の実験候補の詳細は `docs/backlog/<candidate>.md`。`KAGGLE_DIRECTION.md` の「アイデアバックログ」は優先度と要約を示す索引とします。
  - 公式資料の要約は `docs/official/`、ディスカッションの保存は `docs/discussions/`、論文単位のメモは `docs/papers/`。
  - 取得した公開 Notebook と metadata は `docs/notebooks/`。
  - 完了した調査レポートは `docs/surveys/`、検索入口は `docs/surveys/README.md`。上位仮説を直接扱うレポートはfront matterの`hypotheses`へIDを記録し、上位仮説別索引から検索できるようにします。
  - docsから参照する説明用の図は`docs/images/`。調査で生成した未整理の図は`studies/`、実験生成物は`experiments/<exp>/artifacts/`へ置きます。
  - `docs/analysis/` は旧形式の保存場所です。新しい完了レポートは追加せず、既存文書は次に更新または再利用するときに内容を確認して `docs/surveys/` へ移します。一括移行や推測による補完は行いません。
  - 再利用するコードは `src/`、その場限りの調査コードと生の表・図は `studies/`。
  - 公式から取得した生のコンペデータは `data/raw/`、外部データは `data/external/`、再利用する加工済みデータは `data/processed/`。
  - 実験固有のテストは `experiments/<exp>/tests/`、複数実験やリポジトリ全体に関わるテストはルートの `tests/`。
  - 実験で参照する小規模な固定データは`experiments/<exp>/assets/`へ置き、トップレベルの`assets/`は作りません。
  - 新規実験、steering、調査レポートを作成するための雛形は`templates/`。
  - リポジトリが管理する自動化と外部ツールの起動ラッパーは`scripts/`、Gitで追跡しない外部ツールのcloneやローカル配置は`tools/`に置きます。
- トップレベルの`artifacts/`は使いません。実験生成物は`experiments/<exp>/artifacts/`へ集約し、必要な分類はその下のサブディレクトリで表します。旧実験の`features/`と`variants/`は履歴として残せますが、新規作成せず、次にその生成物を更新するとき`artifacts/`配下へ移します。その場限りの調査表・図は`studies/`、確認済みの調査結論は`docs/surveys/`へ保存します。
- 提出監視中のpollingログは一時生成物とし、Gitへ保存しません。CV/LBなど機械処理する数値とkernel version・Kaggle Notebook実行時間・生成物SHAなどの構造化された実行証拠は対応する実験の`metrics.json`、submission ref・提出日時・submission scoring status・監視開始からscore確定までの所要時間を含む詳細な時系列は`SESSION_NOTES.md`、証拠への参照と結果の解釈は`result.md`を正とします。リポジトリ直下の`SUBMISSIONS.md`はsubmission refを専用列に持つ横断履歴であり、`metrics.json`から取得したCV/LBと、必要な場合だけ採点状態・Notebook実行時間・採点所要時間の最終スナップショットを保持します。`SUBMISSIONS.md`のスナップショットは正本の代わりにせず、状態遷移、観測時刻、実行コマンドを重複記録しません。スコア確定時は先に`record-exp`で`metrics.json`を更新し、その値を`record-submission`が読み取って提出履歴へ記録します。同じsubmission refを再記録した場合は新しい行を作らず、既存行のCV/LBと明示されたメモを更新します。code competitionの出力をローカル取得していない場合は、Kaggle側で対象ファイルを確認したうえで`record-submission`に`--allow-missing-file`を渡し、ローカル行数・列・SHAを未取得として記録します。Notebook実行時間は`notebook_runtime_seconds`、submission scoring所要時間は`scoring_elapsed_minutes`とし、どちらも`runtime`と呼びません。
- 実験記録は、`metrics.json` を機械処理する数値、実験の唯一のstatusフィールド、構造化された実行証拠、`config.yaml`をroute・設定・系譜と再現性方針、`result.md` を証拠への参照・解釈・ユーザーの採否判断、`SESSION_NOTES.md` を実行中の時系列ログの正とします。実験の `README.md` は目的、差分、リスク、次アクションとこれらへのリンクに留め、設定、系譜、実験status、CV/LB、SHAを複数ファイルへ手作業で重複記録しません。submission scoring statusは時系列イベントであり、`metrics.json`の実験statusとは別物です。
- 旧形式の実験READMEにroute、status、CV/LBや詳細結果が残っていても一括削除しません。その実験を次に更新するとき、routeが`config.yaml`、status・数値・構造化された実行証拠が`metrics.json`、証拠への参照と解釈が`result.md`、時系列の作業履歴が`SESSION_NOTES.md`に揃っていることを確認してから、READMEを概要と正の記録へのリンクへ簡素化します。
- `docs/surveys/` を完了した調査・分析・統合説明の正とします。実験前の文献・公開Notebook調査と、実験後の分析・実験横断の結論のどちらも対象です。`studies/` や `experiments/` を完了した調査レポートの検索入口にしません。
- 生のコンペデータ、モデル重み、大きな生成物を Git に保存しません。

## 実験の状態

- 実験の状態は当面 `metrics.json` の単一の `status` で管理します。このフィールドは実験statusだけを表し、Kaggle submissionのscoring statusには使いません。
- `planned`、`running`、`debug_completed`、`scaffold_completed`、`failed` は実行状態です。
- `usable`、`completed`、`deprecated`、`discarded` はユーザーの判断を表します。エージェントはユーザーの明示判断前にこれらへ変更しません。
- `leak-risk` は検証リークの注意表示で、採用・不採用・完了の判断ではありません。
- 既存実験の状態は一括変更せず、次に状態を更新するときにこの定義へ合わせます。

## 仮説とアイデアバックログの引き継ぎ

- 上位仮説は、複数の未着手候補や実験で検証する反証可能な問いです。未着手候補は、その一部を1回の実験として検証できる具体的な設計です。`docs/backlog/` を上位仮説単位にまとめず、1候補1ファイルを維持します。
- `KAGGLE_DIRECTION.md` の「検証中の仮説」「アイデアバックログ」節と `docs/backlog/` の作成、内容更新、対応関係・状態・優先度変更、削除は `kaggle-strategy` だけが担当します。`kaggle-idea-forge`、`kaggle-review-exp`、`kaggle-oof-readout` など他のskillは候補と根拠を作り、バックログへ反映する場合は同じターンで `kaggle-strategy` を使って引き渡します。
- 上位仮説のIDは `HYP-YYYYMMDD-NN` 形式とし、同じIDを再利用しません。発行前にリポジトリ全体を検索して未使用であることを確認します。`HYP-19000101-90`から`HYP-19000101-99`は自動テスト専用の予約範囲とし、実運用では発行しません。新しい上位仮説を追加するときは、`KAGGLE_DIRECTION.md` の「検証中の仮説」に、placeholderではない反証可能な仮説、対応する未着手候補、対応する実験、残っている問いを記録します。実験の`lineage.hypothesis_id`には、「検証中の仮説」または`final` / `superseded`の調査レポートに登録されたIDだけを設定します。調査レポートの状態と`superseded_by`の運用は`docs/surveys/README.md`を正とします。
- 各候補は主に検証する上位仮説を1つだけ持ちます。別の仮説にも関係する場合は関連仮説として記録できますが、主仮説を複数にしません。候補詳細には、上位仮説ID、この候補が直接検証する範囲、この候補だけで上位仮説を判断できるか、判断に残る検証を記録します。
- 未着手候補の機械処理用優先度は `P0`、`P1`、`P2`、`P3`、`P4` のいずれかとします。`P0`は他作業を止めて最初に扱う候補、`P1`は次に扱う候補、`P2`はP1の後または先行証拠の成立後に扱う候補、`P3`は後続候補、`P4`は再開条件が成立するまで保留する候補です。`KAGGLE_DIRECTION.md` の優先度列にはこのtokenだけを記録し、日本語の高・中・低や理由を混在させません。優先度の理由は `docs/backlog/<candidate>.md` の別フィールドへ記録します。戦略文脈の自動収集は `P0` から `P2` の候補詳細を先に読み、`P3` と `P4` は対象候補を検討するときに読みます。
- ユーザーが壁打ち結果を「バックログ化」「バックログへ追加」と依頼した場合は、同じセッションで次の両方を更新します。
  - `KAGGLE_DIRECTION.md` の検証中の仮説と未着手バックログへ、上位仮説ID、優先度、短い要約、依存関係、状態、詳細ファイルへの相対リンクを追加する。既存仮説に属する候補なら新しい仮説IDを発行しない。
  - `docs/backlog/_TEMPLATE.md` を使って `docs/backlog/<candidate>.md` を作り、壁打ち時の根拠と実験境界を保存する。
- `docs/backlog/<candidate>.md` を未着手候補の詳細の正とします。`KAGGLE_DIRECTION.md` に同じ長文を複製しません。1候補につき1ファイルとし、ファイル名は未着手表の候補名と一致させます。
- 未着手候補が0件の状態は正常です。`KAGGLE_DIRECTION.md` の未着手表はヘッダーだけを残し、検証を通すためのダミー候補や空の詳細ファイルを作りません。
- バックログ化だけを依頼された時点では、exp番号の採番、`.steering/`、実験フォルダ作成、コード実装、Kaggle実行を行いません。同時に実験化や実装も依頼された場合は、その依頼範囲に従います。
- 候補の状態は、このリポジトリ内の管理用語として次のいずれかを使います。これは実験結果の採用・不採用・完了を表しません。
  - `検討メモ・設計不可`: 結果や実装方針に影響する未決事項が残っている。次セッションは推測で補完せず、設計や実装の前にユーザーへ確認する。
  - `設計可能・実験化未承認`: 仮説、根拠、親実験、変更するもの、固定するもの、最小検証、成功条件、停止条件、禁止する代替実装、未決事項が記録され、未決事項が`なし`である。ただし実験化はまだ承認されていない。
- 詳細ファイルには、観測事実と仮定を分け、関連する `result.md`、metrics、保存済み生成物、一次資料へのパス、壁打ちで採らなかった案と理由も記録します。情報不足を推測で埋めず、未決事項として残します。
- 別セッションで候補を設計・実装するときは、バックログ表だけから内容を再構成しません。最初に対応する `docs/backlog/<candidate>.md` と根拠ファイルを読み、固定するもの、変更するもの、最小検証、成功条件、停止条件、実行しないこと、未決事項を短く提示します。詳細記録と異なる解釈または重要な未決事項があれば、コード作成前にユーザーへ確認します。
- ユーザーが候補の実験化を承認したら、exp番号を採番して `.steering/YYYYMMDD-expXXX-title/` を作ります。候補詳細の上位仮説ID、契約、根拠、判断履歴は`requirements.md`、その契約に対する実装方法と承認済みの差分は`design.md`、作業順序と確認項目は`tasklist.md`へ分担して欠落なく移し、同じ契約本文を複製しません。実験の `config.yaml` では `lineage.hypothesis_id` と `lineage.backlog_candidate` に引き継ぎます。移行を確認後、`kaggle-strategy` が `docs/backlog/<candidate>.md` と未着手バックログの行を削除し、「検証中の仮説」の対応候補を、実験ディレクトリ名をバッククォートで囲んだコード表記のラベル、同名の`experiments/<exp>/`をリンク先とする対応実験へ更新します。複数実験は`<br>`で区切ります。以後は `.steering/` と `experiments/<exp>/` を正とします。
- 1件の実験が終了しても、残る検証があれば上位仮説を終了扱いにしません。関連実験の証拠をまとめ、ユーザーが上位仮説を支持、棄却、保留のいずれにするか判断した後、上位仮説IDを本文とfront matterの`hypotheses`に持つ実験横断の結論を `docs/surveys/` に保存し、上位仮説別索引への反映を確認してから「検証中の仮説」から外します。
- この規則の導入前から存在する候補や実験には、上位仮説IDを推測で一括付与しません。既存候補は索引上で `未整理` とし、次に更新または実験化するとき、根拠を確認して既存または新規の上位仮説へ紐づけます。導入前から詳細ファイルなしで存在する候補も、コード作成前に詳細ファイルを作ってユーザー確認を得ます。

## Skill 入口

- ユーザーが実験化を承認した場合のsteering作成、実験作成、コピー、実験契約に必要な Kaggle Notebook の実装・実行、実験記録、再現性確認、実験レビューは、backlog経由か直接承認かを問わず `kaggle-review-exp` を使います。backlog経由では検証中の仮説とバックログ側の対応変更や削除に `kaggle-strategy` を併用します。
- Kaggle API、Kaggle CLI、kernel push/pull/logs/output、slug/title 問題、network escalation、data sync は `kaggle-platform` を使います。
- Kaggle GPU quota が限られる場合の Colab notebook 作成、Google Drive 入出力、Colab runtime / session 対応は `colab-notebook-runner` を使います。
- 提出前の `submission.csv`、Kaggle Notebook output、`kernel-metadata.json`、sample submission 互換性の検証は `kaggle-submit-check` を使います。
- `kaggle competitions submit` 後の scoring 監視と LB 記録は `kaggle-submit-monitor` を使います。
- コード、notebook、失敗実行、OOF、leakage、実行時間、メモリのレビューは `kaggle-review` を使います。
- OOF、feature importance、feature cache、by-well metrics を結合した誤差分析は `kaggle-oof-readout` を使います。分析から出た候補を検証中の仮説またはバックログへ反映する場合は `kaggle-strategy` を併用します。
- 現在路線の停滞時や次実験の発想時に、既存案の微調整に偏らない問題表現、情報源、候補生成、融合、data generation、validation 案を独立生成する場合は `kaggle-idea-forge` を使います。このskillは反証可能な実験案を生成するだけで、検証中の仮説やバックログを作成・更新しません。選択した候補をバックログ化する場合は `kaggle-strategy` を使います。
- 検証中の上位仮説の追跡、壁打ち結果や他skillが生成した候補のバックログ作成・更新、次実験、ロードマップ、CV/LB 整合性、失敗パターン、優先順位付けの整理は `kaggle-strategy` を使います。
- 論文、過去解法、公開 notebook、ディスカッション、外部実装の調査は `kaggle-survey-papers` を使います。
- Kaggle の上位公開 notebook をメタデータ付きでローカル保存する場合は `kaggle-notebook-fetch` を使います。
- Kaggle discussion / forum を検索可能な Markdown として保存する場合は `kaggle-discussion-archive` を使います。

## 常時品質基準

- Kaggle Notebook の最初のフル実行と公式評価を Kaggle 上で行います。local smoke に必要な入力、依存関係、生成物がローカルに揃っている場合は、別途のユーザー承認なしに `task train-local` / `task infer-local` / `task execute-notebook-local` による smoke debug を許可します。local smoke の結果だけで公式スコアや Kaggle 実行完了を判断しません。
- ハイパーパラメータは `config.yaml` に置きます。notebook や補助モジュール内の暗黙の定数は避けてください。

## GitHub 自動化

- このテンプレートは CI で `.github/workflows/template-check.yml` を使います。
- Anthropic の GitHub アプリと必要なシークレットをこのリポジトリで使う予定がない限り、Claude Code の GitHub Actions は追加しないでください。
