# exp507_exp504_nested_rank_compact_addonly_on_exp413 セッションノート

## 目的

exp504のH512 pairwise rank面をhard-selected TVTではなくstrict nestedなcompact45特徴として、
exp413 final370へadd-onlyで渡す実験の設計を確定する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage N technical PASS・Stage D technical PASS / scientific FAIL・終端閉鎖
- 親 / matched control: exp413、保存OOF RMSE `7.884802794404715`
- rank source: exp504 v1、hard RMSE `8.114276980`、anchor比`-0.124054566 ft`
- feature formula: `final370 + rank_compact45 = final415`
- 実装・Stage N / Stage D train: 完了。inference / submissionは未実施のまま禁止
- 正規train Notebook: compact self-contained版を採用済み。inference Notebookはplaceholder
- 別名compact self-contained train source / Notebook: `.py` / `.ipynb`作成済み

## 2026-08-03 Stage D実行承認

- ユーザーの「次に進んでください」を、Stage N technical PASS後に提示した次工程である
  Stage D private Kaggle T4 package、push、run、固定all-AND gate判定の承認として記録した。
- 承認記録時刻: `2026-08-03 14:49:34 UTC`（`2026-08-03 23:49:34 JST`）。
- 実行対象は1 scientific treatment / 3 LightGBM configs / outer 5 = 15 GPU boostersのみ。
- 保存exp413 control再学習0、Stage N再実行0、候補/PF/HMM/Beam再生成0、inference 0、submission 0。
- GPU quotaは45.00h中25.90h使用、残り19.10h。類似exp502 Stage Dの実績は約5時間45分で、
  見積もり上は残量内である。
- kernel inputはexp072 / exp099 / exp111 / exp413 Stage C / Stage S / Stage D / exp507 Stage Nの
  7 kernelと、exp404 frozen predictions dataset 1件に固定する。
- planned kernelは`kentookumura/exp507-exp504-nested-rank-compact-stage-d`、private、T4、
  internet off、run-on-push。inference / submissionはStage D結果にかかわらず別承認とする。
- packageは21 cells / 10 code cells / output 0、support 45 files。bootstrap ZIPの全byte count / SHA、
  loose/package/embedded config、11 dependency destinationのsource byte一致を確認した。
- package Notebook SHA256: `ffcdbad040b98819eda29861cddfe2973ad53850b86239110641569fc56b4109`。
- kernel metadata SHA256: `e520e83e2a2e8e617d53b1883af60cab256fc85ee928b2edaf6322deb31796b5`。
- embedded/local config SHA256: `a569f04866035086f7801af469b3f9d475a39a082c225497897d1bebee06eb18`。
- scientific source SHA256: `a0ec0bb586a5c0cbecc301f9b4509bdacc2e2a430c2f4e0abe71d1bebe0490e8`。
- canonical slugの事前検索でStage D exact matchは0件。`2026-08-03 14:53:03 UTC`までに
  Kaggle kernel version 1をpushし、private T4 runを開始した。remote id_noは`129584313`。
- pullしたremote metadataでprivate、T4、internet off、7 kernel + 1 dataset、competition source 1、
  model source 0を確認した。remote Notebookは21 cells / 10 code / output 0で、Kaggleによる
  code filename/source表現の正規化を除き、全cell sourceがlocal packageと一致した。
- remote metadata SHA256: `38ec5fba3323d7b19b633f48e6776e49cf2c668fbc37dd0264b273be1a5477be`。
- remote Notebook SHA256: `06b8dfa98812de324a9397d10a7dd39d2e0c1997da75eacdf201891a95ac68bb`。

## 2026-08-03 Stage D完了

- Kaggle private T4 version 1（id_no `129584313`）は`COMPLETE`。terminal status検出は
  `2026-08-03 19:39:48 UTC`、最終log timeは`17,145.681811 sec`（約4時間45分46秒）。
- actual executionは1 treatment / 3 LightGBM configs / 5 folds = 15 GPU boosters、
  best-iteration tree合計`32,711`。保存exp413 control再学習0、Stage N再実行0、
  候補/PF/HMM/Beam再生成0、inference 0、submission 0。
- `3,783,989 rows / 773 wells / final415`、10 train/valid matrix partitions、15 model、
  final schema一意、Stage N manifest、control SHA、artifact SHAをすべてPASS。Tracebackは0。
- pooled OOFはexp507 `7.889515565580203`、保存exp413 `7.884802794404715`で、
  gainは`-0.004712771175488406 ft`（exp507が`0.004712771 ft`悪化）。必要な`>=0.03 ft`をFAIL。
- fold delta（exp507-exp413）はfold 0 / 1 / 2 / 3 / 4 =
  `+0.031800597 / +0.002376523 / -0.028155181 / -0.043276780 / +0.056461120 ft`。
  非悪化は`2/5`で必要な`>=3/5`をFAIL。
- scope deltaは`md_since 0--250 +0.036938807`、`250--1000 +0.012362003`、
  `1000+ +0.003913383`、hidden-like spatial `+0.024333965`、typewell-purged
  `+0.021782450 ft`。最大`+0.036938807 > +0.02 ft`で固定5 scope gateをFAIL。
- by-well deltaはmedian `-0.008439353`、p90 `+0.359385171`、p95 `+0.514122918`、
  p99 `+0.988264358 ft`。worstは`fd8f77fa +2.038361827 ft`、`+1/+3/+5 ft`悪化wellは
  `8 / 0 / 0`。tailは必須readoutでpromotion gateではない。
- technical conditionだけPASSし、pooled / fold / scopeの性能3条件はすべてFAIL。
  `FAIL_CLOSE_WITHOUT_PAIR_FEATURE_SUBSET_TEMPERATURE_OR_GATE_RESCUE`を適用して終端閉鎖する。
  45 rank特徴は全て少なくとも1 modelでgain importanceが正だったが、昇格根拠には使わない。
- output archive全体は取得せず、metrics / fold / scope / hidden / by-well / importance /
  model manifest / schema / reproducibility manifestとlogだけを
  `kaggle/output/stage_d_v1_metadata`へ取得した。OOFはremote artifact SHAだけを記録した。
- final415 logical schema SHA: `491fccff6222d5d5180a80b31a8f727046243f438419866bf96af7d196065bea`。
- Stage D metrics / model manifest / OOF SHA:
  `76b25364...a7c5e` / `9b5386b6...6faa` / `bc4a90c9...9074a`。
- fold / scope / hidden / by-well / importance SHA:
  `67ff6804...9860` / `7cafad4a...d118` / `2dac1a04...5286` /
  `5942f797...5a18` / `85ac1568...09e3`。
- reproducibility manifest file SHA: `6cfe6a919a205c3141967862b5dfa53f98bfa7c9472c1afbd03d2ec74d5ed142`。
- 完了後にStage D run flagとtraining flagをfalseへ戻した。same-OOF rescue、
  inference、submissionは行わない。

## 2026-08-03 Stage N実行承認

- ユーザーの「実行してください」を、直前に提示したStage Nの正規train Notebook採用、
  Kaggle CPU package、push、runの承認として記録した。
- 承認時刻: `2026-08-03 10:10:23 UTC`（`2026-08-03 19:10:23 JST`）。
- 実行対象は1 scientific variant / 1 rank config / outer 5 × inner 4 = 20 CPU boosters、
  25 compact partitionsのみ。
- 保存exp504 outer models再学習0、exp413 control再学習0、候補/PF/HMM/Beam再生成0、GPU 0。
- Stage D 15 GPU boosters、inference、submissionは今回の承認対象外。
- credential checkerはOAuthとlegacy credentialをPASS。exp504 kernel outputの存在をKaggle APIで
  再確認した。確認時点でexp497 Stage M fold4がRUNNINGだが、exp507は独立したCPU kernelとして
  同一canonical slugへ1回だけpushする。
- 正規train Notebookは20 cells / 9 code cells / output 0でcompact候補とbyte一致。
- Kaggle packageはbootstrap込み21 cells / 10 code cells / output 0、support 34 files。
  bootstrap ZIPの全byte count / SHAとloose/package/embedded config SHA一致を確認した。
- kernel metadata: `kentookumura/exp507-exp504-nested-rank-compact-stage-n`、private、CPU、
  internet off、run-on-push、competition source 1、kernel sourceはexp504だけ、dataset/model source 0。
- package Notebook SHA256: `b5ecf6209193354124979587d64e43d92683beac2ed1176f6d3e1c92d66dc610`。
- kernel metadata SHA256: `036a263df1c7e814c49682601316d9221dc71dd479f4c5e08f57d22a772e7b08`。
- embedded/local config SHA256: `ad6e8be8d660f206c6de4f51944a2ec955234d8a12901d03b045624968bf8ffa`。
- scientific source SHA256: `a0ec0bb586a5c0cbecc301f9b4509bdacc2e2a430c2f4e0abe71d1bebe0490e8`。
- canonical slugの事前検索は`Not found`。初回versionとしてpushする。
- `2026-08-03 10:17:33 UTC`にKaggle kernel version 1をpushし、CPU runを開始した。
- remote kernel id_no: `129565024`。push直後のstatusは`RUNNING`。
- pullしたremote metadataでslug/title、private、CPU、internet off、competition source 1、
  exp504 kernel source 1、dataset/model source 0を再確認した。remote Notebookは21 cells / output 0で、
  normalized cell source、support manifest、pushed config SHA、scientific source SHAがlocal packageと一致。

## 2026-08-03 Stage N完了

- Kaggle private CPU version 1（id_no `129565024`）は`COMPLETE`。
- Stage N PASS log time `10,263.367916 sec`、最終log time `10,273.817938 sec`。
- actual executionは1 scientific variant / 1 rank config / outer 5 × inner 4 = 20 models、
  全model 800 trees。保存exp504 outer model再学習0、exp413 control再学習0、GPU/PF/HMM/Beam 0。
- 25 compact partitions（train 20 / outer-valid 5）、合計`18,919,945` row-role、45特徴、
  feature importance `79,440` rowsを生成した。
- technical gateはinput SHA、20 models、25 partitions、row-role、45列一意、held outer/inner
  overlap 0、forbidden feature 0、保存outer surface parityの8条件すべてPASS。
- 保存outer parityはBorda max abs `0.0`、provisional / fallback exact match。Tracebackは0。
  sklearnのfeature-name warningは出たが、固定順arrayでのpredictに対するwarningであり、
  model/partition/technical gateは完走した。
- Stage N manifest SHA: `9a126024f0a67ab571e053038aa4a36e8b6773b6f0ff839d1fdf9ec63bcb7735`。
- rank model manifest SHA: `ccce7aa350ec9de526823a1a9023588e8e55f94747eb1ebb7b972128b1bce364`。
- partition manifest SHA: `86419566d6e9ad51d2be269946108d25df1bd20c2ca9853d586866533076df2e`。
- leakage ledger SHA: `187ded6ca4636b0186b16f7651492115634f89d0288627b5fc4acd89700abd18`。
- feature importance SHA: `7d29fa91211066713ba5ac99cde16b3f1062fa41d825d79cb8cd9027a40a112e`。
- output archive全体は取得せず、後続入力凍結に必要なmanifest / ledger / importance 5点だけを
  `kaggle/output/stage_n_v1_metadata`へ取得し、内部SHAを再照合した。
- Stage D、inference、submissionは0。Stage N run flagは完了後にfalseへ戻した。

## 2026-08-03 実装

- ユーザーの「exp507を実装してください」をtrain-side実装承認として記録した。
- `kaggle-review-exp`を使い、steering、exp504、exp413、exp502、再現性契約を再確認した。
- 設計で未確定だったexp504 v1生成物をKaggle kernel version 1から必要ファイルだけ取得した。
  model/PF/HMM/Beam再学習・再生成は0。
- source file SHA:
  - candidate bank: `1281c1ec501c21bdee9d918d805eb320b3e0e82715d993c2cfd0f9dda18909bc`
  - candidate block: `b23521c6d524fad712927808cb5ef04b89921ce11c6d0770e003ee1eb8872581`
  - shared block: `044efa09f529a3451ca76f4c8ce60b1ac4afab2823759bf59db3514d033de352`
  - block context: `1b01d03b6401e55c525dee259293de2bfc377ddeb0944b7005ea910bc4593186`
  - block metadata: `2417ef9661f61cd3892e02853e92a81408372bbbff1ec6a96de8691630caa181`
  - block selection: `7a53818a1d96fa2601eb2dc4043633ea9a65d22baa47551df207715f9bd38dea`
  - pair probability: `48b1466f278c63f7ffed069e7994348067ff708214cc00d491194a50b3aaf78a`
  - row metadata source: `6dfdccfa0baf0a21a4e4fb9fb8cd026063c595ff5857f83a885c381261019181`
- logical SHA:
  - block metadata: `dc50b0d65b347675a1485466379637443bbd2a0db255f87640ce0a02f76cc735`
  - block selection: `0cd97d79b2925e2ad9ed0b7fc5ec70f6fbb74d8156a056839130435f8b7b4f8b`
  - pair probability: `fb1697339f41db0de9c0f67c67edb92c74381bd611924314d25ca891f1678272`
  - row allowlist 6列: `30870f5c137ebe77eaf0b7683c1f9c5aee3ca8a8af07ac9d56668e7c57581a3a`
- 実ファイルpreflightは`3,783,989 rows / 773 wells / 7,787 blocks / 1,986 pair features`、
  candidate logical SHA、全array shape、candidate/pair/model contractをPASSした。row metadataは
  `id / well / well_row_idx / outer_fold / md_since / h512_group`だけを読み、truth/selected列を
  feature freeze前に読まない。保存outer surfaceとのparityはBorda max abs `0.0`、
  provisional / fallbackともexact match。
- compact45 builder、anchor-first tie、0.5 strict guard、Borda sum=6、weighted moment、
  H512 relative position、raw66一時利用後discardを実装した。
- nested planはouter 5 × inner 4 = 20新規CPU rank model、4 inner-train + 1 outer-valid × 5 =
  25 partitions。outer-validは保存exp504 surfaceを再利用し、outer model再学習0。
- Stage Dはexp413 clean273 + compact74 + signed23 + rank45 = final415を実装した。
  Stage N manifest SHA未固定ならfail closedし、control再学習0、1 treatment × 3 configs × 5 =
  15 GPU boosters以外を許さない。
- 設計文の`block-constant 44 / row-varying 1`は4.6のrow単位weighted moment式と矛盾したため、
  数式を正として`42 / 3`へ訂正した。総数45、列名、順序、数式は維持した。
- 親compact sourceとの比較: exp413は9章 / 766行、exp507候補は9章 / 1,807行。
  exp507はStage N、compact builder、Stage D、metrics/SHAをNotebookセル上で追える。

## 実装検証

- `pytest .../test_exp507_contract.py -q`: `7 passed`
- `ruff ... --select F821,F401,F841,E712`: PASS
- `py_compile`: PASS
- `jupytext --to ipynb` / `jupytext --to ipynb --test`: PASS
- `rg __file__`: train候補に該当0
- `task validate-exp ...`: 環境に`task`がなく未実行
- `make validate-exp EXP=exp507_exp504_nested_rank_compact_addonly_on_exp413`: strict PASS
- `review_exp_docs.py exp507 --root .`: core evidence categoryあり。設計時requirements/tasklistに
  next-action等の形式警告があったため、追加承認範囲と次アクションを追記した。

## 2026-08-03 設計

- ユーザーの依頼をbacklog、steering、実験scaffold、設計契約の作成承認として記録した。
- `kaggle-review-exp`と`kaggle-strategy`を使用し、`KAGGLE_DIRECTION.md`、
  `experiment_summary.md`、`SUBMISSIONS.md`、exp413/exp504/exp502、
  `docs/06_reproducibility.md`を確認した。
- strategy context collectorはsystem `python`が無く1回失敗し、`.venv/bin/python`で再実行した。
- 現行最大exp506を確認し、新規番号をexp507とした。
- `make new-steering EXP=exp507_exp504_nested_rank_compact_addonly_on_exp413`、
  `make new-exp EXP=exp507_exp504_nested_rank_compact_addonly_on_exp413`でscaffoldを作成した。

## 設計判断

- downstreamへ入れるのは12 Borda、11 anchor勝率、Borda要約5、anchor rank 1、
  provisional one-hot 12 + fallback 1、weighted TVT moment 2、H512相対位置1の計45列。
- anchor scoreは12 Bordaのanchor列と同一なので別の重複列を作らない。
- provisionalをordinal IDにしない。
- raw 66 pairはinner rank推論でBorda生成時に一時利用してもdownstream artifactへ保存・投入しない。
- exp413 final370は置換せずfinal415にする。exp502のreplacement失敗と仮説を分離する。
- exp413はnestedだがexp504保存OOFはstandard outer OOFであるため、outer-train側に20 inner
  rank modelsを新規生成する。outer-validの完了済み5 outer modelは再学習しない。

## 実行量契約

- scientific treatment variant: 1
- Stage N rank config: 1
- Stage N outer / inner: 5 × 4
- Stage N new CPU models / boosters: 20 / 20（version 1で完了）
- reused exp504 outer models: 5、再学習0
- Stage D TVT configs / outer folds: 3 × 5
- Stage D new GPU models / boosters: 15 / 15（version 1で完了）
- total new boosters: 35
- exp413 control retrain: 0
- candidate / PF / HMM / Beam regeneration: 0
- inference / submission: 0 / 0

AGENTS.mdのGPU baseline guardに従い、保存exp413 OOFをmatched controlとして使う。Stage Nは
固定20 CPU modelsで完了した。Stage Dはユーザー承認済みで、push前に
1 treatment / 3 configs / 5 folds / 15 GPU boosters、control再学習0を再確認した。

## 再現性メモ

- seed policy: seed 42、outer/inner/candidate/pair/block/row順を固定。
- stochastic components: 実行済みCPU rank LightGBMとGPU TVT LightGBM。
- CPU rank: exp504 deterministic / force_col_wise / 4 threadsを継承。
- GPU TVT: exp413 configを継承し、bitwise deterministicとは断言しない。
- input evidence: exp504 v1 kernel/version、candidate/row/block/model/OOF SHA、exp413 OOF/model SHAを固定。
- artifact gap: 解消。exp504 block / pair / row allowlist / array file・logical SHAをconfigへ固定した。
- feature evidence: rank_compact45 schema、25 partition content / manifest SHA、final415 schemaと
  10 matrix partition SHAを記録済み。
- model/prediction evidence: 20 rank model SHA、15 TVT model manifest / OOF SHAを記録済み。
- submission SHA: inference/submission未承認のため対象外。
- deterministic anchor: false。独立rerun一致前は主張しない。

## 未実行・禁止

- same-OOF feature/pair subset、temperature、weight、threshold、model、gate rescue
- current-test / hidden inference、submission、外部提出

## 次のアクション

1. exp507はscientific FAILとして閉じ、exp413をselected anchorのまま維持する。
2. same-OOF rescue、inference、submissionへ進まない。
3. 必要なら別承認の0-model / saved-artifact-only原因readoutで、fold 0/1/4とshort/hidden-like
   scope悪化がどのrank45 regimeへ集中したかだけを調べる。
