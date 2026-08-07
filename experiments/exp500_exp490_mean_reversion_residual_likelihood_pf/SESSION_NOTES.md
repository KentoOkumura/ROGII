# exp500_exp490_mean_reversion_residual_likelihood_pf セッションノート

## 目的

exp490のK16区間half-life geometry平均回帰を、exp486 residual-state likelihood-PFへ
1変数アブレーションとして移植し、Stage 0 fixed44と明示override下のStage 1 full OOFで監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage1_fail_closed_under_override`。Stage 0 fixed44のFAILを保存したまま、
  2026-08-02のユーザー明示オーバーライドで同一1 variantのStage 1 full OOFを完了
- CV / Public LB / Private LB: `8.813504627432842` / なし / なし
- scientific variant: `1`
- Stage 0 / Stage 1 PF well-runs: 成功実績`44 / 773`。
  Stage 0 version 1 technical retryを含む累積operational量は`88 / 0`
- LightGBM config / trained fold / booster / HMM / Beam / GPU: すべて`0`
- control再実行: `0`
- inference / submission: 未承認・無効
- 実装承認: 2026-08-01のユーザー依頼「exp500を実装してください」
- 正規Notebook採用 / Kaggle package / push / Stage 0 run承認:
  2026-08-01のユーザー依頼「実行してください」
- Stage 1 implementation / run / Stage0 gate override承認:
  2026-08-02のユーザー依頼「Stage1を実装・実行してください」
- override scope: 変更なしの単一variant full OOFのみ。Stage 0 failをPASSへ変更せず、
  inference / submission / same-OOF rescueは承認対象外

## 2026-08-02 Stage 1明示オーバーライドとpush前監査

- Stage 0 technicalは13/13 PASSだったが、matched-control pooled、matched-control by-well p95、
  PF sentinel worst-wellの3 safety gateがFAILした事実を保持する。
- ユーザーの明示依頼を、同じ`k16_half_life_mean_reverting_residual_likpf`を変更せず
  773 train wellへ広げるStage 1実装、正規train Notebook更新、Kaggle private CPUでの
  4 shard + strict merge実行の承認として扱う。
- scientific variant: `1`。parameter / emission / seed / particles / temperature / gate変更なし。
- deterministic shard: `sha256("exp500::full_pf_shard::<well>")`先頭8 byteのlittle-endian整数を
  4で割った余り。well数は`[200, 182, 181, 210]`、suffix row数は
  `[983,418, 906,216, 898,293, 996,062]`。
- Stage 1 candidate PF well-runs: `773`、seeds/well: `128`、seed-well trajectories:
  `98,944`、particles/seed: `500`、particle starts: `49,472,000`。
- reporting folds: `5`。LightGBM config / trained fold / booster / model:
  `0 / 0 / 0 / 0`。saved control PF rerun / HMM rerun / Beam / GPU: `0 / 0 / 0 / 0`。
- shardはraw deployable入力とexp226 geometry allowlist 4列だけを読み、prediction、
  residual ledger、seed evidence、K16/rho contract、well audit、manifest、SHAをfreezeする。
- mergeは4 shard summaryと全artifactのraw/decompressed/logical SHA、manifest、union coverageを
  再検証し、773 well / 3,783,989 rowのunion freeze後にのみtruth、saved exp404 / exp486 /
  exp209、fold、hidden-like role、exp408 / exp410 episodeをattachする。
- Kaggle kernel id:
  `kentookumura/exp500-mean-revert-resid-likpf-full-shard0`--`shard3`、
  mergeは`kentookumura/exp500-mean-revert-resid-likpf-full-merge`。
- runtime: Kaggle private CPU、GPU off、internet off。各shard上限`30,600 sec`、
  peak RSS上限`25 GiB`。controlを学習・再実行しない。
- 推論・submissionは無効のまま。Stage 1 PASS時も別承認なしに進まない。
- 実装後の静的検証: dedicated contract `9 passed`、py_compile、ruff F821、
  Jupytext round-trip、strict validate-exp PASS。compact / canonical train Notebookは
  同じJupytext sourceから生成しbyte-identicalに採用した。
- `--no-src` package監査: shard Notebookは`966,848--966,852 bytes`、mergeは
  `966,864 bytes`でKaggle 1 MB code limit未満。private / CPU / GPU off / internet off /
  run-on-push、competition source 1、shard kernel source 3、merge kernel source 7、
  dataset source 2を確認した。
- 全packageのbootstrap source SHAは
  `31487dea80063e43aad2f9b317adf2a17e2dcafefd05a01007b3c5bd72591f73`。
  config SHAはshard 0--3が順に
  `7dada56bd20ee37ac052e6a4530dd3e9a85981d4965980c88d722447ea095666`、
  `1815858c228f0d2911b0835a3b840d577d6a9a982ca469d78619276794b0aafd`、
  `63021c246e90198641699e9b016b850a7e11e34b159072cbbc4a3e08c629ed39`、
  `ad0f4b67ecf1b7bb5cf0033326bd7283b1ecce0e64a1fe44b12f638395e38940`、
  mergeが`daa71130a1013fe62fb3c5b71f8062bd9bf50bfd8497fab7090831d4d83a59b7`。
  Notebook内bootstrap manifestとpackage直下fileのSHA一致を全packageで確認した。
- canonical base configはexecution flagsを全falseへ戻し、SHAは
  `a6926e9aab688230e5fadf00265664e702d51b3cf7cf8e80ec48476b5de78770`。

## 2026-08-01 Stage 0実行承認とpush前監査

- 今回の「実行してください」を、別名candidateの正規train Notebook採用、Kaggle package、
  `kentookumura/exp500-mean-revert-residual-likpf-train`へのpush、Stage 0 fixed44実行の
  明示承認として扱う。Stage 1、inference、submissionへは進まない。
- canonical id / title:
  `kentookumura/exp500-mean-revert-residual-likpf-train` /
  `exp500 mean revert residual likpf train`。title由来slugとid末尾は一致する。
- runtime: Kaggle private CPU、GPU off、internet off、run-on-push。
- scientific variant: `1`。
- Stage 0 candidate PF well-runs: `44`、seeds/well: `128`、
  seed-well trajectories: `5,632`、particles/seed: `500`、particle starts: `2,816,000`。
- LightGBM config / reporting fold training / booster / model: `0 / 0 / 0 / 0`。
- control PF rerun / HMM / Beam / GPU: `0 / 0 / 0 / 0`。
- exp226 OOF geometry、saved exp209 HMM、exp115 role assignmentは既存Kaggle Notebook source、
  saved exp404 / exp486は既存Kaggle Dataset sourceを使い、controlは再実行しない。
- fixed32、PF sentinel、exp408 / exp410 episode境界の4資産はbootstrap dependencyへ固定する。
- Kaggle credential checker: API Tokenは未設定だが、OAuth credentialsとlegacy CLI credentialは有効。
- package後にbootstrap config/source、metadata、CPU/internet/run-on-push、input sourceを再監査してからpushする。
- 正規train Notebookはcandidateとbyte-identicalに採用済み。package生成後のmetadataは
  private / CPU / internet off / run-on-push、competition source 1、kernel source 3、
  dataset source 2で契約どおり。
- bootstrap manifest上のconfig SHA `fe32077d11ea15a9896aab53ebe6e297b3e04703a918027516c088bc653234b8`、
  source SHA `d1c78fc1243326e0f60b563e9b42c58e867e0ab5f47d82d5b3a0e7861bc66ecf`、
  local asset 4点のSHAはすべてlocal canonicalと一致した。
- push直前検証: dedicated contract `6 passed`、py_compile / ruff / Jupytext test / strict validate-exp PASS。
- 初回pushはKaggle `SaveKernel 400`。同じkernel idのmetadata pullは`GetKernel 500`で、
  診断用にrequest本文を非表示のままresponse本文だけ取得した結果、原因は
  `The kernel source must be less than 1 megabytes in size.`と確定した。
- 初回packageはrepository-wide `src/` bootstrapを含み`1,353,126 bytes`だった。
  self-contained source / Notebookに`src` importがないことを確認し、科学コード・設定・資産・入力を
  変えず`--no-src`で再packageした。再package後は`878,947 bytes`、metadataとbootstrap SHAは維持。
- 別slugは作らず、同じcanonical idへ再pushする。
- `--no-src`再pushはKaggle CLI終了0だったが、API応答は
  `Maximum batch CPU session count of 5 reached.`で未開始。
- 読み取り確認では、exp497 Stage P fold1--4とexp501 trainの計5 CPU sessionがすべて
  `RUNNING`。他実験を停止する権限は今回の依頼に含めず、自然完了を監視して空き発生後に
  同じexp500 canonical slugを再pushする。
- 14:42 UTCにexp501が`COMPLETE`となりCPU slotが解放された。同じcanonical idへ再pushし、
  exp500 kernel version 1を正常作成・実行開始した。
- push後metadata pull: id_no `129380054`、private / CPU / internet off、dataset source 2、
  kernel source 3、competition source 1を確認した。
- version 1は約`1,995 sec`で`ERROR`。44 wellのtarget-free PF、prediction / residual /
  evidence / K16-rho SHA freezeまでは完了し、truth-late readoutのsaved exp486 path解決で停止した。
- 原因はKaggle Datasetが元の`.csv.gz`を展開し、
  `exp486_exp226_geometry_residual_likelihood_pf_stage1_predictions.csv`として提供すること。
  source manifestで元gzip raw SHA `0fe0...65de3`、展開content SHA
  `05f692...153e6`を確認した。Stage 0 gateは未評価で、科学的PASS/FAILではない。
- 技術修正はsaved exp486のfilenameを展開後`.csv`へ合わせ、gzip / plain CSVを同じcontent SHAで
  検証し、`pandas.read_csv(compression="infer")`で読むことだけ。PF、variant、seed、gate、入力内容は不変。
- version 1で実行済みcandidate PF well-runsは`44`。version 2の技術retryでも同じ`44`を再実行するため、
  完了時の累積operational candidate PF well-runsは`88`となる。control PF / HMM / Beam / GPU rerunは`0`。
- 技術修正後はdedicated contract `7 passed`、ruff / Jupytext / strict validate-exp PASS。
  final package `879,765 bytes`、config SHA `63afe82198d69ef2bddfb48c9cfe704980583d0f58192d8be5bff46e78b4322e`、
  source SHA `de9f5c7da2657f5ea24fb496ef3f6e5cfb71693ff926b5a9b07a0ad289aaeb9a`。
- 同じcanonical slugへkernel version 2をpushし、`RUNNING`を確認した。

## コマンドログ

- 2026-08-01:
  `make new-steering EXP=exp500_exp490_mean_reversion_residual_likelihood_pf`でsteeringを作成。
- 2026-08-01:
  `make new-exp EXP=exp500_exp490_mean_reversion_residual_likelihood_pf`で空の実験scaffoldを作成。
- 2026-08-01:
  exp486 / exp490 / exp498、`docs/06_reproducibility.md`、現行backlogを読み、
  状態式、Stage 0 / 1、gate、実行量、禁止救済、再現性契約を確定。
- 設計ターンではPFコード実装、Notebook変換・実行、Kaggle package / push / run、
  output取得、提出を行わなかった。

## 2026-08-01 Stage 0実装

- ユーザーの明示依頼を、Stage 0 fixed44の別名compact self-contained train候補と
  専用contract testの実装承認として扱った。
- 正規train / inference Notebookは既存placeholderのまま上書きしていない。
- `exp500_exp490_mean_reversion_residual_likelihood_pf_compact_selfcontained_train.py`
  と同名candidate Notebookを作成した。23 cells、出力0。
- exp486 residual PFから、初期化、Gaussian GR emission、500 particles、128 seeds、
  process noise、systematic resampling、roughening、temperature-5集約、float32出力を維持した。
- 科学的変更は、destination K16区間の
  `rho_t=2**(-dMD_t/L_segment)`をrate / offset遷移中心へ加える1点だけとした。
- fixed32 identity 32件とPF sentinel identity 12件だけをcandidate生成前に読み、
  truth / exp404 / exp486 / role / fold / cause / exp408・exp410 episode境界を
  prediction、residual ledger、seed evidence、K16/rho contract、SHA freeze後に読む実装とした。
- per-seed total log evidence、temperature weight、ESS、resampling、clip count、
  offset / rate、particle weight sum、support / edge mass、per-well runtimeを保存対象にした。
- Stage 1、raw-test regeneration、inference、submissionの実装は追加していない。

### 実装時の固定実行量

| 項目 | 値 |
| --- | ---: |
| scientific variant | 1 |
| Stage 0 candidate PF well-runs | 44 |
| seed-well trajectories | 5,632 |
| particle starts | 2,816,000 |
| saved control PF rerun | 0 |
| LightGBM config / trained fold / booster | 0 / 0 / 0 |
| HMM / Beam / GPU | 0 / 0 / 0 |

### 静的検証

- `py_compile`: PASS
- `ruff check --select F821`: PASS
- `ruff check`: PASS
- 専用contract test: `6 passed`
- Jupytext `--to ipynb` / `--to ipynb --test`: PASS
- `make validate-exp EXP=exp500_exp490_mean_reversion_residual_likelihood_pf`: strict PASS
- `make test`: PASS
- `__file__` sentinel: candidate sourceに残存なし
- fixed44 asset contract: fixed32 32 + sentinel 12、重複0、union 44、PASS
- K16 destination ownership、各区間rho積0.5、zero-state identity、
  `rho=1` exp486 float32 parity、intervening well実行後のstable RNG再現、
  truth/control/role-before-freeze fail-closedを専用testで確認した。
- 親compact比較: exp486は3,647行 / 12章、exp500候補は2,309行 / 11章。
  exp500はabsolute-unaryとStage 1を範囲外として除き、入力、PF kernel、freeze、
  truth-late readout、fixed gate、生成物、実行guardまでNotebook上に展開した。
- ローカルNotebook実行、Kaggle package / push / run、output取得は行っていない。
- ローカル`.venv`にはNumbaがないため、専用kernel smokeは同一関数のpure-Python fallbackで
  確認した。Kaggle package前にNumba compile smokeを必須とする。

## 変更点

exp486 `slow_residual_offset_state`のtransition centerへexp490の`rho_t`だけを追加する。

- `rho_t = 2 ** (-dMD_t / destination K16 segment MD span)`
- rate center: `0.998 * rho_t * previous_rate`
- offset center: `rho_t * previous_offset + current_rate * dMD_t`
- particles / seeds / noise / initialization / emission / resampling / roughening / temperatureは固定
- exp490 Huber / exact-HMM smoothing / prediction、adaptive gate、mixture、proposal-onlyは不使用

## 実行前契約

### Stage 0 fixed44

- scientific variants: 1
- candidate PF well-runs: 44
- seed-well trajectories: 5,632
- particle starts: 2,816,000
- control PF / HMM / Beam / model / booster / GPU rerun: 0
- role / truth / episodeはcandidate、diagnostic、contract、SHA freeze後だけattach
- technical / mechanism / safety全AND PASS時だけStage 1を別承認で検討

### Stage 1 full OOF

- scientific variants: 1
- candidate PF well-runs: 773
- seed-well trajectories: 98,944
- particle starts: 49,472,000
- reporting folds: 5
- LightGBM config / trained fold / booster / HMM / Beam / GPU: 0
- control再実行: 0
- Stage 0 PASSと別承認なしには実装・実行しない

## 再現性メモ

- seed policy:
  exp486実装と同じ
  `int(sha256("likpf::train::<well_id>").hexdigest()[0:16],16) % 2147483647 + 1 + seed_index`
- stochastic components:
  particle initialization、rate / offset noise、systematic resampling、roughening
- parallel RNG:
  global RNG禁止。well / seed streamをworker / shard順序から独立させる。
- stable order:
  well id、row、seed、particle。
- runtime:
  Kaggle private CPU、GPU off、internet off。Stage 0 per-well秒/rowのp95と固定4 shardの
  row数から最大shard runtimeを投影し、`30,600 sec`を上限とする。design-only時点の
  kernel id / versionはなし。
- SHA:
  raw input、exp226 geometry、manifest、saved control、config、code、scientific contract、
  K16/rho、diagnostic、prediction raw/decompressed/logicalを分けて記録する。
- deterministic anchor:
  いいえ。full coverage、固定probe rerun、raw-test regeneration rerunまで禁止。
- model / submission SHA:
  学習モデルなし、inference / submission未承認のため非該当。
- reproducibility gap:
  pure-Python fallbackではRNG独立性とprobe parityをcontract testで確認済み。Numba compile、
  fixed probe rerun、Kaggle full coverage、raw-test regeneration rerunは未実施である。

## 戦略上の位置づけ

- Phase: Late。
- 優先度: P3、高リスクCPU PF。
- direct PFの比較基準: exp404 `10.914522073`。
- mechanism evidence: exp490 `8.4801552596`だがwell-tail fail-close。
- overall / ML Public-LB基準: exp413 CV `7.884802794` / Public LB `7.201`。
- 本設計だけではroute anchor、CV、LB、提出候補を更新しない。

## 次のアクション

Stage 0のfail-closeを維持し、Stage 1のtail FAILに従ってinference、submission、
same-OOF rescueへ進まない。

## 2026-08-01 Kaggle Stage 0 fixed44結果

- Kaggle kernel `kentookumura/exp500-mean-revert-residual-likpf-train` version 2、
  id_no `129380054`がprivate CPU / internet off / GPU offで`COMPLETE`。
- result status: `stage0_fail_closed`。
- stage: `stage0_fixed44_mechanism_preflight_not_cv`。
- next action: `terminal_close_without_same_fixed44_rescue`。
- completed at: `2026-08-01T22:42:48.420036+00:00`。
- fixed44はCVではなく、CV / Public LB / Private LB / submissionはない。

### Gate

- technical checks: `13/13 PASS`。
  finite prediction / weight / ESS / ledger、fixed32+sentinel identity/SHA/union、
  geometry allowlist / row coverage、K16 segment coverage、positive dMD/span、rho finite/bounded、
  zero-state identity、rho=1 exp486 float32 parity、stable seed/count、half-life、runtime projection、
  peak RSS、truth/control/role before freeze=0を全PASS。
- mechanism all-pass: `false`。以下3件だけがFAIL。
  - matched-control pooled vs exp404: `+1.0794140845859523 ft`
  - matched-control by-well p95 vs exp404: `+7.468536390741729 ft`
  - PF sentinel worst-well vs exp404: `+9.159571432315374 ft`
- persistent episode SSE reduction vs exp486: `0.5094123902047187`。
- persistent improved wells / folds: `13/16` / `5/5`。
- persistent pooled RMSE: candidate `9.479700331476007 ft`、exp486 `13.867164812579553 ft`。
- fold 0--4 candidate / exp486:
  `7.246985166 / 15.382623464`、`18.001520147 / 24.672852036`、
  `5.949660730 / 7.158078236`、`6.165026818 / 7.789370917`、
  `6.770172193 / 7.069413623`。
- PF sentinel pooled candidate RMSE: `27.881350297492673 ft`。
- exp410 PF episode SSE reduction: `0.4330608351297952`。
- exp408 episode count delta: `-5`、recovery delta 256 / 512: `+0.20 / +0.12`。

### 実行量とresource

- version 2成功実績: 1 variant、44 PF well-runs、5,632 seed-well、
  2,816,000 particle starts。
- version 1 technical retry込み累積: 88 PF well-runs、11,264 seed-well、
  5,632,000 particle starts。科学variantは同じ1件で、control再実行はない。
- LightGBM config / trained fold / booster / HMM / Beam / GPU: `0 / 0 / 0 / 0 / 0 / 0`。
- pre-freeze wall / summed well: `1701.405 / 1652.670 sec`。
- per-suffix-row p95: `0.0074462433 sec`。
- full 4-shard projection: `7322.770 / 6747.905 / 6688.908 / 7416.920 sec`、
  最大`7416.920 < 30600 sec`でPASS。
- peak RSS gate / runtime ledger: `0.676270 / 0.638615 GiB`。
- K16 half-life最大絶対誤差: `1.0381e-14`。
- runtime versions: Python 3.12.13、NumPy 2.0.2、pandas 2.3.3、Numba 0.60.0、
  PyYAML 6.0.3。

### Leakage / SHA

- freeze前: geometry 224,400 rows、forbidden geometry / truth / control / role readsは全`0`。
- freeze後: truth 224,400、controls 448,800、role/fold/episode 85 rows。
- scientific contract SHA:
  `50934030ce250943199e0b6194a82bf85504b95f4eeb114d98bf36b8afdbbb26`。
- 実行bootstrap config / source SHA:
  `63afe82198d69ef2bddfb48c9cfe704980583d0f58192d8be5bff46e78b4322e` /
  `de9f5c7da2657f5ea24fb496ef3f6e5cfb71693ff926b5a9b07a0ad289aaeb9a`。
- prediction raw / decompressed / logical / schema SHA:
  `7b844ca1f42ee246d6d9aa75d18909962bb080ad14f9126c4a90b68174d173ca` /
  `b746f537b89718809cb7903074407cb2fc34311a9f5e24b9ee9472ec49cc1730` /
  `7bcc3311f1d047731b02f43c107febfd6e17f77d54551242f12dc013cfe76f7e` /
  `8b0ac39fbac56618bfcb1a1ab844109fa54ace2481fed4082d558fe705c507b3`。
- gate report / input manifest / runtime ledger / freeze manifest SHA:
  `b448a587cfbf1a0d0a7ed0580f5d0ca3e738238f45fc2ac83af8152e2ea19552` /
  `73ec0ac4c8c14aefca7a1509b319b0744fed138e529d24423a26181eb87f9a9d` /
  `4e337e5c6d893026ee20d54d1f264ba1abe950c57bcb5145acbb7735b3e47505` /
  `7013feee98418e5810a7f213f8ef76038008adeaa4367b6e9ff0c062195a7101`。
- logsにgate値、artifact path、SHAが揃っているため、運用規則どおりKaggle output archive全体は
  downloadしていない。

### 判定

平均回帰はpersistent basinで強く効いたが、matched controlとPF sentinelのwell-tailを壊した。
効果は非一様で、exp490 HMMのpooled改善 / tail悪化問題はPFへ移しても解消しなかった。
全AND gateに従いterminal closeとする。half-life / noise / temperature grid、adaptive gate、
mixture、blend / selector、same-fixed44 rescue、Stage 1、inference、submissionは禁止したままとする。

### 完了後のローカル検証

- exp500専用contract test: `7 passed`。
- `py_compile`、ruff、Jupytext `--to ipynb --test`: PASS。
- `make validate-exp EXP=exp500_exp490_mean_reversion_residual_likelihood_pf`: strict PASS。
- `make test`: `1821 passed / 8 skipped / 4 failed`。FAILは今回変更していない既存exp293の
  downstream contract SHA 2件と、既存exp296の完了後status / run guard期待値2件で、exp500専用
  testと本実験のstrict validationには影響しない。範囲外の既存実験ファイルは変更しない。

## 2026-08-02 Kaggle Stage 1 full OOF結果

- Stage 0 `stage0_fail_closed`を再分類せず、ユーザー明示overrideの範囲内で変更なしの
  `k16_half_life_mean_reverting_residual_likpf` 1 variantだけを実行した。
- shard kernel 0--3は各version 1でCOMPLETE。well / rowは
  `200 / 983,418`、`182 / 906,216`、`181 / 898,293`、`210 / 996,062`。
- 773 PF well-runs、98,944 seed-well trajectories、49,472,000 particle starts。
  control PF / HMM / Beam / LightGBM / booster / GPU再実行は全て0。
- merge kernel `kentookumura/exp500-mean-revert-resid-likpf-full-merge` version 3、
  id_no `129465486`がprivate CPU / internet off / GPU offでCOMPLETE。
- merge version 1はpandas readbackのfloat文字列表現によるCSV logical SHA不一致でtechnical error。
  保存CSV payloadそのものをhashする修正だけを行い、PF shardを再実行しなかった。
- merge version 2はcandidate評価を完走したが、事前登録exp226 finalをgeometry `tvt_geop`で
  比較していたためtechnical parityがFAIL。union freeze後に`fold, tvt_pred`だけを読み、
  final predictionと比較する修正だけをversion 3へ適用した。candidate、gate、PF artifactは不変。

### RMSE

- candidate: `8.813504627432842 ft`
- saved exp404 likelihood-PF: `10.914522073423171 ft`、gain `2.1010174459903297 ft`
- saved exp486 residual-PF: `11.139812021086678 ft`、gain `2.326307393653836 ft`
- exp226 final: `9.427109596582222 ft`、gain `0.61360496914938 ft`
- candidate + exp209 HMM fixed 50:50: `8.661349061843389 ft`
- exp404 + exp209 HMM fixed 50:50: `10.084909848760013 ft`
- fold 0--4 candidate / exp404:
  `8.844861303 / 9.360014232`、`9.059117751 / 10.979418534`、
  `9.210184188 / 10.694277027`、`8.129743309 / 10.747502029`、
  `8.778094572 / 12.482449117`。5/5 folds改善。

### Gate

- technical checks: `18/18 PASS`。4 shard SHA、union coverage、773 wells / 3,783,989 rows、
  5 folds、有限性、weight normalization、K16 half-life、saved RMSE parity、runtime / RSS、
  Stage 0 FAIL保存と明示override、freeze前leakage 0を確認した。
- scientific checks: `12/14 PASS`。pooled、exp486 / exp226比較、absolute、fold、全6 scope、
  fixed 50:50、exp408 / exp410 episode checksはPASS。
- FAILはby-well tailの2件だけ。
  - by-well delta RMSE p95 vs exp404: `+6.653601018697123 ft`、上限`0.0 ft`
  - worst well `389ae58f`: `+46.15467103223343 ft`、上限`+0.25 ft`
- exp408 SSE reduction `0.48933732104561833`、episode count delta `-142`、
  recovery delta 256 / 512 `+0.029780564263322873 / +0.04231974921630094`。
- exp410 SSE reduction `0.5252543406664276`。
- status: `stage1_fail_closed_under_override`。
- next action: `terminal_close_without_same_oof_rescue`。

### Runtime / leakage / SHA

- shard wall seconds: `7366.198 / 7777.175 / 6406.646 / 8465.693`。
  全shardが`30,600 sec`上限をPASSし、最大RSSは`1.299 GiB`。
- merge/evaluation wall `956.790 sec`、peak RSS `5.072 GiB`。
- union freeze前のtruth / control / role-fold-episode / forbidden geometry readは全て0。
  freeze後はtruth 3,783,989、control 11,351,967、role/fold/episode 3,786,239 rows。
- scientific contract SHA:
  `dc5c1690312d76964bf4c0dbbb406402509049fa6828c44f6e42f13c0dea2c91`。
- prediction logical SHA:
  `a4bfa0c48203566be31cfefa4c255182c0bec5949056d6ae688b5252b965210a`。
- gate / primary metrics / summary SHA:
  `b9ff9832584d384498be191714241df58119db5462e9a04c284398d7a73b59d5` /
  `886a53bce2f5a95a049d217ba920b7d5d468d2570c2957e9574e34dd28cefbf0` /
  `019e730b6fc7d23017ac681f9a3c0ac4bb39b6673165433bd05808ba82e48680`。
- Kaggle logsに最終summaryとartifact SHAが揃うためarchive全体は取得せず、version 3の
  小さいsummary / gate / primary metricsだけを取得して値とSHAを検証した。

### 判定

固定mean-reversionはfull OOF平均、5 folds、全scope、persistent episodeを強く改善したが、
少数wellの大幅悪化を防げなかった。平均RMSEだけでtail safety契約を上書きせず、
`stage1_fail_closed_under_override`として終端閉鎖する。half-life / noise / temperature探索、
adaptive gate、blend / selector、same-OOF rescue、inference、submissionは行わない。

### Stage 1完了後のローカル検証

- exp500専用contract: `10 passed`。
- `py_compile`、ruff F821、JSON / YAML parse、Jupytext変換 / test: PASS。
- compact / canonical train Notebookは最終sourceから再生成しbyte-identical。
- `make validate-exp EXP=exp500_exp490_mean_reversion_residual_likelihood_pf`: strict PASS。
- final disarmed config SHA:
  `03613f90b733750bc7b600cda1eec4534341054583d7ad5a6995771d21bcc798`。
- final Jupytext source SHA:
  `a1bc11052275dc668aa77d10b3b943e5e1d2e978994da3cb425301fc72896c6d`。
- compact / canonical Notebook SHA:
  `4d848267ad40d2238938d93f37d1ef02a4878083fc207a92b0ba5d1d3b6789a4`。
