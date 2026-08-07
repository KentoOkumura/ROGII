# exp490_geometry_centered_mean_reverting_offset_hmm セッションノート

## 目的

exp357 Huber residual-offset exact HMMへ、exp226 K16区間1つをhalf-lifeとする
geometry-centered mean reversionを追加する独立仮説を、実装前に固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 1 fail-closedを維持・hidden-dynamic version 2提出とLB監査完了
- CV: full OOF `8.48015525957654`（Stage 0はfixed32機構確認でCVではない）
- LB: Public `9.680` / Private未公開（ref `55180208`、version 2）
- 実装承認: 2026-07-30のユーザー依頼「exp490を実装してください」
- Kaggle実行承認: 2026-07-30のユーザー依頼「実行してください」
- full OOF実行承認: 2026-07-31のユーザー依頼「full wellに進んでください」
- inference version 2実行承認: 2026-08-02のユーザー依頼「実行までしてください」
- competition submission承認: 2026-08-02のユーザー依頼「提出してください」

## 2026-07-30 設計記録

- 最新番号exp489の次としてexp490を採番した。
- 科学的親をexp357、構造参照をexp281、geometry参照をexp226、
  失敗機構参照をexp408に固定した。
- exp357のHuber emissionを維持し、residual offsetとrateの遷移中心へだけ
  K16 segment-span half-lifeの`rho_t`を追加する。
- original rate momentum `0.998`は維持し、rate中心を
  `0.998 × rho_t × q_(t-1)`とした。
- hard reset、GR confidence gate、half-life grid、noise/grid変更、
  blend、selectorは範囲外とした。
- Stage 0は保存fixed32、Stage 1はfull 773 wellsとし、別々の承認を必要とする。
- このセッションでは生成スクリプトでsteeringと標準実験雛形だけを作成した。
  notebookと`settings.py`は未実装placeholderのまま変更していない。

## 実行量契約

現在の実行量は全項目0。

| 段階 | candidate variant | HMM well-runs | control再実行 | ML config | trained fold | booster | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 現在 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Stage 0予定 | 1 | 32 | 0 | 0 | 0 | 0 | 0 |
| Stage 1予定 | 1 | 773 | 0 | 0 | 0 | 0 | 0 |

Stage 0はCVではない。Stage 1はStage 0全PASS後の別承認が必要である。

## 2026-07-30 Stage 0実装

- ユーザーの明示依頼をStage 0 fixed32実装承認として扱い、Kaggle package / run、
  Stage 1、inference、submissionは承認範囲外として無効のまま維持した。
- compact self-contained train sourceと同名candidate notebookを新規作成した。
  既存の正規placeholder train / inference notebookは上書きしていない。
- fixed32 manifestはexp411のSHA固定済み32-well manifestをexp490 assetsへ
  byte-identicalに複製した。
- candidate生成前はmanifestの`well`だけを読み、exp226から
  `well_id,row_idx,suffix_offset,tvt_geop`だけを読む。32予測とdecoder contract
  SHAのfreeze後に初めてrole / fold / saved exp357 prediction / truth /
  exp408 episode境界を読むleakage ledgerを実装した。
- exp226 K16と同じequal-row-count segment境界、destination-row ownership、
  positive `dMD`、segment span、`rho`積0.5 sentinelを実装した。
- exp357のHuber delta、known-prefix sigma、欠損GR補完、offset/rate grid、
  process noise、initial prior、posterior meanを固定し、rate / offset遷移中心だけへ
  `rho_t`を追加した。
- persistent episode SSE、persistent well / fold、matched-control pooled / p95、
  dynamic episode count、256/512-row recoveryのtruth-late gateを実装した。
- Stage 1コードは実装していない。

## 実装時の実行量再確認

| 項目 | 値 |
| --- | ---: |
| active variant | 1 |
| candidate HMM well-runs（Stage 0実績） | 32 |
| reporting folds | 5 |
| saved exp357 control HMM再実行 | 0 |
| LightGBM config | 0 |
| trained fold | 0 |
| booster | 0 |
| PF / Beam / GPU | 0 |

control再学習・再decodeは含まない。Kaggle GPUコストは0。

## 静的検証

- `py_compile`: PASS
- `ruff check --select F821`: PASS
- `__file__` sentinel: train sourceに残存なし
- Jupytext `--to ipynb`: PASS
- Jupytext `--to ipynb --test`: PASS
- `pytest -q .../test_exp490_contract.py`: 6件PASS
- `make test`: FAIL（今回の変更外）。collection時に既存のexp297 / exp301 /
  exp333 / exp336 / exp349が各自のconfig contract不一致で停止した。exp490固有testは
  上記の個別実行で6件PASSしている。
- YAML load: PASS
- fixed32 manifest SHA: config期待値と一致
- source比較: exp357 compact train 2,943行 / 12章、exp490 compact train
  2,389行 / 9章。exp490はStage 0専用だが、入力、segment、decoder、
  freeze、truth-late readout、gate、生成物までnotebook上に展開した。
- ローカル環境にはNumba packageがないため、posterior normalization testは
  pure-Python fallbackで実行した。実runではNumba必須guardがfail-closedする。
- この静的検証時点ではローカルnotebook実行、Kaggle package / push / runを
  行っていなかった。後続の実行承認後にKaggle version 1を完了した。

## 再現性メモ

- seed policy: HMM本体は乱数なし。well / row / segment / state順序を固定。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle private CPU、GPU 0。
- Kaggle kernel id / version:
  `kentookumura/exp490-geometry-mean-revert-offset-hmm-train` / version 1。
- input SHA: configにexp226、exp357、fixed32、hidden-likeの期待SHAを固定。
- feature / contract SHA: 実装時にK16境界、span、rho、state遷移契約を保存。
- model manifest / model SHA: 学習モデルなし。decoder manifest SHAで代替予定。
- prediction SHA:
  `0098f0e9ee23e23d6a7f53cd63ae72bcbe3f546fd8c3b425672131560e2d6ca8`
  （decompressed content）。
- submission SHA: inference/submission未承認のため対象外。
- rerun check: 未実行。1回の成功だけでdeterministic anchorとは呼ばない。

## 2026-07-30 Stage 0実行承認

- ユーザーの「実行してください」を、compact self-contained candidateの正規train
  notebook採用と、Stage 0 fixed32のKaggle private CPU 1回実行の承認として扱う。
- 実行範囲はcandidate variant 1件、HMM 32 well-runs、reporting fold 5件、
  保存済みexp357 controlの再decode 0、ML config 0、trained fold 0、
  booster 0、PF / Beam / GPU 0。
- Stage 0はCVではない。Stage 1、inference、submissionは未実装・未承認のまま。
- canonical kernel id:
  `kentookumura/exp490-geometry-mean-revert-offset-hmm-train`
- canonical title:
  `exp490 geometry mean revert offset hmm train`
- Kaggle資格情報checkerはOAuthとlegacy API keyを確認した。headless API tokenは
  未設定だが、Kaggle CLI用OAuth資格情報は有効。

## package前検証

- 実行flagを有効化した直後、契約testによるsource importが本実行を開始する
  entrypoint問題を検出した。`__main__` notebook実行だけがStage 0を開始し、
  module importはside-effect freeとなるguardを追加した。
- guard追加後のexp490契約test: 6件PASS。
- canonical train notebook: compact self-contained sourceからJupytext生成、21 cells
  （Kaggle bootstrap cellを含むpackageでは21、local canonicalは20）、output 0。
- strict package: PASS。
- metadata: private、CPU、internet off、run-on-push、canonical id/title一致。
- dependencies: exp226 train outputとexp357 train outputの2 kernel source。
- current bootstrap config SHA:
  `3c47bd5dc35fd68d4818fb25e49b8dd82170dc60e2551315e0b7a2a6a98cd315`
- bootstrap fixed32 manifest SHA:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
- bootstrap persistent episode SHA:
  `031067fa77c195b77920a0997401310fbdd16532a2d0e99a9c3b5044de28913c`
- 3ファイルともlocal byte SHA、bootstrap manifest SHA、zip内容SHAが一致した。

## 初回push 400とcanonical slug短縮

- 初回planned slug
  `exp490-geometry-centered-mean-reverting-offset-hmm-train`は56文字で、
  title由来slugとの一致確認後もKaggle `SaveKernel 400`。
- 同slugのpullは403で、Kaggle側に利用可能なNotebookは作成されていない。
- 既存exp486 / exp488と同じKaggle slug長境界を避け、同じexp490のまま
  geometry mean-reversion offset HMMという意味を保つ44文字のcanonical slug
  `exp490-geometry-mean-revert-offset-hmm-train`へ一度だけ短縮した。
- canonical titleは`exp490 geometry mean revert offset hmm train`とし、
  title由来slugをid末尾と一致させる。科学契約、入力、実行量は変更しない。
- 短縮後package notebook SHA:
  `b2308d6f138bd0b3062940b2ab869b198654358a9ddca2dab78fd7178ef65023`
- 短縮後kernel metadata SHA:
  `dc6837cf6ae47616303c6f536e60ec1dec097cd139c6d408f75eb5794be00659`
- 再package後もbootstrap config / manifest / episode定義のlocal SHA、
  zip SHA、bootstrap manifest SHAはすべて一致した。

## Kaggle Stage 0 version 1

- canonical kernel version 1 push: 成功。
- push時刻: 2026-07-30。
- kernel id:
  `kentookumura/exp490-geometry-mean-revert-offset-hmm-train`
- Kaggle id_no: `129180511`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp490-geometry-mean-revert-offset-hmm-train`
- pull metadataでprivate / CPU / internet off、exp226 / exp357 kernel source、
  competition source一致を確認した。
- push直後status: `KernelWorkerStatus.RUNNING`。
- 同じversion 1を完了まで監視し、実行中の空logsやstatus 500だけを理由に
  再pushしない。

## Kaggle Stage 0 version 1結果

- 2026-07-30 14:36:57 UTC: `KernelWorkerStatus.COMPLETE`。
- status: `stage0_fail_closed`。
- actual execution: 1 scientific variant、32 candidate HMM well-runs、
  reporting folds 5、保存control再decode 0、model config / trained fold /
  booster / PF / Beam / GPU各0。
- rows / wells: 156,088 / 32。
- candidate runtime: `2130.499953 sec`。
- total runtime: `2169.821896 sec`。
- peak RSS: `0.899147 GiB`。
- technical gate: 12 / 13 PASS。
  - FAIL: full 773 runtime projection
    `51464.889494 sec > 30600 sec`。
- mechanism gate: 6 / 7 PASS。
  - FAIL: matched-control by-well delta RMSE p95
    `+3.118472 ft > +0.25 ft`。
- persistent episode SSE reduction: `69.893385%`（PASS）。
- persistent improved wells / folds: `13 / 16`、`5 / 5`（PASS）。
- persistent fold candidate-minus-parent RMSE:
  `-5.316599 / -12.777886 / -4.596951 / -1.518047 / -0.093476 ft`。
- matched-control pooled parent / candidate:
  `4.871908 / 4.409685 ft`、差`-0.462223 ft`（PASS）。
- persistent episode count delta: `-4`（PASS）。
- recovery rate delta 256 / 512: `+0.080000 / +0.120000`（PASS）。
- posterior normalization最大誤差: `2.44249e-15`（PASS）。
- segment half-life最大絶対誤差: `9.88098e-15`（PASS）。
- finite coverage: `1.0`（PASS）。

## version 1生成物SHA

- executed bootstrap config:
  `3c47bd5dc35fd68d4818fb25e49b8dd82170dc60e2551315e0b7a2a6a98cd315`
- scientific contract:
  `221f6572bc1386475c87ca6db9eccad220ec5ec766e1aa56620611881ee0fbe0`
- decoder contract:
  `ad75aa3190edd0bcee2f6ced088ef535317506eae8a5a660842c9181de2c91cf`
- prediction decompressed content:
  `0098f0e9ee23e23d6a7f53cd63ae72bcbe3f546fd8c3b425672131560e2d6ca8`
- input manifest:
  `d688fa7749ccee356a244142db6e881dcdeb01686227677e7d61dc8c0f184555`
- decoder manifest:
  `631e0d9a8fd7e2d7a63440d1cc9208644624351a1d7570391dd29607e203d44f`
- K16 segment contract:
  `280a5d323299ec97e06776368283a53e193020881118d1b277022c0bf1376270`
- well / episode metrics:
  `819b4749bb8d4603a0f3ca193a0cd984e1b94071512841fefd8a33026290471c` /
  `f334343f37ca942e4bd65c8bcd1fd46dc1c1e4ad17c889336f028c5e9565fad9`

train-side判定に必要な全gate、fold、runtime、SHAが完了logsに出ているため、
Kaggle output archiveは取得していない。

## 解釈

- mean reversionはpersistent側の全主要指標を改善したが、matched controlの一部wellを
  大きく壊した。pooled平均改善とp95悪化の併存は安全性不足を示す。
- full runtime投影も固定上限の約1.68倍で、独立technical FAILである。
- 固定all-AND契約に従い、tail / runtime gateを緩和せず、
  favorable rerun、Stage 1、inference、submissionを行わない。
- 完了済みexp490をアイデアバックログから削除する。このnegative resultだけに
  依存するHMM救済候補は追加しない。

## 次のアクション

1. full shard 0--3をKaggle private CPUで実行し、各prediction/manifest/summary SHAを固定する。
2. 4 shardを0-HMM strict mergeし、full 773-well CVと事前登録gateを評価する。
3. inferenceとsubmissionはfull結果後も別承認なしに実行しない。

## 2026-07-31 Stage 1 full OOF override実装

- Stage 0の`stage0_fail_closed`判定、tail FAIL、runtime FAILは変更していない。
- ユーザーのfull-well依頼を、固定1 variantをfull 773 wellsで評価する
  `explicit_user_override_after_stage0_fail`承認として記録した。
- 科学条件は変更なし。active variant 1、candidate HMM well-runs 773、
  保存exp357 control再decode 0、LightGBM config / trained fold / booster / PF /
  Beam / GPUはすべて0。
- 単体runtime投影`51,464.889494 sec`をKaggle上限内へ収めるため、
  `sha256("exp490::full_well_shard::<well>") mod 4`でtarget-freeに分割した。
- 固定exp226 OOF identityでの分割は、shard 0--3が順に
  `192/204/182/195 wells`、`950,473/986,223/890,131/957,162 rows`。
  比例runtime投影は`12,927/13,413/12,106/13,018 sec`である。
- 元のStage 0 notebookは上書きせず、`train_variant0`--`train_variant3`と
  `train_aggregate`のJupytext self-contained notebookを追加した。
- 各shardはexp226 `tvt_geop`、raw/typewell、known prefixだけを読み、truth、fold、
  saved exp357、hidden-like、episodeを読まない。aggregateは4 shard SHAをconfigへ
  固定した後だけ、保存exp357 truth/controlを後付けする。
- `py_compile`、Ruff F821/F401、5 notebookのJupytext round-trip、既存6件+
  full契約4件の計10 pytestをPASSした。
- ローカルの保存exp226生成物を使うidentity-only smokeで773 wells / 3,783,989 rows、
  4 shardのwell/row数、manifest logical SHA
  `cac549f53ef4a98fce8e3fbf7381c0313f0f28f65409639a0e7d36cd89be7f5f`を確認した。
- strict packageの初回pushは4本ともSaveKernel 400となった。認証情報を出さない
  response-body監査で、原因はKaggleの「kernel source must be less than 1 MB」制限と
  判明した。self-contained notebookが使わない共通`src/`だけを`--no-src`で除外し、
  packageを`951,351 bytes`へ縮小した。科学source、config、bootstrap assetは不変である。
- 4 packageのbootstrap config SHAはすべてlocal config SHA
  `7b0dfb5fbf4baeef8142ec3d5a5662e94f088e4904d861e556a219606b60e6b7`と一致し、
  persistent episode / fixed32 asset SHAも既存固定値と一致した。
- Kaggle private CPU shard 0--3 version 1をpushし、4本とも`RUNNING`を確認した。
  kernel id_noは順に`129283179 / 129283181 / 129283176 / 129283180`。
  remote metadataでGPU/internet off、competition source、exp226 kernel sourceだけを確認した。

## 2026-07-31 Stage 1 full OOF完了

- 4 shardはすべてKaggle private CPU version 1でCOMPLETEした。well / rowは事前固定どおり
  `192/950,473`、`204/986,223`、`182/890,131`、`195/957,162`。
- shard 0--3のcandidate / total秒は順に
  `12,862.277/12,895.451`、`11,338.015/11,371.373`、
  `12,049.571/12,080.013`、`6,772.040/6,790.446`。各technical gateは全PASSし、
  peak RSSは最大`1.277 GiB`だった。
- 4 predictionのraw gzip / decompressed SHA、summary SHA、well manifest SHAを実ファイルで
  検証し、`config.yaml`へ固定した。stable 4-shard merge後のwell manifest SHAは
  ローカルidentity smokeと同じ
  `cac549f53ef4a98fce8e3fbf7381c0313f0f28f65409639a0e7d36cd89be7f5f`。
- strict aggregate packageのnotebook本体は`952,566 bytes`で1 MB制限内。local/package
  config SHAは`71ac72507bda8bf6bd261b9dfe55d4dbebf51f2910fcfd009b9f97bc3086735d`
  と一致した。py_compile、Ruff、Jupytext test、pytest 10件、strict validateを再度PASSした。
- merge kernel `kentookumura/exp490-mean-revert-full-merge` version 1、id_no
  `129321382`をprivate CPU / internet offで実行しCOMPLETE。0 HMM well-runs、
  merge`205.212 sec`、peak RSS`6.154 GiB`。

### full OOF結果

- 3,783,989 rows / 773 wellsのpooled RMSEは`8.48015525957654 ft`。
  保存exp357親`9.737195157482754 ft`から`1.257039897906214 ft`改善し、
  exp226 final`9.427109596582222 ft`からも`0.9469543370056819 ft`改善した。
- fold 0--4のcandidate RMSEは
  `8.935035 / 8.659383 / 8.922330 / 7.928528 / 7.913022 ft`。
  exp357比はfold 0のみ悪化し、4 / 5 folds改善。
- MD 1000+、hidden-like spatial、hidden-like typewell-purgedのexp357比delta RMSEは
  `-1.434059 / -1.306581 / -1.267906 ft`で、3 scopeとも改善した。
- persistent episode SSEは`41.409965%`削減、episode数delta`-59`、recovery rate deltaは
  @256 `+0.036050`、@512 `+0.025078`。長く続く誤差への効果はfullでも再現した。
- by-wellは449改善 / 324悪化。delta RMSE中央値`-0.057105 ft`に対し、p90
  `+3.059139 ft`、p95`+7.257814 ft`、p99`+17.501421 ft`。
  worst well `389ae58f`は親`3.764936 ft`からcandidate`53.367497 ft`へ
  `+49.602560 ft`悪化した。
- 事前登録14 gate中12 PASS。`by_well_p95_nonworse`と`worst_well_regression`がFAILし、
  判定は`stage_1_failed_close_without_rescue`。inferenceとsubmissionは実行しない。
- full predictionのraw gzip / decompressed SHAは
  `99030b33d493cc5f195f7d1a867f0d812a539143da9e1f59277e53779261b72c` /
  `e020e82e748a7836085657c4058070ff7853ed285639f2c2555cab721f9e9a07`。
  summary SHAは`e10658a37ab2252018496f2393dacf7b83e449e42483090b8432c3d9f9a4ba2a`。

### 解釈と次

- 固定強度のmean reversionはpooled・scope・persistentを強く改善する一方、正しい長期offset
  までgeometryへ戻すwellを作る。物理モデルの「復元力」は有効だが、その強度を全wellで
  固定する定式化が不十分である。
- 次は保存full OOFのみを用いる0-HMM readoutで改善wellと悪化wellを比較する。
  segment span、GR情報量、geometry不確実性、初期offset、suffix horizonを候補とし、
  truthで選択するselectorではなく、観測可能な物理量から復元力を決める次実験へつなげる。

## 2026-08-01 current-test inference override実装

- ユーザーの「念のためLBも見たいので推論に進んでください」を、Stage 1 fail-closeを
  維持したまま固定候補を現行testへ1回推論し、`submission.csv`候補を作る承認として
  記録した。competition submitは別判断のままである。
- 現行sampleは14,151 rows / 3 wellsで、SHAは
  `7498f19ba1be281328c31a39044d4ba5f84e71c8f4115c613b5531f42aaff85a`。
- 幾何はexp226 inference kernelのsource/config SHAを固定し、最終`pred`ではなく
  `PredictionResult.geop`だけを再生成する。その上へfull OOFと同一のscientific
  contract SHA `6398bbac380d3eca3a6255681b22c44c26de268ce6d4fad9dd242c066f2b9a35`
  を持つmean-reverting HMMを適用する。
- 実行量はscientific variant 1、exp226 full train fit 1、exp226 geometry 3 well-runs、
  exp490 HMM 3 well-runs。LightGBM config / fold / booster / PF / Beam / GPUはすべて0。
  testは3 wellsだけなのでshard分割しない。
- Jupytext percent形式のcompact self-contained sourceを先に生成し、14章 / 1,576行の
  candidateを正規inference notebookへ採用した。学習aggregateは12章 / 3,154行で、
  inferenceではOOF truth readout、fold/gate、4-shard mergeを除いている。
- py_compile、Ruff `F821`、Jupytext変換とround-trip、inference契約test 5件をPASSした。
  testは実行量、competition submit無効、scientific SHA、sample/source SHA、K16
  half-life、zero-state geometry identity、target/truth read禁止を確認する。
- 過去Stage 0/1契約は今回の承認済みinference flagと分離して静的評価し、exp490の
  Stage 0 / full OOF / inference契約testは合計15件PASSした。strict experiment
  validationもPASSした。
- strict packageはprivate / CPU / internet off / run-on-pushで、kernel sourceは
  `kentookumura/exp226-k16-kappa-repro-inference`だけ。competition sourceはROGIIだけ。
  code notebookは919,967 bytes、26 cells、output 0で1 MB制限内である。
- package/local config SHAはともに
  `244b7041e661a27fc0fa031f59f0291b845ad62dc735c0b6f448c95b24b6e30a`。
  package notebook SHAは
  `69481e22a414ced3342caf5c41ac6b1390cd0628b7730f62a82dd003a70ffcc7`。

## 2026-08-01 current-test inference完了

- Kaggle private CPU kernel
  `kentookumura/exp490-geometry-mean-revert-offset-hmm-inference` version 1、
  id_no `129323029`を実行しCOMPLETE。remote metadataはprivate / CPU /
  internet off、competition sourceはROGII、kernel sourceはexp226 inference 1本だけ。
- actual executionはscientific variant 1、exp226 full fit 1、exp226 geometry 3 wells、
  exp490 HMM 3 wells。model config / trained fold / booster / PF / Beam / GPUは0。
- 14,151 rows / 3 wellsを処理し、HMM `207.103586 sec`、total `265.468702 sec`、
  peak RSS `1.158676 GiB`。posterior normalization最大誤差`1.55431e-15`、
  segment half-life最大誤差`4.99600e-15`。
- technical gateは13 / 13 PASS。exp226 source/config SHA、scientific contract、positive
  dMD、half-life、zero-state identity、finite、sample row/well/ID順序が一致した。
- outputを`kaggle/output/inference_v1`へ取得し、専用submit checkerと独立監査で
  header / 14,151 rows / ID順序、重複なし、欠損・NaN・Infなしを確認した。
  submit-checkはFAIL 0 / WARN 0。
- inference prediction SHAはraw gzip
  `413fa695ad32385f97c1d18a1947bbd7415687a274d1ecee03f02c22467e1cce`、展開後
  `f5b7da9dc99387fef66a159a61d6e1e3c71368296f3b9cf075ec236bfa5845dc`。
  submission SHAは
  `3970e9ad6d89250e3946f48fa97ed89b6dfd05dd33767514502ca8ca7f3be6e5`。
- Stage 1 fail-closeは保持され、competition submissionは
  `not_approved_not_started`。LBはまだ存在しない。

## 2026-08-01 competition submit承認

- ユーザーの「exp490の推論を提出してください」を、inference kernel version 1の
  competition submit 1件に対する明示承認として記録した。
- submit直前に専用checkerを再実行し、FAIL 0 / WARN 0。14,151 rows、header、
  sample ID順序、unique ID、finiteを再確認した。
- submit対象はkernel
  `kentookumura/exp490-geometry-mean-revert-offset-hmm-inference` version 1、output
  `submission.csv`、local確認SHA
  `3970e9ad6d89250e3946f48fa97ed89b6dfd05dd33767514502ca8ca7f3be6e5`。
- Stage 1 fail-closeは保持し、今回の提出はLB auditとして扱う。

## 2026-08-01 competition submit結果

- Kaggle code submissionを1件送信した。refは`55163886`、submittedは
  `2026-08-01 13:59:07.600000 UTC`、kernelは
  `kentookumura/exp490-geometry-mean-revert-offset-hmm-inference` version 1。
- 監視上は`SubmissionStatus.COMPLETE`になったがPublic scoreは空欄。raw APIの
  `error_description`は、hidden datasetでの再実行中に未処理例外が発生し、hiddenは
  publicより大きい・小さい・異なる可能性があるという内容だった。
- 提出前のlocal/public outputは14,151 rows / 3 wells、SHA
  `3970e9ad6d89250e3946f48fa97ed89b6dfd05dd33767514502ca8ca7f3be6e5`、
  submit-check FAIL 0 / WARN 0のままである。提出CSVの形式不良ではない。
- inference sourceはdecode前に公開sample SHAを一致必須とし、さらに
  `expected_submission_rows=14151`、`expected_test_wells=3`を固定assertする。
  hidden可変サイズと非互換なため、これを構造的な失敗原因と診断した。
- Stage 1のtail FAILは保持する。hidden-dynamic version 2の実装・push・再提出は
  今回の1件提出承認に含めず、実行していない。
- 監視ログ: `logs/submission_exp490_geometry_centered_mean_reverting_offset_hmm.log`

## 2026-08-02 hidden-dynamic inference version 2実装

- ユーザーの修正依頼により、同じexp490内のcompact candidateと正規inference source /
  notebookをhidden可変サイズ対応へ更新した。新しい実験番号は作っていない。
- 変更はruntime test contractだけ。公開sample SHA、14,151 rows、3 wellsは
  `data.public_test_reference`へ移し、audit-onlyで記録するが実行gateには使わない。
- ご指摘どおりhidden test入力は開始時に全件見えるため、exp226 full fit前に全sample、
  horizontal、typewellを一括走査する。sample well集合、raw file well集合、各wellの
  `TVT_input`欠損row、sample IDを完全一致させ、nonempty / uniqueも確認する。
- exp226 geometry ID、exp490 prediction ID、sample順序、finiteは後段でも従来どおり確認する。
  exp226 geometry / exp490 HMM well-runsは固定3ではなくruntime sample well数に等しい。
- train source/notebook、Stage 0/1、4 shard、merge、保存OOFは変更していない。
  scientific contract SHAは`6398bbac380d3eca3a6255681b22c44c26de268ce6d4fad9dd242c066f2b9a35`
  のままで、CVを含むtrain結果は不変。
- source SHAはcompact/canonicalとも
  `d57591459ae1d3ee24e00845395a3615b72437681c36a2710b4bb74cca254e70`。
- py_compile、Ruff F821、2 notebookのJupytext conversion / round-trip、Stage 0 / full /
  inferenceの全18契約test、strict experiment validationをPASSした。
- `--no-src` strict inference packageを再生成した。code notebookは929,764 bytes、
  26 cells、output 0、SHA
  `4537d1009f5b295a042ef233e67884bc6baf0e10e06437555f916aff5a6fb617`。
  bootstrap/local config SHAは
  `7b472ca14d6160064551d699e4f1ba091913c9513e4f8ca57e4e3a5b4893f116`、
  bootstrap/local source SHAは上記`d5759145...`で一致した。
- version 2専用run approvalとpush approvalはfalse。Kaggle push/run/再提出は行っていない。

## 2026-08-02 hidden-dynamic inference version 2実行承認

- ユーザーの「実行までしてください」により、canonical inference kernel version 2の
  private CPU / internet off push/runを承認済みとする。
- 予定実行量はscientific variant 1、exp226 full-train fit 1、exp226 geometryと
  exp490 HMMはruntime sample well数。model config / trained fold / booster / PF / Beam /
  GPUは0。親control、train shard、mergeの再実行も0。
- output取得とsubmission候補のlocal検証までを行う。competition submitは未承認で、
  `execution.competition_submission_approved=false`を維持する。

## 2026-08-02 hidden-dynamic inference version 2実行完了

- 同じcanonical kernel
  `kentookumura/exp490-geometry-mean-revert-offset-hmm-inference`へversion 2をpushし、
  private CPU / internet offで`COMPLETE`。id_noは`129323029`。
- executed bootstrap config SHAは
  `a21fa4b6039c09e9c370d9c16c2759b32ccc8fa620d9f9009c40be6ae18f7153`、
  source SHAは`d57591459ae1d3ee24e00845395a3615b72437681c36a2710b4bb74cca254e70`。
- runtime preflightはsample / horizontal / typewell 3 wellsを全件走査し、unknown rowsを
  `3836 / 6014 / 4301`、合計14,151 rowsとして完全一致確認した。
- technical gateは14 / 14 PASS。scientific contract SHAは
  `6398bbac380d3eca3a6255681b22c44c26de268ce6d4fad9dd242c066f2b9a35`で不変。
- HMM `110.832150 sec`、total `149.097627 sec`、peak RSS `1.155529 GiB`。
  posterior normalization最大誤差`1.776357e-15`、segment half-life最大誤差`4.996004e-15`。
- outputを`kaggle/output/inference_v2`へ取得。submit-check FAIL 0 / WARN 0、sample ID
  内容・順序、unique、finiteを独立確認した。
- prediction raw/decompressed SHAはversion 1と同じ
  `413fa695...e1cce` / `f5b7da9d...8455dc`。submissionもversion 1とbyte-identicalで、
  SHAは`3970e9ad6d89250e3946f48fa97ed89b6dfd05dd33767514502ca8ca7f3be6e5`。
- input / decoder / summary SHAは
  `c78bb0ab...4015 / 4657de76...680 / a6c30d67...52097`。
- version 2のrun/push承認は使用済みとしてfalseへ戻した。competition submitは
  `not_approved_not_started`で、再提出していない。
- 実行完了後にlocal push packageをfail-close状態で再生成した。notebookは930,272 bytes、
  SHAは`dc598b63899ab3a0faea6817995d44fb9b558dba0c13bdc93ff29e58e2adbdee`、
  bootstrap config SHAは`1dd891418acf053ffa3e5af7ccf28fb168d03094cd70e3ed5a7a61bb2640b711`。
  bootstrap config/sourceはlocalと一致し、`run_on_push=false`、version 2 run approval=false、
  competition submit approval=falseを確認した。実行時packageのSHAとconfigは
  `kaggle/output/inference_v2`および`metrics.json`に別途保持している。

## 2026-08-02 hidden-dynamic inference version 2提出承認

- ユーザーの「提出してください」により、canonical inference kernel version 2の
  competition submission 1件を明示承認済みとする。
- 提出直前にKaggle outputの`submission.csv`をsampleに対して再検証し、FAIL 0 / WARN 0、
  14,151 rows、header、ID内容・順序、unique、finiteを確認した。SHAは
  `3970e9ad6d89250e3946f48fa97ed89b6dfd05dd33767514502ca8ca7f3be6e5`。
- Kaggle側のcanonical kernel id_noは`129323029`、private CPU / internet off、
  output作成時刻はversion 2完了時刻と一致する。提出対象versionは2に固定する。
- Stage 1 tail fail-closeは保持し、提出目的はhidden-dynamic修正後のLB auditとする。

## 2026-08-02 hidden-dynamic inference version 2提出受付

- canonical inference kernel version 2を1件提出した。submission refは`55180208`、
  submittedは`2026-08-02 07:23:53.240000 UTC`、受付時statusは`PENDING`。
- messageは`exp490 hidden-dynamic v2 LB audit; CV 8.480155; Stage1 tail FAIL retained`。
- version 2の一回限りのcompetition submission承認は使用済みとしてfalseへ戻した。
  同refを監視し、Public LBまたはhidden rerun error確定後に結果を記録する。

## 2026-08-02 hidden-dynamic inference version 2採点完了

- ユーザーの採点完了連絡後、Kaggle CLIでsubmission ref `55180208`をref指定確認した。
  statusは`COMPLETE`、Public LBは`9.680`、Private LBは未公開。scoreが付いたため、
  version 1で起きたhidden再実行の未処理例外は解消した。
- CV `8.480155260`に対してPublic LBは`+1.199844740 ft`。exp226 direct `9.837`より
  `-0.157 ft`改善したが、direct exact HMM `9.063`より`+0.617 ft`、self-GR HMM
  `9.318`より`+0.362 ft`、direct likelihood PF `8.797`より`+0.883 ft`悪い。
- hidden-dynamic runtime契約の修正は成功したが、fixed-strength mean reversionは
  physical-route Public-LB anchorには昇格しない。Stage 1のby-well p95 / worst FAILと
  terminal fail-closeを保持し、同モデルのparameter救済や追加提出は行わない。
- 汎用monitorはref `55180208`を65分まで`PENDING`として追跡した後、新しいlatest ref
  `55181377`へ切り替わった。そのためexp490固有の正確なscoring elapsedは不明であり、
  monitor末尾の`10.116`はexp490ではない。スコア根拠はKaggle CLIのref `55180208`行とする。
