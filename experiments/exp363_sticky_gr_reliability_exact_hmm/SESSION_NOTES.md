# exp363_sticky_gr_reliability_exact_hmm セッションノート

## 目的

exp209 exact HMM に sticky GR reliability 状態だけを追加する設計を確定し、
Stage 1前の0-HMM identifiability readoutを実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage_0_failed_closed`
- CV: Stage 0 diagnosticのため非該当
- LB: まだなし
- Notebook: compact self-contained train / fail-closed inferenceを実装し、
  placeholderの正規Notebookを置換した。
- Kaggle Stage 0 package / push / run: version 1完了、technical PASS /
  scientific FAIL。
- Stage 1 / inference / submission: 不適格・未実施。

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 2026-07-23 実行済み

```bash
make new-steering EXP=exp363_sticky_gr_reliability_exact_hmm
make new-exp EXP=exp363_sticky_gr_reliability_exact_hmm
```

### 2026-07-23 Stage 0 実装

- ユーザーの `exp363を実装してください` を、Stage 0実装とplaceholder Notebook置換の
  承認として記録した。Kaggle実行承認には拡張していない。
- 保存済みexp209 cacheは`id / well / hmm_mean_tvt / hmm_prefix_sigma`だけを
  pre-freezeで読む。cache全体のdecompressed SHA
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
  をhard guardし、`target`はparseしない。
- horizontal GRはexp209と同じboth-direction linear interpolation後のType Well平均fallback、
  Type Well GRはTVT sort後ffill/bfillとendpoint-hold linear interpolationを使う。
- saved exp209 path上で`-0.5 * min(z^2, 600)`を作り、q初期確率
  `[0.8,0.2]`、固定transition、normal/weak multiplier`1.0/0.25`の
  2-state forward filterを各blockで独立に適用する。
- 512行・stride 256、stride startから始まる短い末尾blockを保持する。
  primary scoreはblock内weak posteriorの行平均、全体weak massはblock overlapを含む
  posterior row総和 / block row総和とした。
- circular controlは`sha256("exp363|<well>")`由来のwell内非ゼロblock shift。
  Q1/Q4境界はtruth前のpooled score 25%/75%点で凍結する。
- posterior / block ledgerのdeterministic gzip content SHAを凍結してから、
  SHA固定exp226 fold/truthとexp115 hidden-like roleを初めて読む。
- bad labelはsaved exp209 pathのblock RMSE `>=10 ft`。pooled real/circular AUC、
  Q4-Q1 mean block RMSE、fold AUC、hidden-like 2面、row-weighted weak massを
  固定AND gateで評価する。
- Stage 1 exact HMM、rate予測、hard mask、sigma/transition/multiplier grid、
  blend/selector、親control再実行はコードに含めていない。

## 実行コスト契約

- Stage 0 diagnostic variant: 1
- reporting fold: 5
- HMM well-run / model config / LightGBM config / trained fold / booster:
  `0 / 0 / 0 / 0 / 0`
- parent control再学習・再生成: 0
- Stage 1予約: 全gate PASSかつ別承認時だけ1 variant / 773 exact-HMM well-runs

## 2026-07-24 Stage 0 Kaggle CPU実行承認

- ユーザーの`実行してください`を、固定済みStage 0のKaggle CPU
  package / push / run承認として記録した。
- 実行対象はdiagnostic 1、reporting folds 5。
- HMM well-run / model config / LightGBM config / trained fold / booster /
  親control再実行は`0 / 0 / 0 / 0 / 0 / 0`。
- GPU/TPU/internetはoff。Stage 1、inference、blend、submissionは未承認のまま。
- canonical kernel id / titleは
  `kentookumura/exp363-sticky-gr-reliability-exact-hmm-train` /
  `exp363 sticky gr reliability exact hmm train`とする。
- Kaggle credential checkerはAPI token未設定を報告したが、CLI OAuth、
  username、legacy keyは利用可能と確認した。credential実値は記録していない。

## 2026-07-24 Stage 0 Kaggle push準備

- canonical kernelへの事前`kaggle kernels pull -m`は403で、利用可能な既存notebookは
  確認できなかった。別slugは作らず、固定canonical idを初回push先とする。
- `make prepare-kaggle-notebooks`をstrict / train / run-on-pushで実行した。
- metadataはprivate、CPU、internet/GPU/TPU off、competition source 1、
  kernel sources 3で、id/title slugは一致している。
- bootstrap ZIPは18 support files。local / loose package / bootstrap ZIP内configは
  byte一致し、config SHAは
  `d4db31930773adc9d3af5508d65752d9366b8ab7955e3849359e3dbd8edeb7aa`。
- 正規train notebook SHAは
  `a2d7317e333db3870b1a07aba41216655f5965ded314f1e3bbb56d74abcc0cae`、
  push notebook SHAは
  `8857536d462d12c5340019592f09330bdc6fa7081edd4ae9e725f01a82ccf950`。

## 2026-07-24 Stage 0 Kaggle CPU version 1実行開始

- canonical kernel
  `kentookumura/exp363-sticky-gr-reliability-exact-hmm-train`
  のversion 1 pushに成功した。
- Kaggle metadata pullでprivate CPU notebook、`id_no=128370770`、
  GPU / TPU / internetはすべて無効、入力sourceはexp209 / exp226 / exp115
  であることを確認した。
- 初回statusは`RUNNING`。以後は同じkernel IDを監視し、空ログや一時的な
  status API errorを理由に再pushしない。

## 2026-07-24 Stage 0 Kaggle CPU version 1完了

- 最終statusは`COMPLETE`。runtimeは`497.082523 sec`。
- 3,783,989 rows / 773 wellsから15,174 blocksを生成し、bad blocksは3,532。
- technical gateはPASS。truth-before-freeze 0、全score finite、expected fold一致、
  circular offset、strict quartile、行/well数、HMM/model/booster/control再実行0を確認した。
- pooled real bad10 AUCは`0.607551995`、circularは`0.583995776`、
  差は`+0.023556219`。Q4-Q1 mean block RMSEは`+4.816306326 ft`、
  fold AUCは5/5で`>0.50`となり、ここまではPASSした。
- hidden-like typewell-purged AUCは`0.552194539`でPASSしたが、
  hidden-like spatial AUCは`0.546057972 < 0.55`でFAILした。
- row-weighted weak massは`0.589440997 > 0.50`でFAILした。
- scientific gateはFAIL、decisionは`stage_0_failed_close_without_rescue`、
  Stage 1 eligibilityはfalse。再push、transition/multiplier/sigma/block/threshold救済、
  Stage 1、inference、submissionは行わない。
- 実ファイル確認が必要なgate/SHA記録のためKaggle output 3.5 MBを
  `kaggle/output/train_v1`へ取得した。

## 変更点

- q の2状態、遷移行列、weak log-emission係数0.25を固定した。
- Stage 0 の block、negative control、全 gateを固定した。
- Stage 1 は1 variant / 5 folds / 773 HMM runs / booster 0 / control rerun 0。
- rate predictability は先行条件に置かない。

## 再現性メモ

- seed policy: RNGなし、well / row / state の stable order。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle CPU single worker、`497.082523 sec`、
  GPU/TPU/internet off。
- SHA: block ledger、weak posterior は content SHAを凍結後に truth joinする。
- gzip: decompressed content SHAを主証拠にする。
- block ledger content SHA:
  `967c495f91a6c4ff1aa5f897207c0bf0437a48cb201d82f8f3680981b4434843`。
- weak posterior content SHA:
  `38e1fdaca08513143e281b53f43c6c4d64a621392fca4e47f2f5a3e682a83337`。
- late-truth block readout content SHA:
  `672ab37aa3b94984e67c842a6c282bbd680cc1018407c6bf86e1fec2df5ed89c`。
- gate raw SHA:
  `0a00934e1b3b3b1175fbe4ff5b7bf1c8a94bce67f4093eb6c32612fa018bb3a5`。
- downloaded final summary raw SHA:
  `248ca4197c29459989eea263ed5123445d1c5d3c0ef980427ce10da43b8834ce`。
- prediction / model / submission SHA: 非該当・未生成。

## 実装検証

```bash
.venv/bin/python -m py_compile <exp363 compact train/inference.py> <exp363 test.py>
.venv/bin/ruff check <exp363 sources and test> --select F821,F401,F841,E722,E501
.venv/bin/pytest -q tests/test_exp363_sticky_gr_reliability_exact_hmm.py
# 8 passed
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact source.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact source.py>
make validate-exp EXP=exp363_sticky_gr_reliability_exact_hmm
make validate-template
```

- 科学的親exp209にはcompact self-contained版がないため、同じexp209 childである
  exp346 compact train（10章、約1,900行）を構成参照にした。
- exp363 compact trainも10章で、runtime/path/SHA、safe input、target-free生成、
  late truth join、metric/gate、orchestrationをNotebook上で追える。
- Stage 0はHMM kernelを実行しないため、exp346のexact-HMM forward-backward章は
  2-state reliability forward filterとblock freeze章に置換した。
- compact/正規Notebookはtrain 21 cells、inference 7 cellsでcell source一致を確認した。
- `make validate-exp` strict validationと`make validate-template`はPASSした。
- repository全体のpytestは`753 passed, 5 skipped, 2 failed`。失敗2件は既存exp296の
  完了後configと旧test期待の不一致で、exp363専用8 testsはPASSしている。
  具体的には`experiment.status=completed_train_side_guard_failed_closed`に対して
  旧testが`kaggle_cpu_*` prefixを要求し、`execution.run_variant=false`の完了状態に対して
  旧testがpush approval guardを先に期待している。exp296は今回変更していない。

## 次のアクション

1. exp363 branchを閉じる。
2. Stage 1 exact HMM、inference、submissionは実装・実行しない。
3. 同じq契約を使うexp368は本結果をnegative dependencyとして扱い、
   独立identifiability根拠なしには実装しない。
