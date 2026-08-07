# exp425_symmetric_datum_reanchor_exact_hmm セッションノート

## 目的

exp408で確認したtranslation-gauge lockを直接扱うため、exp209 exact HMMに
rateとは独立した対称absolute-datum branchを追加する新実験を設計する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0 Kaggle version 1完了・`stage0_fail_closed`
- 優先度: P3・高リスク
- CV / LB: なし
- 実装承認: あり
- 正規Notebook採用承認: あり
- Kaggle Stage 0実行承認: あり（2026-07-28）
- Kaggle Stage 0実行結果: version 1完了、Stage 1不適格
- Kaggle Stage 1実行承認: なし
- inference / submission: 無効

## 2026-07-28 設計確定

ユーザー依頼により、backlog、steering、実験scaffoldをdesign-onlyで作成した。
HMMコード、Jupytext source、test、Notebook実装、Kaggle package / push / run、
inference、submissionは今回の範囲に含めない。

### 根拠

- exp408 exclusive forward transition / prior hysteresis:
  episode SSE `59.3978%`
- exp408 backward smoothing reversal:
  episode SSE `23.0444%`
- exp408でrate massが回復しながらtruth position massが減る行:
  `43.3341% rows / 38.3313% SSE`
- exp412 beta rate方向一致:
  `0.776347`、4 / 5 folds
- exp412 backward-cause SSE reduction:
  `-0.069575`でFAIL
- exp412 full runtime projection:
  `51,753.199176秒`

### 設計前の方向写像診断

`/tmp/exp408_row_ledger_v3.csv.gz`の807,710 episode rowsから、次の単一条件を
read-onlyで計算した。

```text
abs((smoothed_rate - filtered_rate) / max(filtered_rate_std, 0.005)) >= 2
```

- active rows: `39,873 / 807,710`（`0.0493655`）
- rate差符号と`truth - posterior_mean`のdatum修正符号一致: `0.365887`
- SSE加重一致: `0.396557`
- parent RMSE: `28.063421`
- 同方向へ1 filtered-position標準偏差をhard適用: `29.441317`
- parent / 同じ1σ枝のtruth-oracle上限: `26.815358`
- shift中央値 / 90 percentile: `3.116052 / 7.792664 ft`

error-selected episode rowsの設計補助であり、CV、full OOF、parameter selection、
promotion evidenceには使わない。この結果から、exp412のrate方向をdatum方向へ
直接転用せず、正負対称枝をfuture likelihoodでsoft選択する。

### 固定したexact branch

- explicit states: `negative / parent / positive`
- branch prior: `0.10 / 0.80 / 0.10`
- shift:
  `±max(first-pass filtered position std at event, 0.35 ft)`
- event:
  exp412と同じpersistent rate-gap scheduleの最初のfalse→true row
- events per well: 最大1
- event後:
  branch identityを保持し、枝間遷移なし
- rate process:
  exp209から完全固定
- readout:
  branchを含むexact sum-product posterior mean

### Stage 0予定

- fixed32:
  backward 8 / forward 8 / matched control 16
- baseline variants / HMM well-runs: `1 / 32`
- treatment variants / logical HMM well-runs: `1 / 32`
- treatment branch states: 3
- total logical HMM well-runs: 64
- LightGBM config / trained fold / booster / model: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`

fixed32 SHA:
`1edb1e1481af84af4e8178fb6e0743fa40315eab0b7441eeff9232b571f93c30`

fixed32はmechanism診断専用で、スコアをCV、full OOF、promotion evidenceとは呼ばない。
全technical / mechanism / runtime gateがPASSしても、Stage 1は別承認とする。

### Stage 1予定

- first pass: 773 HMM well-runs
- treatment: 773 logical HMM well-runs、3 branch states
- total logical HMM well-runs: 1,546
- reporting folds: 5
- model / booster / PF / Beam / GPU: 0

exp412の2-pass projectionだけで8.5時間上限を超えるため、exp425はまず機構検証用とする。
runtime gateを緩和せず、科学gateだけPASSした場合も近似readoutへ事後変更しない。

### 再現性

- `docs/06_reproducibility.md`確認済み。
- RNGなし。well / row / position / rate / branch順を固定する。
- truth / episodeはevent、shift、branch posterior、prediction freeze後だけjoinする。
- fixed32、saved parent、first-pass message、event、shift、branch posterior、
  prediction、metricsのSHAを記録する。
- gzipはdecompressed content SHAを主証拠にする。
- model / submissionを作らないためdeterministic submission anchorとは呼ばない。

## コマンドログ

- `make new-steering EXP=exp425_symmetric_datum_reanchor_exact_hmm`
- steeringのrequirements / design / tasklistをdesign-onlyで記入。
- `make new-exp EXP=exp425_symmetric_datum_reanchor_exact_hmm`
- scaffoldのconfig、README、SESSION_NOTES、result、metricsをdesign-onlyへ更新。
- scaffold既定Notebookの汎用実行コードを削除し、train / inferenceを
  Markdown-only placeholderへ変更。submission生成コードは残していない。
- `make validate-exp EXP=exp425_symmetric_datum_reanchor_exact_hmm`:
  strict PASS。
- `make validate-template`: PASS。
- config / metrics / 2 NotebookのJSON / YAML parse: PASS。
- train / inference Notebookはcode cell 0、output 0を確認。
- `make update-summary`: 423 experimentsを更新し、exp425のroute、parent、
  status、lineageを登録。
- `review_exp_docs.py exp425_symmetric_datum_reanchor_exact_hmm --root .`:
  core evidence categories present。

## 2026-07-28 実装完了

ユーザーが対象を`exp425_symmetric_datum_reanchor_exact_hmm`と明示したため、
Stage 0コード実装と正規Notebook採用を承認済みとして実装した。Kaggle実行、
Stage 1、inference、submissionの承認には拡張していない。

### 実装したexact branch

- unchanged exp209 first passをzero position-shiftで再実行する。
- exp412と同じrolling beta-filter scheduleから最初のfalse→true eventだけをfreezeする。
- event shiftはfirst-pass filtered position stdから
  `max(std, 0.35 ft)`で一意にfreezeする。
- negative / parent / positiveの3 conditional exact-HMMを、
  event transitionのposition kernel shiftだけ変えて実行する。
- 固定prior `0.10 / 0.80 / 0.10`とfull-sequence log evidenceからbranch posteriorを
  計算し、conditional posterior meanをsoft周辺化する。
- conditional factorizationは、persistent branch stateを持つ明示3枝のexact
  sum-productと代数的に同値である。hard MAP / Viterbi branchは使わない。
- no-event wellは追加HMMを実行せず、parent predictionとbranch mass
  `[0, 1, 0]`を返す。

### 実行量の再確認

- active scientific variant: 1
- baseline first-pass HMM: 32 well-runs
- treatment: 32 logical HMM well-runs、3 branch states
- total logical HMM: 64 well-runs
- LightGBM config / trained fold / booster / model: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`
- parent/control再実行: first-pass internal messageが必要なため32
- Kaggle Stage 0 run: `false`、未承認

### Notebook構成比較

構成参照元のexp412 compact trainは2,333行・9節、exp425 compact trainは
2,661行・同じ9役割節である。exp425は親と同じ入力、parity、truth-late、
gate、生成物保存の章を維持し、datum branch marginalizationとbranch SHAを追加した。
同一実験helper import、`__file__`、薄い`main()` entrypointにはしていない。

### 実装検証

- `.venv/bin/pytest -q tests/test_exp425_symmetric_datum_reanchor_exact_hmm.py`:
  `12 passed`
- `.venv/bin/ruff check <exp425 train/inference/test>`: PASS
- `.venv/bin/python -m py_compile <exp425 train/inference>`: PASS
- `jupytext --to ipynb --test` train / inference: PASS
- compact / canonical Notebookのcell source一致: train / inferenceともPASS
- 正規Notebook:
  train 20 cells / code 9 / output 0、inference 8 cells / code 3 / output 0
- `make validate-exp EXP=exp425_symmetric_datum_reanchor_exact_hmm`:
  strict PASS
- `make validate-template`: PASS
- `make update-summary`: 424 experimentsへ更新
- `task` executableはローカルにないため、規約どおりMakefile同等commandを使用した。
- 現在の`.venv`にはNumbaがないため、直接module importは
  `ModuleNotFoundError: numba`。専用testは既存repoと同じNumba decorator stubで
  exact loopと独立exp209 parityを検証した。Numba JIT compile、runtime、memoryは
  未承認のKaggle Stage 0 technical gateで初めて確認する。
- 現在のtreeには`review_exp_docs.py`が存在しないため再実行できず、
  strict experiment validationとYAML / JSON parseを文書整合性の確認に用いた。

## 未実施

- Stage 1
- inference / submission

## 2026-07-28 Stage 0実行承認

ユーザーの「実行してください」を、exp425のcanonical Kaggle private CPU
Stage 0 package / push / runの明示承認として記録した。承認範囲は
`stage_0_fixed32`だけであり、Stage 1、inference、submissionへは拡張しない。

### push前の実行量確認

- active scientific variant: 1
- LightGBM config / trained fold / booster / model: `0 / 0 / 0 / 0`
- reporting fold: 5（診断集計のみ、学習なし）
- baseline parent exact-HMM: 32 well-runs
- treatment: 32 logical well-runs、3 branch states
- logical total: 64 well-runs
- physical exact-HMM call上限: 96
  - parent first pass 32
  - eventが全wellで発火した場合のnegative / positive conditional pass各32
  - parent conditionalはfirst passを再利用する
- 親control再実行: あり、32 well-runs
  - 保存済みpredictionだけではevent freeze、filtered-position std、
    full-sequence branch evidenceに必要な内部messageを復元できないため
- PF / Beam / GPU run: `0 / 0 / 0`
- accelerator / internet: CPU / disabled

親control再実行を含むがGPU学習ではなく、今回のユーザー承認を得たCPU mechanism
diagnosticである。fixed32はCVでもpromotion evidenceでもない。全gateがPASSしても
Stage 1は別承認を必要とする。

Kaggle credential checkerは、OAuth CLI credentialとlegacy credentialが利用可能で
あることを、credential値を表示せず確認した。canonical kernel
`kentookumura/exp425-symmetric-datum-reanchor-exact-hmm-train`はpush前の
`kernels pull`が403、`kernels list --mine`が`Not found`であり、未作成と判断した。

### package前検証

- `make validate-exp EXP=exp425_symmetric_datum_reanchor_exact_hmm`: strict PASS
- 専用test: `12 passed`
- Ruff F821: PASS
- `make validate-template`: PASS
- canonical metadata:
  - id: `kentookumura/exp425-symmetric-datum-reanchor-exact-hmm-train`
  - title: `exp425 symmetric datum reanchor exact hmm train`
  - private / CPU / internet disabled / run-on-push
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp209-joint-exact-parity-train`
- loose / packaged / bootstrap `config.yaml` SHA:
  `ff22cf694e3e7f2afc4fb9780f43825775f156192a29a47cc302e7eb990aa046`
- bootstrap dependency SHA:
  - fixed32 manifest:
    `1edb1e1481af84af4e8178fb6e0743fa40315eab0b7441eeff9232b571f93c30`
  - fixed32 metadata:
    `3c696de9afa5b8782d614664bc65d909a80a086e6a67f54c0be9e732518838ba`
  - exp408 episode summary:
    `b230ffc759e6ee4891f22809b3f3c8a8796681fb461ec0b7215b94a352bf0ab0`
- generated bootstrap manifest内の上記4 SHAがloose sourceと一致した。

### Kaggle Stage 0 push

- command:
  `make push-kaggle-train EXP=exp425_symmetric_datum_reanchor_exact_hmm`
- result: `Kernel version 1 successfully pushed`
- canonical kernel:
  `kentookumura/exp425-symmetric-datum-reanchor-exact-hmm-train`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp425-symmetric-datum-reanchor-exact-hmm-train`
- accelerator: CPU
- Stage 1 / inference / submission: disabled

## 2026-07-28 Kaggle Stage 0完了

canonical private CPU kernel version 1（id_no `128930925`）は
`KernelWorkerStatus.COMPLETE`で完了した。monitor中は同じkernel IDだけを追跡し、
再push、別slug、Stage 1、inference、submissionは行っていない。

### 実行結果

- status: `stage0_fail_closed`
- Stage 1 eligible: false
- fixed32: 32 wells / 152,303 rows
- event wells: 21、全5 reporting foldsをcover
- elapsed: `2,684.506174854秒`（約44.74分）
- peak RSS: `1.0018768310546875 GB`
- runtime versions:
  Python `3.12.13` / NumPy `2.0.2` / pandas `2.3.3` / Numba `0.60.0`
- Numba JIT、CPU runtime、memoryはKaggle上で完走してtechnical確認済み。
- notebook完了後のpandas / nbconvert warningはあり、実行statusと生成物は正常。

### Technical gates

12 / 13 PASS。唯一のFAILはfull runtime projectionだった。

- fixed32 role / unique well: PASS
- cause role / event well fold coverage: PASS
- baseline saved-exp209 parity max abs: `0.0 ft`、PASS
- parent-only branch parity max abs: `0.0 ft`、PASS
- finite coverage: `1.0`、PASS
- normalization max abs error: `2.438424884232404e-08`、PASS
- truth / episode reads before all freeze: `0 / 0`、PASS
- event schedule readback SHA: PASS
- peak RSS: PASS
- full runtime projection:
  `64,847.60228631694秒 > 30,600秒`、FAIL

### Mechanism gates

3 / 7 PASS。

- soft datum direction agreement:
  `0.3965775516538685 < 0.60`、FAIL
- fold別direction:
  `0.439484 / 0.299473 / 0.579400 / 0.200294 / 0.480838`
- passing direction folds:
  `1 / 5 < 4 / 5`、FAIL
- backward-cause SSE reduction:
  `0.0006975834994096264 < 0.10`、FAIL
- forward-cause SSE regression:
  `-0.0029171639393186233 <= 0.02`、PASS
- matched-control pooled RMSE:
  baseline `2.8674709506741363 ft`、
  treatment `2.8653053143391847 ft`、
  delta `-0.002165636334951593 ft`、PASS
- matched-control reanchor mass:
  `0.28563503481676566 > 0.10`、FAIL
- active reanchor mass:
  `0.3288139217119144 >= 0.05`、PASS

設計補助のSSE加重datum方向一致`0.396557`と、exact future evidenceのsoft方向一致
`0.396578`は実質同じだった。対称化でrate符号の直接誤写像は避けたが、現行GR
likelihoodからabsolute datum方向を選ぶ情報は得られなかった。controlにもbranch
massを割り当て、backward-causeの主目的も約0.07%しか改善しなかったため、
trigger / prior / shift / readout / gateをsame-sample救済せず閉鎖する。

### 生成物とSHA

Kaggle logsで結果を確認後、metrics、input manifest、prediction、schedule、
branch posterior、summaryの実ファイル確認が必要なため、outputを
`/tmp/exp425-v1-output`へ一度だけ取得した。次のSHAはKaggle summaryと
downloaded fileで一致した。

- executed loose / bootstrap config:
  `ff22cf694e3e7f2afc4fb9780f43825775f156192a29a47cc302e7eb990aa046`
- scientific contract:
  `7471662e4ef0e347db76f17e0443e52a17e3f181d6b891667bc01b6323a994c1`
- baseline message manifest:
  `3f46ffa166364a5078b0f7137e583eb98bca2234b70f3e633b613bd0c83c1992`
- baseline prediction manifest:
  `3391bb9290942164a72b3a9d9c8f6e6b76ea08c01ce72678baed50b73ff0386c`
- treatment prediction manifest:
  `fb46d0332bb0adf6b8b0b88639d17548da3abc521866544ee532aa592ef877af`
- activation schedule manifest:
  `210ec11a46279a1b7e32ad965471be6d4be7dd3ceabbb6a319c6b8ae86993d76`
- shift schedule manifest:
  `fc023dc681a045f54089dca42c74febcdcdc15da0e123090da5b729292ab0f36`
- branch posterior manifest:
  `06c478a546afbcda059d37f58c69d7f817ddfde32098117c5e53c8e1abe25697`
- prediction decompressed content:
  `728cf7448ae52147719dcb5cc16e95e4349a9afdc610fe00bbe0d74fa3545319`
- event schedule decompressed content:
  `87a405ff92804b078b19d9b1e9b9f01b2d67d3012388f1778c8aca28e8452559`
- direction readout:
  `0ce9a5e44bdb8f035a2d626a0a5990ba874e977590d2195614f235e01ec01a43`
- cause episode readout:
  `5f3c799e195a76030a7ac1e47b4838fed50d3371598777f99fe69892941c0512`
- well metrics:
  `ebe27f89342886dcbc980507f6d3a95dc344ccb3da2c688de0815a92bdac6a79`
- input manifest:
  `ddc59fa557982918da16069336279ffeb887bbf23a1a5788329ca29dd699b1ec`
- Kaggle output metrics:
  `32431724e5e8308d4ef457794cad4f15a423d421cc0c55e6515e3f749d7b17a7`
- Stage 0 summary:
  `914ad2575fd294e4ac38740c1de03330de3aa77fcf321072f1168cc128a4816e`

### post-run fail-close

- loose configを`stage0_fail_closed`、`run_hmm=false`、
  `create_prediction=false`へ戻した。
- local generated train packageを再生成し、Kaggleへは再pushしていない。
- generated metadata: `run_on_push=false`
- post-run loose / packaged / bootstrap config SHA:
  `3ac3e38f9b0498b68dfb9ea6b08ffce1436c1bd9e5622f391092ed516a7c85f8`
- version 1で実行したconfig SHA
  `ff22cf694e3e7f2afc4fb9780f43825775f156192a29a47cc302e7eb990aa046`
  はKaggle version履歴、downloaded output、上記再現性記録で保持する。
- 誤ってlocal packageを再pushしても自動Stage 0実行されない状態を確認した。

## 次のアクション

exp425を`stage0_fail_closed`として閉じる。Stage 1、inference、submissionへ
進まない。完了済みexp425をアイデアバックログから削除し、同じtrigger /
symmetric-datum scheduleに依存する候補へnegative evidenceを反映する。
