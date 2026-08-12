# exp424_exp209_momentum1_exact_hmm_ablation セッションノート

## 目的

exp209 exact HMMの`mom=0.998`による0方向rate mean reversionが、
exp408で観測されたrate under-responseとpersistent offsetへ因果的に寄与するかを、
`mom=1.0`だけの単一変更で検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0 technical PASS / mechanism FAIL・`stage0_fail_closed`
- 優先度: P3
- CV / LB: なし
- inference / submission: 無効
- 実装承認: あり（2026-07-28 ユーザー依頼）
- Kaggle実行: Stage 0のみ承認・完了（2026-07-28）

## 2026-07-28 設計確定

ユーザー依頼により、backlog、steering、実験scaffoldをdesign-onlyで作成した。
実装、正規Notebook編集、Kaggle package / push / run、inference、submissionは
今回の範囲に含めない。

### 根拠

- exp408 filtered zero-directed rate under-response:
  `70.9074% rows / 70.3580% SSE`
- exp408 persistent-offset exclusive forward cause SSE:
  `59.3978%`
- `mom=0.998`のdMD 1 ftでのrate半減期:
  約346 rows
- exp338の実質global`sig_r=0.004`:
  direct HMM `+2.124061 ft`、改善fold`0/5`
- exp411のdirectional trigger:
  future-rate方向一致`0.225397`、fold`0/5`でFAIL

### 固定した単一変更

- parent: `mom=0.998`
- treatment: `mom=1.0`
- `sig_r=0.002`とそれ以外のHMM grammarは全固定

`mom=1.0`はrate変化へのdiffusionを増やさず、0方向mean driftだけを除く。
一方、stale initial rateを長く保持するリスクがあるため、改善を前提にしない。

### Stage 0予定

- baseline variants / HMM well-runs: `1 / 32`
- treatment variants: 1
- target wells / treatment HMM well-runs: `32 / 32`
- total HMM well-runs: 64
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`
- fixed32:
  persistent 16 / matched control 16、SHA
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`

fixed32はmechanism診断専用であり、Stage 0のスコアをCV、full OOF、
promotion evidenceとは呼ばない。

### Stage 1予定

Stage 0全gate PASSと別承認後だけ実行する。

- treatment variants: 1
- target wells / treatment HMM well-runs: `773 / 773`
- saved parent control HMM reruns: 0
- reporting folds: 5
- model / booster / PF / Beam / GPU: 0

### 再現性

- `docs/06_reproducibility.md`確認済み。
- RNGなし。well / row / state / fold順を固定する。
- input、fixed32、prediction、rate readout、metricsのSHAを記録する。
- gzipはdecompressed content SHAを主証拠にする。
- submissionを作らないためdeterministic submission anchorとは呼ばない。
- 実装後もKaggle push前にloose / bootstrap configとNotebook bodyを照合する。

## コマンドログ

- `make new-steering EXP=exp424_exp209_momentum1_exact_hmm_ablation`
- steeringのrequirements / design / tasklistをdesign-onlyで記入。
- `make new-exp EXP=exp424_exp209_momentum1_exact_hmm_ablation`
- config、README、SESSION_NOTES、result、metricsをdesign-onlyへ更新。
- `KAGGLE_DIRECTION.md`の未着手バックログへP3として追加。
- 設計レビューでsample-matched parent rate momentが保存されていないことを確認し、
  Stage 0をbaseline 32 + treatment 32 = 64 HMM runsへ訂正。
  Stage 1は保存済みparent predictionを使うためtreatment 773 runsのまま維持。
- `make validate-exp EXP=exp424_exp209_momentum1_exact_hmm_ablation`: strict PASS。
- `make validate-template`: PASS。
- `make update-summary`: 420 experimentsを更新。
- `review_exp_docs.py exp424 --root .`: core evidence categories present。

## 2026-07-28 Stage 0実装

ユーザーの「exp424を実装してください」を実装・正規Notebook採用の承認として、
Kaggle実行を伴わないStage 0コードと契約テストを追加した。

### 実装内容

- `exp424_exp209_momentum1_exact_hmm_ablation_compact_selfcontained_train.py`
  - exp209 exact HMMの入力準備、rate / position transition、
    forward-backward、posterior-mean readoutを自己完結実装。
  - parent `mom=0.998`とtreatment `mom=1.0`のleaf差分を`mom`だけに固定。
  - baselineを保存済みexp209 float32 predictionへ`<=1e-5 ft`で照合し、
    fail-close後にだけtreatmentを実行する。
  - predictive / filtered / smoothed rate mean、filtered / smoothed rate std、
    rate-grid両端1 state massを両variantで保存する。
  - 32 wells全prediction / rate readoutをfreezeしてcontent SHAを作るまで、
    suffix truthとpersistent episode ledgerを読まない。
  - persistent episode SSE、0方向under-response SSE share、persistent well / fold、
    matched control、rate-edge mass、runtime / RSSをAND gate化。
- compact / 正規train NotebookをJupytextから生成・採用。
- inferenceはfail-closed placeholderだけを実装し、正規Notebookへ採用。
- `experiments/exp424_exp209_momentum1_exact_hmm_ablation/tests/test_exp424_exp209_momentum1_exact_hmm_ablation.py`を追加。

### コスト再確認

- active treatment variant: 1
- Stage 0 baseline / treatment / total HMM well-runs: `32 / 32 / 64`
- Stage 1 planned treatment HMM well-runs: 773（未実装・未承認）
- LightGBM config / trained fold / booster / model: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`
- parent control再実行: Stage 0は32、Stage 1予定は0

Stage 0はsample-matched internal rate momentが必要なためparent 32を再実行する。
このKaggle CPUコストは実装承認に含めず、package / push / run前に別承認を得る。

### Notebook比較

親exp209にはcompact self-contained sourceがないため、同じfixed32 exact-HMM
Stage 0構成のexp411 / exp412を実装参照とした。

- exp411: 9章 / 2255行
- exp412: 9章 / 2333行
- exp424: 9章 / 2237行

exp424はpredictionに加えてbaseline / treatmentのfiltered / smoothed rate readout、
truth-late episode gateを持ち、薄いhelper呼び出しNotebookではない。

### 検証

- `py_compile` train / inference / test: PASS
- `ruff --select F821,F401`: PASS
- 専用`pytest`: `10 passed`
- Jupytext train / inference変換と`--test`: PASS
- 正規Notebook: train 20 cells（Markdown 11 / code 9）、
  inference 8 cells（Markdown 5 / code 3）、output 0
- fixed32 persistent 16 wellsに対するepisode ledger:
  25 episodes / 20,669 rows、全16 wellsをcoverし、
  `end_row_idx_exclusive - start_row_idx == rows`を確認
- `make validate-exp EXP=exp424_exp209_momentum1_exact_hmm_ablation`:
  strict PASS
- `make validate-template`: PASS
- `__file__`、同一exp helper import、exp411 trigger、exp412 beta schedule:
  train sourceに残存なし
- `make test`: exp424 test実行前のcollectionで既存5実験が停止したため全体FAIL。
  - exp297: Stage-2 scientific contract mismatch
  - exp301: `execution.implementation_authorized` KeyError
  - exp333: frozen Stage 0/1 contract mismatch
  - exp336 / exp349: experiment name contract mismatch
  - exp424専用testは単独再実行で`10 passed`。上記既存実験には変更を加えない。

## 実装完了時点の未実施事項

- Kaggle package / push / run
- Stage 0 / Stage 1
- inference / submission

## 2026-07-28 Stage 0実行承認

ユーザーの「実行してください」を、事前登録済みStage 0だけのKaggle CPU
package / push / run承認として記録した。

- baseline variants / HMM well-runs: `1 / 32`
- treatment variants / HMM well-runs: `1 / 32`
- total HMM well-runs: 64
- LightGBM config / trained fold / booster / model: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`
- Stage 1 / inference / submission: 未承認・無効
- canonical kernel:
  `kentookumura/exp424-exp209-momentum1-exact-hmm-ablation-train`
- runtime: Kaggle private CPU、internet無効

OAuth credential、legacy identity/keyは利用可能。API tokenは未設定だが、
Kaggle CLIはOAuth credentialで利用可能と確認した。

次はstrict packageを生成し、bootstrap config / asset SHA / metadataを確認して
canonical kernel idへpushする。Stage 0全gate PASS時だけ、別承認後にStage 1
実装・実行を検討する。

### push前package検証

- `make prepare-kaggle-notebooks ... --run-on-push --strict`: PASS
- metadata id/title slug: canonical一致
- private CPU / GPU false / internet false / run_on_push true
- kernel source:
  `kentookumura/exp209-joint-exact-parity-train`
- bootstrap: 32 files
- embedded config:
  Stage 0 authorized / `run_hmm=true` / `create_prediction=true`
- embedded execution:
  64 HMM well-runs、model / booster / PF / Beam / GPU 0
- embedded fixed32 SHA:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
- embedded episode SHA:
  `031067fa77c195b77920a0997401310fbdd16532a2d0e99a9c3b5044de28913c`
- Stage 1 / inference / submission: false

### Kaggle Stage 0 push

- push: 2026-07-28 11:34:10 UTC
- kernel version: 1
- canonical kernel:
  `kentookumura/exp424-exp209-momentum1-exact-hmm-ablation-train`
- Kaggle kernel id: `128924158`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp424-exp209-momentum1-exact-hmm-ablation-train`
- pulled metadata:
  private / GPU false / internet false / `machine_shape=None`
- Docker image:
  `gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`
- kernel source:
  `kentookumura/exp209-joint-exact-parity-train`
- initial status: `KernelWorkerStatus.RUNNING`

### Stage 0完了

- final status: `KernelWorkerStatus.COMPLETE`
- completion確認: 2026-07-28 12:09:12 UTC
- executed:
  baseline 32 + treatment 32 = 64 HMM well-runs
- model / booster / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0`
- suffix rows / wells:
  `156,088 / 32`
- elapsed:
  `2,077.533832秒`
- peak RSS:
  `1.030926 GB`
- full 773-well treatment runtime projection:
  `24,402.685805秒`
- technical gate:
  13 / 13 PASS
- mechanism gate:
  3 / 7 PASS
- final:
  `stage0_fail_closed`
- Stage 1 eligible:
  false

mechanism gateの詳細:

| gate | observed | threshold | result |
| --- | ---: | ---: | --- |
| persistent episode SSE reduction | `0.475550%` | `>=5%` | FAIL |
| persistent improved wells | `8 / 16` | `>=10 / 16` | FAIL |
| persistent improving folds | `3 / 5` | `>=4 / 5` | FAIL |
| under-response SSE share reduction | `9.849995 points` | `>=2 points` | PASS |
| control pooled RMSE delta | `-0.054769 ft` | `<=+0.02 ft` | PASS |
| control by-well RMSE delta p95 | `+0.157066 ft` | `<=+0.25 ft` | PASS |
| smoothed rate edge mass delta | `+0.000377954` | nonworse | FAIL |

fold別persistent episode SSEはfold 0 / 1が悪化し、fold 2 / 3 / 4が改善した。
baseline readbackは保存済みexp209 predictionと最大差`0.0 ft`、
posterior normalization最大誤差は`2.18467e-08`、finite coverageは`1.0`、
nonfinite rate momentは0だった。

### output実体とSHA

Kaggle outputは`/tmp/exp424-output.ECih0s`へ一時取得し、実体SHAを
Notebook summaryと照合した。大きなprediction / rate readoutはGit管理対象へ
コピーせず、実験直下には`metrics.json`だけを反映した。

- `metrics.json`:
  `54f2c0e7395c8986556aebcd215e146ccff6b0adfdc13fbc2c0fbeb2b1d02ccd`
- summary:
  `a3b99f3864201e431477f22135f92c2d796af1f4f2cdb9e72c4c799829075af4`
- prediction raw gzip:
  `7f3a43299c27b3d2e5a7198aeabf19c5b2f79ad2a8cfd4be02ca1b671dac1898`
- prediction decompressed/logical:
  `77a26a70dcbfe64e1ff9b75df3a65ec4f65d3e0f3bbc7da01c2d5884784e3271`
- rate readout raw gzip:
  `255aeedf59d42bf30d80c6faaab9e475390b70fc5e2007fca751616f45889f16`
- rate readout decompressed/logical:
  `9eb313efcefc4fe90374a9d05b60df1a387a9e7a00eb519718017747c9abd270`
- well metrics:
  `d33001a341f2c0f597aa424dc029818ad6c07dc5a04a0b010955f90aca9ef250`
- episode truth-late readout:
  `a378b7897158a3ac623bb531d22fb9a541976a4cecef2551fdc5c81c20403236`
- input manifest:
  `704b173ae59756400db22a058b9ddeda37a377bba19a43d3433771681dc495e4`
- scientific contract:
  `8d2d6ab5376ae78bdd8c3128da8499cae701e4d375d53ad2aefd5203a838c6cd`

### 判断

rate under-response SSE shareは改善しcontrol safetyも満たしたが、主目的の
persistent episode SSE、改善well数、fold一貫性、rate edge massが事前gateを
満たさなかった。固定sampleでも安定したTVT修復へ転移しないため、
`mom=1.0`単独branchを閉じる。設計どおりsame-OOFのparameter / gate / sample /
blend rescueを行わず、Stage 1、inference、submissionは実行しない。

### 結果反映後の検証

- local `metrics.json` SHA:
  Kaggle outputの
  `54f2c0e7395c8986556aebcd215e146ccff6b0adfdc13fbc2c0fbeb2b1d02ccd`
  と一致
- compact train / inference Jupytext `--test`: PASS
- train / inference / dedicated test `py_compile`: PASS
- Ruff `F821,F401`: PASS
- dedicated pytest: `10 passed`
- `make validate-exp`: strict PASS
- `make validate-template`: PASS
- `review_exp_docs.py`: core evidence categories present
- `make update-summary`: `stage0_fail_closed`を反映
