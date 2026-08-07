# exp427_affine_ar1_whitened_gr_likelihood_readout セッションノート

## 目的

単純な行別GR差、Pearson / ZNCC、heuristic window scoreではなく、
known-prefix affine posteriorとfold-safe AR(1) residual covarianceを統合した
proper block predictive likelihoodに、追加のshift識別力があるかを原因分離可能な
0-HMM readoutとして設計する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0 technical / scientific FAIL・terminal close
- 優先度: 低-中 P3
- 親: `exp280_exp226_shift_likelihood_separability_readout`
- CV / LB: なし
- implementation: ユーザー依頼により承認・完了
- 正規train Notebook採用 / Kaggle Stage 0 run: 完了
- package / push / rerun: terminal close後に再無効化
- inference / prediction / submission: 未承認・無効

## 2026-07-28 設計確定

ユーザー依頼によりbacklog、steering、実験scaffoldをdesign-onlyで作成した。
exp425とexp426は既存steering・backlogに予約されていたため、衝突を避けてexp427を採番した。

### 根拠

- exp280 raw Gaussian shift score:
  MRR `0.389626`、top3 `0.452421`、shuffle差は5/5 foldsで正。
- exp343:
  GR residualの自己相関自体は強いが、per-well tauはjoint evaluable
  `0.381630`、fallback `0.618370`、ほぼ全wellで上限clipとなり不安定。
- exp345:
  affine HMMはlast-640で`+0.169505 ft`、4/5 folds改善したが、
  worst well `+9.354827 ft`でFAIL。
- exp359:
  heuristic 500-row window scoreは保存GaussianよりMRR `-0.022264`、
  top3 `-0.033496`、改善0/5 folds。
- exp360:
  ZNCC bad10 AUC `0.505164`でraw Gaussian `0.549949`より弱かった。
- exp374 / exp389:
  Student-t / Huberは平均を改善したがby-well tail gateをFAIL。

これらを受け、per-well rho、correlation score、rowwise heavy-tail replacement、
heuristic score weightを使わず、fold共通AR1とprefix affine uncertaintyを含む
Gaussian posterior predictive densityだけを新しい仮説とする。

### 固定した要因分解

- matched control: `identity_iid_matched`
- affine単独: `affine_iid`
- AR1単独: `identity_ar1`
- primary: `affine_ar1`
- strong saved reference: exp280 raw Gaussian

primaryがmatched / saved controlだけでなく両single-factor variantも上回ることを
AND gateに含め、複合変更の原因を分離する。

### Stage 0予定実行量

- scientific primary scores: 1
- diagnostic ablation scores: 2
- matched control scores: 1
- saved control scores: 1
- reporting folds: 5
- HMM / PF / Beam: `0 / 0 / 0`
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- GPU: 0
- parent control再生成: 0

Stage 0 PASSでもHMM / PFの実装権限は発生しない。decoder化は別番号、
別steering、別承認を必要とする。

### 再現性

- `docs/06_reproducibility.md`確認済み。
- real scoreはRNGなし。
- negative controlだけimmutable block key由来のstable SHA256 local RNGを使う。
- global RNGとthread scheduling依存を禁止する。
- input、prefix posterior、fold rho、eligibility、4 score、negative control、
  manifest、metricsのcontent SHAを記録する。
- gzipはdecompressed content SHAを主証拠にする。
- prediction / submissionを作らないためdeterministic submission anchorではない。

## コマンドログ

- `make new-steering EXP=exp427_affine_ar1_whitened_gr_likelihood_readout`
- steeringのrequirements / design / tasklistを記入。
- `make new-exp EXP=exp427_affine_ar1_whitened_gr_likelihood_readout`
- config、README、SESSION_NOTES、result、metricsをdesign-onlyへ更新。
- `KAGGLE_DIRECTION.md`の未着手backlogへ低-中P3として追加。
- `make validate-exp EXP=exp427_affine_ar1_whitened_gr_likelihood_readout`:
  strict PASS。
- `make validate-template`: PASS。
- `make update-summary`: `experiment_summary.md`を423実験へ更新。
- `review_exp_docs.py exp427 --root .`:
  core evidence categories present。

## 2026-07-28 Stage 0実装

ユーザーの実装依頼を受け、設計済みのscientific contractを変更せず、
compact self-contained候補として実装した。既存の正規Notebookは
明示的な採用承認がないため上書きしていない。

### 実装内容

- `exp427_*_compact_selfcontained_train.py` / `.ipynb`
  - current-well known-prefix finite pairだけのBayesian affine posterior
  - known-prefix identity residual population std、固定clip `[10, 60]`
  - contiguous finite run内だけのlag-1 Yule-Walker rho
  - outer-valid wellを除外したfold共通Fisher-z median rho
  - stationary AR(1) whitening
  - rank-2 Woodbury / matrix determinant lemmaによるGaussian predictive density
  - `identity/affine × iid/AR1`の固定2×2 score
  - exp280保存controlのexact well/block/shift alignment
  - immutable block key由来のstable negative control
  - target-free bundle content SHA freeze後だけtruth / hidden-like roleを読むledger
  - pooled / 5 folds / 1000+ / hidden-like / top1-regret p90の固定AND gate
- `exp427_*_compact_selfcontained_inference.py` / `.ipynb`
  - prediction / submissionを拒否するfail-closed contract
- `tests/test_exp427_affine_ar1_whitened_gr_likelihood_readout.py`
  - affine posterior、missing run、outer-fold exclusion、dense/Woodbury parity、
    factorial score、truth rejection、stable permutation、truth-late、AND gate、
    inference refusal、Notebook境界を専用test化

### 実行契約

- scientific primary: 1
- diagnostic ablation: 2
- matched control: 1
- saved control: 1
- reporting folds: 5
- HMM / PF / Beam: `0 / 0 / 0`
- model / trained fold / booster / GPU: `0 / 0 / 0 / 0`
- parent control再生成: 0

### Notebook比較

- 親exp280 train source: 9章、1,165行
- exp427 compact train candidate: 12章、約2,200行
- 親のscore surface、truth-late readout、metrics / 生成物保存を維持し、
  affine posterior、fold-safe AR1、2×2 factorial、saved-control alignment、
  technical/scientific gateをNotebook上で追えるように展開した。
- 同一実験helper importとscript-file path依存はない。

### 検証コマンド

- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...train.py`
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...inference.py`
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`
- `.venv/bin/python -m py_compile ...`
- `.venv/bin/ruff check ...`
- `.venv/bin/pytest -q tests/test_exp427_affine_ar1_whitened_gr_likelihood_readout.py`
- `.venv/bin/pytest -q tests/test_kaggle_notebooks.py tests/test_scaffold.py`
- `make validate-exp EXP=exp427_affine_ar1_whitened_gr_likelihood_readout`
- `make validate-template`
- `make test`

### 検証結果

- 専用pytest: `14 passed`
- synthetic dense / Woodbury最大絶対差: `1.4210854715202004e-14`
- Jupytext round-trip: train / inference PASS
- py_compile / Ruff: PASS
- strict experiment / template validation: PASS
- 共通Notebook / scaffold test: `11 passed`
- `make test`: exp427 test実行前のcollectionで既存5実験がFAIL。
  - exp297: Stage-2 scientific contract mismatch
  - exp301: `execution.implementation_authorized`欠落
  - exp333: frozen Stage 0/1 contract mismatch
  - exp336 / exp349: experiment name contract mismatch
  - 今回変更したexp427専用test、config、Notebook検証とは独立した既存failure。

## 今回未実施

- HMM / PF / Beam / model実行
- prediction / inference / submission

## 2026-07-28 Stage 0実行承認

ユーザーの「実行してください」を、compact self-contained train候補の正規
`*_train.ipynb`採用、Kaggle package、同じcanonical kernelへのpush、Stage 0 CPU
runの明示承認として記録する。正規inference Notebook、prediction、submissionは
承認範囲に含めない。

push前の固定実行量:

- scientific primary scores: 1
- diagnostic ablation scores: 2
- matched control scores: 1
- saved control scores: 1
- reporting folds: 5
- HMM / PF / Beam: `0 / 0 / 0`
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- GPU: 0
- parent control再生成: 0

Kaggle credential checkerではAPI Tokenは未設定だが、Kaggle CLIで使用できるOAuth
credentialとlegacy credentialを確認した。実行はinternet無効・GPU無効で行う。

## 次のアクション

fixed Stage 0を実行し、結果に基づきterminal closeする。exp427内のparameter /
support / gate救済、rerun、decoder、inference、submissionは行わない。

## Kaggle package / 初回push

- compact self-contained trainを正規`*_train.ipynb`へ採用した。
- 専用pytest `14 passed`、Jupytext round-trip、strict experiment validation、
  template validationを再PASSした。
- 初回package:
  `kentookumura/exp427-affine-ar1-whitened-gr-likelihood-readout-train` /
  `exp427 affine ar1 whitened gr likelihood readout train`
- metadataはid/title slug一致、CPU、internet無効、run-on-push、3 kernel sources、
  loose/package config完全一致を確認した。
- 初回pushはKaggle `SaveKernel 400`で実行前に拒否された。直後に同IDを
  `kaggle kernels pull -m`した結果も403で、kernel/versionは未作成、Stage 0
  score / HMM / PF / Beam / model / booster / GPU runはすべて0。
- slug本体54文字で、repo内の既知のKaggle 50文字上限パターンと一致する。
  科学契約、Notebook、入力、実行量、exp番号を変えず、意味を保った43文字の
  canonical pairへ同時短縮する:
  `kentookumura/exp427-affine-ar1-whitened-gr-readout-train` /
  `exp427 affine ar1 whitened gr readout train`。
- 短縮canonical packageはstrict生成、loose/package config一致、slug 43文字、
  CPU / internet無効、run-on-pushを確認した。
- push結果: `Kernel version 1 successfully pushed`。
- Kaggle側metadata pull成功:
  kernel `kentookumura/exp427-affine-ar1-whitened-gr-readout-train`、
  version 1、`id_no=128931242`、`machine_shape=None`、GPU / internet無効、
  3 kernel sources。
- 開始記録: `2026-07-28 13:04:04 UTC`、status `RUNNING`。
- push package SHA:
  - config: `70b94a04fa2dcdce5bdc71e7459a87e066d529b0e94b5f279fb45225b7dde0c1`
  - notebook: `62c2c0d12621826553d17f6b976b607f7d5f65e05b18c068b56208f4ef7d47b6`
  - metadata: `898b738a0e97ba2f1a35c8f16deddd486bd6ef0001ea41af4ae115755a919471`

## Kaggle version 1 ERROR

- 終了: `2026-07-28 13:09:53 UTC`、status `ERROR`。
- bootstrap、scientific contract、承認、入力解決を通過し、prefix affine posteriorを
  `773/773 wells`まで完了した。
- 最初のscore対象外wellで`score_rows=[]`をschemaなしDataFrameへ変換し、
  `sort_values("well_id")`が`KeyError: 'well_id'`となった。
- 科学仮説・入力・fold・score式・gateの失敗ではなく、eligible blockが0のwellを
  扱う空table schemaの実装欠陥である。
- version 1の実行量:
  prefix posterior 773 wells、factorial score 0 completed wells、
  HMM / PF / Beam / model / booster / GPU各0。
- 修正:
  eligibility schemaからscore / negative controlの型付き空DataFrameを作り、
  score対象外wellでもmanifestとeligibilityを保持する。専用回帰testを追加してから
  同一kernelへversion 2をpushする。

## Kaggle version 2

- 空well回帰testを追加し、専用pytest `15 passed`、Jupytext round-trip、
  py_compile、Ruff、strict experiment / template validationをPASSした。
- scientific contract SHA、score式、入力、fold、eligibility、gate、実行量は不変。
- version 2 package SHA:
  - config: `7fa5b1d7b111785e2b0d3252c2edd42cd7779cfe73e1add6471e44d0860d8073`
  - notebook: `ab071183b31f8e484fee1f36a7f62d3a667650f18fc4ce2ca017e741bb9afa03`
  - metadata: `898b738a0e97ba2f1a35c8f16deddd486bd6ef0001ea41af4ae115755a919471`
- 同じcanonical kernelへversion 2 push成功。post-push metadata pullで
  `id_no=128931242`、CPU / internet無効、3 kernel sourcesを再確認した。
- 開始記録: `2026-07-28 13:12:38 UTC`、status `RUNNING`。

## Kaggle version 2 COMPLETE / terminal close

- 完了: `2026-07-28 14:25:45 UTC`、Kaggle status `COMPLETE`。
- runtime `4,358.768411秒`、peak RSS `1.264053 GB`。
- 3,783,989 rows、773 wells、7,787 blocksを処理。eligibleは697 wells /
  5,615 blocks。
- 実行量は事前契約どおり:
  scientific primary 1、diagnostic ablation 2、matched control 1、
  saved control 1、reporting folds 5、HMM / PF / Beam / model config /
  trained fold / booster / GPU各0、parent control再生成0。

### Technical gate

- FAIL: eligible block fraction `0.721073584 < 0.75`。
- PASS:
  eligible well fraction `0.901681759 >= 0.90`、affine eligible well fraction
  `1.0`、score finite / row identity coverage `1.0 / 1.0`、candidate count 13、
  maximum abs fold rho `0.754092147 < 0.8`、outer-valid source overlap 0、
  dense/Woodbury差`1.4210854715202004e-14`、truth / hidden pre-freeze read
  `0 / 0`、runtime / RSS。

### Scientific gate

| family | MRR | top3 | top1-regret p90 |
| --- | ---: | ---: | ---: |
| matched identity-iid | 0.388002620 | 0.450400712 | 38.577259 |
| affine-iid | 0.385603633 | 0.437577916 | 39.802520 |
| identity-AR1 | 0.386476559 | 0.451291184 | 38.712802 |
| primary affine-AR1 | 0.386090045 | 0.439180766 | 39.852949 |
| saved exp280 | 0.388146378 | 0.449866429 | 38.499431 |
| shuffled | 0.235872482 | 0.223152271 | 75.357668 |

- primaryはmatched比MRR `-0.001912575`、top3 `-0.011219947`。
- primaryはsaved exp280比MRR `-0.002056333`、top3 `-0.010685663`。
- affine-iid比MRRは`+0.000486412 < +0.005`、identity-AR1比MRRは
  `-0.000386514`。
- matched / saved比の改善foldはMRR各`2/5`、top3各`1/5`。
- shuffle比だけはMRR / top3とも`5/5`。
- long-tailはMRR / top3とも両controlを下回った。hidden-like 2面はMRRを
  両controlより改善したがtop3を両controlより悪化した。
- primary top1-regret p90 `39.852949`はsaved `38.499431`より悪化。
- scientific gateは17 FAIL / 3 PASS。

### 再現性 / 生成物

- scientific contract:
  `75241052d0bdeba3dcbad6548167bb1193f4375b1035e8de625591d4fdb24773`
- target-free bundle:
  `3cae530e8c2629eea16468383ae06edc3e971d1ed77fb3a4d8d71d4043ba8a4d`
- target-free score content:
  `62f8d44475666552ad046e3c093f10f66ed7f62f00376266d117aebc93d87050`
- eligibility content:
  `6bc27561921873603e45293ebaff82a242e7cbbb9e20a67c6bc5cca7ed5cfa65`
- prefix posterior content:
  `6f751896ac059877ad7455cd898aa8572e16e346672b448e545a17e4ebaeb855`
- fold rho content:
  `e9f042b05cc12240b2fd08cd02cf9a5f54b1c5d10ab53779f51d1a7c2cf77aeb`
- score / eligibility / posterior / rho / manifest freeze前のtruth / hidden role readは0。
- Kaggle output archive全体は取得せず、factorial / fold / scope / rho / gate /
  summaryの小さい6ファイルだけを`artifacts/`へ保存した。
- 保存した小さい生成物のraw SHA:
  - factorial metrics:
    `d91b30abf5ed62590711ee92fd86830199f740619f66696a60e504d86341241d`
  - fold rho:
    `6e869f18d3361f120f8658c9d5636cdf0baf4f691988dceb3b112a37b4d17cb8`
  - fold metrics:
    `d9ef02e44a38215c903572becbe2ebc19acde25c5dd0ffbaa7714b6c8257e214`
  - gate:
    `35d32b87bf9076ab174f7d17caa6b856a847522a0fb18372cac07f5a9c42a2d6`
  - scope metrics:
    `1ce5f31b02842422648cc85ad82caef514290c0c3c597ebb2de8f97281fc559a`
  - summary:
    `ab4566ca6f6be9435999342632f6b29ce529b81f98ef0185d7663c343cdcb0a4`

### 判断

`stage_0_failed_close_without_rescue`。technical coverageだけでなく、eligibleな
5,615 blocks上でもprimaryがmatched / saved controlを下回るため、coverage gateを
緩和しても仮説は支持されない。prior、rho、clip、support、block、shift、score
family、gateをsame-OOF救済せず、rerun、HMM / PF / Beam、prediction、inference、
submissionへ進まない。exp427完全PASSを前提にした条件付きexp431も閉じる。

次候補は低優先度P4のsaved-artifact-only失敗原因分解に限定する。これはaffine /
AR1のtop3・tail悪化を説明するためであり、exp427の再開・parameter探索・昇格には
使わない。

## 最終検証

- 専用pytest: `15 passed`
- 共通Notebook / scaffold pytest: `11 passed`
- Jupytext train / inference round-trip: PASS
- py_compile / Ruff F821: PASS
- strict experiment / template validation: PASS
- `metrics.json` JSON / primary差分再計算: PASS
- canonical train / compact train cell一致: PASS
- `make update-summary`: `experiment_summary.md`を429実験へ更新し、
  exp427 status `stage_0_completed_gate_failed_closed`を確認
- `review_exp_docs.py exp427 --root .`: core evidence categories present
