# exp307_finite_only_robust_sigma_gr セッションノート

## 目的

exact-HMMの`σ_GR`推定から欠損GRの0補完を除き、finite std diagnosticとfinite MAD primaryを固定条件で評価する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train v2完了、全promotion gate FAILで閉鎖
- variants / HMM well-runs: `2 / 1546`
- model / LightGBM / trained folds / PF / Beam / boosters: `0 / 0 / 0 / 0 / 0 / 0`
- control再実行: 0

## 2026-07-21 設計

```bash
make new-steering EXP=exp307_finite_only_robust_sigma_gr
make new-exp EXP=exp307_finite_only_robust_sigma_gr
```

- `kaggle-review-exp`に従いsteeringを先に作成した。
- `docs/06_reproducibility.md`、exp209 code/config/result、exp264物理モデル解説を確認した。
- finite std/MAD、fallback/clip、実行量、gate、禁止事項を固定した。
- Notebookはdesign-only guardとし、実装・Kaggle実行を行っていない。

## 再現性メモ

- RNGなし、well/variant固定順。
- expected exp209 prediction SHA `8e2f4236...7ae5`、raw identity `bbb687a1...b32`。
- 実行時はscale audit、prediction、metricsのdecompressed/content SHAを保存する。
- model/submission SHAは非該当。

## 次のアクション

version 2のpromotion gate FAILを最終判断とし、sigma/clip/likelihood/HMM/blend救済、inference、submissionへ進まない。exp307 PASSを固定依存にしたdescendantは未実行のまま閉鎖する。

## 2026-07-21 実装

- ユーザーの「exp307を実装してください」を実装承認として記録した。Kaggle package/push/run承認は含めていない。
- `exp307_finite_only_robust_sigma_gr_compact_selfcontained_train.py`をJupytext percent形式で実装し、正規train Notebookを含むtrain 2冊を生成した。
- finite-only population std diagnosticと`1.4826 * MAD` primary、20 pair未満fallback 30、clip `[10,60]`、`a=1,b=0`をscale auditへ固定した。
- horizontal rawは`MD/Z/GR/TVT_input`だけを読み、scale/prediction gzipのdecompressed content SHAをfreezeした後にだけexp226の`tvt_true`を読むlate-joinにした。
- exp209の`_hmm2_fb`をself-containedに抽出し、AST同一性`True`を確認した。evaluation GR補間、typewell処理、grid/rate/transition/prior/posterior meanは固定した。
- saved exp209 exact-HMM、saved exp072 LikPF、exp226 fold/truth、exp115 hidden-like assignmentのSHA/row/order preflightを実装した。
- primary gateはdirect RMSE `>=0.05 ft`、4/5 folds、1000+、hidden-like 2面、by-well p95、worst `<=+0.25 ft`を必須とし、fixed LikPF 50:50はnon-regressionだけを必須にした。finite stdはpromotion不可。
- `exp307_finite_only_robust_sigma_gr_compact_selfcontained_inference.py`はinference/submissionをfail-closedにした。
- 親exp209にはcompact sourceがない。正規Jupytext source比較はexp209 `174行/6章`、exp307 `1,632行/10章`で、exp307はhelper importだけの薄いNotebookではない。

### 実行量ガード

- active variants: 2 (`finite_std_diagnostic`, `finite_mad_primary`)
- HMM well-runs: `2 x 773 = 1,546`
- model / LightGBM configs / trained folds / PF / Beam / boosters: `0 / 0 / 0 / 0 / 0 / 0`
- 親control再実行: 0
- Kaggle GPU: 0、予定runtimeはCPU 2 outer workers x 2 Numba threads

### 検証コマンド

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_inference.py
.venv/bin/python -m py_compile experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_train.py experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_inference.py
.venv/bin/ruff check experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_train.py experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_inference.py tests/test_exp307_finite_only_robust_sigma_gr.py
.venv/bin/pytest -q tests/test_exp307_finite_only_robust_sigma_gr.py
make validate-exp EXP=exp307_finite_only_robust_sigma_gr
make validate-template
```

- exp307対象test: `10 passed`
- shared Notebook/scaffold test: `11 passed`
- experiment validation: strict PASS
- template validation: PASS
- 全repo test: `415 passed, 1 skipped, 2 failed`。2 failureは既存exp296が`completed_train_side_guard_failed_closed`/`run_variant=false`へ更新済みなのに、testが実行承認中status/flagを期待している不整合で、exp307 testは全PASS。exp296は本実装のscope外なので変更していない。
- Notebook実行、Kaggle package/push/run、output取得は未実施。

## 2026-07-21 Kaggle CPU train v1 実行承認

- ユーザーの「実行してください」をKaggle CPU package/push/runの明示承認として記録した。
- active variants: 2 (`finite_std_diagnostic`, `finite_mad_primary`)
- HMM well-runs: `2 x 773 = 1,546`
- model / LightGBM configs / trained folds / PF / Beam / boosters: `0 / 0 / 0 / 0 / 0 / 0`
- 保存済みexp209 exact-HMM / exp072 LikPFをcontrolとして読むだけで、親control再実行は0。
- GPU 0、CPU、outer workers 2、Numba threads 2、internet off、runtime limit 30,600秒。
- canonical kernel id/title: `kentookumura/exp307-finite-only-robust-sigma-gr-train` / `exp307 finite only robust sigma gr train`。
- credential check: OAuth PASS、legacy CLI credential PASS。API Tokenは未設定だがCLI OAuth/legacy経路を使用する。
- inference、submission、exp308以降は未承認のまま。

### package / push / start確認

```bash
make prepare-kaggle-notebooks EXP=exp307_finite_only_robust_sigma_gr EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp307-finite-only-robust-sigma-gr-train --title 'exp307 finite only robust sigma gr train' --run-on-push --strict"
make push-kaggle-train EXP=exp307_finite_only_robust_sigma_gr
kaggle kernels pull kentookumura/exp307-finite-only-robust-sigma-gr-train -p /tmp/kaggle-pull/exp307-finite-only-robust-sigma-gr-train-v1 -m
kaggle kernels logs kentookumura/exp307-finite-only-robust-sigma-gr-train
```

- package metadata: private / CPU / internet off / run-on-push。
- metadata/bootstrap config: 2 variants、1,546 HMM well-runs、control再実行0、booster 0、inference/submission falseを確認。
- push: `Kernel version 1 successfully pushed`、2026-07-21 05:37 UTC。
- URL: `https://www.kaggle.com/code/kentookumura/exp307-finite-only-robust-sigma-gr-train`
- pull metadata: canonical id/title一致、`id_no=128085112`、GPU false、internet false、kernel sources 3件一致。
- push直後の通常logsは空。Kaggle CLIは実行中logsが空のことがあるため失敗とは判定せず、同一kernelを監視する。

## 2026-07-22 Kaggle CPU train v1 failure / v2修正

- v1 status: `KernelWorkerStatus.ERROR`。
- Kaggle logsでは773 / 773 wellsを処理し、2 variants x 773 = 1,546 HMM well-runsを完了した。
- 失敗時刻: notebook elapsed `38,852.478 sec`（約10時間47分32秒）。
- 最初の意味のあるerror: `ValueError: Usecols do not match columns, columns expected but not found: ['likpf_mean']`。
- failure stage: prediction/scale freeze後の`load_late_readout_frame`。分類は`code_saved_control_schema_contract`で、HMM数式、scale生成、data path、memory、networkの失敗ではない。
- 原因: 保存済みexp072 full cacheは絶対値`likpf_mean`ではなく`last_known_tvt`と差分`likpf_mean_d`を保存している。親exp209のcomparisonと既存exp279/exp305も`last_known_tvt + likpf_mean_d`で絶対TVTへ復元している。
- `kaggle kernels files`は空で、ERROR versionから再利用できる公開output fileはなかった。

### v2最小修正

- `data.saved_controls`を`likpf_anchor_column: last_known_tvt` / `likpf_delta_column: likpf_mean_d`へ修正した。
- `materialize_saved_likpf_tvt`を追加し、absolute LikPF TVTをanchor + deltaで復元する。
- HMM実行前のpreflightでsaved HMM / exp072 cacheの必須列をheader-onlyで検証する。今後の列契約ミスは高コスト計算前にfail-fastする。
- 2 variants、1,546 HMM well-runs、HMM数式、scale、truth late-join、gate、control再実行0、booster 0は変更しない。

### v2修正後の検証

```bash
.venv/bin/pytest -q tests/test_exp307_finite_only_robust_sigma_gr.py
.venv/bin/python -m py_compile experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_train.py experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_inference.py
.venv/bin/ruff check experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_train.py experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_inference.py tests/test_exp307_finite_only_robust_sigma_gr.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp307_finite_only_robust_sigma_gr/exp307_finite_only_robust_sigma_gr_compact_selfcontained_train.py
make validate-exp EXP=exp307_finite_only_robust_sigma_gr
make validate-template
```

- exp307 tests: `11 passed`。
- py_compile / ruff F821 / Jupytext round-trip / strict experiment validator / template validator: PASS。
- v2予定量: 2 variants、1,546 HMM well-runs、0 model / LightGBM / fold / PF / Beam / booster、control再実行0、CPU、internet off。

### v2 package / push / start確認

```bash
make prepare-kaggle-notebooks EXP=exp307_finite_only_robust_sigma_gr EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp307-finite-only-robust-sigma-gr-train --title 'exp307 finite only robust sigma gr train' --run-on-push --strict"
make push-kaggle-train EXP=exp307_finite_only_robust_sigma_gr
kaggle kernels pull kentookumura/exp307-finite-only-robust-sigma-gr-train -p /tmp/kaggle-pull/exp307-finite-only-robust-sigma-gr-train-v2 -m
kaggle kernels status kentookumura/exp307-finite-only-robust-sigma-gr-train
kaggle kernels logs kentookumura/exp307-finite-only-robust-sigma-gr-train
```

- package内config/source/Notebookに`last_known_tvt` + `likpf_mean_d`復元、HMM前schema preflight、2 variants / 1,546 runs / control rerun 0が含まれることを確認した。
- push: `Kernel version 2 successfully pushed`（2026-07-22）。
- canonical id/title: `kentookumura/exp307-finite-only-robust-sigma-gr-train` / `exp307 finite only robust sigma gr train`。
- id_no: `128085112`（同じcanonical kernel）。
- metadata: private / CPU / internet off / competition source 1 / kernel sources 3、canonical設定一致。
- initial status: `KernelWorkerStatus.RUNNING`。
- push直後のlogsは空。完了判定には同一version 2の後続logs/statusを使う。

## 2026-07-22 Kaggle CPU train v2完了

ユーザーの完了連絡後にstatusとlogsを取得し、`KernelWorkerStatus.COMPLETE`を確認した。logsにfold別実数がなかったため、predictionを除外し、overall/fold/scope metrics、promotion gate、summary、scale audit、by-well metrics、runtime、contract、manifestの小型生成物だけを選択取得した。取得した全小型生成物のraw SHAはNotebook summary記録値と一致した。

- Runtime: `27,402.239090 sec`（約7時間36分42秒）、上限30,600秒以内。
- Rows / wells / HMM runs: `3,783,989 / 773 / 1,546`。
- Finite coverage / ID mismatch / posterior normalization max error: `1.0 / 0 / 2.89e-15`。
- finite std direct: `14.209717676`対control `11.938287235`、改善`-2.271430441 ft`、0/5 folds。
- finite std fixed blend: `10.767489650`対control `10.269692505`、改善`-0.497797145 ft`、1/5 folds。
- finite MAD direct: `15.661340918`対control `11.938287235`、改善`-3.723053683 ft`、0/5 folds。
- finite MAD fixed blend: `11.187332636`対control `10.269692505`、改善`-0.917640131 ft`、1/5 folds。
- primary direct fold candidate-control差: `+3.746712 / +1.161175 / +3.790092 / +5.526046 / +4.123333 ft`。
- primary blend fold差: `+0.970488 / -0.381148 / +0.846989 / +1.505009 / +1.537513 ft`。
- primary 1000+ / hidden-like spatial / hidden-like typewell-purged / by-well p95 / worst差: `+3.968383 / +2.559348 / +2.348994 / +5.235247 / +77.405013 ft`。

scale auditでは旧0-fill std中央値`38.6418`、finite std中央値`13.8957`、finite MAD中央値`10.1367`。finite MADは365/773 wellsで下限10にclipされ、well別は299改善 / 474悪化だった。0補完除去はscale inflationを消したが、GR emissionを過度に鋭くし、誤modeを過信させたと解釈する。

strict technical gateはsaved LikPFとblend baselineの差`3.28e-6 / 3.64e-6 ft`が許容値`1e-6`を超えてFAILした。raw HMM parity、coverage、identity、runtime、normalizationはPASSしており、この微小差は候補悪化`0.498--3.723 ft`を救済しないためnegative decisionは信頼できる。

主要SHA:

- scientific contract: `abb340cee25878ede3c87a0017e02920952d1ba01680748b36010430387f6ce2`
- input/control manifest: `0382500b48ec41ee30a91bc6b843e7cb602dcf590ba561eb58a9296f6c67f8fc`
- prediction raw/content: `76bacfb2cb5c85f081ab84620a53097d3a35cbbee8560ff6edc84d20abf03486` / `8138303bcbb86bf804983496403cb68896a95295d3bec03d11b9534a2122e522`
- scale audit raw/content: `2d1b6f8be357a1800b2e85bfd8357bdcc5040661229c76ea997f852292a694aa` / `edde07fb51505ae186b8188d3c3688ff6a80dad22aef96e24c92bbc48f221e19`
- promotion gate / overall metrics / by-well metrics: `a8f134f6d1e7a296e93016f5902f5805970a274511c262c95d4d502c074fc68e` / `c02cd669ecdbdcbcf4d941f7fad5d5cb751cd65f62c70844317f4366aff2bee5` / `6edb4b82dfdbcc81c33e617d59ff6a995bdde4f77375fc01aad53cdd3193ea0e`

事前登録どおり`finite_mad_primary_failed_close_without_rescue`を適用する。sigma/clip/likelihood/HMM/blend救済、inference、submissionは行わない。exp307 PASSを固定依存にするexp308/310、exp308 PASS依存のexp309、同chainのexp323--328は未実行のまま閉鎖する。
