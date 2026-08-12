# exp247_missing_gr_masking セッションノート

## 目的

`missing_gr_masking` backlogを、exp221 exact HMMの固定controlに対する1変更ablationとして実装する。raw horizontal GR欠損rowを補間観測として扱わず、GR emission contributionを0にする。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle CPU train v1完了・一律mask不採用・branch closed
- CV / LB: mask-only 8.322894658（fixed control 8.327728213、差 -0.004833555）/ なし
- inference / submit: disabled

## 実装前コストガード

- active variant: 1 (`mask_only`)
- LightGBM config: 0
- model training fold: 0
- booster: 0
- exp221 control HMM再生成: 0（保存済みtrain v3 predictionを固定入力化）
- exp148 control/model再学習: 0
- GPU: なし、Kaggle CPU
- hidden-test inference / submit: なし

## 変更点

- exp221 `run_hmm2`のGR emission構築を関数化した。
- raw evaluation GRが非有限のrowだけ、LGB unary追加前のGR log-likelihoodを全stateで厳密に0にする。
- prefix calibration/sigma、GR補間値、grid、rate lattice、transition、固定LGB unaryはexp221と同じ。
- raw train/test missing-run分布をHMM生成前に集計する。
- fixed controlに対するmissing-run、post-gap 128/256 rows、distance、hidden-like、by-well、finite coverage、連続divergence segmentを保存する。
- inference notebookはtrain-side-only guardで停止する。

## 再現性メモ

- `docs/06_reproducibility.md`を2026-07-14に確認。
- seed policy: `no_new_rng_exact_hmm_deterministic_ablation`。新規RNGなし。
- well順: sort済み。outer workers 2、Numba threads 2をconfigとsummaryに記録する。
- runtime: Kaggle CPU、GPU disabled、internet disabled。
- input evidence: raw horizontal/typewell file SHA inventory、exp221 control raw/decompressed SHA、exp148 OOF SHA、exp115 assignment SHA、config SHA。
- output evidence: mask predictionとrow auditはraw gzip SHAとdecompressed content SHA、他CSVはfile SHAを記録する。
- model / submission SHA: 対象外。学習・submissionなし。
- deterministic anchor: false。train-side prediction ablationでありhidden-test submission anchorではない。

## コマンドログ

### 2026-07-14 steering・実験作成

```bash
make new-steering EXP=exp247_missing_gr_masking
make new-exp EXP=exp247_missing_gr_masking SOURCE=experiments/exp221_lgb_oof_gaussian_emission_hmm_on_exp148
```

- `docs/legacy/steering/20260714-exp247-missing-gr-masking/`に要件、1変更設計、再現性方針を記載した。
- exp221 sourceからexact HMM kernelを継承し、実験固有のnotebookとconfigへ分離した。

### 2026-07-14 実装・静的検証

```bash
.venv/bin/python -m py_compile experiments/exp247_missing_gr_masking/exp247_missing_gr_masking_train.py experiments/exp247_missing_gr_masking/exp247_missing_gr_masking_inference.py experiments/exp247_missing_gr_masking/exact_hmm_smoother.py experiments/exp247_missing_gr_masking/settings.py
.venv/bin/ruff check experiments/exp247_missing_gr_masking/exp247_missing_gr_masking_train.py experiments/exp247_missing_gr_masking/exp247_missing_gr_masking_inference.py experiments/exp247_missing_gr_masking/exact_hmm_smoother.py experiments/exp247_missing_gr_masking/settings.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp247_missing_gr_masking/exp247_missing_gr_masking_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp247_missing_gr_masking/exp247_missing_gr_masking_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp247_missing_gr_masking/exp247_missing_gr_masking_train.py
make validate-exp EXP=exp247_missing_gr_masking
make validate-template
make prepare-kaggle-notebooks EXP=exp247_missing_gr_masking EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp247-missing-gr-masking-train --title 'exp247 missing gr masking train' --run-on-push --strict"
```

- py_compile: pass。
- Ruff default rule set: pass。
- Jupytext convert/test: train / inference pass。
- strict experiment validation / template validation: pass。
- helper synthetic assertion: observed GR unaryはcontrolと一致し、raw-missing rowは全stateで0。
- notebook embedded synthetic assertion: mask後もexp148 LGB unaryがmissing rowへ残ることを確認する。
- notebook sourceに`__file__`なし。
- 親exp221にcompact self-contained版はなく、正規train sourceは188行/6章。exp247 trainは1196行/11章で、raw input、missing inventory、fixed control、mask生成、metrics、SHAをnotebook上に展開した。重いNumba forward-backwardだけを`exact_hmm_smoother.py`へ残した。
- canonical package: `kentookumura/exp247-missing-gr-masking-train`、title `exp247 missing gr masking train`。
- package metadata: CPU、GPU false、internet false、run-on-push true、competition source 1、kernel sources exp221 / exp148 / exp115の3本。
- package/bootstrap config: active variant 1、LightGBM config / fold / booster `0 / 0 / 0`、parent/control retraining false、inference false。
- 初回notebook実行はKaggleを正とするため、ローカルnotebook smoke/full executionは行っていない。

## 次のアクション

- 一律missing-GR maskは不採用として閉じる。run-length gate、追加threshold grid、raw-test inference、selector、submitへ進まない。
- この結果から新しいbacklogは追加しない。別の欠損処理を再検討する場合は、tiny aggregate gainではなくworst-well回帰を抑える独立根拠を先に求める。

## 2026-07-14 Kaggle CPU train v1 pre-push

- ユーザー指示によりKaggle full auditの実行を開始する。
- canonical kernel: `kentookumura/exp247-missing-gr-masking-train`
- title: `exp247 missing gr masking train`
- active variant: 1 (`mask_only`)
- LightGBM config / fold / booster: `0 / 0 / 0`
- exp221 control再生成・再学習: なし。保存済みtrain v3 predictionを固定入力として読む。
- runtime: CPU、GPU false、internet false、run-on-push true。
- kernel sources: exp221 train / exp148 train / exp115 train の3本。
- strict experiment/template、package py_compile、Ruff、kernelspec、id/title、bootstrap configの整合を再確認した。
- push後は同じcanonical IDをpullして存在確認し、通常logsとstatusを併用して完了まで監視する。実行中logsが空でも再pushやslug変更はしない。

## 2026-07-14 Kaggle CPU train v1

```bash
make push-kaggle-train EXP=exp247_missing_gr_masking
kaggle kernels pull kentookumura/exp247-missing-gr-masking-train -p /tmp/kaggle-pull/exp247-missing-gr-masking-train -m
kaggle kernels status kentookumura/exp247-missing-gr-masking-train
```

- push result: version 1 successfully pushed。
- URL: https://www.kaggle.com/code/kentookumura/exp247-missing-gr-masking-train
- Kaggle `id_no`: `127064272`。
- pull result: canonical IDのsource/metadata取得に成功。
- pulled metadata: private、CPU、GPU/TPU/internet false、machine_shape `None`、competition source 1、kernel sources exp115/148/221。
- initial status: `KernelWorkerStatus.RUNNING`。

## 2026-07-15 Kaggle CPU train v1完了・監査

```bash
kaggle kernels status kentookumura/exp247-missing-gr-masking-train
kaggle kernels logs kentookumura/exp247-missing-gr-masking-train
kaggle kernels pull kentookumura/exp247-missing-gr-masking-train -p /tmp/kaggle-pull/exp247-v1 -m
kaggle kernels output kentookumura/exp247-missing-gr-masking-train -p /tmp/kaggle-output/exp247-v1 --file-pattern '.*(group_metrics|by_well_metrics|summary|finite_coverage|divergence_segments).*'
```

- final status: `KernelWorkerStatus.COMPLETE`。
- version / id_no: `1 / 127064272`。
- runtime: 11,409.172秒（約3時間10分9秒）、Kaggle CPU、773 wells / 3,783,989 rows。
- cost guard: active variant 1、LightGBM config / fold / booster `0 / 0 / 0`、parent/control再学習なし、GPU/inferenceなし。
- raw GR missing: 1,200,837 rows / 773 wells。mask predictionは3,782,870 rows / 773 wellsで固定controlから変化した。
- finite coverage: control / maskともprediction・std 3,783,989 / 3,783,989 rows、finite wells 773 / 773。
- divergence: 1,797 segments、最長10,052 rows。

### paired metrics

| slice | rows | control RMSE | mask RMSE | ΔRMSE mask-control | ΔMAE | Δwithin10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 3,783,989 | 8.327728213 | 8.322894658 | -0.004833555 | +0.042731938 | +0.000220667 |
| raw GR missing | 1,200,837 | 8.258335070 | 8.253294362 | -0.005040709 | +0.066277985 | +0.000492989 |
| missing run 1-4 | 907,465 | 7.661102194 | 7.660639390 | -0.000462804 | +0.063732750 | +0.000134440 |
| missing run 5-31 | 290,718 | 9.884724690 | 9.872591029 | -0.012133661 | +0.077927662 | +0.001214235 |
| missing run 1-31 guard | 1,198,183 | 8.255837996 | 8.251989133 | -0.003848864 | +0.067176896 | +0.000396434 |
| post-gap 1-128 | 2,555,191 | 8.373273302 | 8.368494772 | -0.004778531 | +0.032130143 | +0.000095492 |
| post-gap 129-256 | 20,524 | 7.366781488 | 7.364965862 | -0.001815626 | -0.000997128 | -0.000097447 |
| post-gap 257+ | 7,437 | 5.955891709 | 5.960079203 | +0.004187495 | +0.004025675 | +0.000134463 |
| distance 1000+ | 3,012,442 | 9.130472317 | 9.124143987 | -0.006328330 | +0.048925680 | +0.000367144 |
| hidden-like spatial | 972,463 | 9.572230856 | 9.578192456 | +0.005961600 | +0.047406615 | -0.001595948 |
| hidden-like typewell-purged | 976,449 | 9.545375249 | 9.540505814 | -0.004869435 | +0.040814351 | -0.000937069 |

- by-well RMSEは改善386 / 悪化387 / 同値0、median ΔRMSEは+0.000063584 ft（悪化側）。
- worst well `c66be2b8` は7.491665913 -> 10.068646557、ΔRMSE +2.576980644 ft、ΔMAE +2.030790328 ft、within10 -0.088830043。longest missing runは8 rowsで、長欠損だけの問題ではない。
- RMSEのaggregate gainは0.0048 ftと小さく、overall / missing / short-runでMAEが悪化し、hidden-like spatialとworst wellも悪化した。改善・悪化well数も均衡しており、一律maskを採用する根拠にはならない。

### SHAとartifact監査

- input config: `ff459e3aa20dee883a37d3f488148ac5b675c77c8b847a6f08899316c094fcc2`
- exp221 control raw / decompressed: `8027ac8840d1048cfbda8377bcbae6a9b47b50ad7765da38eebb7f1df57d0a54` / `ceca23fbd6b2f85a4e2d7e351f6922de41dd244f8eff2a03c282aed742dcd2b8`
- exp148 OOF source: `12f2980972c19ef72a88b198efa0f5329ee3614a21b269f1bebc5a37b3ac21b5`
- exp115 hidden-like assignment: `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`
- prediction raw / decompressed: `626c903d89c85b58b823254a3069b8c86bc6d32dbf490bf3945ded0a2b9d2a81` / `a3ac4983c52d77cb324c981bca072e8a8aef86d8efbd8c23e0cc0271eef4f339`
- row audit raw / decompressed: `da171cff90a91a91e3b3e520756b8e392405726052e5e9f70d20629d1609f3c0` / `1a7a706c053a7b5105e8559862b8e407315d7ac651910bfdaab6808c990fc421`
- downloaded small artifactのSHAはnotebook summaryと一致: group metrics `ca04f0c82721f78ba35c03b41c3e4cfc4868815981215bc9efb9d1814efaea3b`、by-well `04d8c2e8dc9aa19d8c7094353fb3fc3c9e9191b2ae26cae914ea4ae3a3078e86`、finite `7f0cf11e8c407c7b232690fc4a69194835c8d48ae0c69b8586f2c5b71024a9e5`、divergence `f24d1e376319c099f9ff5020a4faae4db9d8d78aafba3544369da76a92d36553`。
- 全773 wellsにevaluation GR missingがあるため、`no_eval_missing_control_parity_max_abs`は`null`。synthetic contractはログでPASSし、observed GR unary不変・raw-missing GR unary 0・LGB unary保持を確認した。
- v1実行時の`config.yaml`では未引用bucket labelがYAML数値として読まれ、artifact上で`1_4 -> 14`、`5_31 -> 531`、`32_127 -> 32127`、`128_255 -> 128255`、`256_999 -> 256999`、distanceも同様の表示になった。edgesとrow assignmentは正しく、上表では意図したlabelへ読み替えた。実行後のlocal configはstatus更新とlabel引用だけを行い、科学設定は変更していない。v1実行configは上記SHAを正とし、post-run local config SHAは`66f4a070a2465fca93cf1160d97b950195356058a63773b19c7d38e3ecd088b8`。

### 採否

- decision: `reject_uniform_missing_gr_mask_close_branch`。
- 一律mask、run-length gate/threshold grid、raw-test inference、selector、submissionは行わない。
