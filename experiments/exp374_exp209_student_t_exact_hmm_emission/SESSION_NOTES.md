# exp374_exp209_student_t_exact_hmm_emission セッションノート

## 目的

他のexp209系実験と同様に、exp209 absolute-TV​T exact HMMを親とし、
Gaussian emissionだけを固定`df=4` Student-t尤度へ置換する独立実験を実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: terminal close（by-well tail gate FAIL、no rescue）
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- negative reference: `exp342_exp226_student_t_residual_offset_emission_audit`
- CV / LB: `11.720478702142103` / 未提出
- Notebook: compact self-contained train候補を正規trainへ採用、inferenceはplaceholder維持
- Jupytext source / helper / tests / Kaggle package: あり / なし / 9件PASS / あり
- implementation / train / inference / submission承認:
  `true / true / false / false`

## 2026-07-24 Kaggle完了

- canonical private CPU kernel version 1、id_no `128436182`を完了した。
- runtimeは`19,662.082424 sec`（約5時間27分42秒）。
- 実行量は1 variant / 773 HMM well-runs / LightGBM config・trained fold・
  booster・control再実行各0で、事前契約どおり。
- technical gateはPASS。3,783,989 rows / 773 wells、finite 1.0、
  ID mismatch 0、posterior normalization誤差最大`4.22e-15`、
  truth-before-freeze 0、input/control parityを確認した。
- directは`11.938287234887435 → 11.720478702142103`、
  `+0.21780853274533207 ft`改善し、4/5 folds改善。
- fixed LikPF/HMM 50:50も`10.269692505026358 → 10.12538554487205`、
  `+0.14430696015430833 ft`改善。
- raw observed / raw missing / high missing / 1000+ / hidden-like spatial /
  hidden-like typewell-purgedはそれぞれ`+0.131927 / +0.404486 / +0.503873 /
  +0.236105 / +0.488873 / +0.415856 ft`改善した。
- well単位では430/773改善、343/773悪化。delta p95は`+0.982661 ft`で
  必須`<=0`をFAILした。
- worst well `a6f967fb`は`12.785496 → 47.801459`、
  `+35.015963 ft`悪化し上限`+0.25 ft`をFAILした。
- decisionは`student_t_exp209_failed_close_without_rescue`。
  df/scale/temperature/clip/mixture/Huber/sigma/transition/grid/blend救済、
  再実行、inference、submissionは行わずterminal closeする。
- logsを一次証拠とし、fold/scope/tail数値がlogsに不足したため
  overall/fold/scope、by-well、gate、summaryの小規模4ファイルだけを取得した。
  86 MBのprediction archiveは取得していない。
- prediction content SHA:
  `668fe87da902955acee742c72d30724abb53f32050bb5d0a5c1b3dee0cbd626e`
- scientific contract SHA:
  `1425655ef89d0b7f887480a28a74f747115df9104a158cc821aa27b58b5ba0e5`
- promotion gate SHA:
  `d8334237a3da5e3e8deee159971dfc7fbe50a2793332ba93c10311c879298d4a`

## 2026-07-24 実行承認

- ユーザー指示「実行してください」により、正規train Notebook採用、
  Kaggle package/push/runを承認済みと記録した。
- 実行量は`1 variant / 773 HMM well-runs / LightGBM config 0 /
  trained fold 0 / booster 0 / Gaussian control再実行0`。
- CPU、internet off、`num_workers=1`、Numba thread 2、上限30,600秒。
- canonical kernel:
  `kentookumura/exp374-exp209-student-t-exact-hmm-emission-train`
- canonical title:
  `exp374 exp209 student t exact hmm emission train`
- exp209、exp226、exp115の3 kernel sourceと必要成果物の存在をKaggle CLIで確認した。
- package metadataはcanonical id/title、private、CPU、internet off、run-on-push、
  competition source 1、kernel source 3を満たす。
- package notebookはbootstrap hash検証セルと最終`run_full_experiment(CONFIG)`セルを持ち、
  package内configは生成時点の正本configとSHA一致を確認した。
- 正本/package config SHA:
  `f9303cb4da2f3012e8b5ca7547b7b7aee2f78dd62c063fa6c64a30e739449c66`
- 正規train Notebook SHA:
  `18bbe93f0f33b65f792b4651d0647fbc6f70b9c2784e87312fb1f77a52f084d2`
- push用train Notebook SHA:
  `3d3b9e8e1f94385a7d9dcaf379014aa1bc732842a1883f96bc4c1cfbabe60371`
- kernel metadata SHA:
  `2fc5f753a278667090dc4ec4fa50c85c0239224d0e8ea31eab98f3c5b4a260a7`
- package検証は専用9件と共通Kaggle notebook 4件の計`13 passed`。
- raw-test inferenceとsubmissionは今回の承認に含めず、引き続きfail-closedとする。
- 2026-07-24 04:52:50 UTCにcanonical kernel version 1をpushした。
- Kaggle pullでid_no `128436182`、private、GPU/internet off、3 kernel sources、
  canonical id/titleを確認した。
- 起動直後と約1分後のstatusはいずれも`KernelWorkerStatus.RUNNING`。
- 実行中の通常`kaggle kernels logs`は空で返った。既知挙動として扱い、
  再pushやslug変更は行っていない。

## 2026-07-24 実装

- ユーザー指示「exp374を実装してください」をimplementation承認として記録した。
- exp346のcompact self-contained監査構成を参照し、次だけを実装した。
  - exp209 known-prefix zero-fill population std、raw GR補間、Type Well GR、
    absolute-TV​T grid、41 rate states、transition、prior、momentum、
    posterior meanを固定。
  - 行emissionだけを`-2.5 * log1p(z^2 / 4)`へ置換。
  - raw observed/missing、missing-fraction、1000+、hidden-like、by-well、
    fixed LikPF/HMM 50:50の固定readoutとAND gate。
  - prediction・raw-GR/emission contract・observation auditをcontent SHA付きで
    freezeした後だけtruth/controlをjoinする。
- exp226 pre-freeze allowlistは`well_id,row_idx,suffix_offset,fold`だけとし、
  `tvt_geop,tvt_pred,gr_delta,tvt_true,error,abs_error`をdecoder入力から禁止した。
- synthetic wellで親exp209 `run_hmm2(emission="t", df=4)`とのmean/std/loglik一致、
  exact forward-backward kernel parityを確認した。
- fail-closed inference候補はtrain-side promotionと別承認までRuntimeErrorで停止する。
- 実装時点では正規Notebookを上書きせず、compact candidateだけを生成した。
- 実装時点ではKaggle package/push/run、ローカルNotebook実行、推論、提出を
  行っていなかった。その後の別承認で正規train採用とKaggle runを行った。

## 2026-07-24 設計確定

- exp372は使用済み、exp373は既存steeringで予約済みのためexp374を採番した。
- 科学的親はexp209。exp342のexp281 residual-offset HMMへreparentしない。
- 変更は行log emissionのみ。
  - Gaussian: `-0.5 * min(z^2, 600)`
  - Student-t: `-0.5 * (df + 1) * log1p(z^2 / df)`、`df=4`
- absolute-TV​T grid、41 rate states、transition、prior、sigma、missing-GR処理、
  Type Well GR、momentum、weight、posterior meanはexp209から変更しない。
- exp226`tvt_geop`、residual-offset、segment/rate prior、GR affine、
  missing-distance weight、ACF temperingは導入しない。
- exp342 shift-rank Stage 0は使わず、将来の承認後に1 variantを773 wellsへ
  直接適用する。
- controlは保存済みexp209 Gaussian exact HMM
  `11.938287234887435`であり、control HMMは再実行しない。
- secondary readoutは保存LikPFと候補HMMの固定50:50。Gaussian基準は
  `10.269696146642758`で、weight searchはしない。
- technical/scientific gateはAND。FAILなら
  `student_t_exp209_failed_close_without_rescue`として閉じる。
- df、scale、temperature、clip、mixture、Huber、sigma、missing weight、
  transition、grid、blend weightによる同一OOF救済は禁止する。

## 実行量ガード

- scientific variant: `1`
- HMM well-runs: `773`
- Gaussian parent HMM再実行: `0`
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- PF / Beam run: `0 / 0`
- GPU: `false`
- inference / submission: `0 / 0`
- 将来想定runtime: CPUで約3.2〜8.5時間、上限`30,600 sec`

実装承認はrun承認を兼ねない。Kaggle run前に
`1 variant / 773 HMM / model 0 / booster 0 / control rerun 0`
を再確認して別承認を得る。

## 再現性メモ

- seed policy: RNGなし。well、row、grid、rate、variant順を固定する。
- stochastic components: なし。
- exp209 HMM decompressed SHA:
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- exp072 LikPF decompressed SHA:
  `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`
- exp226 fold OOF decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- hidden-like assignment SHA:
  `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`
- 将来はkernel version、input/contract SHA、prediction raw/decompressed/logical
  SHA、metrics/gate SHAを記録する。
- inference/submission未実装なのでdeterministic submission anchorとは呼ばない。

## コマンドログ

```text
make new-steering EXP=exp374_exp209_student_t_exact_hmm_emission
make new-exp EXP=exp374_exp209_student_t_exact_hmm_emission
make update-summary
.venv/bin/python scripts/validate_experiment.py --experiment exp374_exp209_student_t_exact_hmm_emission
.venv/bin/python scripts/validate_project.py --strict
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp374 --root .
.venv/bin/python -m py_compile experiments/exp374_exp209_student_t_exact_hmm_emission/*compact_selfcontained*.py
.venv/bin/ruff check experiments/exp374_exp209_student_t_exact_hmm_emission/*compact_selfcontained*.py experiments/exp374_exp209_student_t_exact_hmm_emission/tests/test_exp374_exp209_student_t_exact_hmm_emission.py --select F821,F401,F841
.venv/bin/pytest -q experiments/exp374_exp209_student_t_exact_hmm_emission/tests/test_exp374_exp209_student_t_exact_hmm_emission.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp374_exp209_student_t_exact_hmm_emission/*compact_selfcontained*.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp374_exp209_student_t_exact_hmm_emission/*compact_selfcontained*.py
make prepare-kaggle-notebooks EXP=exp374_exp209_student_t_exact_hmm_emission EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp374-exp209-student-t-exact-hmm-emission-train --title 'exp374 exp209 student t exact hmm emission train' --run-on-push --strict"
make push-kaggle-train EXP=exp374_exp209_student_t_exact_hmm_emission
kaggle kernels status kentookumura/exp374-exp209-student-t-exact-hmm-emission-train
kaggle kernels logs kentookumura/exp374-exp209-student-t-exact-hmm-emission-train
kaggle kernels output kentookumura/exp374-exp209-student-t-exact-hmm-emission-train --file-pattern '<metrics/gate/summary only>'
make update-summary
```

設計時は`task`実行環境がないため同じrepository automationのMakefile入口を使った。
実装時の専用pytestは`9 passed`、py_compile / Ruff / Jupytext変換・round-tripはPASS。
`validate_experiment.py`、`validate_project.py --strict`、`make validate-exp`、
`review_exp_docs.py exp374 --root .`もPASSした。
構成参照元exp346は1,900行、exp374 train候補は1,880行で同じ10章構成を維持した。
ローカルNotebook実行、推論、提出は行っていない。

## 次のアクション

1. 事前登録どおり同一OOF救済、再実行、inference、submissionを行わない。
2. 同familyの新規backlogは追加せず、既存P1/P2を維持する。
