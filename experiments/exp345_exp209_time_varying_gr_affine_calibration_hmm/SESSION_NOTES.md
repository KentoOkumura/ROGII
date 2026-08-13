# exp345_exp209_time_varying_gr_affine_calibration_hmm セッションノート

## 目的

閉鎖済みexp328のcurrent-well causal GR affine仮説を、信頼できるexp209 exact-HMMへ直接接続した独立実験として設計固定し、exp338を進めても候補が埋もれない状態にする。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0 technical PASS / scientific FAIL / `stage_0_full_failed_closed`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- sibling: `exp338_exp209_well_adaptive_transition_noise`。相互依存なし
- CV / LB: last-640 Stage 0は`+0.169505 ft`・4/5 foldsだがscientific AND gate FAIL / LBなし
- Notebook: compact self-contained train候補を正規trainへ採用。inferenceはtemplate placeholderのまま
- 科学実装: 承認済み・完了
- 正規Notebook採用、microbenchmark、Stage 0 full: 完了。Stage 1、inference、submission: 未実施・無効。branchは救済なしで閉鎖

## 2026-07-22 設計作成

実行済み:

```bash
make new-steering EXP=exp345_exp209_time_varying_gr_affine_calibration_hmm
make new-exp EXP=exp345_exp209_time_varying_gr_affine_calibration_hmm
```

- 旧exp328はterminal closedのまま維持し、本実験をその明示的な独立再検証入口にした。
- 親をexp209へ固定し、観測中心の`a_t,b_t` scheduleだけを変更対象にした。
- exp209 zero-fill std `sigma_GR`、missing weight 1、`sig_r=0.002`、`sig_p=0.02`、grid、momentum、prior、posterior meanを固定した。
- exp307 finite-only/MAD scale、exp308 missing weight、exp338 well別`sig_r`、Type Well群priorを持ち込まない。
- current-well visible prefix初期fit、frozen base path、one-pass causal filter、schedule freeze、exact HMM 1回という旧exp328の主要契約を維持した。

## 実行量契約

- runtime microbenchmark: 32 wells、親/variant合計64 HMM runs、model/config/fold/booster各0。
- Stage 0: runtime PASSと別承認後のみ、1 scientific variant、親/variant合計1,546 HMM runs、model/config/fold/booster各0。
- Stage 1: Stage 0全gate PASSと別承認後のみ、1 scientific variant、773新規HMM runs、control再実行0、model/config/fold/booster各0。
- GPU: 0。CPU、internet off、Kaggle Notebook実行を正とする。

## 判定契約

- Stage 0: 親比`>=0.05 ft`、4/5 folds、GR NLL改善、boundary jump p95 `<=3 sigma`、hidden-like非悪化、worst `<=+0.25 ft`、fallback `<=50%`、projected runtime `<=8.5 h`をAND gateにする。
- Stage 1: exp209 raw HMM `11.9382872349`比`>=0.05 ft`、4/5 folds、1000+、hidden-like spatial/typewell-purged、by-well p95非悪化、worst `<=+0.25 ft`をAND gateにする。
- exp209 saved HMM decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`。
- exp209 saved LikPF decompressed SHA: `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`。

## exp338との関係

- exp338とはexp209を共有する独立兄弟で、exp338のPASS/FAILを本実験の依存条件にしない。
- 本実験のPASS/FAILも、exp338後の新exp323相当および新exp324--327相当の作成条件にしない。
- 本実験はexp338 successor chainへ合流させず、同時に複数変更した比較を作らない。

## 再現性メモ

- RNGなし。outer fold、well ID、raw row、variantの順序を固定する。
- HMMはexp209採用の`outer_workers=2`、Numba threads `2`を開始点とする。
- base path、affine schedule、process-noise contract、fallback flag、candidate predictionをtruth/fold score/hidden-like role接続前にfreezeする。
- gzip生成物はdecompressed content SHAを主証拠とし、kernel version、input/schema/content SHA、prediction SHAを記録する。
- model/submission SHAは非該当。本実験をdeterministic submission anchorとは扱わない。

## 設計時点の次のアクション

別途Kaggle実行承認が得られた場合だけstable SHA順32-well runtime microbenchmarkを行う。Stage 0 full、Stage 1、inference、submissionへ自動移行しない。

## 2026-07-22 科学実装

ユーザーの「exp345を実装してください」を科学実装の明示承認として、次を追加した。

- `exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.py`
- `exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.ipynb`
- `experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/tests/test_exp345_exp209_time_varying_gr_affine_calibration_hmm.py`

実装内容:

- stable SHA256順32-well runtime microbenchmark、last-640 Stage 0、saved exp209 base pathを使うStage 1を相互排他的なrun flagで実装した。
- Stage 0はmasked parentを先に1回実行してbase mean/stdを凍結し、affine schedule凍結後にvariant HMMを1回だけ実行する。
- process noiseはfinite pair 40件ごとのexpanding robust-prefix stateと最終stateの隣接差二乗をraw row差で正規化し、held-out wellのmasked visible prefix推定をouter-train fold medianへ`n/(n+100)`で線形縮約する。
- 初期covarianceはretained-pair OLS covarianceを`[b, log(a)]`へ変換し、EKFはJoseph form、slope clip、有限raw GRだけのcurrent-row updateを使う。
- exp209 zero-fill std、欠損row HMM weight 1、GR補間、Gaussian emission、41 rate states、`sig_r=0.002`、`sig_p=0.02`、position floor、momentum、prior、posterior meanを固定した。
- prediction、affine schedule、process noise、fallback、runtime、scientific contract、input manifestのSHAを保存し、truth、fold score、hidden-like role、saved LikPFはprediction freeze後にだけ接続する。
- Stage 1のsaved LikPFと固定50:50は診断値だけで、variantやgate救済には使わない。

実行量再確認:

- 現在有効なrun flag: 0。Kaggle HMM run 0。
- microbenchmark予定: 1 variant、親32 + variant32 = 64 HMM runs。
- Stage 0予定: 1 variant、親773 + variant773 = 1,546 HMM runs。
- Stage 1予定: 1 variant、variant 773 HMM runs、親control再実行0。
- LightGBM config 0、学習fold 0、booster 0、PF 0、Beam 0、GPU 0。

親Notebook比較:

- 親exp209にはcompact self-contained train sourceがない。正規train sourceは174行・6章で、`exact_hmm_smoother.py`など同一exp helperへ主要処理を委譲している。
- exp345 compact train sourceは約2,000行・11章で、config/path/SHA、process noise、causal EKF、exact HMM、stage orchestration、late readout、gate、生成物保存をNotebookセルで追える。
- 親のexact HMM kernelは内容を維持してself-contained候補へ持ち込み、identity schedule parityを専用testで確認した。

静的検証:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.py
.venv/bin/python -m py_compile experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.py experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/tests/test_exp345_exp209_time_varying_gr_affine_calibration_hmm.py
.venv/bin/pytest -q experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/tests/test_exp345_exp209_time_varying_gr_affine_calibration_hmm.py
make validate-exp EXP=exp345_exp209_time_varying_gr_affine_calibration_hmm
```

- Jupytext round-trip: PASS。
- `py_compile`: PASS。
- ruff: PASS。
- 専用test: `7 passed`。identity scheduleはexp209親HMMと`atol=1e-10`で一致。
- strict experiment validation: PASS。
- `__file__`依存: なし。
- ローカルfull notebook実行、Kaggle package/push/run、output取得: 未実施。

## 2026-07-22 microbenchmark push前監査

ユーザーの「実行してください」を、直前に提示したstable SHA順32-well runtime microbenchmarkと、その実行に必要なcompact self-contained候補の正規train notebook採用に対する明示承認として記録する。

- 実行stage: `stage_0_microbenchmark`のみ。
- scientific variant: 1 (`one_pass_causal_affine_schedule_on_exp209`)。
- control再実行: masked exp209 parentを32 wellsで実行。variantと同一条件のruntime比較およびbase mean/std凍結に必要で、今回の承認範囲に含む。
- candidate実行: affine variantを同じ32 wellsで実行。
- 合計: 64 HMM runs。LightGBM config 0、学習fold 0、booster 0、PF 0、Beam 0、GPU 0。
- runtime gate: 64 runの実測から773 wellsへ射影し、`<=8.5 h`だけを判定する。
- Stage 0 full (`1,546 HMM runs`)、Stage 1 (`773 new HMM runs`)、inference、submissionはrun flagをfalseのまま維持し、自動実行しない。
- Kaggle設定: CPU、internet off、input kernelはexp209 / exp226 / exp115の3件。
- 初回packageの58文字slug/titleはKaggle API `400 Bad Request`で実行前に拒否された。3つのinput kernelの存在を確認後、短いIDで再packageし、version 1のpushに成功した。
- Kaggleはtitle由来のURL slug `kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark`へ解決した。指定した`...-train`はstatus取得不可、URL slugは`RUNNING`を返したため、URL slugを唯一のcanonical kernel IDとして以後の監視・記録に使う。重複pushはしない。

## 2026-07-22 microbenchmark完了

Kaggle kernel `kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark` version 1が`COMPLETE`になったことを、同じcanonical slugのstatusと完了後logsで確認した。

確認コマンド:

```bash
kaggle kernels status kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark
kaggle kernels logs kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark
kaggle kernels output kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark -p /tmp/exp345-kaggle-output.eAbYXk
```

実行結果:

- kernel version: 1、status: `COMPLETE`、Kaggle CPU、GPU off、internet off。
- 実行scope: `stage_0_microbenchmark`だけ。parent 32 + variant 32 = 64/64 HMM runs。
- prediction: 20,480 rows / 32 wells、全finite。
- fallback: 0/32、`0.0%`。
- posterior row-sum最大絶対誤差: `2.220446049250313e-15`。
- 全体runtime: `364.48950481414795 s`。
- measured HMM parallel time: `191.7565702199936 s`。
- 773 wells外挿: `4632.11964937672 s = 1.2866999 h`、上限`8.5 h`以内。
- predeclared technical/runtime gate: PASS。
- promotion decision: `runtime_gate_passed_wait_for_stage_0_approval`。

runtime gate証拠:

- promotion gate raw SHA: `744e545edeca0864ae1b595bc30062457bf742083bcab908484e1fb53b32aca1`。
- scientific contract SHA: `90883b1a017ad5285eed1e6fcc810a4cc2cecb1d1bb600f5b0c0efe8ace6a9e8`。
- input manifest SHA: `ea8c88adaf0ccbdcd6e60a9bd3a8df27c8e537bbaf153ed86c7cb28312bbd3dc`。
- freeze manifest SHA: `985f53790c39a6ba90811f45f24cc11bd6819bc0442c8e4c6120d7055086a085`。
- prediction decompressed SHA: `e1b645f548cc3865f787edf9b5be06b5b6eade614a98b4a0ed86acaa63069423`。ローカル再計算一致。
- affine schedule decompressed SHA: `643bfe29537ea91b3d2c79c115568cdd579116e2cafaa9f1f84820da71aa5e41`。ローカル再計算一致。
- 小さいpromotion gate JSONとpaired metrics CSVだけを実験配下`artifacts/`へ保存した。その他のoutputは一時領域で確認し、リポジトリへ複製しない。

科学preview（正式microbenchmark gate外）:

- overall parent RMSE: `3.365505859689715`。
- overall candidate RMSE: `3.585675529160236`。
- improvement: `-0.22016966947052063 ft`、すなわちcandidateは`+0.220170 ft`悪化。
- fold改善: 1/5。fold 3だけ`+0.706838 ft`改善、他4 foldは悪化。
- well改善: 12/32、非改善20/32。
- best well delta candidate-parent: `-2.933062 ft`。
- worst well delta candidate-parent: `+2.385809 ft`。
- fallback 0%なので、preview悪化はfallbackだけに起因しない。

判定と次:

- runtime gateだけは契約どおりPASS。Stage 0 fullを技術的に解錠するevidence SHAをconfigへ記録した。
- 32-well subsetは科学gate用ではないため正式FAILにはしないが、overall、4/5 folds、20/32 wells、worst tailがnegativeで、次段の便益は低い。
- Kaggle push承認と`run_microbenchmark`をfalseへ戻し、`run_stage_0` / `run_stage_1`もfalseを維持した。
- 次はStage 0 full（1,546 HMM runs）を別承認するか、negative previewを根拠にfamilyを終了するかのユーザー判断待ち。自動実行とpost-hoc parameter/grid救済は行わない。

## 2026-07-22 Stage 0 full push前監査

ユーザーの「Stage 0へ進んでください」を、negative 32-well previewを提示・記録した後のStage 0 full実行と、そのmatched baselineに必要なmasked parent再実行の明示承認として記録する。

- 実行stage: `stage_0_full`だけ。
- scientific variant: 1 (`one_pass_causal_affine_schedule_on_exp209`)。
- parent control: last-640 maskを適用したexp209 parentを773 wellsで再実行する。保存済みexp209 full-suffix cacheではmasked base mean/stdを代替できず、同一mask条件のpaired baselineに必要。
- candidate: frozen masked parent base pathからcausal affine scheduleを作り、variant HMMを773 wellsで実行する。
- 合計: parent 773 + variant 773 = 1,546 HMM runs。
- LightGBM config 0、学習fold 0、booster 0、PF 0、Beam 0、GPU 0。Kaggle CPU、internet off。
- microbenchmark外挿: `4,632.1196 s = 1.2867 h`、上限8.5時間以内。
- runtime evidence SHA: `744e545edeca0864ae1b595bc30062457bf742083bcab908484e1fb53b32aca1`。
- canonical kernel: `kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark`。push前pullで`id_no=128227099`、version 1の存在、GPU off、internet offを確認した。
- 同じcanonical slugへversion 2としてpushする。別slugを作らない。
- Stage 1、inference、submission、affine/process-noise/grid救済はfalse/禁止のまま維持する。

### Stage 0 version 2 push

```bash
make prepare-kaggle-notebooks EXP=exp345_exp209_time_varying_gr_affine_calibration_hmm EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark --title 'exp345 GR affine HMM runtime microbenchmark' --run-on-push --strict"
make push-kaggle-train EXP=exp345_exp209_time_varying_gr_affine_calibration_hmm
kaggle kernels status kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark
```

- `Kernel version 2 successfully pushed`を確認した。
- canonical status: `KernelWorkerStatus.RUNNING`。
- URL: `https://www.kaggle.com/code/kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark`。
- 別slug、version 2以外の追加run、Stage 1は作成していない。
- ユーザーが完了を通知するまで長時間pollingは行わない。

## 2026-07-22 Stage 0 full完了・閉鎖

ユーザーの完了通知後、canonical kernel version 2のstatus、完了logs、必要なoutputを取得した。

確認コマンド:

```bash
kaggle kernels status kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark
kaggle kernels logs kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark
kaggle kernels output kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark -p /tmp/exp345-stage0-output.ifkAh4
```

実行scopeと状態:

- kernel version 2、status `COMPLETE`、Kaggle CPU、GPU off、internet off。
- Stage 0 fullだけを実行。parent 773 + variant 773 = 1,546/1,546 HMM runs。
- prediction 494,720 rows / 773 wells、全finite。
- fallback 0.0%、posterior row-sum最大絶対誤差`3.219646771412954e-15`。
- runtime `4871.012860536575 s = 1.3531 h`、上限8.5 h以内。
- technical gate: PASS。

科学readout:

- overall parent RMSE `14.50104778333667`、candidate `14.33154269727665`、改善`+0.16950508606002046 ft`。
- fold改善4/5。fold 1だけ`-0.00025577656705877416 ft`悪化。
- GR predictive NLLはidentity `4.6516703356317`からaffine `4.646152133963695`へ改善。
- boundary jump p95 `0.010089394905331438 sigma`でPASS。
- hidden-like checksは空で、必須spatial / typewell-purged 2 scopeの非悪化証拠がないためFAIL。
- by-well 373改善 / 400悪化。best `0dc835f3`はdelta`-17.88088390240066 ft`、worst `c03b9305`は`+9.354827458796363 ft`で上限`+0.25 ft`をFAIL。
- scientific AND gate: FAIL。最終decision: `stage_failed_close_without_rescue`。

Stage 0証拠:

- promotion gate raw SHA: `39296d1b900463c27f1fd65fbaa265e3c1a3a6b9d42afd9322eb03ac6140525a`。
- paired metrics raw SHA: `25ac09ed67108a18cc3427beb74b80434528f5ef2cd2547bd4004c099ca910f8`。
- by-well metrics raw SHA: `ad230342bcb2c950b9c52ec1ed67223ace07fea5047bccb825287ec02853d89a`。
- input manifest raw SHA: `fc81201b445b86561c851d4e4c8fc8612d852652f444fb2733d76d507ff31de9`。
- scientific contract declared SHA: `0814b2b8788204fc3561bbe37c7dc64b46f79bad353865f300272a7d6cc73b47`。
- freeze manifest SHA: `1f4aed295a28a744de92a076ff1c21babf8841f947966a7c98f35d0f854c3509`。
- prediction raw / decompressed SHA: `8e038204dc5768dc68c77931f72bddf883f1afe67ed3461119892e80800467d0` / `f2ff65b78a66c88e9993f2c362fbd9db445061670980cfffccf449ef81d4bfbc`。ローカル再計算一致。
- affine schedule raw / decompressed SHA: `0470f6ee70d91eb9a7501f5b58c6c4e4de89e1eab0b9b1b6a94eb5342c8c10b9` / `51827246e6b7154ff39d3d6a8c07d1bd0dd43715090b9f11036b67960d9bf0f0`。ローカル再計算一致。
- 小さいpromotion gate JSONとpaired metrics CSVだけを実験配下`artifacts/`へ保存し、大きなoutputは一時領域で確認した。

閉鎖処理:

- `config.yaml`を`stage_0_full_failed_closed`へ更新し、Kaggle push承認と全run flagをfalseへ戻した。
- failed Stage 0 evidence SHAが存在してもStage 1を有効化できないよう、`stage_0_gate_passed is True`を追加guardにした。
- Stage 1、affine/process-noise/grid救済、inference、submissionは実行しない。
- 実装済みexp345を`backlog/KAGGLE_DIRECTION.md`のアイデアバックログから削除し、完了判断を判断メモへ移す。
- 同familyの救済backlogは追加しない。再訪には独立根拠、別実験の事前設計、ユーザー確認を必要とする。

## 2026-07-22 完了後検証

fail-closed configとStage 1 guardをJupytext source、compact notebook、正規train notebookへ同期した。

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.py
.venv/bin/python -m py_compile experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.py experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/tests/test_exp345_exp209_time_varying_gr_affine_calibration_hmm.py
.venv/bin/pytest -q experiments/exp345_exp209_time_varying_gr_affine_calibration_hmm/tests/test_exp345_exp209_time_varying_gr_affine_calibration_hmm.py tests/test_kaggle_notebooks.py tests/test_scaffold.py
make validate-exp EXP=exp345_exp209_time_varying_gr_affine_calibration_hmm
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp345_exp209_time_varying_gr_affine_calibration_hmm --root .
```

- Jupytext round-trip、`py_compile`、ruff: PASS。
- 関連test: `18 passed`。
- strict experiment validation: PASS。
- experiment docs reviewer: core evidence categories present、exit 0。
- final fail-closed packageを同じcanonical id/titleで再生成したがpushはしていない。source configとpackage configのSHA256はともに`bd632770cf266ab9ceb9f312ca147c28738fd893e688eeccf97c9555cea619d7`で一致し、`kaggle_push_approved=false`、全run flag=false、`stage_0_gate_passed=false`、`stage_1_eligible=false`を確認した。
