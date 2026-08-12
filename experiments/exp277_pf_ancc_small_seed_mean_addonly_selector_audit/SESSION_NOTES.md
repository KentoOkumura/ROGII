# exp277 PF ANCC small-seed mean add-only selector audit セッションノート

## 目的

`KAGGLE_DIRECTION.md` の `pf_ancc_small_seed_mean_addonly_selector_audit` を実装する。
exp263/exp264系selectorの既存`pf_ancc`をexp271の保存済みmean4/mean8 candidateへ差し替え、
target-free disagreementをboth variantだけへ加え、exp264 fixed controlに対するdownstream add-only価値をfold-safeに監査する。

## 現在の状態

- route: `ensemble`
- status: corrected親へのport・静的検証・design-only package完了、corrected Kaggle run再承認待ち
- CV / LB: corrected run未実行。旧mean4 v1 CVは無効 / 未提出
- inference / submission: disabled
- parent/control再学習: なし

## 固定設計

- candidate input: exp271 version 2 candidate gzip
  - raw SHA: `a01f2082717c17c5c22ef26dc91f7f87cc98cb48e2d4e0c92dc0a9a0b922590a`
  - decompressed SHA: `a7c48204d6782e62941e433b5d47ba5e03f6e441b8e601461be1c63ebcdca336`
  - schema SHA: `9037c3e40cd7a4ad8535479dcad7ee16885c2214940a6c357915e8ec8b2a5ba9`
- core bank: exp263固定12候補、manifest SHA `85e60ac1...a26bb9e`、60 partitions。
- fixed baseline: corrected exp264 Stage D v3 clean-273 `matched_control__lgb_mean__pred_tvt` OOF、
  SHA `7367983f3053186e0a6adf18c0f145302b0451332625fb679357f3c1326dafee`。control modelは再学習しない。
- selector: corrected exp264 raw-test-only 88特徴、outer 5 × inner 4、`pred_abs_error` / `p_within10`。
- downstream: exp218 clean 273 base features、compact 74との最終347列、固定3 LightGBM configs、5 outer folds。
- hidden-like assignmentは事後評価だけに使う。

## Variant

| variant | candidate | disagreement feature |
| --- | --- | --- |
| `mean4_only` | core12の`pf_ancc` → mean4、計12候補 | なし |
| `mean8_only` | core12の`pf_ancc` → mean8、計12候補 | なし |
| `mean4_mean8_disagreement` | core12の`pf_ancc` → mean4 + mean8、計13候補 | seed std4/8、particle std4/8、mean差signed/absolute |

single-candidate controlはmean4を既定とする。mean8/bothがmean4を更新しなければ4 seed契約へ縮約する。

## 学習コスト契約

| Stage | 1 runのvariant | objective/config | fold | booster | device | control再学習 |
| --- | ---: | ---: | --- | ---: | --- | --- |
| design | 0 | 0 | 0 | 0 | CPU | なし |
| nested selector | 1 | 2 objectives | 5 outer × 4 inner | 40 | CPU | なし |
| downstream add-only | 1 | 3 configs | 5 outer | 15 | GPU | なし |
| aggregate compare | 0 | 0 | 0 | 0 | CPU | なし |

3 nested stageをすべて実行する場合は合計120 CPU selector boosters、3 downstream stageは合計45 GPU
boosters。ただし一括pushせずvariant単位で実行し、各Kaggle push前にscopeを再確認して明示承認を得る。

## リーク・禁止事項

- target / error / oracle / hidden-like roleをfeature、gate、candidate選択へ使わない。
- exp271 path凍結後にだけraw TVTをlabel/評価へjoinする。
- outer-valid targetをselector/downstream fit、calibration、early stoppingへ使わない。
- hard top1、oracle routing、candidate平均、control再学習を行わない。
- train-side PF再生成、raw-test PF再生成、inference、submissionを行わない。

## 再現性

- `docs/06_reproducibility.md`を2026-07-18に確認した。
- stochastic PF feature generationはなく、exp271固定gzipだけを読む。
- selector samplingはexp264と同じseed 42 / stable SHA256、LightGBM deterministic設定を維持する。
- gzipはdecompressed SHAを主証拠、raw SHAを副証拠にする。
- nested feature/schema/compact/model manifest、downstream model/prediction/metrics、Kaggle kernel versionを記録する。
- submission SHAはscope外。

## 実装

- `docs/legacy/steering/20260718-exp277-pf-ancc-small-seed-mean-addonly-selector-audit/`へ要件・設計・tasklistを作成。
- `src.candidate_selector_pipeline.run_stage_a/run_stage_c`へoptional cache factoryを追加し、exp264既存挙動を維持したまま外部candidate cacheを注入可能にした。Stage Cには既定trueのhard readout opt-outを追加し、exp277ではfalse固定でhard top1を計算・出力しない。
- `src/pf_ancc_selector_audit.py`へ次を実装。
  - exp271 gzip raw/decompressed/schema/coverage guard。
  - exp263 fold bundleへのID/well/row fail-closed joinと既存`pf_ancc` slot置換。
  - 12/12/13候補の3 variant contractとdisagreement feature block。
  - variant単位のStage A + nested 40-selector実行。
  - exp264 fixed-control OOFと比較する15-model downstream実行。
  - 3 variantの0-booster aggregateと4-seed縮約readout。
- canonical Jupytext train notebookはcost、input、variant、execution、metrics、importance、SHAをセル展開した。
- inference notebookはdisabled guardで停止する。

## コマンドログ

```bash
make new-steering EXP=exp276_pf_ancc_small_seed_mean_addonly_selector_audit
make new-exp EXP=exp276_pf_ancc_small_seed_mean_addonly_selector_audit SOURCE=experiments/exp264_exp263_candidate_confidence_dual_selector
.venv/bin/python -m py_compile src/candidate_selector_pipeline.py src/pf_ancc_selector_audit.py experiments/exp277_pf_ancc_small_seed_mean_addonly_selector_audit/*.py
.venv/bin/ruff check src/candidate_selector_pipeline.py src/pf_ancc_selector_audit.py experiments/exp277_pf_ancc_small_seed_mean_addonly_selector_audit/*.py experiments/exp277_pf_ancc_small_seed_mean_addonly_selector_audit/tests/test_exp277_pf_ancc_small_seed_mean_addonly_selector_audit.py --select F821,E9,F401,F841
.venv/bin/pytest -q experiments/exp277_pf_ancc_small_seed_mean_addonly_selector_audit/tests/test_exp277_pf_ancc_small_seed_mean_addonly_selector_audit.py experiments/exp264_exp263_candidate_confidence_dual_selector/tests/test_exp264_candidate_selector_pipeline.py
```

- py_compile: PASS。
- Ruff selected checks: PASS。
- targeted unit/parent regression: 22 tests PASS。
- 初回notebook/full trainはローカル実行していない。

作業中に別実験が`exp276_exp264_compact_tail_risk_target_free_gate_audit`として追加されたため、
番号衝突を避けて本実験を`exp277`へ付け替えた。既存exp276の内容は変更していない。

## 2026-07-18 最終静的検証とKaggle package

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <inference.py>
make validate-exp EXP=exp277_pf_ancc_small_seed_mean_addonly_selector_audit STRICT=1
make validate-template
make test
make prepare-kaggle-notebooks EXP=exp277_pf_ancc_small_seed_mean_addonly_selector_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp277-pf-ancc-mean-selector-audit-train --title 'exp277 pf ancc mean selector audit train' --strict"
```

- Jupytext train/inference convert/test: PASS。
- py_compile / full Ruff: PASS。
- strict experiment validation: PASS。
- template validation: PASS。
- full repository test: 147 tests PASS。
- notebook sourceの`__file__`依存: 0。
- 親exp264 trainは447行/7章、exp277 trainは407行/7章。exp277はruntime、cost、input、
  PF contract、nested/downstream orchestration、metrics/importance、SHAをセル上に展開し、重いfold学習を
  `src/pf_ancc_selector_audit.py` / `src/candidate_selector_pipeline.py`へ残した。
- package kernel: `kentookumura/exp277-pf-ancc-mean-selector-audit-train`、private、CPU、internet off、
  run-on-push false。loose/packageのconfig、PF helper、selector pipeline SHAは一致した。
- package configは`design_only`、`mean4_only`、`run_approved=false`、40 nested / 15 downstream
  boosters per run、control retraining false。Kaggle pushは行っていない。
- `runtime.kaggle.stage_kernel_sources`へstage別入力を固定した。nestedはexp263+exp271、downstreamは
  exp072+exp145+exp264 fixed control+対応nested kernel、aggregateは3 downstream kernelsだけを使う。
  push前に`kernel_sources`を対象stageのallowlistへ合わせ、不要な巨大outputをmountしない。

## 2026-07-18 candidate置換契約の修正

初回実装は既存`pf_ancc`を残してmean pathを末尾追加する13/13/14候補としていたが、ユーザー確認により
意図は既存`pf_ancc` slotの差し替えであると確定した。single variantを12候補、both variantを13候補へ
修正し、candidate order、両legal domain、fold bundle values/availability/confidenceから旧`pf_ancc`を
除外する回帰guardを追加した。`add-only`はdownstream compact追加を指し、candidate追加を意味しない。

## 次

1. `nested_selector_mean4_only`を最初の候補とする場合も、40 CPU boostersの明示承認前にpushしない。
2. nested score/leakage guard通過前にdownstream、aggregate、inference、submissionへ進まない。

## 2026-07-18 mean4 nested selector実行承認

ユーザーの「実行してください」を、直前に提示した`nested_selector_mean4_only`の明示承認として記録する。

- active variant: 1 (`mean4_only`)
- selector objective: 2 (`pred_abs_error`, `p_within10`)
- folds: outer 5 × inner 4
- total boosters: 40 CPU
- parent/control再学習: 0
- downstream GPU booster: 0
- inference / submission: なし
- Kaggle inputs: exp263 candidate cache + exp271 fixed PF mean pathのみ

configを`stage=nested_selector_mean4_only`、`active_variant=mean4_only`、`run_approved=true`へ変更し、
kernel sourcesをnested stage用のexp263 + exp271だけへ絞る。canonical kernelは
`kentookumura/exp277-pf-ancc-mean4-selector`、private、CPU、internet offとする。

### Push

- version: 1
- id_no: `127737879`
- pushed kernel: `kentookumura/exp277-pf-ancc-mean4-selector`
- push result: success / run-on-pushで実行開始
- metadata pull: success。同じslug、private、CPU、internet off、inputはexp263 + exp271のみ。
- 初回CLI logs: 空。実行中は空になり得るため、失敗・再pushとは判定しない。
- push約1分後のstatus: `KernelWorkerStatus.RUNNING`。
- ユーザー指示によりCodex側の継続監視を停止。Kaggle実行は停止せず、完了連絡後に同じversion 1を監査する。

## 2026-07-18 version 1完了監査・output quarantine

ユーザーの完了連絡後、同じkernel version 1を監査した。

- Kaggle status: `KernelWorkerStatus.COMPLETE`
- stage完了log時刻: 6,984.277秒
- 実行量: `mean4_only` 1 variant / 2 objectives / outer 5 × inner 4 / 40 CPU boosters
- generated artifact files: 85
- output取得先: `kaggle/output/nested_mean4_v1/`（約3.3GB、Git管理外）
- 40 model SHA: 全一致
- 25 compact partition SHA: 全一致
- compact: 18,919,945 rows / 74 features
- outer-valid candidate-long: 45,407,868 rows
- candidate contract: 12候補、旧`pf_ancc`なし、slot 4は`pf_ancc_seed_mean_4`
- hard readout / inference / submission: なし

notebook内ではexpected-error MAE、within10 logloss、within10 Brierがpriorに対してpooledかつ5/5 foldsで
改善しscore guard PASSとreportされた。ただし監査時点で親exp264のfeature availability leakageが確定済みであり、
出力`feature_schema.json`にも以下のtraining-only formation raw/delta 12特徴が含まれることを直接確認した。

- raw: `ctx__raw__ancc`, `astnu`, `astnl`, `egfdu`, `egfdl`, `buda`
- delta: `ctx__raw_delta_last__ancc`, `astnu`, `astnl`, `egfdu`, `egfdl`, `buda`

hidden testでは元の6 formation列が利用できないため、内部well分離guardがPASSしても有効なselector OOFにはならない。
score、compact、importance、modelをすべてquarantineし、mean8/both/downstream/aggregateへ進めない。
local `run_approved=false`を維持し、raw-test-only selector schemaをStage Aから再構築するまでBLOCKEDとする。

## 2026-07-19 修正版親完了監査・exp277 port

ユーザーからraw-test-only selector schemaの再構築完了連絡を受け、修正版exp264を監査した。

- exp264 status: corrected Stage A v4 / Stage B v5 / Stage C v6 / Stage D v3 / inference v4完了。
- selector schema: 88特徴。file SHA `b91ec151...f1035`、logical SHA `aaef4ffd...ddd3a4`一致。
- raw context: `MD/X/Y/Z/GR`のみ。train 773/773・current-test 3/3 filesでavailability PASS。
- formation raw/delta: 0件。`id__candidate__pf_ancc`は候補IDでありformation raw contextではない。
- Stage C: 40/40 model SHA一致。model manifest `3f28b04a...2422d2`、compact manifest
  `f4855726...aecf1c`一致。
- downstream: source 380から非fold-safe 107列を除いたclean 273。allowlist SHA
  `d01a73cc...677bf`。
- fixed control: corrected Stage D v3 `matched_control` lgb_mean RMSE `10.476169179272501`、
  OOF SHA `7367983f...6dafee`。

exp277を次の契約へ更新した。

- configを`design_only` / `run_approved=false`へ戻し、旧version 1の承認を流用しない。
- nested run前にactual train/current-test header availabilityを検証する。
- formation 6列のraw/delta 12特徴を生成schema上でもfail-closed拒否する。
- downstreamを旧380列からclean 273へ変更し、compact 74との最終347列をassertする。
- fixed controlを修正版Stage D v3 OOF SHA / RMSEへ差し替える。
- clean 273 allowlistをbootstrap dependencyへ追加する。
- notebook source / ipynbを更新し、旧version 1 outputはquarantineしたまま維持する。

次の実行候補はcorrected `nested_selector_mean4_only` 1 variant / 2 objectives / outer 5 × inner 4 =
40 CPU boosters。control再学習0、downstream GPU 0。新たな明示承認なしに実行stageへ切り替えず、
run-on-push packageの再生成、Kaggle push、runは行わない。

### corrected design-only package

- kernel id/title: `kentookumura/exp277-pf-ancc-mean4-selector` / `exp277 pf ancc mean4 selector`
- private / CPU / internet off / `run_on_push=false`
- competition source: `rogii-wellbore-geology-prediction`
- kernel sources: exp263 Stage 0 cache + exp271 fixed PF mean pathのみ
- package config: `design_only` / `mean4_only` / `run_approved=false`
- loose/package config、PF helper、selector pipeline: byte一致
- config SHA: `a9074a8b...086`
- PF helper SHA: `d7cb10e8...a4af`
- selector pipeline SHA: `382ed81c...05c4`
- notebook SHA: `8cddd8ee...b9be`
- metadata SHA: `1661978b...2c0e`

これはlocal package更新だけで、Kaggle pushと実行は行っていない。

### validation

- actual raw header audit: train 773/773、current-test 3/3で5列すべてPASS
- targeted exp277 + exp264 regression: 26 tests PASS
- full repository: 195 tests PASS
- py_compile / Ruff F821,E9,F401,F841: PASS
- Jupytext ipynb conversion/test: PASS
- strict experiment validation: PASS
- template validation: PASS

## 2026-07-19 corrected mean4 nested selector再実行承認

ユーザーの「corrected mean4 selectorの40 CPU boosters」「実行してください」を、修正版exp264
raw-test-only 88特徴を親とする`nested_selector_mean4_only`の明示承認として記録する。今回の明示指示を
優先し、`exp276_corrected_exp264_parent_revalidation`はこのrunの前には挟まない。

- active variant: 1 (`mean4_only`)
- selector objectives: 2 (`pred_abs_error`, `p_within10`)
- folds: outer 5 × inner 4
- total model trainings: 1 × 2 × 5 × 4 = 40 CPU boosters
- per-model upper bound: 1,200 boosting rounds、early stopping 80
- parent/control再学習: 0
- downstream config / fold / GPU booster: 0 / 0 / 0
- PF生成: 0（exp271 version 2の固定mean4 pathを入力利用）
- inference / submission: 0 / 0
- Kaggle inputs: exp263 candidate cache + exp271 fixed PF mean pathのみ
- 旧version 1 output: input利用禁止、quarantine維持

push前にconfigを`stage=nested_selector_mean4_only`、`active_variant=mean4_only`、
`run_approved=true`へ変更する。canonical kernelは
`kentookumura/exp277-pf-ancc-mean4-selector`、private、CPU、internet off、run-on-pushとする。

### corrected version 2 push

- package contract: mean4 1 variant × 2 objectives × outer 5 × inner 4 = 40 CPU models
- config SHA: `ba7b59d2c4f21c89c4fec3747c8d835a83e5f3b1db13acd591eb7c9b73a2fe59`
- PF helper SHA: `d7cb10e8d3f73446731afa53a7db540c7c3e9cc240d6bd169d6d90df1c9ca4af`
- selector pipeline SHA: `382ed81c5f6d3b9e76ea4eaf34af57b9ae1d6db3f0822c39dae5a736b96305c4`
- notebook SHA: `f12ff10b9ffb64fe99e7f7b4c327bddf95a5b4b15ed9d82c10d6cf656935231d`
- metadata SHA: `fb9ccc1d95bf9914aca18e660e313980ffee4644f13ec10aaf66526e5273ff20`
- loose/package config、PF helper、selector pipeline: byte一致
- kernel: `kentookumura/exp277-pf-ancc-mean4-selector` version 2 / id_no `127737879`
- push result: success / run-on-push
- post-push metadata: same slug、private、CPU、internet off、inputsはexp263 + exp271のみ
- initial status: `KernelWorkerStatus.RUNNING`
- initial logs: 空（実行開始直後）
- local `run_approved`: push消費後の重複実行防止としてfalseへ戻した
- monitoring: 継続監視なし。完了後に同じversion 2を監査する

## 2026-07-19 corrected mean4 version 2完了監査

ユーザーの完了連絡後、canonical version 2を監査した。Kaggle statusは
`KernelWorkerStatus.COMPLETE`、stage完了log時刻は5,707.598秒、notebook最終処理は約5,718.663秒。
cost contractは`mean4_only` 1 variant / 2 objectives / outer 5 × inner 4 / 40 CPU models、
control再学習0のまま完走し、86生成物を出力した。

### schema / availability / candidate contract

- Stage A audit: 600,000 rows、143生成特徴からall-missing 34、constant 4、exact duplicate 17を落とし88特徴。
- feature schema logical SHA: `0e92f643c8b1f078332123160a4827b671accaa50907bf53e7b79ade57328e36`
- feature schema file SHA: `9b753a53825e09a9605680a01711a5e5fbed21c7aa8624df06a6771915b04cee`
- raw context: `MD/X/Y/Z/GR`。train 773/773・current-test 3/3 filesで全列availability PASS。
- formation raw/delta: `ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`由来hit 0。
- corrected exp264 88特徴との差は`id__candidate__pf_ancc`を
  `id__candidate__pf_ancc_seed_mean_4`へ置換した1列だけ。
- candidate orderは12候補、旧`pf_ancc`なし、0-based slot 4が`pf_ancc_seed_mean_4`。
- feature importanceにもformation raw/delta hit 0。

### nested selector / integrity

- model: 40 / expected 40。5 outer × 4 inner × 2 objectivesの40 unique combinations。
- model実体40本、合計30,797,334 bytesを選択取得し、manifest SHAと40/40一致。
- best iteration: 83〜224。上限1,200、early stopping 80の範囲内。
- model manifest SHA: `6cf60fa8c5ee2ed6f32c758f9e30a9575ac0353d8c2678daef9a6f7c8b0563d9`
- compact manifest SHA: `50cf8e0d5f1f02ef24b15eecdd0b5e82c25d3cfb251b0e9559c95460f9b1298e`
- outer-valid candidate score SHA: `e37bb872f1f3709ccdef91927e1fd57b3875c8a9e01f299dbf507be9f9dc8b3e`
- compact: 25 unique partition keys / 18,919,945 rows。各downstream outer foldは3,783,989 rows、
  train 4 partitions × selector model count 1、valid 1 partition × model count 4。
- outer-valid candidate-long: 45,407,868 rows。
- outer-valid wellのinner assignment除外、inner train/valid well disjoint、train=`inner_oof`、
  valid=`four_inner_model_ensemble`を確認し、leakage audit PASS。
- compact parquet約3GB本体は取得せず、manifest file SHA、partition key/row/role/model-countを監査した。

選択取得した主要file SHA:

- `nested_selector_metrics.json`: `2efe03f40f02da929fab4d0d7c26c8f3503de03d6146f4b2e5b8689bd70be216`
- `exp277_nested_summary.json`: `44e103aee01daee115a2ffd15e7b0982f79f829a66b238a641839d4476c5438a`
- `stage_a_summary.json`: `0b8d1e7f527e3c78d57262548a8659c72c17ffd9dddf3849d2c0426abbb430cc`
- `raw_context_availability_audit.csv`: `cead0e03624f5b9401c8bca28febe1965dd830ac90bd50b73ef88870d56ae1c3`
- `exp277_nested_reproducibility_manifest.json`: `04d8aa3850c3dd8a01950d3028f758f765be3736758e5fea003d9ac473e6a0c2`
- `reproducibility_manifest.json`: `be1f15f6a8bcdc39cde5da4a72093a51f918e024c3dfdd72c5c3d2afedfbbd38`

### selector score

| metric | mean4 selector | outer-train prior | guard | corrected exp264 original `pf_ancc` | delta vs exp264 | improved folds vs exp264 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| expected-error MAE | 3.793764044 | 5.708749073 | pooled + 5/5 PASS | 3.798819493 | -0.005055449 | 4/5 |
| within10 logloss | 0.360023958 | 0.509003394 | pooled + 5/5 PASS | 0.359411822 | +0.000612135 | 2/5 |
| within10 Brier | 0.112271517 | 0.164559781 | pooled + 5/5 PASS | 0.111830419 | +0.000441097 | 2/5 |

事前定義したselector score guardとraw-test/leakage/integrity guardはすべてPASSし、version 2 compactを
有効なdownstream入力候補として採用する。一方、original `pf_ancc`比はobjective間でmixedであり、
mean4の一様優位やdownstream TVT改善を主張しない。hard top1、downstream、PF再生成、inference、
submissionは未実行。

次に`downstream_mean4_only`を行う場合は、保存済みversion 2 compactを入力とし、1 variant ×
3 LightGBM configs × 5 folds = 15 GPU boosters、control再学習0を別途明示承認する。

### 完了後validation

- targeted exp277 + corrected exp264 regression: 26 tests PASS
- full repository: 227 tests PASS
- strict experiment validation: PASS
- template validation: PASS
