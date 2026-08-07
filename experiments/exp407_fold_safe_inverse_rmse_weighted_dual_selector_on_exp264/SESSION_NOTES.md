# exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264 セッションノート

## 目的

exp264 corrected Stage B v5の12候補共有dual selectorに対し、候補を削除せず、
各modelのfit partition内で計算した候補別TVT RMSEの逆数に比例するtask weightだけを
追加し、弱いcandidate taskによるnegative transferを検証する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle Stage B v1完了・technical PASS / scientific FAIL・no-rescue閉鎖
- CV: hard-primary OOF RMSE `8.66814102464331`
- LB: まだなし
- train notebook: compact self-contained候補を正規Notebookへ採用済み
- inference notebook: template placeholderのみ
- package: 実行時packageはrun-on-push付きで生成済み。Kaggle v1は`COMPLETE`。
  Stage C/D、推論、提出: scientific FAILにより閉鎖・未実施

## 2026-07-26 設計確定

- 親は`exp264_exp263_candidate_confidence_dual_selector`。
- 12候補、候補順、88列raw-test-safe schema、2 legal domain、2 objectives、
  5 folds、sampling、LightGBM paramsを固定する。
- 変更はcandidate-long training sample weightだけ。
- Stage Bは各outer foldのdeterministic sampled outer-train fit rowsだけで候補別RMSEを計算する。
- 将来Stage Cへ進む場合は各outer × inner modelのsampled inner-train fit rowsだけで再計算する。
- weightはinverse RMSE、mean-one normalize、`[0.5, 1.5]` clip、再normalizeで固定した。
- 同じweightを`pred_abs_error`と`p_within10`のtraining datasetへ適用する。
- validation/early stopping/OOF metricsはunweightedとする。
- global OOF weight、inverse-square、clip/exponent grid、候補別手調整、
  objective別weight、Beam削除、candidate subsetを禁止した。
- 初回Stage Bは新規variant 1 × objective 2 × outer fold 5 = 10 CPU boosters。
- 親/control再学習0、PF/HMM/Beam再生成0、GPU booster 0。
- Stage B実装・実行、Stage C、Stage D、inference、submissionはすべて別承認とした。

## 2026-07-26 Stage B implementation-only

ユーザーの「exp407を実装してください」をStage Bのimplementation-only承認として扱った。
Kaggle実行、正規Notebook採用、Stage C/D、inference、submissionは承認範囲外とした。

- 親と同一SHA
  `4f4d3f77db01d7477f9e73066ac311cfdc2c14b15eba84fab9830f4cf5486c20`
  の`candidate_contract.yaml`を実験内に固定した。
- 8章・親trainと同じ465行のJupytext compact self-contained train候補を別名で作成した。
- `src/candidate_task_weighting.py`へ、fit labelsだけを入力にする候補別RMSE、
  inverse、mean-one normalize、clip、再normalize、final range fail-closedを実装した。
- `src/candidate_selector_pipeline.py::run_stage_b`へoptional hookを追加した。
  exp407だけが明示configを渡し、既存callerはdefault unweightedのままとした。
- 同じsample weight SHAをfold内の2 objectivesへ渡す一方、eval setへweightを渡さない。
- fold別weight table、sampling manifest、truth-read ledger、fit row ID /
  candidate error / feature content / model / OOF SHAを保存する。
- 保存済みparent v5とのfold、near、1000+、hidden-like 2面、by-well比較と
  technical/scientific全AND gateを`src/exp407_inverse_rmse_selector.py`へ実装した。
- `execution.run_approved=false`、Stage B enabled false、run-on-push falseを維持した。

## 参考値（fitには使用禁止）

全OOFから計算したillustrative mean-one inverse-RMSE weightは、`beam_mean`が約0.6555、
`pf_ancc`が約0.7134、fixed formulaが約1.2551である。Beamを0にせず穏やかに下げる尺度で、
clip範囲内に収まる。実際の学習weightは必ずfold別fit partitionから再計算する。

## 実行予算（push前に再確認必須）

| 段階 | variant | config/objective | folds | 合計booster | control再学習 | 状態 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Stage B | 1 | 2 objectives | 5 outer | 10 CPU | 0 | 2026-07-26実行承認済み |
| Stage C | 1 | 2 objectives | 5 outer × 4 inner | 40 CPU | 0 | 条件付き・未承認 |
| Stage D | 1 | 3 configs | 5 | 15 GPU | 0 | 条件付き・未承認 |

Stage B push前には上表を再確認し、`execution` flagをユーザー承認なしに開かない。

## 2026-07-26 Stage B実行承認

ユーザーの「実行してください」を、直前に提示した正規Notebook採用とStage B Kaggle実行の
明示承認として受領した。承認範囲は新規variant 1、dual objectives 2、outer folds 5、
合計10 CPU boosters、親control再学習0、PF/HMM/Beam再生成0、GPU 0である。
Stage C、Stage D、inference、submissionおよび技術エラー時の再pushは承認範囲外のままとする。

- 承認受領: `2026-07-26 11:14:46 JST`
- execution scope:
  `inverse_rmse_weighted_stage_b_1_variant_2_objectives_5_outer_10_cpu_boosters_no_control_retraining`
- credential: Kaggle CLI OAuth、legacy credentialとも利用可能。credential実値は記録しない。
- 固定bootstrap dependency 5ファイルはすべて存在し、configに固定したSHAと一致した。
- 正規train notebookをJupytext sourceから生成し、18セルのcompact構成を採用した。
- canonical kernel id/titleは
  `kentookumura/exp407-inverse-rmse-dual-selector-exp264-train` /
  `exp407 inverse rmse dual selector exp264 train`でslug一致。
- Kaggle metadataはprivate、CPU、internet off、run-on-push true、competition input 1、
  保存済みexp263 kernel source 1で生成した。
- package config SHA:
  `0d7252823acb5d97cb5cb8782fb174f11d5debfdbc037616755143085b4493d9`
- push前の同一kernel pullは`GetKernel` 403だったが、canonical idへの初回pushは成功し、
  Kaggle kernel version 1が作成された。403だけを理由にslug変更や再pushはしていない。
- kernel:
  `kentookumura/exp407-inverse-rmse-dual-selector-exp264-train` v1
- URL:
  <https://www.kaggle.com/code/kentookumura/exp407-inverse-rmse-dual-selector-exp264-train>
- `2026-07-26 11:25:15 JST`、ユーザーの「監視は止めていいです。完了したら連絡します。」
  に従ってpollingを停止した。最終確認statusは`KernelWorkerStatus.RUNNING`。
  再push、output取得、Stage C/D、inference、submissionは行っていない。

## 2026-07-26 Stage B v1完了・科学ゲートFAIL

ユーザーから完了連絡を受け、同じcanonical kernel version 1のstatus、logs、
必要な小容量artifactだけを確認した。kernelは`KernelWorkerStatus.COMPLETE`、
実行時間は`1,531.430秒`だった。再push、retry、Stage C/D、inference、
submissionは行っていない。

- technical gate: PASS
  - base rows `3,783,989`、candidate-long rows `45,407,868`
  - 12 candidates、88 features、10 models / model SHA 10-of-10
  - fold別weight平均1、全fold min `0.6468215060740257`、
    max `1.2626658474376724`
  - fit-valid well overlap 0、forbidden truth read 0
  - 両objectiveで同一fold weight SHA、validation weightなし、metricはunweighted
- scientific全AND gate: FAIL
  - expected-error MAE `3.7986703583181702`、親比`+0.002869195694088`、
    改善2/5 folds
  - within10 logloss `0.3604606479455683`、親比`+0.000489053142799`、
    nonworse 1/5 folds
  - within10 Brier `0.11264822387628641`、親比`+0.000197244656241`、
    nonworse 2/5 folds
  - hard-primary OOF RMSE `8.66814102464331`、親比`+0.081136637939888`、
    nonworse 1/5 folds
  - near 0--250親差`+0.005087332884175622`はPASS
  - 1000+ `+0.0912276539147765`、hidden-like spatial
    `+0.1037590194668816`、typewell-purged `+0.07905226513054942`はFAIL
  - worst well `52f1e77a`は親比`+16.226862522494358`でFAIL
- decision: `fail_close_exp407_without_rescue`
- inverse-square、weight強度、clip/exponent grid、candidate subset / Beam削除、
  同じOOFを用いた救済は行わない。
- exp264 corrected Stage B v5をselector anchorとして維持する。
- 保存済み親/exp407 OOFだけを使う0-booster candidate-switch tail attributionを
  低優先度の原因診断としてbacklogへ追加する。これはexp407を再開しない。

小容量artifactは
`kaggle/output/train_v1_small/artifacts/`へ保存した。約930 MBの
candidate-score OOF parquetはダウンロードせず、runtime manifestに記録された
content SHAを確認した。

## コマンドログ

### 2026-07-26 実行済み

```bash
make new-steering EXP=exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264
make new-exp EXP=exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264 SOURCE=templates/experiment
make validate-exp EXP=exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264 --root .
```

`make validate-exp`はstrict PASS。文書reviewは全体としてcore evidence categoriesを確認した。
YAML/JSON parse、route、implementation disabled、候補12本、Stage B 10 boosters、
control再学習0、notebook 6-cell placeholderの静的assertもPASSした。

### implementation-onlyで実行

```bash
.venv/bin/python -m py_compile \
  src/candidate_task_weighting.py \
  src/exp407_inverse_rmse_selector.py \
  src/candidate_selector_pipeline.py \
  experiments/exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264/exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264_compact_selfcontained_train.py
.venv/bin/ruff check \
  src/candidate_task_weighting.py \
  src/exp407_inverse_rmse_selector.py \
  src/candidate_selector_pipeline.py \
  experiments/exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264/exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264_compact_selfcontained_train.py \
  tests/test_exp407_inverse_rmse_weighted_dual_selector.py \
  --select F821,F811,F401,E501
.venv/bin/pytest -q tests/test_exp407_inverse_rmse_weighted_dual_selector.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264/exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264_compact_selfcontained_train.py
```

専用testは9件PASS、親`candidate_selector_pipeline`との関連回帰を含む26件もPASSした。
py_compile、Ruff F821/F811/F401/E501、Jupytext変換/round-trip、strict
`validate-exp`、実験文書reviewもPASSした。

`make test`は追加gate test前の全1,170件中、exp407 8件、exp264共有selector 17件、
fixed13 selector群を含む1,159件PASS・7件skipだった。残る4件は今回変更していない
exp293の既存contract file SHA drift 2件と、exp296の完了後status/approval flagに対する
stale assertion 2件で失敗した。exp407または共有selector hook由来の失敗は0。

### 承認後の実行予定として記載したコマンド（v1で実行済み）

```bash
# 正規Notebook採用とStage B実行を別承認された後だけ使用する。
task prepare-kaggle-notebooks EXP=exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264 \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp407-inverse-rmse-dual-selector-exp264-train --title 'exp407 inverse rmse dual selector exp264 train' --strict"
task push-kaggle-train EXP=exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264
```

### 2026-07-26 Stage B v1 push

```bash
make prepare-kaggle-notebooks EXP=exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264 \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp407-inverse-rmse-dual-selector-exp264-train --title 'exp407 inverse rmse dual selector exp264 train' --run-on-push --strict"
kaggle kernels pull kentookumura/exp407-inverse-rmse-dual-selector-exp264-train \
  -p /tmp/kaggle-pull/exp407-inverse-rmse-dual-selector-exp264-train -m
kaggle kernels push \
  -p experiments/exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264/kaggle/train
```

- prepare: PASS
- canonical pull before first push: `GetKernel` 403
- push: `Kernel version 1 successfully pushed`
- runtime: Kaggle CPU、internet off、private
- 実行結果: `KernelWorkerStatus.COMPLETE`、`1,531.430秒`

### 2026-07-26 完了監査

```bash
kaggle kernels status kentookumura/exp407-inverse-rmse-dual-selector-exp264-train
kaggle kernels logs kentookumura/exp407-inverse-rmse-dual-selector-exp264-train
kaggle kernels files kentookumura/exp407-inverse-rmse-dual-selector-exp264-train
kaggle kernels output kentookumura/exp407-inverse-rmse-dual-selector-exp264-train \
  -p experiments/exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264/kaggle/output/train_v1_small \
  --file-pattern '<small-artifact>'
```

- log SHA:
  `8b4a0fca872bfa88fb056cac2b621195478dfced587031733a906f05283cd2da`
- gate SHA:
  `2ae8cb3eaafd3f11558035f27c4afdf0be70468e8edacb0c8703c6b949a99962`
- scientific decisionとlocal `metrics.json` / `result.md`を照合した。

## 再現性メモ

- seed policy: 親のseed 42とdeterministic sampled row IDsを継承
- stochastic components: LightGBM subsample / colsampleのみ
- CPU/GPU runtime: Stage B/C CPU、Stage Dは条件付きGPU
- PF/HMM/Beam: 保存済みcandidate値をload-only、新規生成0
- Kaggle kernel id / version:
  `kentookumura/exp407-inverse-rmse-dual-selector-exp264-train` / v1
- input / feature schema SHA: 親logical schema
  `aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`
- parent Stage B candidate-score OOF SHA:
  `9a91b62599278d4e56d57074df4725d4a09391460458b8eccc02dd50af34d48a`
- parent Stage B model manifest SHA:
  `d5159ed142a055865ddd94a66d89e8242cb720b28dddf175d313ee32125ca07d`
- fold weight table SHA:
  `ecf3e93b161e2a173ed3cadbf69cc369d367f38d939d8463be1624e4c851922b`
- fold weight logical SHA:
  `072613c59535e2f2968223f30b7563a314ee1e9b6725dd7829cb5cbec733997e`
- weight manifest SHA:
  `7234351edcebbd852ea8d4b771258e676bbb5295939105226fb61c92033be067`
- model manifest SHA:
  `1fce3716fc7f545e0ea883e8ee71b05174d141212334f7c01913b32ef38adfd4`
- candidate-score OOF content SHA:
  `d993b806d92c2462c1509f110669b272b27d48806c0280a2cf54e87c7f32f1e8`
- compact-meta OOF SHA:
  `a88503f506985ae1b25391234abc753e39bd1d81b52b98e827486fa6102b9672`
- prediction / submission SHA: 対象外
- deterministic anchor: false。同一package rerun parity未確認

## 次のアクション

1. exp407はno-rescueで閉鎖し、Stage C/D、inference、submissionへ進めない。
2. selector anchorはexp264 corrected Stage B v5のままとする。
3. 独立した必要性とユーザー承認がある場合だけ、保存済みOOFによる
   candidate-switch tail attributionを0-booster原因診断として検討する。
