# exp388_exp374_fixed13_dual_selector_on_exp264 セッションノート

## 目的

exp374の固定`df=4` Student-t absolute-TVT exact-HMM train予測を、
corrected exp264 fixed12 candidate-long dual selectorへ13本目として追加する。
候補単独の平均改善をselectorが安全に利用できるかを評価し、exp374のtail gate
失敗自体はPASSへ再分類しない。

## 事前固定設計

- Route: `ensemble`
- selector parent: `exp264_exp263_candidate_confidence_dual_selector`
- candidate parent: `exp374_exp209_student_t_exact_hmm_emission`
- 追加候補: `student_t_exact_hmm`
- fixed fallback: 既存7候補のまま
- selector fold: exp263 outer 5 fold
- active variant: `1`
- LightGBM objectives: `2`
- outer / inner folds: `5 / 4`
- planned CPU selector boosters: `40`
- parent/control retraining: `0`
- GPU boosters: `0`
- downstream TVT / inference / submission: `0 / 0 / 0`

ユーザーの「selectorに含めることに進みたいです」を、上記Stage A + Stage C
CPU実行までの承認として記録した。

## exp374入力契約

- kernel: `kentookumura/exp374-exp209-student-t-exact-hmm-emission-train`
- version / id_no: `1 / 128436182`
- file:
  `artifacts/exp374_exp209_student_t_exact_hmm_emission_predictions.csv.gz`
- rows / wells: `3,783,989 / 773`
- raw gzip SHA:
  `ea6f95334b2d75ab8c96f705c453f66b856397fd222770cd15f3ae0b7fef221e`
- decompressed content SHA:
  `668fe87da902955acee742c72d30724abb53f32050bb5d0a5c1b3dee0cbd626e`
- allowlist:
  `id,well_id,row_idx,student_t_*_hmm_tvt,student_t_*_hmm_std,
  student_t_*_hmm_loglik`

exp374はwell間学習を持たず、各wellのknown prefixだけで生成したtarget-free候補。
source foldは存在しないため合成せず、global `(well_id,row_idx)` join後に
exp263 selector foldへpartitionする。truth、error、Gaussian/LikPF control、
scope、gate診断はloaderで開かない。

## 科学gate

- selector score guardをPASS
- added candidate primary top1 fraction `>= 0.005`
- positive usage folds `>= 4`
- parent fixed12 hard selectorに対しpooled非悪化
- parent fixed12に対し`4/5` folds改善
- near 0-250 / 1000+ / hidden-like delta `<= 0.02 ft`
- by-well p95 / worst delta `<= 0.25 ft`
- same-OOF rescue、weight、threshold調整は禁止

## 実装

- steering:
  `docs/legacy/steering/20260724-exp388-exp374-fixed13-dual-selector-on-exp264/`
- experiment:
  `experiments/exp388_exp374_fixed13_dual_selector_on_exp264/`
- reusable loader/cache:
  `src/exp374_fixed13_candidate_cache.py`
- dedicated tests:
  `experiments/exp388_exp374_fixed13_dual_selector_on_exp264/tests/test_exp388_exp374_student_t_fixed13_dual_selector.py`
- Jupytext source:
  `exp388_exp374_fixed13_dual_selector_on_exp264_compact_selfcontained_train.py`

exp375の10章相当のfixed13 selector構成を参照し、candidate sourceだけを
exp374の実列契約へ置き換えた。Notebook内で入力/SHA、cost contract、
global key join、Stage A、Stage C、paired readout、novelty、生成物を追跡する。
同一exp helper importと`__file__`は使用しない。

## 確認済みコマンド

```bash
kaggle kernels output \
  kentookumura/exp374-exp209-student-t-exact-hmm-emission-train \
  -p /tmp/exp374-selector-source \
  --file-pattern 'artifacts/exp374_exp209_student_t_exact_hmm_emission_predictions.csv.gz'

.venv/bin/python -m py_compile \
  src/exp374_fixed13_candidate_cache.py \
  experiments/exp388_exp374_fixed13_dual_selector_on_exp264/exp388_exp374_fixed13_dual_selector_on_exp264_compact_selfcontained_train.py

.venv/bin/pytest -q \
  experiments/exp388_exp374_fixed13_dual_selector_on_exp264/tests/test_exp388_exp374_student_t_fixed13_dual_selector.py
```

- dedicated tests: `10 passed`
- Kaggle credential: OAuth / legacy CLI credential利用可能
- API tokenは未設定だがKaggle CLI操作には影響しない

## 現在状態

`completed_fixed13_selector_scientific_gate_failed_closed`

## Kaggle push

- Jupytext `--test`: train / inference PASS
- ruff `F821,F401,F841,E501`: PASS
- strict experiment validation: PASS
- dedicated + common selector tests: `27 passed`
- exp375構成参照比較: train sourceは双方`540`行、8つの役割章を維持
- package configと正本config: byte一致
- metadata: GPU off / internet off / run-on-push
- kernel sources: exp263 cache / exp374 prediction / exp264 parent score

最初のid/title
`kentookumura/exp388-exp374-fixed13-dual-selector-on-exp264-train` /
`exp388 exp374 fixed13 dual selector on exp264 train`は双方同じ51文字slugだったが、
Kaggle SaveKernel APIの詳細なし400で拒否され、runは開始しなかった。
既知のexp373復旧例と同じ長さ制限と判断し、科学条件を変えず46文字の
`kentookumura/exp388-exp374-student-t-fixed13-selector-train` /
`exp388 exp374 student t fixed13 selector train`へ揃えてpackageを再生成した。

短縮slugのversion 1 pushに成功した。

- kernel id: `kentookumura/exp388-exp374-student-t-fixed13-selector-train`
- version / id_no: `1 / 128464582`
- status: `KernelWorkerStatus.RUNNING`
- approval: version 1で消費済み
- packaged config SHA:
  `5ffa9d504af511faccccdfe8ec81511e91cb984586811b2e3bf0bcfb78096959`
- packaged metadata SHA:
  `40ae0731695eb8646692e1356e26ff20c12bc98dba44cb93f9b0da7efd8075ac`
- packaged notebook SHA:
  `f6e495d5b5e21fdba7faf7f72152efbcb53b80cbb76ac1691ce2560ac6ecd725`

空logsやstatus一時障害を理由に再pushせず、同じversion 1を完了まで監視した。

## Kaggle version 1 結果

- status: `KernelWorkerStatus.COMPLETE`
- notebook scientific runtime: `7253.168438142 sec`
- selector models: `40 / 40`
- parent/control再学習 / GPU / downstream TVT / inference / submission:
  `0 / 0 / 0 / 0 / 0`
- technical checks / leakage audit: 全PASS
- Stage A: 650,000 audit rows、153 -> 90 features、compact 77
- Stage C: 25 partitions、18,919,945 compact rows、
  49,191,857 outer-valid candidate-score rows
- selector score guard: PASS
  - expected-error MAE: `5.844224569 -> 3.838679871`
  - within10 logloss: `0.509949108 -> 0.358495164`
  - within10 Brier: `0.164938162 -> 0.111596203`
  - pooledかつ各5/5 folds改善

## Fixed13 integration

- fixed13 hard RMSE: `8.736104109`
- parent fixed12 hard RMSE: `8.652531956`
- delta: `+0.083572154 ft`悪化
- fixed fallback RMSE: `8.238331546`
- fold delta:
  `-0.021005498 / +0.200900144 / +0.304599290 /
  -0.127122022 / +0.048977581 ft`
- improved folds: `2 / 5`
- Student-t top1: `692,647 / 3,783,989 rows = 0.183046779`
- positive usage folds: `5 / 5`
- near 0--250 delta: `+0.009049238 ft`（PASS）
- 1000+ delta: `+0.089723756 ft`（FAIL）
- hidden-like spatial / typewell-purged:
  `+0.088184377 / +0.091252200 ft`（FAIL）
- improved / regressed wells: `366 / 407`
- by-well median / p95: `+0.006057640 / +0.910123172 ft`
- worst `d2f3b1ab`: `+6.708956173 ft`

利用率とnearだけをPASSし、pooled、改善fold数、1000+、hidden-like 2面、
by-well p95、worst-wellをFAILした。decisionは
`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`。

## Post-freeze novelty

診断専用oracleでは補完性が残った。

- H512: `3.700319996 -> 3.603021059`、`0.097298936 ft`改善、
  unique-best `1,127 / 7,787 groups`
- whole-well: `4.801786361 -> 4.728378750`、`0.073407611 ft`改善、
  unique-best `112 / 773 wells`

候補の局所補完性はあるが、現行hard selectorは18.3%まで選びながらtailを
抑えられず、deployable gainへ変換できなかった。

## 再現性SHA

- exp374 decompressed:
  `668fe87da902955acee742c72d30724abb53f32050bb5d0a5c1b3dee0cbd626e`
- post-read prediction:
  `506842291a539c9c8dd4da1a0a1eb6a11bfb2c8466fb022f4410b9bfdc234854`
- feature schema:
  `66568a948768a8dd4953a404b6b88f8e7f58c5ecf6f5afe22ca8bcd0a4b881fe`
- model manifest:
  `9acdc4a165b69f737ee4807e953653070b0f3db0647514554744b7790d573044`
- compact manifest:
  `5e9a1242c73963f5947955a653604633d9034ab136ef9b6f56c4154d78e08968`
- outer-valid candidate score:
  `7ad8419f299419824447ad2b500ebdb353af4a39773ef5aefc702692ef36ecd8`
- summary:
  `bc5ce77913862c963d6a65ff4491a3f193405b8e6fdf83c9d648615b16b74b99`

## 終了判断

same-OOF上のcandidate weight、usage threshold、candidate除外、gate緩和、
downstream TVT、inference、submissionへ進めずbranchを閉じる。

再訪候補はStudent-t TVTのhard selectionではなく、Gaussian--Student-t
disagreement、posterior std、log-likelihoodをcontinuous risk featureとして
downstream MLへadd-onlyする独立仮説に限定し、低・P4とする。
