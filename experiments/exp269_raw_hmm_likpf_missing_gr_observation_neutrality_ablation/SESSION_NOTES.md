# exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation セッションノート

## 2026-07-17 実装

### 親実験の確定

- scientific parentは`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`のraw typewell-GR exact HMM。
- exp209 `metrics.json`でHMM generated/reference decompressed SHAがともに`8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`で、exp205 v2とのexact content parityが記録済み。
- exp223 self-GR HMMは親にしない。self-GRは補間window descriptorまで変更するため、本件のmissing observation neutralityだけを原因分離できない。
- exp247はmask境界と診断構成の参照に限定し、exp221 controlとexp148 LGB OOF unaryは除外した。

### 実装内容

- `.steering/20260717-exp269-raw-hmm-likpf-missing-gr-observation-neutrality-ablation/`で要件、1変更、再現性、Stage 1 guardを事前固定。
- raw GR欠損判定は補間前horizontal `GR`のnon-finiteだけから作成。
- exp209と同じGR emissionを構築した後、raw-missing evaluation rowの全stateを`0.0f`へ置換。
- notebookのHMM呼び出しにはself-GR/LGB unaryを渡さず、configにも`lgb_emission`を持たせない。
- 保存済みexp209/exp205 controlを読み、decompressed SHAをstrict検証。ID mismatchはguardへ渡してfail-closeする。
- overall、GR availability、missing-run、post-gap、distance、1000+、hidden-like 2群、focus well `11d0f5ac`、by-well、posterior std/loglik、finite coverage、divergence segmentを出力する。
- guardはthresholdだけでなく必要groupの欠落もfailureとして扱う。
- inference notebookは`inference.enabled=false`と`pf_stage.enabled=false`を検証して停止する。

### Kaggle push前コストガード

| 項目 | 実行値 |
|---|---:|
| active variant | 1 (`raw_hmm_missing_gr_neutral`) |
| LightGBM config | 0 |
| fold training | 0 |
| booster | 0 |
| parent/control再学習・再生成 | 0 |
| likelihood-PF Stage 2 | disabled |
| inference/submission | disabled |
| GPU | false |
| outer workers / Numba threads | 2 / 2 |

既存controlを再生成しないため、追加承認が必要なcontrol GPU再学習は含まれない。Stage 1 variantのKaggle CPU生成だけが次の実行対象。

### 再現性

- Stage 1は新規RNGなし。seed policyは`no_new_rng_raw_exact_hmm_missing_gr_neutrality_ablation`。
- sorted well順、outer workers 2、Numba threads 2、Python/NumPy/pandas/kernel versionをsummaryへ保存。
- raw train/test file SHA、control raw/decompressed SHA、prediction/row/group/by-well/finite/divergenceのSHAを記録。
- gzip間の主比較はdecompressed content SHAを使う。
- deterministic submission anchorではない。PF、current-test prediction、submissionは生成しない。

### 静的検証コマンド

```bash
.venv/bin/python -m py_compile experiments/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation_train.py experiments/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation_inference.py experiments/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation/exact_hmm_smoother.py experiments/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation/settings.py
.venv/bin/ruff check --select F821,F401,E9 experiments/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation
.venv/bin/pytest -q experiments/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation/test_observation_neutrality.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation_train.py
make validate-exp EXP=exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation
```

### 実行状態

- localでnotebook本体は実行していない。Kaggle Notebookを正とする。
- targeted testは6件pass。exp209とのHMM config、prefix statistics、Numba transition kernelの静的一致も確認した。
- Kaggle CPU train packageを`experiments/exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation/kaggle/train/`へstrict生成した。
- kernel IDは`kentookumura/exp269-raw-hmm-missing-gr-neutrality-train`。metadataはCPU、GPU false、internet false、run_on_push true、kernel sources exp209/exp205/exp115。
- package内configもactive variant 1、LightGBM/fold/booster 0、control再生成false、self-GR/LGB unary false、PF/inference falseと照合済み。
- Kaggle push/trainは未実行。
- Stage 1結果、guard、PF eligibilityは未確定。
- PF Stage 2はStage 1が通過しても別承認まで実装・実行しない。

## 2026-07-17 Kaggle Stage 1実行開始

- ユーザーの「実行してください」をKaggle CPU Stage 1 push/runの明示承認として記録した。
- 実行対象: active variant 1 (`raw_hmm_missing_gr_neutral`)、LightGBM config 0、fold 0、booster 0。
- 保存済みexp209/exp205 controlを読み、親/control再生成は行わない。
- metadata: CPU、GPU false、internet false、run_on_push true。
- PF Stage 2、inference、submissionは無効のまま。
- canonical kernel: `kentookumura/exp269-raw-hmm-missing-gr-neutrality-train`。
- push直前にsource/packageの`config.yaml` SHA一致、および`exact_hmm_smoother.py` SHA一致を確認した。
- `make push-kaggle-train EXP=exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation`でversion 1をpushした。
- Kaggle URL: `https://www.kaggle.com/code/kentookumura/exp269-raw-hmm-missing-gr-neutrality-train`。
- Kaggle metadata pull成功: `id_no=127592556`、private、CPU、GPU/internet false、kernel sources exp209/exp205/exp115。
- 初回statusは`KernelWorkerStatus.RUNNING`。通常logsは空だったが、実行中はCLI logsが空になる既知挙動のため再pushしない。

## 2026-07-18 Kaggle Stage 1完了

### 実行確認

- canonical kernel version 1が`KernelWorkerStatus.COMPLETE`になったことを確認した。
- Kaggle logsと、summary、metrics、group/by-well、finite coverage、generation-by-wellの小規模成果物だけを取得した。prediction/row audit archive全体はダウンロードしていない。
- 3,783,989 rows / 773 wellsを完走し、raw GR missingは1,200,837 rows / 773 wellsだった。
- runtimeは19,573.731秒（約5時間26分14秒）。Kaggle CPU、outer workers 2、Numba threads 2、Python 3.12.13、NumPy 2.0.2、pandas 2.3.3。
- control decompressed SHAは`8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`で、事前固定値と一致した。

### 数値結果

| 評価面 | control RMSE | neutral RMSE | delta |
|---|---:|---:|---:|
| overall | 11.938287 | 13.348499 | +1.410212 |
| raw GR missing | 11.948064 | 14.496321 | +2.548257 |
| raw GR observed | 11.933740 | 12.779855 | +0.846115 |
| distance 1000+ | 13.135431 | 14.719236 | +1.583805 |
| hidden-like spatial | 12.564491 | 16.027490 | +3.462999 |
| hidden-like typewell-purged | 12.367244 | 15.923789 | +3.556545 |
| focus `11d0f5ac` | 21.160939 | 21.514271 | +0.353332 |

- predictionは3,783,349 rows / 773 wellsで変化。mean absolute delta 2.298263 ft、max absolute delta 80.893875 ft。
- worst wellは`e03b45fd`: 10.657176 -> 61.824631、delta `+51.167455 ft`。missing fraction 0.446462、longest missing run 56。
- missing-run `1_4`は`+1.958686 ft`、`5_31`は`+4.222560 ft`。主要な短い欠損でも悪化した。
- posterior std meanはoverallで2.940914 -> 3.671249、missing rowsで3.271540 -> 4.413686。
- divergence segmentは1,354、最長10,052 rows。

### Guardと判断

- FAIL: overall、raw-missing、observed、1000+、hidden-like 2面、worst-well。
- PASS: prediction finite coverage 1.0、std finite coverage 1.0、ID mismatch 0。
- `stage1_guard.passed=false`、`pf_stage_eligible=false`、`pf_stage_executed=false`。
- final decisionは`stage1_fail_pf_closed`。likelihood-PF Stage 2、raw-test inference、submissionは実行しない。
- blanket neutralityは欠損rowだけでなくHMM smoothingを通じてobserved rowにも悪影響を伝播させた。現行raw exact HMMでは補間GR emissionがpathの重要な拘束として機能していると解釈する。
- 一部bucketの局所改善を使った事後run-length gate、sigma/temperature救済、mask gridは行わない。missing-GRを固定した既存`exp270_exact_hmm_posterior_mode_candidate_audit`を別仮説の次監査として維持する。

### 主なSHA

- config: `d14d3974e5b12d09f19ff2c553f162f08be92789f4421032fd207ee172c70a56`
- 上記はKaggle実行時config SHA。完了記録後に`experiment.status`だけを`completed_stage1_guard_failed_pf_closed`へ更新したローカル/package config SHAは`1917cf6353c00e5eea9b6b788bd269f0b4a0e7648307345f6cc31776d3d984b1`。
- control cache raw: `a483e2b544021048dfb224db8306142ae0c802a7fe8303b302efa198e0ed17a5`
- prediction decompressed: `4dfcceccccb1496e89601566019f0dd8f649cb4c5711f1d9a2f83be617a39976`
- row audit decompressed: `818126ff9f538429befefa2890e4e0c7ef9a03d6255f00634336630dc941c88b`
- group metrics raw: `436fddd5dad0c9c5e1cb7471c849d254f484d71d589877890c85476f298dc6ea`
- by-well raw: `64546144faad2cdf675a3cf6259341ec86454744a56cbd840672e3121ed9d5dc`
- finite coverage raw: `7ab7e74940a3cd299b5fa477628723dbe2bc4703ff116f1465bf52bbe9cba586`
