# exp338_exp209_well_adaptive_transition_noise セッションノート

## 目的

旧exp309で未評価のwell別`sig_r`仮説を、信頼できるexp209 exact-HMMへ直接reparentし、観測モデルを変えずに独立再検証できる設計として固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train version 3完了 / promotion gate FAIL / terminal close
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- CV / LB: `14.062348052437676` / 未提出
- 実行量: 1 variant / 773 HMM well-runs / LightGBM config・fold・booster・PF・Beam・control再実行すべて0
- 科学実装: 完了
- 正規train Notebook採用: 完了
- Kaggle CPU train package/push/run: version 3 COMPLETE
- inference、submission、後継実験: 未実施・branch closeにより無効

## 2026-07-22 設計確定

```bash
make new-steering EXP=exp338_exp209_well_adaptive_transition_noise
make new-exp EXP=exp338_exp209_well_adaptive_transition_noise
```

- 科学的親をexp209へ固定した。
- 旧exp309からはknown-prefix rate innovation MAD、20未満fallback、`n/(n+100)` log shrink、clip `[0.001,0.004]`の式だけを参照する。
- exp307 finite-only/MAD `sigma_GR`とexp308 missing-distance confidenceは使用しない。
- exp209 zero-fill std `sigma_GR`、GR/typewell preprocessing、Gaussian emission、41 rate states、`sig_p=0.02`、position floor `0.1225`、momentum `0.998`、prior、posterior meanを固定した。
- transition audit/prediction freeze後だけtruth/fold/hidden-like/LikPFを読むlate-join境界を固定した。
- direct、fold、1000+、hidden-like、p95、worst、fixed LikPF 50:50、fallback/clipのAND gateを固定した。

## 後続依存契約

- exp338全gate PASS時だけ、旧exp323を再開せず、exp338を親にした新exp323相当を新番号・別steering・別承認で作る。
- 新exp323相当全gate PASS時だけ、新exp324、325、326、327相当を兄弟分岐として別番号・別steering・別承認で作る。
- exp338または新exp323相当がFAILした場合、対応する後続は作らない。
- 旧exp323--328は閉鎖履歴として維持し、parent fieldの差し替えや実装再開を行わない。
- 旧exp328相当は後続chainに含めず、exp209直系の独立兄弟`exp345_exp209_time_varying_gr_affine_calibration_hmm`を明示的な再検証入口とする。exp338とexp345は相互非依存とする。

## 再現性メモ

- RNGなし。well ID、raw row、transition計算、variant順を固定する。
- HMMはexp209採用の`outer_workers=2`、Numba threads `2`を開始点とする。
- saved HMM cache decompressed SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`。
- saved LikPF cache decompressed SHA: `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`。
- raw/input/dependency/scientific contract、transition audit、prediction、metricsはdecompressed content SHAを主証拠にする。
- model/submission SHAは非該当。本実験をdeterministic submission anchorとは扱わない。

## 2026-07-22 科学実装

- ユーザーの`exp338を実装してください`という依頼を実装承認として記録した。Kaggle実行承認には拡張していない。
- `exp338_exp209_well_adaptive_transition_noise_compact_selfcontained_train.py` / `.ipynb`を追加した。
- exp209のzero-fill population std `sigma_GR`、unit observation weight、GR補間、Gaussian emission、41 rate states、`sig_p=0.02`、position floor、momentum、prior、posterior meanを固定した。
- 旧exp309からはknown-prefix U-rate innovationの`1.4826*MAD`、20未満fallback、`n/(n+100)` log shrink、clipだけを移植した。MD normalizationに未登録の1 ft floorは入れていない。
- saved exp209 HMM/LikPF、fold、hidden-likeはprediction freeze前にSHAとheaderだけを確認する。fold/hidden-like値とsuffix truthはtransition audit/predictionのdecompressed content SHA固定後だけ読む。
- direct、5 folds、1000+、hidden-like 2面、by-well p95/worst、fixed LikPF 50:50、fallback/clip、runtime、baseline parityを単一AND gateへ実装した。
- `exp338_exp209_well_adaptive_transition_noise_compact_selfcontained_inference.py` / `.ipynb`は常にfail-closedとした。
- 既存の正規`*_train.ipynb` / `*_inference.ipynb` placeholderは上書きしていない。
- 実装量は1 variant / 773 HMM well-runs / LightGBM config・fold・booster・PF・Beam・control再実行すべて0のまま。

実施した検証:

```bash
.venv/bin/python -m py_compile experiments/exp338_exp209_well_adaptive_transition_noise/exp338_exp209_well_adaptive_transition_noise_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp338_exp209_well_adaptive_transition_noise/exp338_exp209_well_adaptive_transition_noise_compact_selfcontained_train.py --select F821
.venv/bin/pytest -q experiments/exp338_exp209_well_adaptive_transition_noise/tests/test_exp338_exp209_well_adaptive_transition_noise.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp338_exp209_well_adaptive_transition_noise/exp338_exp209_well_adaptive_transition_noise_compact_selfcontained_train.py
```

- 専用pytest: `10 passed`。
- synthetic parity: exp209親`run_hmm2`と候補のposterior mean/stdが`atol=1e-10`で一致。
- formula/fallback/clip、zero-fill std、truth freeze、dependency SHA、全gate、disabled inferenceを検証した。
- `make validate-exp EXP=exp338_exp209_well_adaptive_transition_noise` strictと`make validate-template`をPASSした。
- 全体`make test`は610 collected中605 passed / 3 skipped / 2 failed。失敗2件はいずれも既存exp296の完了後config状態と旧test期待値の不一致で、exp338専用testは全件PASSした。無関係なexp296は変更していない。
- 章立て参照元の旧exp309 compactは10章・1,994行、exp338 compact trainは10章・1,889行で、runtime/config/preflight、exp209 observation、HMM、freeze、late join、metrics/gate、生成物保存をすべて維持した。

## 2026-07-22 Kaggle CPU train実行承認

- ユーザーの`実行してください`を、exp338の正規train Notebook採用、Kaggle package/push/run、および同一kernel完了監視の承認として記録した。
- inference、submission、PASS後の後続実験作成には承認を拡張しない。
- 実行対象は科学variant 1件、well別`sig_r`候補のexact-HMM 773 well-runs。
- LightGBM/model config 0、trained fold 0、booster 0、PF 0、Beam 0、親control HMM再実行0。
- 比較baselineは保存済みexp209 HMM/LikPFをSHA照合してread-only利用する。
- Kaggle CPU、GPU off、internet off、`outer_workers=2`、Numba threads 2、runtime limit 30,600秒。
- kernel id: `kentookumura/exp338-exp209-well-adaptive-transition-noise-train`
- kernel title: `exp338 exp209 well adaptive transition noise train`
- kernel version: `3`（version 1は起動前、version 2はHMM完走後にエラー）
- kernel id_no: `128226900`
- kernel URL: `https://www.kaggle.com/code/kentookumura/exp338-exp209-well-adaptive-transition-noise-train`
- push後にKaggle側metadataをpullし、private CPU、GPU/TPU/internet off、competition sourceと3 kernel sourcesがpackage契約に一致することを確認した。

## 次のアクション

exp338は完了・terminal close。inference、submission、新exp323相当以降の後続chainは作成しない。transition-noise適応を独立に再訪する場合だけ、HMM前にtarget-free proxyのwell間識別力とclip率を検査するpreflightを別途設計する。

## 2026-07-22 Kaggle CPU train version 1 起動前エラー

- status: `KernelWorkerStatus.ERROR`
- id_no: `128226900`
- bootstrap、実行契約表示、raw well identity確認までは通過した。
- exp209 control HMM/LikPFのdecompressed SHAと列検査は通過したが、親`metrics.json`契約で停止した。candidate HMMは開始前で、実行数は0。
- 原因はexp338 preflightが、Kaggle inputにあるexp209生成時のnested raw metricsへ、Kaggle完了後にローカル`metrics.json`へ追記した`route`、`hmm_feature_parity_pass`、`metric_parity_accepted`等のflat keyを要求していた実装不整合。
- 親Kaggle outputから診断に必要な`metrics.json`だけを取得した。raw SHAは`ebf7e24624d8b98f41cdda9c90a5df488031673f44d59a7485c9b2b62e6d67ae`。
- 修正方針は、raw metrics SHA、nested HMM parity、生成LikPF SHA、行/well数、best candidate一致、およびexp209でユーザー確認済みのRMSE差許容`1e-5`をfail-closedに検証すること。strict raw flagの偽装や閾値緩和は行わない。
- 修正・専用test・package再照合後に、同じkernelのversion 2として再実行する。version 1で科学計算を消費していないため、実行量は引き続き1 candidate / 773 HMM well-runs / control再実行0。

## 2026-07-22 Kaggle CPU train version 2

- 同じkernelへversion 2をpushし、`KernelWorkerStatus.RUNNING`を確認した。
- Kaggle側metadataはid_no `128226900`、private CPU、GPU/TPU/internet off、competition source、3 kernel sourcesがpackage契約と一致した。
- package内configとtrain sourceのSHAは元ファイルと一致し、bootstrap後のnotebook bodyは正規train Notebook 20セルと完全一致した。
- 同じversion 2を完了まで監視し、空logsを理由に再pushしない。

## 2026-07-22 監視引き継ぎ

- ユーザーの依頼により、Codex側のstatus監視だけを停止した。Kaggle kernel version 2自体は停止していない。
- 監視停止直前の状態は`KernelWorkerStatus.RUNNING`。
- 完了連絡後に同じversion 2のlogsを取得し、promotion gate、metrics、artifact SHA、runtimeを記録する。
- 完了前に再pushせず、inference/submission/PASS後の後続実験作成も行わない。

## 2026-07-23 Kaggle CPU train version 2 エラー

- status: `KernelWorkerStatus.ERROR`
- candidate HMM: `773/773` wells完走。最後のwell開始表示はKaggle elapsed `19,394.190 sec`。
- notebook error到達: `19,666.731 sec`（約5時間27分47秒）。
- HMM完走後、prediction/audit freeze後のlate joinで`ValueError: late hidden-like role contract mismatch`となった。科学計算、メモリ、runtime limitの失敗ではない。
- exp115 assignment SHA `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`と773 wellは設定通り。
- 実role件数はspatial=`train:573, valid:200`、typewell-purged=`train:557, valid:200, purged_train_excluded:16`。`purged_train_excluded`はexp115がgroup purgeのため意図的に生成する正式値だった。
- exp338は両role列を一律`train/valid`だけに制限していたため、typewell-purgedの正式な16 wellsを誤って拒否した。
- Kaggle `kernels files`は空で、失敗runのcandidate prediction/auditを取得できなかった。再計算回避用の保存生成物はない。

### version 3修正と実行量ガード

- spatialは`train/valid`だけ、typewell-purgedは`train/valid/purged_train_excluded`だけを明示許可する。
- 許容集合だけでなく、上記role件数をlate joinで完全一致検証する。未知roleを広く許容しない。
- hidden-like scope判定は従来通り`role == valid`だけであり、`purged_train_excluded`をvalidationへ含めない。
- 専用testに実exp115 artifact SHAとrole件数の回帰検証を追加し、`12 passed`を確認した。
- version 3も科学variant 1件 / candidate HMM 773 well-runs / LightGBM config 0 / trained fold 0 / booster 0 / PF 0 / Beam 0 / 親control再実行0。
- inference、submission、PASS後の後続実験作成は引き続き無効。
- 同じcanonical kernelへversion 3をpushし、Kaggle側metadataのid_no `128226900`、private CPU、GPU/TPU/internet off、competition source、3 kernel sourcesを再確認した。
- push直後は`KernelWorkerStatus.RUNNING`。ユーザーの前回依頼どおりCodex側の継続監視は行わず、kernel自体は停止していない。

## 2026-07-23 Kaggle CPU train version 3 完了結果

- status: `KernelWorkerStatus.COMPLETE`
- kernel: `kentookumura/exp338-exp209-well-adaptive-transition-noise-train` version 3、id_no `128226900`
- generated at: `2026-07-23T00:40:27.171017+00:00`
- runtime: `11,376.512313604355 sec`（約3時間9分37秒）
- prediction freeze: `11,276.093342065811 sec`
- rows / wells / HMM runs: `3,783,989 / 773 / 773`
- status record: `train_side_adaptive_sig_r_gate_failed_closed`
- decision: `adaptive_sig_r_failed_close_without_rescue`

### 科学readout

- direct candidate RMSE `14.062348052437676`、parent raw HMM `11.938287234887435`、delta `+2.124060817550241 ft`。
- direct fold deltaはfold 0--4で`+3.738900559 / +1.429482294 / +2.866298472 / +0.148951688 / +2.454391900 ft`。改善は`0/5 folds`。
- MD 1000+ delta `+2.3773986647535725 ft`。
- hidden-like spatial delta `+3.278598240816672 ft`。
- hidden-like typewell-purged delta `+3.362722713313005 ft`。
- fixed LikPF 50:50 candidate `11.184021746721717`、parent blend `10.269692505026358`、delta `+0.9143292416953592 ft`。
- by-well RMSE p95はcandidate `30.215993485792644`、control `25.425746549947814`、delta `+4.790246935844831 ft`。
- worst wellは`a645da9a`、delta `+54.81883754730505 ft`。

### technical gateと失敗原因

- raw identity、parent dependency、3 baseline metrics parity、行/well/ID、finite coverageはPASS。
- posterior normalization max abs errorは`3.774758283725532e-15`。
- fallback fractionは`0.0`。
- total clip fractionは`1.0`で、事前上限`0.5`をFAIL。
- transition auditの全773 wellsがhigh clipとなり、最終`sig_r`は全件`0.004`。innovation medianは全件0、absolute medianはほぼ0.01だった。
- known-prefix `U=TVT_input+Z`の有限差分proxyは量子化成分に支配され、well間transition noiseを識別できなかったことを強く示す。観測モデルやbaseline artifactの不整合ではなく、proxyとmappingの科学的失敗として扱う。

### 再現性SHA

- scientific contract: `4d21c3f89a190833b1c201bfc9f3867c638943e4f1ec99f6a2a9d101ec7c6760`
- input control manifest raw: `99e12e4f0c2099e687fad5eba63f1fe43e04cc624effa37d877304e3ebdfc131`
- prediction content / raw gzip: `bf426bcf5b0452004ca0a3d6626c1f7e476f005740a8a283b1f035e782286838` / `0177e2a738309b97cdc170c2851b36422b7752108b4adedb37240a85f6225e08`
- transition audit content / raw gzip: `eaa3956f62b7ca592e97ac9175f4a2ad2c18c4772068a8cd4713318686d19aca` / `51e67534e70f5d8d2eb80d755c786452466b7e72ba8caf72ff114fac9e1bd53c`
- promotion gate raw: `5e99d8298be9bc3643a1f03864de29c639b3791d497e1f34c451c3b0c482d2cd`
- overall/fold/scope metrics raw: `c7c92a13dea24f307b058c258a97951f98c91d7e56547b20f1424c5fcca0d51d`
- diagnostic metrics raw: `1735510892bede583def664f97cbd98f27651f6bc430e8280a21a0f577cc1123`
- by-well metrics raw: `24dc1632bbfdfa8441f7b8c42a059600f7340d83bed5a6b2e494742007db82f5`
- runtime CSV raw: `5461fb0d94faca9ffbf1f59cca0b17377f9242f9240da7a379bea89ac0369b28`

Kaggle outputはlogsと小さいsummary/gate/metrics/auditファイルだけを取得し、86 MBのprediction archiveはダウンロードしていない。

### 閉鎖判断

事前契約どおり`sig_r`式、clip、pseudocount、threshold、`sig_p`、momentum、rate grid、blendによる救済を行わない。inference、submission、新exp323相当、新exp324--327相当を作らず、exp338 branchをterminal closeする。独立兄弟exp345の既存判断は変更しない。
