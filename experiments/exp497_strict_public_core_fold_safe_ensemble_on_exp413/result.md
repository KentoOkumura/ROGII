# exp497_strict_public_core_fold_safe_ensemble_on_exp413 結果

## 状態

Stage P/M/EをKaggle上で完了した。Stage P単一kernel version 1はfold 0..2完了後に12時間上限で
停止したため、同一計算をfold別5 kernelへ分割して完走した。Stage MもKaggle GPUで全5 folds、
合計200 boostersを完走した。Stage Eの事前固定promotion gateはFAILし、selected train anchorは
exp413を維持した。その後ユーザー承認により、昇格や提出を伴わないStage I current-test診断推論を
同じexp内で追加実行し、version 3で診断予測を完了した。続いてユーザー承認によりmodel
serializationを追加したversion 4を同じKaggle GPU kernelで再実行し、40 boosterとRidge 2本を
保存・検証した。さらに保存modelだけを読むhidden-safe inference version 2を同じKaggle T4で完了し、
14,151行の`submission.csv`と再現性artifactを検証した。推論時学習と外部competition submitは0。
Stage Eの科学gate FAILとselected train anchor exp413は変更していない。Colabは使用していない。

## 仮説

Public LB固有処理を除いた独立public-coreはexp413と異なるtrajectory inductive biasを
保ち、保存OOF同士のconstant convex blendでexp494より安全な相補性を示す。

## 設定

- 親: exp413_scale5_likpf_full_replacement_on_exp335
- Route: ensemble
- 検証: exp413 outer 5 / public-core inner 4 / ensemble meta 5、Group=well
- メトリック: suffix-row unweighted RMSE
- シード: 42、stable SHA256 key
- 予定学習: 1 variant、2 branches、200 boosters、Ridge 10
- 親/control再学習: 0

## 変更点

exp413へpublic featureを追加せず、公開pipeline coreを別trajectoryとして生成する。
Public固有overlayを除外し、公開固定selector/weightはouter-train内fitへ置換する。
最終段だけでexp413とのpublic-core上限0.30のconstant convex blendを行う。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage P/M/E実装 | 完了 |
| source SHA | `88c7b99e234fdbd5620c0045df294d9167eac84e56f538ceb3f2449a677a5454` |
| source feature数 | SP45 195 / learned 205 |
| LikPF実行量 | 1,546 banks / 197,888 seed-well / 98,944,000 particle starts |
| Beam実行量 | selector 10,822 / learned 5,411 / total 16,233 well-config |
| contract test | 23件PASS |
| Stage M fold0 | 757,738 rows / 155 wells / 40 boosters / 12,396.747秒 |
| fold0 strict public-core RMSE | 9.281962 |
| Stage M fold1 | 756,650 rows / 155 wells / 40 boosters / 15,179.320秒 |
| fold1 strict public-core RMSE | 8.413250 |
| Stage M fold2 | 756,255 rows / 154 wells / 40 boosters / 14,041.621秒 |
| fold2 strict public-core RMSE | 8.467883 |
| Stage M fold3 | 757,101 rows / 155 wells / 40 boosters / 16,074.250秒 |
| fold3 strict public-core RMSE | 8.732576 |
| Stage M fold4 | 756,245 rows / 154 wells / 40 boosters / 14,201.086秒 |
| fold4 strict public-core RMSE | 10.121002 |
| exp413 CV | 7.884803 |
| exp497 cross-fit blend CV | 7.874488 |
| exp497 − exp413 | -0.010315 ft |
| nonworse folds | 3 / 5 |
| by-well delta p95 / worst | +0.700720 / +7.541588 ft |
| promotion gate | FAIL |
| selected prediction | exp413 OOF |
| Stage I version 4 | COMPLETE / 14,151 rows / 3 wells |
| Stage I runtime | 9,204.737秒（約2時間33分） |
| Stage I fit | LightGBM 24 + CatBoost 16 / Ridge 2 |
| Stage I exp413再学習・再推論 | 0 / 0 |
| Stage I prediction contract | sample ID順序一致 / 重複・欠損・非有限値0 |
| Stage I prediction SHA | `6abd8b1d2c73d88cd8d8cfa0863cc9d08e89dbd97d1d7892d278c0d23e83f98e` |
| Stage I model serialization | 40件PASS（LightGBM 24 / CatBoost 16、335,918,672 bytes） |
| Stage I reload parity | 最大絶対差0.0（許容値1e-5） |
| Stage I model-set SHA | `dcc2166f4bd5731364efe0b3fb848a46cf87f8133cbe78890658a1062c604626` |
| Stage I Ridge weights SHA | `34aa73067d6e67b98eb72c40035b5065d6721674af89982a1089f1d803a6c727` |
| v4 − v3 strict component MAE / max | 0.024010 / 0.099000 ft |
| v4 − v3 final blend MAE / max | 0.003291 / 0.014000 ft |
| Saved-model inference version 1 | `ERROR` / visible parity guard |
| Saved-model inference version 2 | Kaggle T4 `COMPLETE` / output検証PASS |
| Saved-model inference kernel | `kentookumura/exp497-strict-public-core-saved-inference` / id_no `129666751` |
| Saved-model inference fit / load | fit 0 / exp497 40 + Ridge 2 / exp413 75 |
| version 1 visible strict / blend max差 | 0.001281 / 0.014195 ft（共通許容0.001） |
| version 2 visible strict / blend max差 | 0.001281 / 0.014195 ft（許容0.002 / 0.020、PASS） |
| version 2 runtime | kernel本体約684.077秒 / exp413 391.418秒 / strict特徴231.471秒 / 保存model推論33.509秒 |
| version 2 output | 14,151行 / sample ID順一致 / 重複・欠損・非有限値0 / submit-check FAIL・WARN 0 |
| version 2 submission SHA | `04ca2e2f80f45bced1e22bd68a58002b4cb7c7e5b19510932375cdccafa6680a` |
| 外部competition submit | 未実行 |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: false
- seed policy: stable_sha256_per_stage_split_fold_family_well_seed_index（実装済み）
- kernel version: Stage P fold別 version 1 / Stage M fold0 version 3・fold1..4 version 1 / Stage E version 1
- fold0 prediction SHA: `36ed4827e432f214c725034c09bedce057ff49b87c134f56fdea8be9f56b91bd`
- strict public-core OOF SHA: `7a7f55fafade7fa5af9c3ac10a30d5d795ca2b0cd4ba5c86b0048b497c329147`
- cross-fit blend OOF SHA: `e716cdbac014c47a92152b6144b7512077037d78808730348d03b5054dc4c632`
- selected OOF SHA: `85fe52ac68b15a7460d1bc19c3852eadc4f92233551e3ae9edddcbd3895e23ee`
- Stage I v4 rerun: COMPLETE、GPU学習のためbitwise deterministic anchorではない
- Stage I v4 model-set SHA: `dcc2166f4bd5731364efe0b3fb848a46cf87f8133cbe78890658a1062c604626`
- Stage I v4 strict public-core prediction SHA: `27641aa6d28204a855b38e4debf0059031727b701066df75e19dad9902378885`
- Stage I v4 blend prediction SHA: `c939c9f8edf83628f610d1ec85988aeb0d7e7ebcd168d5d5aa2e0805fbe56f72`
- Saved-model inference v2 kernel: COMPLETE、推論時fit 0、visible component parity PASS
- Saved-model inference v2 model-set SHA: `dcc2166f4bd5731364efe0b3fb848a46cf87f8133cbe78890658a1062c604626`
- Saved-model inference v2 prediction decompressed SHA: `db120520f8575409c1ff3043fbc4b381a3d92cedc9850b083fbda3ed47d2dc7c`
- Saved-model inference v2 strict / blend prediction SHA: `2ba49e4442e789d83033fdd95659aaf4aca019b6ddb9875f66728e7d1569ce3c` / `a16d00d97c1156f532146cd7fb469b9614f39efdab0fe462ae6a4f049155ddef`
- Saved-model inference v2 submission SHA: `04ca2e2f80f45bced1e22bd68a58002b4cb7c7e5b19510932375cdccafa6680a`

## 解釈

source identity、禁止処理のdecontamination scan、exp413のOOF/fold/scope/hidden/by-well
SHA、outer/inner split、spatial pool除外、truth-late freeze、nested dual branch、meta5の
held-fold除外はすべてfail-closed契約を通過した。5 meta-foldのpublic-core weightもすべて
正で0.30未満となり、独立trajectoryの平均的な相補性は確認できた。

ただしpooled gainは必要な0.03 ftに対して0.010315 ftに留まり、fold 0 / 4はそれぞれ
`+0.025357 / +0.139179 ft`悪化した。hidden-like spatial / typewell-purgedも
`+0.105138 / +0.097410 ft`、by-well p95 / worstは`+0.700720 / +7.541588 ft`悪化した。
平均改善は一部well・foldへの大きな害を相殺できず、安全なensemble anchorにはならない。

## 次

exp497は再weight、scope gate、same-OOF rescueを行わず、selected final anchorはexp413を維持する。
Stage Iではstrict public-coreだけを40 boostersでfull-train inner-4 fitし、保存済みexp413 current-test
予測と固定係数`0.13716473330712417`で診断blendした。version 4ではLightGBM 24本、CatBoost 16本を
保存直後に再読込し、全40本で予測最大絶対差0.0、SHA/bytes/count/path契約PASSを確認した。Ridge 2本も
JSONへ保存した。v3とv4の予測はGPU学習の非bitwise決定性により完全一致しないが、final blend差は
MAE `0.003291 ft`、最大`0.014000 ft`だった。model artifactは揃ったため、同じexp内で保存model
だけを読むhidden-test推論専用候補を実装した。候補はexp497 40 booster + Ridge 2、exp413 75 boosterを
SHA検証して読み、推論時学習0でdynamic sampleへ適用する。exp413の公開test固定sidecarは使わず、
hidden-safe dynamic runtimeを再利用する。Jupytext round-trip、py_compile、Ruff、専用test 29件、
実model artifact 40件/335,918,672 bytesの読込契約はPASSした。候補を正規Notebookへ採用し、
76 support filesのpackage readbackとremote T4 metadataを検証してKaggle version 1をpushした。
version 1はdynamic exp413、strict特徴、全保存model推論まで完了したが、旧visible predictionとの
共通`0.001 ft` parity guardで停止した。dynamic exp413 content SHAも旧referenceと異なり、blend差を
支配している。OOM・入力欠落・model破損ではなく、外部submitも行っていない。修正時はstrict成分と
dynamic exp413/blendを別監査へ分離し、parent-only中間submissionをfinal名から隔離する。

version 2ではstrict `0.002 ft`、blend `0.020 ft`へ分離し、中間exp413 submissionを検証後に
`artifacts/exp413_intermediate_submission.csv`へ隔離した。科学式、保存model、weight、dynamic runtimeは
不変。30 testsと76-file package/remote marker readbackをPASSし、同一private T4 kernelで完了した。
exp497 40 booster + Ridge 2、exp413 75 boosterをloadし、fitは0。visible parityはstrict
`0.001281 <= 0.002 ft`、blend `0.014195 <= 0.020 ft`でPASSした。`submission.csv`は14,151行で
sample ID順序、重複、欠損、finite、serialized blendとの一致、SHAをすべてPASSし、submit-checkも
FAIL/WARN 0だった。外部competition submitは未実行であり、提出する場合は別途明示承認を要する。
