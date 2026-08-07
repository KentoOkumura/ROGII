# exp305_tempered_raw_smoothed_exact_hmm_emission 結果

## 状態

Kaggle CPU train version 3は正常完了した。固定tempered emissionはdirect、fixed likPF 50/50 blendともに全promotion gateをFAILしたため、救済せずnegative resultとして閉じる。inferenceとsubmissionは行わない。

## 仮説

exp304選択SWT emissionをrawへ15%だけ混ぜると、rawを保持したままexact-HMM posterior meanとfixed likPF blendを改善できる。

## 設定

- 親: `exp304_gr_denoiser_emission_separability_readout`
- HMM/control参照: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- emission: `ell_beta = 0.85 * ell_raw + 0.15 * ell_swt`
- 保存likPF: `last_known_tvt + likpf_mean_d`
- 検証: 保存controlとのtrain-side paired readout
- 実行量: 1 variant、773 HMM well-runs、control再実行0、model/LightGBM/PF/Beam/booster 0

## 結果

| 比較 | candidate RMSE | control RMSE | 改善量 | 改善fold | 判定 |
|---|---:|---:|---:|---:|---|
| direct tempered HMM | 13.218199 | 11.938287 | -1.279912 ft | 1/5 | FAIL |
| tempered HMM / likPF 50/50 | 10.767674 | 10.269693 | -0.497982 ft | 1/5 | FAIL |

directのfold別candidate-control RMSE差は`+2.973011 / -0.032811 / +2.677078 / +0.381420 / +0.457403 ft`、blendは`+1.443477 / -0.150180 / +1.013859 / +0.096239 / +0.180397 ft`だった。負値はcandidate改善を表すが、改善したのは両比較ともfold 1だけだった。

必須stress scopeもすべて悪化した。

| scope | direct差 | blend差 |
|---|---:|---:|
| MD since 1000+ | +1.395238 | +0.542714 |
| hidden-like spatial | +1.052477 | +0.442637 |
| hidden-like typewell-purged | +0.993719 | +0.429642 |
| by-well RMSE p95 | +1.906563 | +1.868961 |
| worst-well RMSE | +56.989605 | +29.298736 |

入力preflight、3,783,989 rows / 773 wells、finite coverage 100%、ID mismatch 0、silent fallback 0、773 HMM runs、8.5時間runtime上限はPASSした。実行時間は15,983.840秒（約4時間26分24秒）。

strict technical gateは、raw HMM parityは`1.3e-11 ft`差でPASSした一方、正しいdelta復元後のsaved likPFと50/50 controlが保存baselineからそれぞれ`3.28e-6 / 3.64e-6 ft`ずれ、固定許容値`1e-6 ft`を超えたためFAILした。この差は科学的悪化`0.498--1.280 ft`より十分小さく、全fold/scopeでの大幅悪化を救済しない。positive resultの根拠にはできないが、候補を棄却するnegative decisionは信頼できる。

## 再現性

- Kaggle kernel: `kentookumura/exp305-tempered-raw-swt-exact-hmm-emission-train` version 3、`id_no=128079137`、status `COMPLETE`
- scientific contract SHA256: `343084494621e1a3bb15899f6c3c441507f4dfd3af671e8de7c08f3f8867bd1b`
- input/control manifest SHA256: `48f3c68cc1d02f60871a111f0cb473cc9a67d29963cc5d4c79143c69c00d15f7`
- prediction raw gzip SHA256: `6419657d633564325ced8cabfde22532737cdabd1c7844686317d8dd7efe2552`
- prediction decompressed/content SHA256: `86b1768f18d31ba296774054c14b24e2e4650ddc74d9858d9baea8b534027302`
- promotion gate SHA256: `9edc8fa51f070d38420bc6b84ec5e329a1b5c7425a46234f4e85cf5741ee1c71`
- overall/fold/scope metrics SHA256: `efe59c08753072d658eac1138cb1702ac4579939734de68e1aec89a615f4532e`
- by-well metrics SHA256: `4b810d2620172427e6da2373a0a23813d127325f9f3beb9db4d85943ec1070b2`
- 取得した小型metrics生成物はKaggle summary記録SHAと一致した。46 MBのprediction本体はダウンロードせず、Notebookが出力したraw/content SHAを記録した。

## 解釈

exp304でshift separabilityが良かったSWTでも、raw exact-HMM likelihoodへ15%混ぜるとwrong modeを安定化し、decoder RMSEとtail safetyを広く悪化させた。単一foldの小改善以外は一貫して不支持であり、beta、sigma、clip、HMM、blend weightの同一OOF救済を行う根拠はない。

v1/v2は入力契約参照ミスで失敗したが、v3はschema preflightと全科学計算を完走している。事前登録どおりexp304 reserved案2をFAIL、案3と案4を閉じる。

## 次

exp305由来のSWT tempered-emission救済は追加しない。独立仮説であるexp307のfinite-only robust sigma routeと、exp305完了待ちだったexp321 Stage A/Bを既存条件のまま優先し、exp305のinference/submissionは行わない。
