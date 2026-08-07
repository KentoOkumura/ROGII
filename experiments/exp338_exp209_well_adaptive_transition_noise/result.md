# exp338_exp209_well_adaptive_transition_noise 結果

## 状態

Kaggle CPU train version 3は完了したが、promotion gateはFAIL。`adaptive_sig_r_failed_close_without_rescue`としてbranchを閉じた。推論、提出、後継実験は作成していない。

## 仮説

exp209の観測モデルを固定し、known-prefix U-rate innovationから推定したwell別`sig_r`だけを使えば、固定`sig_r=0.002`よりsuffixのrate変化へ適応できる。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- 変更: well別`sig_r,w`だけ
- 固定control: exp209 raw HMM
- 実行量: 1 variant / 773 HMM well-runs / 0 model / 0 booster / control再実行0
- kernel: `kentookumura/exp338-exp209-well-adaptive-transition-noise-train` version 3、id_no `128226900`
- runtime: `11,376.512秒`（約3時間9分37秒）

## 結果

| 比較 | Candidate RMSE | Control RMSE | Candidate - Control |
| --- | ---: | ---: | ---: |
| direct HMM | 14.062348 | 11.938287 | +2.124061 ft |
| fixed LikPF 50:50 | 11.184022 | 10.269693 | +0.914329 ft |

directは0/5 folds改善で、fold deltaは`+3.738901 / +1.429482 / +2.866298 / +0.148952 / +2.454392 ft`だった。

| 必須scope / tail | Candidate - Control | Gate |
| --- | ---: | --- |
| MD 1000+ | +2.377399 ft | FAIL |
| hidden-like spatial | +3.278598 ft | FAIL |
| hidden-like typewell-purged | +3.362723 ft | FAIL |
| by-well RMSE p95 | +4.790247 ft | FAIL |
| worst well `a645da9a` | +54.818838 ft | FAIL |

## Technical gate

- rows / wells / HMM runs: `3,783,989 / 773 / 773`
- finite coverage: `1.0`
- ID mismatch: `0`
- posterior normalization max error: `3.77e-15`
- exp209 raw HMM / LikPF / blend baseline parity: 全PASS
- fallback fraction: `0.0`
- total clip fraction: `1.0`（上限`0.5`に対してFAIL）
- 最終`sig_r`: 全773 wellsで`0.004`

transition auditでは全wellのrate innovation medianが0、absolute medianがほぼ0.01となり、shrink前後の値が全件で上限clipを超えた。known-prefix `U=TVT_input+Z`の有限差分proxyは量子化成分に支配され、well間のtransition-noise差を識別できなかったことを強く示す。

## 再現性

- scientific contract SHA: `4d21c3f89a190833b1c201bfc9f3867c638943e4f1ec99f6a2a9d101ec7c6760`
- input control manifest SHA: `99e12e4f0c2099e687fad5eba63f1fe43e04cc624effa37d877304e3ebdfc131`
- prediction content SHA: `bf426bcf5b0452004ca0a3d6626c1f7e476f005740a8a283b1f035e782286838`
- transition audit content SHA: `eaa3956f62b7ca592e97ac9175f4a2ad2c18c4772068a8cd4713318686d19aca`
- promotion gate raw SHA: `5e99d8298be9bc3643a1f03864de29c639b3791d497e1f34c451c3b0c482d2cd`
- submission SHA: 非該当

## 実行履歴

| Kaggle version | 到達点 | 結果 |
| --- | --- | --- |
| 1 | 親artifact preflight | raw/local metrics schema差によりHMM前ERROR |
| 2 | candidate HMM 773/773完走 | exp115の正式な`purged_train_excluded` roleをlate contractが拒否してERROR |
| 3 | candidate HMM 773/773、全readoutとgate | COMPLETE、promotion FAIL、branch close |

## 判断

事前契約どおり、clip、shrinkage、`sig_p`、momentum、rate grid、blendによるpost-hoc救済は行わない。exp338 PASSを先行条件としていた新exp323相当以降の後続chainも作成しない。旧exp323--328は閉鎖履歴のまま維持し、独立兄弟exp345の判断には影響させない。

## 次

exp338は完了・terminal close。transition-noise適応を将来独立に再訪する場合は、HMM実行前にtarget-free proxyのwell間識別力とclip率を検査するpreflightを先行させる。
