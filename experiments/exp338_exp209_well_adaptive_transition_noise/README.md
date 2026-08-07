# exp338_exp209_well_adaptive_transition_noise

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU train version 3完了、promotion gate FAIL、terminal close
- CV / Public LB / Private LB: `14.062348 / - / -`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 式の参照元: `exp309_well_adaptive_transition_noise`のwell別`sig_r`推定だけ
- inference / submission / successor: 未実施

## 仮説と単一変更

exp209の観測モデルを完全に固定し、known prefixの`U=TVT_input+Z`から得るrate innovationのrobust scaleだけをwell別`sig_r`へ使う。変更はwell共通`sig_r=0.002`からwell別`sig_r,w`への置換だけで、exp209のzero-fill std `sigma_GR`、GR補間、Gaussian emission、state/rate grid、`sig_p`、position floor、momentum、prior、posterior meanは固定した。

## Kaggle v3結果

- 実行: 1 candidate / 773 HMM well-runs / 3,783,989 rows
- runtime: `11,376.512秒`
- direct: `14.062348` vs parent `11.938287`、`+2.124061 ft`
- fixed LikPF 50:50: `11.184022` vs parent blend `10.269693`、`+0.914329 ft`
- folds improved: `0/5`
- 1000+ / hidden-like spatial / typewell-purged: すべて悪化
- by-well p95 delta: `+4.790247 ft`
- worst well delta: `+54.818838 ft`
- fallback / clip fraction: `0.0 / 1.0`

## 検証方針

- transition auditとcandidate predictionをunknown-suffix truth接続前に凍結する。
- 保存済みexp209 HMM/LikPFをSHA照合し、controlを再実行しない。
- direct gain、5 folds、1000+、hidden-like 2面、by-well p95/worst、fixed LikPF 50:50、fallback/clipを単一AND gateで判定する。
- baseline parity、有限性、行/well/ID、posterior normalizationをtechnical gateとして分離する。

## 所見

全773 wellsの最終`sig_r`が上限`0.004`に張り付いた。known-prefix finite-difference proxyは量子化に支配され、事前に意図したwell-adaptive transition noiseとして機能しなかったことを強く示す。baseline parity、finite、ID、posterior normalizationはPASSしており、結果悪化を親artifact不整合や数値破綻では説明できない。

## 結論

decisionは`adaptive_sig_r_failed_close_without_rescue`。事前契約に従い、clip/threshold/shrinkage/gridの変更、`sig_p`/momentum救済、PF/Beam化、推論、提出、後継実験作成は行わない。exp338を先行条件とする新exp323--327相当chainは不成立。

正規train入口は`exp338_exp209_well_adaptive_transition_noise_train.ipynb`。inference候補は記録として残すが、terminal failure後は実行しない。
