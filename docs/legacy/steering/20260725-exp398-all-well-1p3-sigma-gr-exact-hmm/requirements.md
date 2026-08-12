# 要件

## 依頼

exp209 exact HMMのGRノイズ幅を、selectorなしで全773 wells一律に`1.3`倍し、
保存済みexp209 Gaussian HMMをcontrolとしてtrain-side CVを評価する。

## 制約

- Route: `pf_beam`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- exp209のknown-prefix zero-fill population stdを`[10,60]`でclipした後、全wellで
  `1.3`を正確に1回掛ける。再clipしないため有効範囲は`[13,78]`。
- Gaussian capped emission、absolute-TVT grid、41 rate states、transition、prior、
  missing-GR処理、Type Well補間、posterior meanは変更しない。
- scientific variant 1、HMM well-runs 773、reporting folds 5。
- model config、trained fold、booster、PF、Beam、親control再実行は各0。
- 予測とcontent SHAをunknown-suffix truth読込前にfreezeする。
- multiplier、clip、emission、transition、grid、blendの同一OOF救済をしない。
- 最初のfull runはKaggle private CPUで行う。GPUとinternetは使わない。

## 受け入れ基準

- input SHA、3,783,989 rows / 773 wells、row/fold identity、finite prediction、
  posterior normalization、truth-before-freeze 0を満たす。
- exp209 direct HMM比RMSE gain `>=0.05 ft`、4/5 folds、raw-observed gain
  `>=0.05 ft`を満たす。
- raw-missing、high-missing、1000+、hidden-like 2面、by-well p95、worst well、
  fixed LikPF 50:50の全guardを満たす。
- predictionのraw gzip SHAとdecompressed/logical content SHA、scientific contract SHA、
  Kaggle kernel versionを記録する。
- FAIL時はinference、submission、parameter rescueなしでcloseする。
