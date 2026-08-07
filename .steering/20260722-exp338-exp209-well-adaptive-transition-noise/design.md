# 設計

## アプローチ

旧exp309はwell別`sig_r`仮説を実装したが、観測モデルとしてexp307 finite-MAD `sigma_GR`とexp308 missing-distance confidenceを固定していた。exp307がdirect `+3.723054 ft`悪化してpromotion gateをFAILしたため、旧lineageでは`sig_r`仮説自体が未評価のまま閉じた。

exp338では、exact HMM cache parityが確認済みのexp209へ直接戻し、観測モデルを含む全条件を固定した上で`sig_r=0.002`だけをwell別値に置き換える。これにより、ControlとCandidateの差をrate diffusionだけに限定する。

known prefixの`U=TVT_input+Z`から区間rateを作り、隣接区間rate差を区間中点間のMD距離平方根で標準化する。robust MADを固定prior `0.002`へlog shrinkし、support不足はcontrolへfallbackする。式は旧exp309から移植するが、旧exp309のGR scale、missing confidence、parent preflight、HMM observation codeは移植しない。

## 実験範囲

- 対象実験: `exp338_exp209_well_adaptive_transition_noise`
- Route: `pf_beam`
- 科学的親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 実装参照: `exp309_well_adaptive_transition_noise`のtransition-scale式のみ。
- negative evidence: `exp307_finite_only_robust_sigma_gr`、`exp273_two_dimensional_formation_gradient_transition`。
- 変更する変数: well別`sig_r,w`。
- 固定する変数: exp209 zero-fill std `sigma_GR`、GR/typewell preprocessing、Gaussian emission、position/rate grid、`sig_p`、momentum、start/rate prior、posterior mean。
- 実行量: 1 scientific variant、773 HMM well-runs、LightGBM config/fold/booster、PF、Beam、control再実行はすべて0。

## 検証方法

1. raw well identity、exp209 HMM cache `8e2f4236...f7ae5`、exp209-generated exp072/LikPF cache `0503de05...a536`、exp226 fold、exp115 hidden-like assignmentをpreflightする。
2. raw horizontalからunknown-suffix truth、formation、fold/hidden-like scoreを接続せず、known prefixだけでwell別transition auditを作る。
3. `n`、innovation median/MAD、raw scale、shrinkage alpha、unclipped/clipped `sig_r,w`、fallback/clipを773 wellsで凍結し、decompressed content SHAを保存する。
4. exp209 exact-HMM code contractを維持し、candidateだけ1回生成する。candidate prediction/content SHAを凍結する。
5. その後だけunknown-suffix truth、fold、distance、hidden-like、saved exp209 control、saved LikPFをjoinする。
6. overall、5 folds、1000+、hidden-like 2面、by-well p95/worst、fixed 50:50 LikPF blendをAND gateで判定する。

## 生成物契約

- input/dependency/scientific contract JSON。
- well別transition audit CSV.gz。
- candidate exact-HMM prediction/posterior diagnostic CSV.gz。
- overall/fold/distance/hidden-like/by-well/transition-support metrics。
- promotion gate JSONとmetrics.json。
- gzipはraw SHAとdecompressed content SHAを分け、後者を科学的identityの主証拠とする。

## 後続依存グラフ

```text
exp209
  `-- exp338 well-adaptive sig_r
        `-- NEW time-varying exp226 dip-rate prior (旧exp323相当)
              |-- NEW donor-covariance segment sig_r,t (旧exp324相当)
              |-- NEW window-likelihood HMM factor (旧exp325相当)
              |-- NEW residual-rate momentum_t (旧exp326相当)
              `-- NEW position sigma_t (旧exp327相当)
```

- exp338 FAIL時はこの後続を作らない。
- exp338 PASS後も新exp323相当は自動作成せず、別steeringと承認を要求する。
- 新exp323相当 FAIL時は324--327相当を作らない。
- 324--327相当は新exp323 PASS後に兄弟分岐として別々に採番する。
- 旧exp323--328は旧lineageの閉鎖履歴として維持する。
- 旧exp328のcausal affine仮説は本chainから独立し、exp209直系の兄弟`exp345_exp209_time_varying_gr_affine_calibration_hmm`へ切り出し済みとする。exp338とexp345は相互非依存で、自動合流させない。

## 再現性設計

- seed policy: RNGなし。well ID、raw row、transition calculation、variant順を固定する。
- stochastic処理: なし。
- PF/Beam/likelihood-PF: 新規生成なし。saved LikPFはreadout専用。
- 並列処理: 実装時はexp209採用値`outer_workers=2`、Numba threads `2`を開始点として固定し、変更時はprediction parityを要求する。
- runtime: Kaggle CPU、GPU/internet off。exp209 HMM実績`11,285.868 sec`を基準に8.5時間上限とする。
- SHA: raw/input/dependency/scientific contract、transition audit、prediction、metricsのdecompressed content SHAを記録する。
- model/submission SHA: 非該当。deterministic submission anchorとは扱わない。
- package: canonical kernel id/title、metadata、bootstrap config/source、実行量をpush前に照合する。実装・pushは未承認。

## リスク

- リークリスク: `sig_r,w`はknown `TVT_input`と`Z/MD`だけから凍結し、suffix truth/errorを使わない。
- process-scale mismatch: 観測rate innovationには測定noiseや局所離散化も含まれる。log shrink、support fallback、clipを事前固定し、clip/fallback率でtechnical FAILさせる。
- tail risk: exp273のtransition変更はworst `+36.118726 ft`だったため、global平均だけでなくp95/worst/hidden-likeをhard gateにする。
- objective mismatch: prefix rate scaleが物理的でもsuffix TVT RMSEを改善するとは限らない。directとfixed blend両方を要求する。
- runtime: controlを再実行せず1 candidateだけに限定する。
- CV/LB: train-side PASS後もinference/submissionを自動許可しない。

## 優先度

Lateフェーズの`低-中・P2`。仮説は未評価で単一変更として明確だが、exp268のinitial-rate direct gainが`0.042706 ft`に留まり、exp273のtransition変更がtailを悪化させたため、厳しいpromotion gateを維持する。

## 次のアクション

version 3は同一Kaggle CPU kernelで773/773 HMMを完了した。candidate `14.062348`はparent `11.938287`より`+2.124061 ft`悪化し、全773 wellsの`sig_r`が上限`0.004`へclipされた。設計済みFAIL条件に従い、救済変更、inference、submission、後続chainなしでterminal closeする。

## 実行後の設計評価

known-prefix finite-difference innovationはwell間のprocess scaleではなく量子化成分を主に反映したことを強く示す。将来このfamilyを独立に再訪する場合、HMM前にproxyのdistinct値、分位点、IQR、well間変動、mapping後clip率をtarget-freeに検査し、clip率`<=0.5`を満たさないproxyを実行対象にしない。
