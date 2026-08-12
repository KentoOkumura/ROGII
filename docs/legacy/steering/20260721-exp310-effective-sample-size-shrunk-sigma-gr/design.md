# 設計

## アプローチ

exp307のfinite-MAD scaleをraw estimateとして固定し、scale推定の不確実性だけを自己相関由来`n_eff`で表現する。well間priorはprefix残差だけからleave-one-well-outで作り、各wellをlog scale上で縮約する。observation center、欠損補間、state transitionは変更しない。

## 実験範囲

- 対象: `exp310_effective_sample_size_shrunk_sigma_gr`
- Route: `pf_beam`
- 親: `exp307_finite_only_robust_sigma_gr`
- 変更: exp307 `σ_GR`へのsupport/自己相関依存shrinkageだけ。
- 固定: finite pair/MAD定義、fallback、clip、evaluation GR補間、typewell、Gaussian emission、HMM grammar/transition/posterior mean。
- 除外: GR downweight、state noise、affine、Student-t、mixture、row-wise sigma、PF/LikPF/Beam、inference、submission。

## 固定推定式

exp307と同じ有限prefix残差`e_i`を使う。欠損をまたいだ見かけのlag pairを作らず、raw行番号が連続するfinite residual run内だけで自己相関pairを集計する。

- `n`: finite residual数。
- 各lag `k=1..20`で全contiguous runの有効pairを集約し、demeaned Pearson `rho_k`を計算する。pair 20未満またはzero varianceは`rho_k=0`。
- `tau=max(1,1+2*sum_{k=1}^{20} max(rho_k,0))`。
- `n_eff=clip(n/tau,1,n)`。
- `s_w`: exp307 finite-MAD raw scale。20 pair未満はraw fallback 30。
- `s_prior,-w`: 対象wellを除く772 wellsの`clip(s_w,10,60)`中央値。
- `alpha_w=n_eff/(n_eff+50)`。
- `sigma_w=clip(exp(alpha_w*log(s_w)+(1-alpha_w)*log(s_prior,-w)),10,60)`。

hidden test inferenceを将来設計する場合はfull 773 train prefix priorを保存して使うが、本実験ではinferenceを行わない。

## 実行trigger

exp307 scale auditをtruth/error join前に読み、次のORを判定する。

- median `n_eff/n <= 0.5`
- fraction of wells with `n_eff < 50 >= 0.20`

trigger FAILならHMMを開始せず、exp310を`target_free_trigger_fail`として閉じる。trigger後にlag/k/priorを調整しない。

## 検証方法

1. exp307 dependency/raw identity/scale audit SHAをpreflightする。
2. ACF pair count、rho、tau、n_eff、LOO prior、alpha、shrunk sigmaをtruthなしでfreezeする。
3. target-free triggerを判定する。
4. PASS時だけexp307 decoderへshrunk sigmaを渡し、773-well predictionをfreezeする。
5. freeze後にtruth/folds/hidden-like/saved LikPFをlate joinし、overall、support/ACF/shrinkage quintile、distance、by-wellを評価する。

## 実行量

- active variants: trigger PASS時1、FAIL時0
- HMM well-runs: 最大773
- model / LightGBM config / trained fold / PF / Beam / booster: `0 / 0 / 0 / 0 / 0 / 0`
- parent/control再実行: 0
- Kaggle CPU、internet off、8.5時間上限

## 生成物契約

- dependency/input/scientific contract JSON
- ACF/effective sample size/shrinkage audit CSV.gz
- target-free trigger JSON
- trigger PASS時だけprediction/posterior diagnostics CSV.gz
- overall/fold/support/ACF/shrink/distance/hidden-like/by-well metricsとgate summary

## 再現性設計

- RNGなし。contiguous runs、lag 1--20、LOO median、固定式で決定的。
- exp307 scale audit、ACF/n_eff/prior/shrunk sigma、trigger、prediction、metricsのcontent SHAを保存する。
- parent control再実行なし。Kaggle kernel/bootstrap SHAを記録し、model/submission SHAは非該当。

## リスク

- ACFはscale値のbiasではなく推定不確実性に効く。本案はsigmaをtau倍するのではなくshrinkage weightだけへ使う。
- 長いfinite runが少ないwellではrhoが不安定。pair 20未満を0とし、最終的にglobal priorへ縮約する。
- exp240のrow-wise residual scale shrinkageはmixed guardでclosed。本案はraw GRのwell-scalar uncertaintyに限定するが、同様に小さいoverall gainとtail regressionの可能性がある。
- global priorがhidden distributionに合わない可能性がある。本実験はtrain-side gateだけで、PASS後もinferenceを自動許可しない。

## 次のアクション

design-onlyのままexp307結果待ちとする。trigger/PASS後も実装・Kaggle実行は別途承認制とする。
