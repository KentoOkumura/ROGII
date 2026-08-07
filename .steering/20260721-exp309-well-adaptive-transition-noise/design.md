# 設計

## アプローチ

exp308の観測モデルを固定し、known prefix `U=TVT_input+Z`からrate diffusionだけを推定する。current init rateは末尾medianとして引き続きwell別に使い、本案はその周囲でrateがどれだけ変化できるかだけを適応させる。

## 実験範囲

- 対象: `exp309_well_adaptive_transition_noise`
- Route: `pf_beam`
- 親: `exp308_imputed_gr_confidence_downweight`
- 変更: HMM `sig_r`のwell別値だけ。
- 固定: exp307 `σ_GR`、exp308 missing confidence、GR補間、typewell、grid 0.35、41 rates、rate span 0.10、`sig_p=0.02`、position floor、`mom=0.998`、start prior、posterior mean。
- 除外: state数/range/center、sig_p、momentum、GR側、PF/LikPF/Beam、inference、submission。

## 固定推定式

known prefixの連続行で`ΔMD_i>0`かつ`TVT_input/Z`がfiniteなpairから

`q_i = Δ(TVT_input+Z)_i / ΔMD_i`

を作る。連続するrate pairについて`h_i=max(0.5*(ΔMD_i+ΔMD_{i-1}),1)`、

`a_i=(q_i-q_{i-1})/sqrt(h_i)`

とし、`s_w=1.4826*median(|a_i-median(a)|)`をraw diffusion scaleとする。

- valid `a_i`数を`n_w`とする。
- `alpha_w=n_w/(n_w+100)`。
- `sig_r_w=clip(exp(alpha_w*log(max(s_w,1e-6))+(1-alpha_w)*log(0.002)),0.001,0.004)`。
- `n_w<20`またはnonfiniteなら`sig_r_w=0.002`。

`sig_p=0.02`はexp209 kernelで`max(sig_p,0.35*step)=0.1225`に置換されるため、通常範囲のwell適応は実質無効である。position floorまで変えると別仮説になるため本実験では固定する。

## 依存gate

- exp308が全promotion gateをPASSしている。
- exp308 scientific contract、input、prediction expected SHAは実行前にconfigへ固定する。先行実装中は明示的なpending値とrun guardを置く。
- dependency FAIL/SHA mismatch時はHMMを開始しない。

## 検証方法

1. dependency/raw identityをpreflightする。
2. truthなしでq/a/support/raw scale/alpha/final sig_r/fallback/clipをfreezeする。
3. parent decoderでsig_rだけをwell別値へ置換し、773-well predictionをfreezeする。
4. freeze後にtruth/folds/hidden-like/saved LikPFをlate joinする。
5. overallに加えsig_r quintile、support、trajectory turning、distance、by-wellをreadoutする。

## 実行量

- active variants: 1
- HMM well-runs: 773
- model / LightGBM config / trained fold / PF / Beam / booster: `0 / 0 / 0 / 0 / 0 / 0`
- parent/control再実行: 0
- Kaggle CPU、internet off、8.5時間上限

## 生成物契約

- dependency/input/scientific contract JSON
- well別transition-scale audit CSV.gz
- adaptive prediction/posterior diagnostics CSV.gz
- overall/fold/sig-r/support/turning/distance/hidden-like/by-well metrics
- fixed blend readoutとgate summary

## 再現性設計

- RNGなし。rate statisticsとshrinkageは固定式で決定的。
- dependency、q/a summary、sig_r、prediction、metricsのcontent SHAを保存する。
- parent controlはexpected SHA固定で再実行しない。
- Kaggle kernel/bootstrap source/config SHAを記録する。model/submission SHAは非該当。

## リスク

- prefix rate diffusionがsuffixのfault/turningを表さない可能性がある。
- qの微分は測定丸めに敏感でMADが0またはclipへ寄る可能性がある。fallback/clip率をtechnical gateにする。
- `sig_r`を広げるとwrong GR mode間移動も増える。worst/p95/hidden-like guardを必須にする。
- exp273の2D formation gradient transitionはnegativeだった。本案は方向gradientを加えずscalar diffusionだけを変えるが、transition変更全般のriskは高い。

## 次のアクション

self-contained実装と静的検証は完了。exp308 PASS後にstatus、prediction SHA、parent/direct/blend metricsを固定し、別途承認後だけKaggle CPU実行へ進む。
