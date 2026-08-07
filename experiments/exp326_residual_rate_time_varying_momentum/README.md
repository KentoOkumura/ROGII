# exp326_residual_rate_time_varying_momentum

## 状態

- Route: `pf_beam`
- 状態: 閉鎖済み・未実装・未実行
- 親: `exp323_time_varying_exp226_dip_rate_prior`

親exp323がterminal closeしたため、2026-07-22に本実験も閉鎖した。新exp326相当は、exp338 PASS後に作る新exp323相当がさらにPASSした場合だけ、新番号で設計する。

## 仮説

rateを`r_t=μ_r,t+δr_t`と表した後なら、exp226 priorが大きく変わる場所で`δr`のmomentumを弱め、安定区間で0.998を維持することに物理的意味がある。

```text
s_t = clip(|Δμ_r,t| / (sig_r sqrt(ΔMD)), 0, 4)
L0 = -median(ΔMD) / log(0.998)
L_t = L0 / (1+s_t)
m_t = exp(-ΔMD/L_t)
δr_t = m_t δr_(t-1) + ε_t
```

`sig_r`やprior平均は変えない。

## 段階

- Stage 0: known-prefix backtestで固定momentumよりrate residual NLLを1%以上改善し、4/5 folds、発火1〜50%を要求する。
- Stage 1: 全PASSと別承認後だけ1 variant / 773 HMM runs。

## 検証方針

- Stage 0: 固定momentum比でknown-prefix residual-rate NLLを1%以上改善し、4/5 folds、時間変化の発火率1--50%、tail scope非悪化を要求する。
- Stage 1: 保存済みexp323親HMM比0.05 ft以上、4/5 folds、1000+・hidden-like・p95・worst非悪化を要求する。
- `mu_r,t`、`sig_r`、`delta_md`からmomentum scheduleを決定論的に生成し、suffix truth結合前にSHAを固定する。

## 所見

残差rate座標へ移した後にmomentumだけを変えるため、時間変化priorとの二重計上を避けられる。sigmaやposterior feedbackを同時に変えない設計で、現在は結果なしとする。

本実験の実装、実行、inference、submissionは今後行わない。
