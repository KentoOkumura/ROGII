# exp323_time_varying_exp226_dip_rate_prior

## 状態

- Route: `pf_beam`
- 状態: 閉鎖済み・未実装・未実行
- 親: `exp309_well_adaptive_transition_noise`
- CV / LB / submission: なし / なし / なし

旧exp307/308/309 lineageが成立しないため、2026-07-22にterminal closeした。parent差し替えや再開は行わない。exp338が全gateをPASSした場合だけ、exp338を親に同じ役割の実験を新しい番号で設計する。

## 仮説

exp226のfold-safe K16 geometry fieldが表す区間別dip-rate変化をexact HMMの遷移平均へ入れると、自well prefixの絶対rateを維持しながら、一定rateの累積driftを減らせる。

```text
r_geo,t = Δ(TVT_geop + Z) / ΔMD
μ_r,t = r_parent + (r_geo,t - r_geo,first)
r_t = μ_r,t + δr_t
```

exp226の絶対TVTや最終予測は使わない。対象wellの絶対rateは親HMMの初期rateに固定し、exp226からは時間変化分だけを移植する。

## 固定した段階

1. Stage 0: HMMを走らせず、凍結した`r_geo,t-r_geo,first`が真の区間rate変化と累積経路を改善するか監査する。
2. Stage 1: Stage 0全gate PASSと別承認後だけ、残差rate状態のexact HMMを1 variant実装する。

Stage 0は1 diagnostic / HMM 0、Stage 1最大1 variant / 773 HMM well-runs、model・booster 0。親controlは保存済み結果を使い再実行しない。

## 検証方針

- Stage 0: 定数prior比でsegment rate-change RMSEを5%以上、累積経路RMSEを0.05 ft以上改善し、4/5 folds、1000+・hidden-like 2面・worst well `<=+0.25 ft`をすべて満たす。
- Stage 1: Stage 0 PASS時だけ、保存済み親HMM比0.05 ft以上、4/5 folds、hard-tail非悪化を1 variantで確認する。
- geometry fieldとscheduleをsuffix truthの結合前に凍結し、fold/segment/donor/fallback/content SHAを保存する。

## 所見

絶対rateをexp226から借りず、親のwell別rateへanchorして時間差分だけを移植するため、dip-rate priorの時間変化を最も直接に検証できる設計である。現在は設計確定のみで結果はない。

## 禁止事項

- exp226最終予測、GR補正、U projection、absolute unary、固定shape、blendを使わない。
- suffix truthでdonor、fallback、segment、weightを選ばない。
- Stage 0 FAIL後のwindow、weight、gate、grid救済を行わない。
- 実装、Kaggle実行、inference、submissionは別途承認制。
