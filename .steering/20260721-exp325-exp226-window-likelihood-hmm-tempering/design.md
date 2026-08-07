# 設計

> **閉鎖済み（2026-07-22）**: 本文は旧lineageの設計履歴であり、実装入口ではない。後継資格はexp338の`successor_policy`を正とする。

## アプローチ

親の時間変化rate priorからwindow内relative pathを固定し、各候補center TVTに対してexp226 correlation/MSE/level scoreを計算する。state方向へ標準化し、`stride/window`とexp226 posterior-SD shrinkを掛け、window centerにだけ疎なlog potentialを追加する。

## 実験範囲

- 対象: `exp325_exp226_window_likelihood_hmm_tempering`
- Route: `pf_beam`
- 親: `exp323_time_varying_exp226_dip_rate_prior`
- 変更: sparse window observation factorと`lambda_t`だけ。
- 固定: row Gaussian emission、transition、grid、momentum、posterior mean。

## 再現性設計

real scoreはRNGなし、shuffleだけstable local RNG。window/profile/state score/lambda scheduleをfreezeし、その後だけtruth rankを評価する。CPU/internet off、0 booster。metadata/bootstrapとcontent SHAを確認する。

## リスク

- row GRとの二重計上: factorをstride centerだけに置きoverlap 0.25で正規化する。
- repeated mode過信: posterior-SD shrinkとshuffle/hidden-like gateを要求する。
- runtime: Stage 0で反証してから773 HMM runsを許可する。
