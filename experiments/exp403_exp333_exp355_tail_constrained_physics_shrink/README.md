# exp403_exp333_exp355_tail_constrained_physics_shrink

## 状態

- ルート: `ensemble`
- 状態: Kaggle train完走・scientific FAIL・terminal close
- CV / Public LB / Private LB: `8.238331667 / なし / なし`
- 作成日: 2026-07-26
- 親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- branch:
  `exp333_exp226_k16_segment_residual_offset_target` /
  `exp355_exp226_dip_rate_prior_on_exp209`

## 仮説

exp263のK16 50%をexp333、exact-HMM 25%をexp355へ固定置換し、
outer-trainのwell-tail制約を満たす最大scalar λだけをheld-out foldへ
適用すれば、平均改善を維持しながらexp263のtail safetyを保てる。

## 変更点

```text
full = 0.50*exp333 + 0.25*LikPF + 0.25*exp355
candidate = exp263 + lambda_fold*(full-exp263)
```

- λは固定9値。
- outer-validを見ず、他4 reporting foldsだけで選ぶ。
- exp263 generation foldとexp226 reporting foldは別ledger。
- 保存済みprediction以外は再実行しない。

## 検証方針

- Fold / Group: exp226 outer 5 group fold / `well_id`
- Control: exp263 `8.238331546`
- 主gate: pooled `>=0.03 ft`改善、4/5 folds、全scope、by-well p95、
  worst、persistent episode、512-row recoveryの全AND
- Leakage: prediction/formula/SHA freeze後だけraw suffix truthをjoin

## 利用可否

利用不可。Kaggle train version 4はtechnical PASSだったが、5 foldsすべてで
positive eligible λがなく0へfallbackし、promotion FAILとなった。
inferenceとsubmissionは実施しない。

## 所見

- full置換の平均gainは検証価値がある。
- full置換は`8.238332→8.159425`だが、最小λ`1/64`でもouter-train gainは
  `0.005785--0.007919 ft`、by-well p95は`+0.023577--+0.026743 ft`だった。
- positive λは`0/5 folds`、cross-fit CVはcontrolと同じ`8.238331667`。
- prediction content SHAは`17a7bae6...618a`、gate SHAは`094c59ee...592`。

## 次

exp403は閉じる。同一OOFでλ、weight、gate、routerを救済しない。
component別tail原因を調べる場合は、予測生成を伴わない別readoutとして扱う。
