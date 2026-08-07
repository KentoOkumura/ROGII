# exp443_mean_preserving_trapezoidal_lattice_hmm

## 状態

- ルート: `pf_beam`
- 状態: Kaggle Stage 0 version 1完了、`stage0_fail_closed`
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-29
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

rateが行間で変化する場合、TVT変位をdestination rateだけでなく
`0.5*(source+destination)`で積分し、連続平均を格子上で厳密保存すると、
position積分・量子化由来のsigned biasを減らせる。

exp439で実現不能だった親varianceを再主張せず、

```text
effective variance = max(parent target variance, lattice minimum variance)
```

を明示的な別表現として固定した。平均は厳密保存し、余分な格子分散は監査する。

## 検証方針

fixed32の32 wellsをmechanism preflightとして実行し、数値contract、runtime、
persistent/control安全性を事前固定AND gateで判定した。これはCVやpromotion
evidenceではない。

### Stage 0結果

- private CPU version 1、32/32 HMM wells、5 reporting foldsを完走。
- mean / effective variance / rate marginal / nonnegative weight / brute-force /
  truth-late / SHA / RSSはPASS。
- runtime projectionは`125,406.237秒`で上限`30,600秒`をFAIL。
- forward-cause SSEは5.517%改善したが必要10%未満。
- persistent SSEは5.766%悪化。10/16 wells、4/5 folds改善でもfold 0の悪化が支配。
- control pooled RMSE delta `+0.093698 ft`、by-well p95 `+1.394368 ft`でFAIL。
- variance-floor active edgeは9,665,508、inflation mean/maxは
  `0.003905 / 0.015619 ft²`。

## 所見

台形平均の数値表現は成立し、one-step grid mean biasもほぼ完全に除去した。
しかし追加されたlattice varianceがcontrol安全性とpersistent pooled SSEを損ね、
full runtimeも非実用的だった。事前登録どおり同一fixed32でのparameter/gate/
blend/selector救済は行わず、Stage 1、inference、submissionへ進まない。

詳細は`result.md`、`SESSION_NOTES.md`、`artifacts/kaggle_v1/metrics.json`を参照する。
