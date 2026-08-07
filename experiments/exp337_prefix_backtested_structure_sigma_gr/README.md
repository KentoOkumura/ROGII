# exp337_prefix_backtested_structure_sigma_gr

## 状態

- ルート: `pf_beam`
- 状態: Stage 0 FAIL・枝を終了
- CV / LB / Submit: なし
- 作成・完了日: 2026-07-22
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 失敗根拠: `exp307_finite_only_robust_sigma_gr`
- Kaggle: version 1 / id_no `128220965` / `COMPLETE`

## 仮説

exp307のfinite-only scaleはGR観測noiseだけを小さく見積もり、typewell不一致やalignment不確実性まで除いてGR emissionを過信した。known prefix後半の予測誤差から構造分散を推定し、

```text
sigma_eff^2 = sigma_finite^2 + tau_structure^2
```

とすれば、欠損GRの0補完に頼らず安全なemission幅を復元できると考えた。

## Stage 0

- origin 60% / 80%以前だけでscaleをfitし、直後20%のfinite residualでGaussian NLLを評価した。
- fit prefixのfinite residualを60% / 40%へ分け、early population varianceとlate zero-center MSE差から`tau_structure^2`を求めた。
- finite pair不足時は同prefixのexp209 zero-fill scaleへfallbackした。
- finite-only、zero-fill、structure-addedを773 wells・保存済み5 foldsで比較した。
- HMM、ML、PF、Beam、booster、親control再実行はすべて0。

## 検証方針

- 保存済み5 foldsをreadout strataに使い、origin以後のknown-prefix blockをfitへ入れない。
- coverage、fallback、finite-only / zero-fill比のpooled・fold NLL、full-prefix `tau_structure`、lower clipを事前固定AND gateで判定する。
- unknown-suffix truthはscaleと判定成果物のfreeze前に読まない。

## 結果

| origin | finite-only NLL | zero-fill NLL | structure-added NLL | structure勝利 vs finite | structure勝利 vs zero-fill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 3.027165 | 3.589239 | 3.073866 | 0/5 | 5/5 |
| 0.80 | 2.971854 | 3.571889 | 3.015784 | 0/5 | 5/5 |

- evaluable coverage: 両origin `773/773`（100%）
- fallback: 両origin `0/773`
- zero-fill比NLL gain: `0.515373` / `0.556105` per residual
- full-prefix median `tau_structure`: `0.0`（gate `>=5.0`をFAIL）
- lower clip: `42/773 = 5.433%`（gate内）
- runtime: `143.899 sec`

## 判断

structure-addedはzero-fillより明確に良かったが、両originでfinite-onlyより悪く、fold勝利も`0/5`だった。さらに典型wellの構造分散は`tau_structure=0`となり、仮説の中心である追加不確実性を確認できなかった。

固定gate不通過として枝を閉じる。split、threshold、scale、likelihoodの救済gridは行わず、Stage 1 HMM、inference、submissionへ進まない。

## 所見

### 良かった点

- coverage 100%、fallback 0で監査自体は安定して完走した。
- structure-addedは広すぎるzero-fill scaleを両origin・5/5 foldsで改善した。

### 悪かった点

- 採用基準のfinite-only比較は両originとも0/5 foldsだった。
- full-prefix median `tau_structure=0.0`で、追加構造分散の中心仮説を支持しなかった。

### 次

- なし。この枝は終了し、同一結果上の救済実験を追加しない。
