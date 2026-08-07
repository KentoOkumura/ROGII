# exp441_full_support_ou_rate_transition_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 1を完走し、固定AND gateで
`stage0_fail_closed`。Stage 1、inference、submissionへ進まない。
fixed32はmechanism preflightであり、CV / LBはない。

## 仮説

exp209のtri-diagonal rate kernelを、同じ`momentum`と`sig_r`から定まる
全support exact OU kernelへ置換すると、1行1binの人工的な伝播制約を除き、
forward transition/prior hysteresisとrate under-responseを減らせる。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- 変更: rate transitionのみ
- Stage 0: 1候補×fixed32、保存control rerun 0
- Stage 1: Stage 0全PASS・別承認時だけ773 wells
- model / booster / PF / Beam / GPU: すべて0

## 結果

| メトリック | 値 |
| --- | --- |
| Kaggle kernel | `kentookumura/exp441-full-support-ou-rate-transition-hmm-train` v1 |
| 状態 | `KernelWorkerStatus.COMPLETE` |
| Stage 0 | `stage0_fail_closed` |
| 行 / well | 156,088 / 32 |
| technical gate | 16 / 17 PASS |
| mechanism gate | 2 / 7 PASS |
| 実行時間 / peak RSS | 1,582.080秒 / 1.123249 GB |
| full 773-well換算 | 38,217.120秒（上限30,600秒、FAIL） |
| under-response SSE share | parent 0.683973 → candidate 0.660999、削減0.022974（必要0.05） |
| forward-cause episode SSE削減 | -0.001635（必要+0.10） |
| persistent episode SSE削減 | -0.016743（必要+0.05） |
| persistent改善 | 8 / 16 wells、1 / 5 folds（必要10 / 16、4 / 5） |
| matched-control pooled RMSE delta | -0.061891 ft（PASS） |
| matched-control by-well delta p95 | +0.037121 ft（PASS） |
| CV / LB | - / - |

## 解釈

exact OU kernel自体は、analytic mass/moment、position parity、dense
brute-force、normalization、truth-late、SHA readbackを全てPASSした。
一方、全support化でzero-directed under-response shareは2.297 pointsしか
下がらず、forward-causeとpersistent episode SSEはわずかに悪化した。
control安全性は保ったが、persistent改善は8 wells / 1 foldに留まり、
仮説の中心だった追従遅れ回復を再現しなかった。

したがって、人工的な1行1bin制限を除くだけでは主要なpersistent lagを
解消できないと判断する。full-support化の計算量もfull換算上限を超えた。
OU parameter、`sig_r`、momentum、support、emission、grid、gateを
同じfixed32で救済しない。

prediction decompressed SHA:
`063ff78b6a5e352681391bf37c1eecec2f841fe477629afa70b36b2065f13c92`。
combined transition / prediction / diagnostic manifest SHA:
`6448b4e8a74f0bd4f670e3c8a1fe872b42f88d1bfe635f8bc57ade66765efc4b` /
`d7bbd2fe08957564575da25f2aa2297170cd3c6be39bfec15608ce122ab96511` /
`331ee69054769d631d5aeebfd918135eee7d08c08ef9c576704db99490665fcf`。

## 次

exp441をterminal closeとし、Stage 1、rerun、inference、submissionへ進まない。
次に原因を掘る場合も、新規HMMを回さず、保存済みrate diagnosticだけを使う
低優先のtruth-late失敗原因readoutを別実験・別承認で行う。
