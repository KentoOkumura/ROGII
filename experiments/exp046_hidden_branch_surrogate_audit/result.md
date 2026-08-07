# exp046_hidden_branch_surrogate_audit 結果

## 仮説

public sample の SHA 一致や `changed_rows=0` は、見えない test well 用処理が安全である証拠にならない。public sample の 3 well は train 由来なので、この処理がそもそも動かないことがある。train well の途中以降を隠した疑似 test rows に見えない test well 用処理を強制適用して、変更量、予測範囲、距離 bucket、層化 fold の危険信号を code submit 前に保存する。

## 設定

- 親: `exp045_public_pf_meta_strict_parity_audit`
- 入力: `exp029_public_sel15_pf_oof_feature_generation` の PF/Beam 生成物。train well の途中以降を隠し、本番 test 風に予測させたもの。
- 監査 split: original-fold、well-hash、stratified-group fold
- reference: `exp026_pseudo_tail_bucket_shrink`
- 監査対象: `pf090_hold010`、`exp033` PF residual、`exp035/045` meta residual

## 結果

Kaggle train version 1 で full audit が完了した。

| メトリック | 値 |
| --- | --- |
| rows / wells | 1,782,279 / 773 |
| original-fold best | 14.313668 (`exp035_ridge_meta_residual_shrink0p75_clip60p0`) |
| well-hash best | 14.172010 (`exp035_ridge_meta_residual_shrink0p75_clip60p0`) |
| stratified-group best | 14.022803 (`exp035_ridge_meta_residual_shrink0p75_clip60p0`) |
| exp033 PF residual original / well-hash / stratified | 14.937393 / 14.844228 / 14.881560 |
| pf090_hold010 all split systems | 15.089532 |
| public PF selector all split systems | 15.172636 |
| 見えない test well 用処理の code submit LB for exp035 candidate | 13.738, exp027 から +4.957 悪化 |
| validation | PASS |
| tests | 10 passed |
| Kaggle train | version 1 COMPLETE |

## 解釈

代理検証では、`exp035_ridge_meta_residual_shrink0p75_clip60p0` が original-fold、well-hash、stratified-group の全てで最良だった。original-fold では exp026 row reference から -1.330065、public PF から -0.858969、`pf090_hold010` から -0.775864 改善した。well-hash と stratified-group でも同じ候補が最良だった。

ただし、この候補は `exp035` の code submit で Public LB 13.738 となり、exp027 8.781 から +4.957 悪化済みである。`exp045` の strict parity でも Public LB 19.177 とさらに悪化した。したがって exp046 の結論は「public sample の `changed_rows=0` blind spot を補う代理監査は作れたが、exp034/035-style の見えない test well 用 meta 処理は本番採点の Public LB に転移しない」である。

exp033 PF residual も代理検証では public PF / `pf090_hold010` を上回るが、実 Public LB は 14.961 で失敗済み。PF/Beam の見えない test well 用処理で次に試すなら、直接残差補正ではなく、保守的な gate / weight 調整に限定する。

## 次

exp027 anchor 8.781 を維持する。exp034/035-style の見えない test well 用 meta 処理と exp033-style residual branch の追加チューニングには進まない。
