# exp368_marginalized_reliability_pf 結果

## 状態

Kaggle private CPU Stage 0 version 1を完了した。technical gateはPASS、
scientific gateはFAILし、decisionは`stage_0_failed_close_without_rescue`である。
Stage 1 PF、inference、submissionは行わない。

## 仮説

sticky reliabilityを粒子ごとに周辺化すれば、追加粒子なしでGR観測過信を
抑えられる。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- route: `pf_beam`
- Stage 0: known-prefix predictive NLL + saved exp072 path bad-block AUC
- q: normal sigma 1倍 / weak sigma 4倍のexact forward recursion
- 対象: 3,783,989 rows / 773 wells / 15,174 suffix blocks
- known-prefix: 49,472 held-out rows
- 実行量: 1 diagnostic / 5 reporting folds
- PF run / PF control / model / LightGBM / trained fold / booster / 親control再実行:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`
- kernel: `kentookumura/exp368-marginalized-reliability-pf-train`
  version 1、id_no `128591117`
- runtime: `630.531264 sec`

## 結果

| メトリック | 値 | 固定gate | 判定 |
| --- | ---: | ---: | --- |
| known-prefix predictive NLL gain | 0.037356% | 1%以上 | FAIL |
| pooled real bad10 AUC | 0.636675 | 0.60以上 | PASS |
| circular bad10 AUC | 0.578411 | - | - |
| real - circular AUC | +0.058264 | +0.02以上 | PASS |
| AUC > 0.50 folds | 5/5 | 4/5以上 | PASS |
| hidden-like spatial AUC | 0.641795 | 0.55以上 | PASS |
| hidden-like typewell-purged AUC | 0.636115 | 0.55以上 | PASS |
| row-weighted weak mass | 0.009689 | 0.02--0.50 | FAIL |
| Q1 mean block RMSE | 5.000769 ft | - | - |
| Q4 mean block RMSE | 10.978203 ft | - | - |
| Q4 - Q1 mean block RMSE | +5.977434 ft | - | - |
| Public / Private LB | - / - | - | - |

fold別real bad10 AUCは`0.627147 / 0.627234 / 0.633846 / 0.653398 /
0.639133`で、全foldが0.50を上回った。

known-prefix base NLLは`225678.675138`、marginalized NLLは
`225594.369615`で、絶対差は改善したが相対改善は`0.000373564`に留まり、
事前下限`0.01`を満たさなかった。

## Technical gate

PASSした。

- expected / observed rows、wells、foldsが一致した。
- known-prefixは773 wellsすべてで64 held-out rows、合計49,472 rows。
- 全score finite、weak scoreは`[0,1]`内。
- truth columns read before freezeは0。
- multi-block circular offsetはすべて非ゼロ、Q1/Q4境界はstrict。
- PF、model、booster、親control再実行はすべて0。

## Scientific gate

FAILした。

- PASS: pooled bad10 AUC、circular差、5/5 fold AUC、hidden-like 2面。
- FAIL: known-prefix predictive NLL gain、weak mass下限。
- Stage 1 eligibilityはfalse。

## 再現性

- deterministic anchor: no（固定入力のdeterministic diagnosticでありsubmission
  anchorではない）
- RNG: なし、CPU single worker
- scientific contract content SHA:
  `dd333d921e377447f1f4c1c49c77bd852122eab059e1532ba9a2f22013ba1314`
- block ledger content SHA:
  `7327ce8e6383d76f99c51cec6982c1db181e6f05257df28e7268d7a0549ba30a`
- known-prefix NLL content SHA:
  `eeb5d7981a8926753a20435b6b816eeb6548877f0171be18f3113069f07a2811`
- weak posterior content SHA:
  `4ffa4fc761fc4db6b1c7de42c132b8102e33f9910bf5dc56752b20e95c2520ae`
- late-truth block readout content SHA:
  `5f90ed658c09c2dc54f52a617f1f2467c46939cd7c955828463129bc7d611189`
- gate raw SHA:
  `bb2e83cbcecdefa9c195987f18d1fa3b58d81bcf33f1f11c6f2ec21dd5d53e48`
- downloaded summary raw SHA:
  `fcf0a17d31ae242fb6bf74bfdf333152ad40c30b22bbe9ab14bc63bf1a7650ae`
- prediction / model / submission SHA: 非該当・未生成

## 解釈

suffixではweak posteriorがbad blockを安定して識別し、全foldとhidden-like 2面へ
転送した。しかしposterior massは約0.97%しかなく、known-prefixの逐次予測尤度を
実質的に改善していない。固定qとsigma 4倍のnormalized Gaussianは「誤差区間を
見つけるreadout」としてはsignalを持つ一方、PFの観測尤度へ組み込むだけの頻度と
予測効用が不足している。

これは技術失敗ではなく科学仮説のFAILである。同じOOF上でtransition、sigma倍率、
block、threshold、gateを調整して救済しない。

## 次

branchを閉じる。Stage 1 PFは実装・実行せず、inferenceとsubmissionも行わない。
再訪する場合は、固定qの調整ではなく、known-prefixとsaved suffixでweak activationが
乖離する原因をtruth-freeで監査する独立仮説を先に要求する。
