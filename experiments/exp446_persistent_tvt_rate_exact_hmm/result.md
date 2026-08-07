# exp446_persistent_tvt_rate_exact_hmm 結果

## 状態

Kaggle private CPUのfixed32 Stage 0を完走し、`stage0_fail_closed`で終了した。
technical gateはruntime projectionだけFAIL、mechanism gateは7項目すべてFAILした。
fixed32はmechanism preflightであり、CVまたはpromotion evidenceではない。

## 仮説

exp209の持続rateを`U-rate=d(TVT+Z)/dMD`から
`TVT-rate=dTVT/dMD`へ変更し、既知`Z`勾配をrate dynamicsから除くと、
rateの0方向under-responseとforward transition/prior hysteresisを減らせる。

## 実行

- kernel:
  `kentookumura/exp446-persistent-tvt-rate-exact-hmm-train`
- version / id_no: `1 / 129106260`
- private CPU、GPUなし、internetなし
- scientific candidate: 1本
- candidate HMM: 32 wells / 156,088 suffix rows
- 保存exp209 parent HMM rerun: 0
- LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`
- elapsed / peak RSS: `1,928.728804 sec / 1.131355 GB`
- CV / Public LB / Private LB: なし / なし / なし

## Gate結果

### Technical

`17 / 18 PASS`。

- finite coverage: `1.0`
- posterior normalization max error: `3.642e-14`
- constant-Z parent parity: prediction/posterior max error `0.0`
- small-state dense reference: prediction max error `4.042e-09`
- position edge residual: `0.0 ft`
- truth/role/fold/episode read before freeze: `0`
- peak RSS: `1.131355 <= 25 GB`
- Stage 1 runtime projection:
  `46,590.855 > 30,600 sec`、FAIL

### Mechanism

`0 / 7 PASS`。

| Gate | 観測値 | 閾値 | 判定 |
| --- | ---: | ---: | --- |
| zero-directed under-response SSE share削減 | `-0.061091` | `>= 0.05` | FAIL |
| forward-cause episode SSE削減率 | `-0.306441` | `>= 0.10` | FAIL |
| persistent episode SSE削減率 | `-0.214831` | `>= 0.05` | FAIL |
| persistent改善well | `5 / 16` | `>= 10 / 16` | FAIL |
| persistent改善fold | `2 / 5` | `>= 4 / 5` | FAIL |
| matched control pooled RMSE delta | `+7.159063 ft` | `<= +0.02 ft` | FAIL |
| matched control by-well delta p95 | `+16.310622 ft` | `<= +0.25 ft` | FAIL |

zero-directed under-response SSE shareはparent `0.242821`からcandidate
`0.303912`へ増え、絶対`0.061091`悪化した。persistent episodeはfold 2と4だけ
改善し、fold 0/1/3で悪化した。

## 再現性

- 初回runのためdeterministic anchorにはしない。
- scientific contract SHA:
  `99ab27aa50ecc38a20f10ae39d8709f55bba3323c28fe9c2b036b2cf417659f1`
- transition / posterior / prediction manifest SHA:
  `5813d4de...49cc3 / 34c94c88...407d9 / 35581c23...931e`
- prediction decompressed SHA:
  `72c40bd73b71e469b49abdb25b0eb3048150b37e3f9624a94787338ad9eb3634`
- diagnostic decompressed SHA:
  `4ad24705604b3895278910ca875b9240fc4f7a2766a0cf6eecc2472e8ad5b4dc`
- package notebook / metadata / pushed config SHA:
  `f0a31dac...631c / 722cff9f...bb3 / 22962bdd...69a`
- model / submission SHA: 対象外

## 解釈と判断

TVT-rateを持続させても、known-Z forcingをrate dynamicsから外すと
exp209のcontrol安全性を大きく失い、狙ったunder-responseとforward/persistent
hysteresisも悪化した。exp435のmemoryless TVT-only失敗とは異なる構造でも、
既知幾何を外すという共通リスクが実データで再現した。

事前契約どおり、rate定義、span、momentum、noise、grid、emission、prior、
gate、blend、selectorによるsame-fixed32救済を行わずbranchを閉じる。
Stage 1、rerun、inference、submissionは実行しない。
