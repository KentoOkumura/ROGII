# exp229_lgb_quantile_band_emission_hmm_on_exp148 結果

## 状態

Kaggle train v3 と HMM audit v3 は完了。train-side で不採用とし、inference / submit は行わない。

## 仮説

LightGBM quantile band を row-wise HMM emission sigma として使えば、exp221 の fixed sigma より予測信頼度を反映できると考えた。

## 実行

- Route: `ensemble`
- Quantile train: exp148 feature surface、`q16/q50/q84`、1 config x 3 alpha x 5 folds = 15 boosters
- HMM: `q50` を emission center、`(q84-q16)/2` を sigma とし、floor/cap は `6/30`
- timeout recovery: 3 lambda の v2 は 12 時間で 652/773 well まで到達したため、部分結果最良の `lambda=0.25` のみを v3 で実行
- HMM audit v3: 3,783,989 rows / 773 wells / 7 features / 0 boosters、elapsed `11,320.273` sec

## 結果

| 候補 | RMSE | exp148 比 | exp193 比 |
| --- | ---: | ---: | ---: |
| exp193 `lgb_mean` | 8.456676 | -0.044615 | 0.000000 |
| exp148 `lgb_mean` | 8.501291 | 0.000000 | +0.044615 |
| exp229 q50 | 8.685006 | +0.183715 | +0.228330 |
| exp229 quantile-band HMM (`lambda=0.25`) | 8.684401 | +0.183110 | +0.227725 |

- HMM は q50 単体から `-0.000605` RMSE とほぼ変化しなかった。
- exp221 fixed-sigma HMM の train-side RMSE `8.327737` より `+0.356664` 悪い。
- corrected central quantile band coverage は `0.525238` で、nominal な中央 68% 区間より明確に低い。
- sigma floor rate は `0.767265`、cap rate は `0.0`。row-wise sigma の大半が floor に張り付き、期待した不確実性の差がほとんど出なかった。
- quantile crossing any rate は `0.038634`。補正自体は動作したが、q50 の点予測性能を補えなかった。

## 解釈

HMM の追加は q50 をほぼ悪化させなかったが、emission center の q50 が exp148 / exp193 の既存 point prediction より弱い。このため、sigma を row-wise にしても既存 anchor を超えなかった。coverage 不足と sigma floor への集中も、quantile band をそのまま信頼度として使えないことを示す。

同じ HMM family では exp221 fixed-sigma が train-side でより良かったものの、Public LB への転移は小さかった。本実験も inference / submit に進める根拠はない。未実行の `lambda=0.50/1.00` は partial run でも劣後しており、追加実行しない。

## 再現性

- quantile train kernel: `kentookumura/exp229-lgb-quantile-band-exp148-train` v3
- HMM audit kernel: `kentookumura/exp229-quantile-band-hmm-exp148-audit` v3
- HMM feature decompressed SHA256: `330d6fc3192b93c26a9ba022486fa5fe0a64d9bd923892f9454e6fad47c55a05`
- HMM audit summary SHA256: `2a620bd72784edbec84d5ec5ad47dd026242a4a33994cb9f36c14ac5c492551f`
- 評価根拠は Kaggle v3 logs。提出物・後続入力は生成しないため output archive は取得していない。

## 次

quantile band 由来 sigma の HMM emission は閉じる。再検討するなら、弱い q50 を center にせず、Public LB 7.843 の現ML anchorである exp218 point OOF を centerとした cross-fitted residual-scale の較正だけを低優先で独立検証する。
