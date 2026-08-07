# exp511_exp413_transductive_k16_neighbor_rate_postprocess 結果

## 状態

Kaggle private CPU Stage A version 4を完走した。技術契約は全件PASSしたが、固定性能gateの
pooled gainとfold方向性をFAILしたため、同一OOFでparameterを救済せず終端閉鎖する。
inferenceとsubmissionは実装・実行しない。

## 仮説

exp413で一度予測した各wellの`TVT + Z`低周波rateをK16へ縮約し、同じ擬似test batch内の
他well予測だけからexp226型local-linear空間合意を作れば、exp413の細かいpathを保持したまま
cross-wellで不整合な低周波rateだけを安全に弱補正できる。

## 変更点

保存exp413 OOFを再学習せず、同じouter-valid batch内の他well予測だけから作る
K16 neighbor-rate correctionを1本評価した。raw readerは`X/Y/Z`だけを許可し、同fold donor
field、自井戸除外、stable tie-break、prediction freeze後のtruth/hidden-like接続をfail closedで
実装した。Kaggle実行中にraw competition mount、距離weight underflow、exp115 source添付の
3件の技術不備を修正したが、科学parameterとgateは変更していない。

## 設定

- 親: `exp413_scale5_likpf_full_replacement_on_exp335`
- route: `ensemble`
- primary: `transductive_k16_neighbor_rate_a005_cap025` 1本
- K16/local-linear: `K=16 / rho=10 / theta0=118.4 / projection>=0.3 / k=50 / bandwidth=500 / ridge=1`
- support: self除外、unique donor wells `>=8`
- correction: first score row 0、`alpha=0.05`、最終cap `±0.25 ft`
- model / booster / PF/HMM/Beam / GPU / parent retraining: すべて0
- metric: suffix-row unweighted RMSE、保存exp413 outer 5-fold OOF

## 結果

| メトリック | exp413 | exp511 | gain / delta |
| --- | ---: | ---: | ---: |
| pooled CV | 7.884802794405 | 7.883964795206 | `+0.000837999 ft` gain |
| fold 0 | 7.919988324359 | 7.921350854543 | `+0.001362530 ft`悪化 |
| fold 1 | 8.377381332841 | 8.377623038423 | `+0.000241706 ft`悪化 |
| fold 2 | 7.539713352261 | 7.536331800684 | `-0.003381552 ft`改善 |
| fold 3 | 7.574331166543 | 7.574793668873 | `+0.000462502 ft`悪化 |
| fold 4 | 7.982868392834 | 7.979871941814 | `-0.002996451 ft`改善 |

- nonworse folds: `2/5`（必要 `4/5`）。
- fixed scope最大悪化: `+0.000416093 ft`で安全gate PASS。
- by-well delta p95 / worst: `+0.003229925 / +0.204191458 ft`でtail gate PASS。
- first-row最大補正 / 全row最大補正: `0 / 0.25 ft`、row-order変更0、nonfinite 0。
- support成立: `350 / 12,368 segments = 2.8299%`。unique donor wells中央値は5で、
  support下限8を下回るsegmentが大半だった。

## Gate判定

- 技術条件: 全件PASS。
- pooled gain `0.000837999 < 0.01 ft`: FAIL。
- nonworse folds `2 < 4`: FAIL。
- scope / by-well tail / continuity: PASS。
- 最終判定:
  `FAIL_CLOSE_WITHOUT_ALPHA_CLIP_K_BANDWIDTH_RHO_THETA_SUPPORT_FADE_SCOPE_OR_GATE_RESCUE`。

## 再現性

- Kaggle kernel: `kentookumura/exp511-exp413-k16-neighbor-rate-postprocess-train` version 4、
  private CPU、internet off、scientific summary 111.71秒。
- input manifest SHA: `15e223ec87be256d54b0c9503cd47b2fe4eb272af430082bfab6f44bc2454298`
- raw geometry logical SHA: `6c199052a573be334d6334e3e0dddcb7a194e898b13b45e87dc90ae4fc3287cd`
- K16 field logical SHA: `7776e69610920dfb66f83123429b8882cb872efbef8fdbb56bca1b5e7c0a7e1a`
- support ledger logical SHA: `de9c5a9510c615ea048522f449c673867aafa65348e57a321b465160eeeff15d`
- prediction logical / content SHA:
  `5415c24c388ffeee857b2a070d9f2f83db34d5a85ad738f9ee3e57dc17652264` /
  `8320ba95188652749861b7a9ace901def8a6faa0b5e41764b3c87c5bbe0b6fbf`
- prediction freeze SHA: `986e0654d83946ac36a247960638e4bf5002a6f9a98ecb89c63824dc9a806e6e`
- deterministic anchor: いいえ。FAIL閉鎖のため独立rerunは行わない。

## 解釈

小幅な平均改善とtail安全性は得たが、改善量は最低基準の約8.4%にすぎず、改善方向もfold 2/4
だけだった。さらに固定support条件を満たしたsegmentは2.83%で、近傍donor well中央値5、選択
距離中央値4,291 ftだった。予測だけから作る局所K16 consensusは、強いexp413へ安定して移せる
ほど密でも一貫的でもない。support、bandwidth、alpha、cap等の変更は同じOOFへの後付け救済に
なるため行わない。

## 次

exp511はここで完了・終端閉鎖する。inference port、submission、同一familyのparameter rescueは
作成しない。独立した新証拠が必要になった場合は、既存artifactを使うtruth-late原因readoutとして
別仮説・別承認で扱う。
