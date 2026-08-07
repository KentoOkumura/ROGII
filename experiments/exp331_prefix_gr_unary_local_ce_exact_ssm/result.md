# exp331_prefix_gr_unary_local_ce_exact_ssm 結果

## 状態

Stage 0はcompute gateをPASSしたが、Stage A fold 0は科学gateをFAILした。事前契約どおりStage B、推論、提出へ進まず、exp331をrescue gridなしで閉じる。

## 仮説

row-independent local CEでexp295のGR unaryを学習し、model freeze後だけfixed exp209 exact SSMを使えば、structured training timeoutを避けながらcomplete-well alignment品質を回復できると仮定した。

## 変更点

exp295のencoder、prefix conditioning、input、decoderを固定し、training/early stoppingのstructured DPだけをhard nearest-state local CEへ置換した。real/shuffle/geometryは学習済みの同一modelをfreezeした後だけexact decodeした。

## 実行契約

- kernel: `kentookumura/exp331-prefix-gr-unary-local-ce-exact-ssm-stage-a` version 1 / `COMPLETE`
- accelerator: Nvidia Tesla T4
- 規模: 1 architecture × fold 0 × seed 42 = 1 neural model
- LightGBM config / booster / PF-Beam / parent-control再学習: `0 / 0 / 0 / 0`
- fold: fit 556 wells / early stop 62 wells / valid 155 wells
- 学習: 8 epochs完了、epoch 7を選択、best early-stop local CE `4.844269`

## Stage A結果

| メトリック | exp331 real GR | 比較対象 | 判定 |
| --- | ---: | ---: | --- |
| fold 0 RMSE | 24.760360 | exp209 12.671087 | FAIL（+12.089273 ft） |
| geometry-only RMSE | 24.760360 | 32.465002 | PASS（7.704642 ft改善） |
| circular-shuffle RMSE | 24.760360 | 57.878820 | PASS（33.118460 ft改善） |
| true-state NLL | 22.972018 | shuffle 27.289439 | PASS（4.317422改善） |
| within10 mass | 0.532114 | shuffle 0.103201 | PASS（+0.428913） |
| well RMSE p95 | 44.560719 | exp209 26.301518 | FAIL（+18.259200 ft） |
| worst-well regression | +63.109520 ft | 上限 +10 ft | FAIL |
| target-in-grid / finite coverage | 1.0 / 1.0 | 下限 0.995 / 1.0 | PASS |
| prefix clamp max error | 0.0 ft | 上限 1e-6 ft | PASS |
| runtime / peak GPU | 4.115497 h / 1.889884 GB | 上限 8.5 h / 14 GB | PASS |

長距離・hidden-likeでもexp209へ大きく回帰した。

| scope | exp331 | exp209 | 差 |
| --- | ---: | ---: | ---: |
| distance 1000+ | 26.016535 | 13.878414 | +12.138121 ft |
| hidden-like spatial | 25.234291 | 12.761284 | +12.473007 ft |
| hidden-like typewell-purged | 24.169723 | 12.046808 | +12.122914 ft |

155 wells中、exp209より改善したのは17 wells、悪化は138 wellsで、well別delta中央値は`+9.831584 ft`だった。geometry-onlyより改善したwellは112、shuffleより改善したwellは128だった。

## Gate判定

PASSはfinite prediction、runtime、memory、prefix clamp、real-vs-shuffle NLL、real-vs-geometry RMSE、real-vs-shuffle within10、target-in-grid。FAILは次の3件だった。

- `real_rmse_vs_exp209`
- `well_p95_non_regression`
- `worst_well_regression`

総合判定は`stage_a_failed_branch_closed`、次動作は`close_stage_b_without_exp331_rescue_grid`である。

## 再現性・leakage監査

- outer-valid truth access before freeze: 0
- hidden-like assignment: prediction freeze後に読込
- forbidden neighbor sources: 0
- summary SHA: `273e51100babeab3554a56d8853e345b1d993ced3c0d23d9568cf60e07dd3356`
- model SHA: `e9cd7404eabf9192a3026184bdffb2de3f585861aa8b0dee2edd777d542fc61b`
- frozen prediction gzip SHA: `120ae26e25d7d598c63b1faa847f869a5441a97853966ce1f8d0ff165e4a82af`
- frozen prediction decompressed SHA: `4fdee845e7c3c2605bc0901db5755de09b599f8f2ff5542b473486a71c86b844`
- input manifest SHA: `1caa1050421377e573df4a70a5011e01fa7599fe00770a0a4cd223d74bedabc2`
- emission/posterior manifest SHA: `14dd9e2ecc7874b1ca8f21d3588fe72b6883200bf31ebc00617edea1918c9dc9`

取得した全artifactについてsummary記載SHAと実ファイルSHAが一致し、frozen predictionの展開後SHA、model SHA、manifest SHAも一致した。

## 解釈

real GRがshuffleとgeometry-onlyを明確に上回ったため、encoder/unaryはGR対応信号を学習している。しかしexp209よりRMSEが約12.09 ft悪く、138/155 wellsで回帰し、tailも大幅に悪化した。したがって「local CEだけで学習したrow unaryをfixed exact SSMへ渡せば、実用的なwhole-well alignment品質になる」という主仮説は不支持である。計算量の問題は解消したが、row-local目的とglobal path品質の隔たりが残ったと判断する。

## 次

exp331内のarchitecture、loss、band、temperature、view、epoch救済は行わない。Stage B、推論、提出は閉鎖する。代替設計exp332はexp331 closeという先行条件だけ成立したが、着手には別のユーザー判断が必要である。
