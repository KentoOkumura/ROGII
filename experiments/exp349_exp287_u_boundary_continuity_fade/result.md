# exp349_exp287_u_boundary_continuity_fade 結果

## 状態

Kaggle private CPU Stage 0 version 2完了。全technical gateとscientific gate 11/12件はPASSしたが、pooled改善量が事前下限に届かず、`FAIL_CLOSE_NO_RESCUE`で終端した。raw-test inferenceとsubmissionは実行していない。

## 仮説と設定

- 親: `exp287_fold_safe_formation_74_addonly_on_exp264`
- Route: `ml_model`
- 検証: 保存済みouter 5-fold OOFへのtarget-free deterministic postprocess audit
- variant: `u_cap8_tau240_always_on` 1件
- 式: `move(d) = -clip(gap_U, -8, 8) * exp(-d / 240)`
- 実行量: 5 reporting folds、model/config/booster/PF/Beam/HMM/control再学習/GPUすべて0
- 行数: 3,783,989 rows / 773 wells

## 結果

| 指標 | 親 | 候補 | 親−候補改善 |
| --- | ---: | ---: | ---: |
| pooled RMSE | 8.136708220 | 8.135096925 | 0.001611295 ft |
| 0--240 MD-ft RMSE | 1.505570413 | 1.395566636 | 0.110003778 ft |
| hidden-like spatial RMSE | 8.799767731 | 8.797668985 | 0.002098746 ft |
| hidden-like typewell-purged RMSE | 8.735403758 | 8.733164055 | 0.002239703 ft |

fold別には5/5で改善した。

| Fold | 親RMSE | 候補RMSE | 改善 |
| --- | ---: | ---: | ---: |
| 0 | 8.070368347 | 8.069360449 | 0.001007899 ft |
| 1 | 8.255838432 | 8.254177476 | 0.001660956 ft |
| 2 | 7.893630011 | 7.891794846 | 0.001835165 ft |
| 3 | 8.106566731 | 8.105078236 | 0.001488494 ft |
| 4 | 8.349625947 | 8.347570193 | 0.002055754 ft |

距離別改善は0--64 / 64--128 / 128--240 / 240--480 / 480--1000 / 1000+で、それぞれ`0.305682 / 0.135888 / 0.068317 / 0.027269 / 0.002901 / 0.000002 ft`だった。by-well deltaのmedianは`-0.000363 ft`、p95は`+0.010451 ft`、worstは`+0.063651 ft`。最大補正絶対値は`6.282791 ft`だった。

## Gate判定

- Technical: PASS。親OOF/model manifest SHA、row/well/ID/CV parity、全well prefix/suffix、raw/OOF alignment、finite、式一致、8 ft cap、単調fade、first-hidden gap非増加、truth-before-freeze 0、SHA readbackを確認した。
- Scientific: 11/12 PASS。改善fold、0--240、遠方3帯、hidden-like 2面、by-well median/p95/worst、大幅悪化well数はPASSした。
- FAIL: pooled改善`0.001611295 ft`が事前下限`0.020 ft`を満たさなかった。

## 再現性

- Kaggle kernel: `kentookumura/exp349-exp287-u-boundary-continuity-fade-train` version 2、id_no `128239658`
- core result / notebook total: 約`156.134 / 165.895 sec`
- executed config SHA: `711b12f7ca7ffc88a6746477329480ce96247316f91cb73550412890ebc37927`
- parent OOF SHA: `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- parent model manifest SHA: `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- Stage A freeze manifest SHA: `a1070b4a7f9bcaaa1d973bd10719ee2fe9daf29dce74899e0c67e5f0f2a64d99`
- target-free candidate SHA: `adf51375d0d9b676dd9b7610528da43a862801632eed3afd61f340b3639fcbce`
- pretruth diagnostic SHA: `b6be5278cafed4cfa5ed51ca754bd58321ac1ffbe0007e5aa847977468e19ef0`
- Kaggle metrics / decision SHA: `9217bac22f5cb053a1e343402dffe8af1cb6e1eb20e4ba2891737f65e3a920b8` / `cbd821a4ae57bcd8cd0d5516d9661e306164f38ad48746ef8a83f8b88431362a`
- 小規模metrics/manifestsを選択取得し、reproducibility manifest記録SHAと全件一致を確認した。大きなcandidate parquet自体は取得していない。
- version 1はKaggle pandas返り値型差によりfreeze前に停止。型互換だけを修正し、仮説・入力・cap/tau・gateを変えずversion 2を実行した。

## 解釈

固定U境界fadeは境界直後には明確に効き、全foldとhidden-like 2面でも悪化しなかった。一方、未知suffixの大半を占める1000+では実質無変化で、pooled改善は必要量の約8.1%に留まった。危険な補正というより、現行anchorを更新するほどの総量がない補正である。

同じOOFを見てcap、tau、threshold、距離範囲、well gate、blend、親を調整するのは事前契約上のpost-hoc救済になるため行わない。公開Notebook全体の低Public scoreを、この単独U補正の昇格根拠にも使わない。

## 次

direct fixed U-boundary fade branchを閉じる。inference・submissionへ進めない。continuityを再訪する場合は、exp349の補正値を救済するのではなく、独立したtarget-free signalをadd-only featureまたはselector補助として事前設計した別仮説に限る。
