# exp435_tvt_memoryless_u_rate_dzonly_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 1（id_no `129049294`）を完了した。
technical gateは全PASSしたが、`memoryless_41rate`と`dz_only_r0`は
mechanism AND gateをともにFAILした。decisionは
`stage0_fail_closed_all_variants`。fixed32はmechanism-onlyでありCVではない。

## 仮説

TVTだけを持続状態とし、41個のU-rate候補を毎行独立に周辺化すれば、
非ゼロrate supportを残しながらexp209のrate履歴によるforward hysteresisを除ける。
同じkernelを`r_U=0`へ固定したdz-onlyとの比較により、非ゼロrate support自体の
必要性を分離する。

## 固定設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- treatments:
  `memoryless_41rate`, `dz_only_r0`
- Stage 0:
  fixed32、2 treatments × 32 wells = 64 HMM well-runs
- parent rerun:
  0
- model / LightGBM config / trained fold / booster / PF / Beam / GPU:
  すべて0
- rate weight:
  zero-centered parent-AR stationary distribution
- dz-only:
  同一kernelの`rates=[0]`, `weights=[1]`
- truth:
  全prediction / diagnostic freeze後にだけjoin

## 実行結果

| 項目 | 値 |
| --- | ---: |
| kernel | version 1 / id_no `129049294` / COMPLETE |
| Stage 0 elapsed | `46.077013096 sec` |
| peak RSS | `0.455474854 GB` |
| wells / suffix rows | `32 / 156,088` |
| Stage 1最大runtime投影 | `379.764049737 sec` |
| finite coverage | `1.0` |
| transition row-sum max error | `4.440892099e-16` |
| posterior normalization max error | `3.330669074e-15` |
| dz parity max abs | `0.0 ft` |
| truth / role-fold / episode pre-freeze reads | `0 / 0 / 0` |
| technical gate | 全PASS |

## Variant別mechanism gate

| variant | forward-cause SSE削減 | persistent SSE削減 | 改善well | 改善fold | control pooled delta | control p95 delta | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `memoryless_41rate` | `27.205050%` PASS | `11.244859%` PASS | `4/16` FAIL | `1/5` FAIL | `+16.151527 ft` FAIL | `+29.129905 ft` FAIL | FAIL |
| `dz_only_r0` | `43.429062%` PASS | `21.835503%` PASS | `5/16` FAIL | `1/5` FAIL | `+13.705216 ft` FAIL | `+24.955652 ft` FAIL | FAIL |

matched-control parent RMSEは`3.428436286 ft`、candidate RMSEは
memoryless / dz-onlyで`19.579963225 / 17.133652291 ft`だった。
両variantともepisode集約SSEは改善したが、改善は1 foldへ集中し、
persistent wellの多数とmatched controlを大幅に悪化させた。

## 再現性

- seed policy:
  RNGなし、固定well / row / TVT grid / rate / variant順
- scientific contract SHA:
  `92f3e307007fa1dc94bd4921f519aa01267f044c0874b31d6581a61a7a356a63`
- input manifest SHA:
  `53a918ba6b7b7fb535cc9358a6402b4e9347bee12d0329d81fe2ed70b05e7950`
- prediction logical/readback SHA:
  `aa79810f6c189dd7fbb9d53b8c172a4a051d29ac1780ee4696237e8c24e214c3`
- rate readout logical/readback SHA:
  `1a554d22071e4d9210808ccbbd6f326257fa5e6265b4e117d4116c4c394f0495`
- well metrics / episode readout SHA:
  `33abe461c48170a21d44084c539e5dc1b1d9a639dab75efe18a4515a6e98e302` /
  `a4e50c688dd64eb6a40e2575fff7e94f622ce4f84df16847837861056f22b0ae`
- output archive:
  logsに必要なmetrics、生成物path、SHAがそろっていたため未取得
- model / submission SHA:
  対象外

## 解釈

rate履歴除去は、exp408で特定したforward-transition/prior-hysteresis episodeを
軽減するsignalを持つ。しかしTVT-only状態への縮約はmatched controlで
桁違いのnegative transferを起こし、well / fold再現性もなかった。
stationary nonzero-rate mixtureはdz-onlyより安全でもなく、
本実験の非ゼロrate supportはrate履歴除去の欠点を補えなかった。

technical gateが全PASSしているため、数値異常、runtime、truth leakage、
SHA不整合ではなく、事前固定した科学介入自体のnegative resultと判断する。

## 次

Stage 1 eligible variantは空。rate重み、support、noise、emission、grid、gate、
blend、well / row selectorを同じfixed32で救済せず、branchを閉じる。
Stage 1、inference、submissionへ進めない。同familyの後続候補は追加しない。
