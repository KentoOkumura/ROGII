# exp405_geometry_reinjected_interval_semimarkov_fusion 結果

## 状態

Kaggle CPU version 2のfull saved-OOFはtechnical PASS /
constrained-oracle PASS / scientific FAILで完了した。
exp405は閉鎖し、exp406 fixed16 Stage 0を解禁する。
current-test実装、inference、submissionは行わない。

## 仮説

exp293 fixed12をH256 block / H512 minimum durationのsemi-Markov posteriorで
融合し、exp226 geometryへdocking非依存で再注入すれば、保存済み物理pathの
oracle headroomをtarget-freeに回収できる。

## 固定比較

- control: exp263 `8.2383315465 ft`
- support reference: exp293 H512 oracle `3.683763 ft`
- candidate: exp405 posterior-mean TVT 1本
- negative controls: circular GR / block-order-permuted GR
- primary gate: `<=6.90 ft`かつexp263比5/5 folds改善

## full saved-OOF結果

| 指標 | exp405 | exp263 anchor | 差 |
| --- | --- | --- | --- |
| pooled RMSE | `8.451060 ft` | `8.238332 ft` | `+0.212728 ft` |
| constrained oracle | `3.606822 ft` | `8.238332 ft` | `-4.631510 ft` |
| by-well delta p95 | - | - | `+1.744584 ft` |
| worst-well regression | - | - | `+3.515814 ft` |

全5 foldsでexp405がanchorより悪化した。主要scopeも
`1000_plus +0.230850 ft`、`hidden_like_spatial +0.373085 ft`、
`hidden_like_typewell_purged +0.354028 ft`とすべて悪化した。

## Gate解釈

- technical 17/17: PASS
- constrained oracle 2/2: PASS
- scientific: FAIL
- runtime / peak RSS: `1,434.099 sec / 2.220737 GB`でPASS
- geometry mass 3項目とphysical continuityはPASS
- circular / block-permutation controlとの差はそれぞれ
  `0.000346 / 0.000230 ft`だけで、要求`>=0.05 ft`を満たさなかった。

候補bank自体にはoracle `3.606822 ft`の大きなheadroomがある一方、
実GR morphology evidenceは正しい候補を識別できていない。
real evidenceが2つのnegative controlとほぼ同じなのが決定的で、
geometry再注入量を確保しても精度改善にはつながらなかった。

## SHA

- decision:
  `e159cfb712a6ed81e78f4524febbf0d995375124a473a5056aad3c1347b648f0`
- summary file:
  `9992612ba22cb615e3fd01795450c5b44bcf9cc24dcf8d6b7b331b3579d7bc77`
- gate file:
  `d601df56ca58ec137a67a622c412750bf3a7a6fc455440d7823c5e14885947bf`
- score logical:
  `7b6f08efc27f2245b48995235a6dfca4ea06aa3d9035385251cfdef85c1920d9`
- posterior logical:
  `598690cb6645f692397e3dbe2ad98c469a2040a5dfb2b45152bec0c30ab908e6`
- prediction logical:
  `02245e4c08e7c93de82cf16051a412ecad51c2dbb0114bd237733b4d78fd41b4`
- prediction decompressed:
  `136f4e0e65c8df9e2e22bc94573948f99c1aa5cf5aa8cd00d6abc248cca2add1`

## 判断

設計で固定した分岐どおり、same-OOF rescueやgate調整は行わずexp405を閉じる。
current-test実装資格は得られない。独立familyの
`exp406_loop_closed_multiwell_rgt_fixed16_stage0`は実装可能状態へ解禁するが、
実装・実行には別途ユーザー承認を必要とする。

実行後のローカルKaggle packageは`run_on_push: false`、
`run_stage: implementation_only`へ戻しており、full OOFを再実行しない
fail-closed状態である。Kaggle version 2へは追加pushしていない。
