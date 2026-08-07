# exp174_typewell_late_range_ml_posthoc_clip_audit 結果

## 仮説

`known_last_pct` が高い well で ML 予測が typewell TVT range の前半へ落ちる行は、target-free な lower-bound shrink / clip で救済できる可能性がある。

## 設定

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 検証: fixed OOF prediction no-training posthoc audit
- メトリック: RMSE / MAE / within10 / changed rows / by-well regression / bucket metrics
- シード: 42
- GPU: なし
- LightGBM config / fold / booster: 0 / 0 / 0
- control 再学習: なし

## 結果

| メトリック | 値 |
| --- | --- |
| baseline CV | 8.501281182 |
| best non-baseline | no-op policy。発火 0 行、baseline と同一 |
| best firing policy | `fixed_lb0p7_klp0p75_a0p25` |
| best firing RMSE | 8.501891、baseline から +0.000609 |
| max firing rows | `known_last_m0p05_klp0p75_a0p25`: 13,657 rows / 14 wells、RMSE 8.518425、+0.017144 |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: false。train-side OOF posthoc audit であり submission anchor ではない。
- seed policy: no_new_rng_posthoc_grid
- kernel: `kentookumura/exp174-typewell-late-clip-train` v1
- source prediction SHA: output summary は OOF gzip 取得中に `kaggle kernels output` が code 137 で落ちたため未記録。small metrics CSV は取得済み。
- generated OOF prediction SHA: OOF gzip は 0 byte partial download のため記録対象外。
- candidate metrics SHA256: `faf8c6f6332cb2dc605f550012f7739bb10b03864f5de2e9c3156e4222fe0328`
- bucket metrics SHA256: `22a9f54901856364383b59c7896ceb7d4d178fbfd3211449e463e36e3d9c0fdc`
- by-well metrics SHA256: `86f43d42c88f5f8f4de8794cc5cfc8b9debb50bf552f65b41be9c802eb2c3bae`
- changed summary SHA256: `b5646c94f821d3fff78f15a26eaa1804e51280ae7befc6ab26aec99100aef005`
- model SHA / manifest SHA: 対象外
- submission SHA: 対象外
- rerun result: なし

## 解釈

ML の exp148 OOF では、late known prefix かつ front-half pred へ落ちるケースは lower bound `0.55/0.60/0.65` では発火しなかった。`0.70` や `known_last_pct - 0.10/0.05` まで強めると 1,325-13,657 rows で発火するが、いずれも global RMSE を悪化させた。

したがって、この仮説は exp148 ML posthoc clip としては不採用。inference port / submit は行わない。typewell late-range prior を使うとしても、ML 予測の hard lower bound ではなく PF/Beam candidate の confidence feature / selector prior に限定する。

## 次

`typewell_late_range_ml_posthoc_clip_audit` は完了/不採用として backlog から外す。PF/Beam 側の `candidate_pct` prior は、exp174 negative を前提に優先度を下げ、hard clip / hard invalid は避ける。
