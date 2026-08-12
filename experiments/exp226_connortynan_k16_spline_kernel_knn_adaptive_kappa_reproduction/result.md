# exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction 結果

## 状態

- Kaggle train v1 完了。
- Kaggle inference v1 完了。
- submit-check PASS。
- Code submit 完了。
- Public LB: 9.837
- 採用判断: 不採用。

## 実装要約

Connor Tynan 公開 notebook の deterministic v6 fallback を source-port した。対象は K=16 segment spline、raw/smoothed donor field、XY local-linear kNN、adaptive kappa、near-strike ANCC local theta、typewell GR correction、U-projection。外部 weight 前提の v7 neural committee と v8 LightGBM meta-layer は Stage 1 では無効化した。

## Kaggle Train v1

- Kernel: `kentookumura/exp226-k16-kappa-repro-train`
- Version: 1
- Status: COMPLETE
- CV: 9.427109596582213
- MAE: 6.148527797393756
- Bias: -0.29961900506691624
- within10: 0.8077095361535142
- within25: 0.9767446469849674
- Rows: 3,783,989
- Wells: 773
- OOF decompressed SHA256: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- Active variants: 1
- LightGBM configs: 0
- Boosters: 0

## Kaggle Inference v1

- Kernel: `kentookumura/exp226-k16-kappa-repro-inference`
- Version: 1
- Status: COMPLETE
- submission rows: 14,151
- submission SHA256: `b71e15f7dc7e66f7be70db4a81d9ec72e1001ff2ba13907c3aba24938e906047`
- TVT range: 11590.507200740965 - 12237.326047949082
- TVT mean/std: 11905.948938813102 / 277.98497549360616
- submit-check: PASS
- local output: `/tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/inference_v1`

## Code Submission

- Ref: `54491603`
- Status: COMPLETE
- Public LB: 9.837
- Private LB: -
- `SUBMISSIONS.md`: v060

## 現時点の判断

group-safe CV 9.427 は exp206 v4 CV 52.507 より大幅に良く、公開 K16 fallback の再現は exp206 線形外挿とは明確に別物として成立した。一方、Public LB 9.837 は exp218 ML anchor 7.843、exp148 CPU runtime anchor 7.921、ensemble anchor exp082 7.601 より悪いため採用しない。

## 2026-07-27 オフセット根本原因

保存済みOOF 3,783,989行 / 773 wellsを、geometry、GR、U projection、
suffix距離、K16/H64--H1024 quotient、persistent episode、donor距離、
raw GR欠損、fold別kappa、公開ソース移植parityの観点で再監査した。

根本原因は、最後の既知TVTを一度だけabsolute anchorとし、その後は空間donor由来の
相対的な`TVT+Z`増分を累積する一方、unknown suffix内で再anchorしない構造にある。
target wellとdonor fieldの小さなsigned rate mismatchが長距離で積分され、
後続K16 segmentへ低周波vertical offsetとして継承される。

主要証拠:

- global bias除去のMSE説明は`0.1010%`だけ。
- K16 segment mean offset除去のMSE説明は`98.5617%`、
  RMSEは`9.427110 -> 1.130603`。
- segment meanと前segment end errorのPearsonは`0.982951`、
  境界jump中央値は`0.008190 ft`。
- RMSEはsuffix 0--50の`1.741257`から2000+の`11.151214`へ成長。
- persistent 645 episodesはrows`18.9943%`でSSE`82.0073%`を占め、
  onset一行jump中央値は`0.021148 ft`。
- geometry / pre-U / final RMSEは
  `10.077950 / 9.500816 / 9.427110`。GRとUはpooledでは改善する。
- donor distance max下位/上位quartileのwell RMSE中央値は
  `4.099483 / 7.774613`。
- 公開deterministic v6とportの9数値核は固定synthetic入力で最大差`0.0`。

したがってglobal calibration、誤anchor、行順、特定fold、K16境界不連続、
K=16単独、GR単独、U projection単独、v6移植ミスは原因として棄却する。
外部weight依存のv7/v8 learned residual layerをexp226が含まないことは性能上限の
一部だが、v6 reproductionの実装不具合ではない。

詳細は
`docs/analysis/exp226_offset_root_cause_audit_20260727.md`、
機械可読生成物は
`studies/exp226_offset_root_cause_audit_20260727/`を正とする。
