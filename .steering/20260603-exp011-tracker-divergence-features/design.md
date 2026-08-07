# 設計

## アプローチ

PF / beam 系の公開 notebook 知見を、standalone 予測や stochastic snapshot ではなく fold-safe な補助特徴に変換する。

実装は exp010 をコピーして、失敗した trajectory full variants を selected path から外す。paired typewell GR に対して scale 別の deterministic candidate TVT path を作り、GR match score、best shift、direction、DTW cost、hold weight、candidate divergence を row-wise features として HGB residual model に渡す。

## 実験範囲

- 対象実験: `exp011_tracker_divergence_features`
- 親実験: `exp010_trajectory_drift_ablation`
- 変更する変数: tracker feature group の有無、all-GR/no-GR placement、限定 trajectory direction の併用
- 固定する変数: GroupKFold、HGB hyperparameters、sampling caps、residual shrink、baseline anchor、GR NCC disabled、formation guide disabled

## Tracker Features

- Scale candidates: `s3`, `s5`, `s8`, `s12`
- Candidate path: `last_known_tvt + recent_slope * slope_scale * direction * delta_md + shift`
- Search: bounded shift grid and optional reverse direction
- Signal columns:
  - scale score, shift, direction, slope scale, DTW cost
  - scale predicted TVT, delta from anchor, difference vs recent-linear prior
  - typewell GR interpolation and absolute GR error
  - best/second score, confidence, entropy, hold weight
  - mean/std/range of candidate TVT deltas
  - eval length and Z span selector flags

## リスク

- リークリスク: typewell GR と 見えない test で使える columns だけを使う。validation fold の target や formation columns は使わない。
- CV/LB 不一致リスク: exp010 audit の hard/no-GR、steep trajectory、high GR missing、long eval group で局所悪化を確認する。
- ランタイム/メモリリスク: stochastic PF、多 seed、多 particle は使わない。DTW cost は downsampled bounded band に限定する。
